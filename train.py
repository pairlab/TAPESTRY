"""TAPESTRY training entry point (Hydra + Accelerate).

    accelerate launch train.py --config-name train_libero                          # the paper's LIBERO recipe from CogVideoX-2b
    accelerate launch train.py --config-name train_tapestry                        # the paper's pretraining recipe
    accelerate launch train.py --config-name finetune init_from=/path/to/tapestry.pt  # fine-tune on your HDF5 demos

Checkpoints (weights, optimizer, scheduler, resolved config) are written to checkpoints/<experiment_name>/.
resume=true continues an interrupted run from its latest checkpoint; init_from=<path> starts from a checkpoint's
weights with a fresh optimizer.
"""


import os
from datetime import timedelta

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from einops import rearrange

from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import lpips

from tapestry.models.base import Tapestry
from tapestry.models.model_loading import load_checkpoint_cpu
from tapestry.models.encoders.text import download_t5
from tapestry.models.unified_model_wrapper import UnifiedModelWrapper
from tapestry.loaders.libero_dataset import LiberoDataset, custom_collate
from tapestry.loaders.video_only_dataset import VideoOnlyDataset
from tapestry.loaders.hdf5_demo_dataset import HDF5DemoDataset
from tapestry.video_diffusion.inference_utils import log_videos_to_wandb
from tapestry.losses import ModalityLossManager, DiffusionLoss
from tapestry.utils.training_manager import ExperimentTracker, TrainingState, load_init_weights
from tapestry.eval_utils import run_direct_libero_evaluation, log_evaluation_results
from tapestry.utils.data_utils import (
    WeightedConcatDataset,
    compute_video_metrics,
    create_dataset_from_config,
)


def setup_accelerator(mixed_precision, static_graph):
    """Accelerator for mixed-precision training. A multi-process launch (accelerate launch, torchrun) runs DDP over NCCL,
    and a single-process launch trains on one GPU however many are visible."""
    # find_unused_parameters: a batch only reaches the heads of its active modalities
    ddp = DistributedDataParallelKwargs(find_unused_parameters=True, static_graph=static_graph)
    # Long timeout: the other ranks wait at a barrier while rank 0 runs LIBERO rollouts between epochs
    pg = InitProcessGroupKwargs(timeout=timedelta(hours=5))
    return Accelerator(mixed_precision=mixed_precision, kwargs_handlers=[ddp, pg])


def split_batch(batch, active_inputs, active_outputs, device):
    """Split a collated batch into model inputs and loss targets, with tensors moved to `device`.

    Every active key brings along its `<key>/attention_mask` and `<key>_mask` (the per-sample modality mask from
    custom_collate) when the batch has them.
    """
    def to_dev(v):
        return v.to(device) if isinstance(v, torch.Tensor) else v

    def pick(keys):
        keys = [k for key in keys for k in (key, f"{key}/attention_mask", f"{key}_mask")]
        return {k: to_dev(batch[k]) for k in keys if k in batch}

    inputs, targets = pick(active_inputs), pick(active_outputs)
    for k in ["active_modalities", "video/view_mask"]:   # loss masks: active modalities per sample, real camera views
        if k in batch:
            targets[k] = to_dev(batch[k])
    return inputs, targets


def _mse_at_t_means(stats):
    """{target: [(t_bucket, mean mse)]} over the non-empty timestep buckets of pop_per_timestep_mse_stats()."""
    out = {}
    for tk, (sums, counts) in stats.items():
        means = (sums / counts.clamp(min=1).to(sums.dtype)).cpu().tolist()
        counts = counts.cpu().tolist()
        out[tk] = [(t, means[bi]) for bi, t in enumerate(DiffusionLoss.T_BUCKETS) if counts[bi] > 0]
    return out


def train_epoch(model, optimizer, train_loader, loss_manager, accelerator, epoch, start, train_global_iter,
                active_inputs, active_outputs, log_interval, tracker, scheduler, step_ckpt_interval):
    """Train for one epoch from its batch `start` on, which is nonzero when resuming from a mid-epoch latest.pt.
    Returns ({target: mean loss over the batches run, 'total': ...}, updated train_global_iter)."""
    model.train()
    sums = {}
    loader = accelerator.skip_first_batches(train_loader, start) if start > 0 else train_loader   # skips without loading
    pbar = tqdm(loader, desc=f"Epoch {epoch}", initial=start, total=len(train_loader)) if accelerator.is_main_process else loader
    for i, batch in enumerate(pbar, start):
        inputs, targets = split_batch(batch, active_inputs, active_outputs, accelerator.device)
        optimizer.zero_grad()
        model_outputs = model(inputs, loss_manager=loss_manager, targets=targets)
        noise_dict = {k: v for k, v in model_outputs.items() if k.endswith("_noise")}
        losses = loss_manager(model_outputs, targets, noise_dict)
        if losses["total"].grad_fn is None:
            raise RuntimeError(f"batch {i} produced no trainable loss (active_modalities={targets['active_modalities']})")

        accelerator.backward(losses["total"])
        if accelerator.sync_gradients:
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad()
                if accelerator.is_main_process:
                    print(f"  [NaN guard] Skipping step {i}: non-finite grad_norm={grad_norm.item():.4g}")
                continue
            if accelerator.is_main_process and i % log_interval == 0:
                print(f"  Step {i:4d} grad_norm (pre-clip): {grad_norm.item():.4f}")
                if wandb.run is not None:
                    wandb.log({"train_step/grad_norm": grad_norm.item()}, commit=False)
        optimizer.step()

        if accelerator.is_main_process:
            train_global_iter += 1
            # mid-epoch latest.pt for preemptible jobs (training.save_interval)
            if step_ckpt_interval > 0 and train_global_iter % step_ckpt_interval == 0:
                tracker.update_state(epoch=epoch, epoch_batches=i + 1, train_global_iter=train_global_iter)
                tracker.save(accelerator.unwrap_model(model), optimizer, scheduler, None, "latest.pt")
                print(f"Saved step checkpoint at iter {train_global_iter} (latest.pt)")

        # Every rank pops the per-timestep MSE buckets so they reset together; only the main rank logs its means
        mse_at_t = _mse_at_t_means(loss_manager.pop_per_timestep_mse_stats()) if i % log_interval == 0 else {}
        if accelerator.is_main_process:
            if i % 10 == 0:
                pbar.set_postfix({"total_loss": f"{losses['total'].item():.4f}"})
            if i % log_interval == 0:
                print(f"  Step {i:4d}: " + " | ".join(f"{k}: {v.item():.4f}" for k, v in losses.items()))
                for tk, pairs in mse_at_t.items():
                    if pairs:
                        print(f"  Step {i:4d} mse_at_t[{tk}]: " + " ".join(f"t{t}={m:.4f}" for t, m in pairs))
                if wandb.run is not None:
                    step_log = {f"train_step/{k}": v.item() for k, v in losses.items()}
                    step_log.update({f"train_step/mse_at_t/{tk}/t{t}": m for tk, pairs in mse_at_t.items() for t, m in pairs})
                    step_log.update({"epoch": epoch, "train_global_iter": train_global_iter})
                    wandb.log(step_log)

        for k, v in losses.items():
            sums[k] = sums[k] + v.detach() if k in sums else v.detach()

    keys = [*loss_manager.target_keys, "total"]
    return {k: (sums[k] / len(loader)).item() if k in sums else 0.0 for k in keys}, train_global_iter


def validate(model, val_loader, loss_manager, accelerator, epoch, val_global_iter, active_inputs, active_outputs,
             log_interval, log_steps, max_batches):
    """Mean validation losses over all ranks. Returns ({target: mean loss, 'total': ...}, updated val_global_iter).

    log_steps: log step losses to wandb every log_interval batches and advance val_global_iter (off for the per-dataset
    passes). max_batches: cap on batches per rank, None for the whole loader.
    """
    model.eval()
    if len(val_loader) == 0:
        raise RuntimeError("Validation dataloader has zero batches on this rank: the validation split is empty or "
                           "drop_last removed a too-small per-rank shard.")
    sums = {k: torch.tensor(0.0, device=accelerator.device) for k in [*loss_manager.target_keys, "total"]}
    n_batches = torch.tensor(0, device=accelerator.device)
    loss_manager.pop_per_timestep_mse_stats()  # discard what training left in the per-timestep MSE buckets

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Validation") if accelerator.is_main_process else val_loader):
            if max_batches is not None and i >= max_batches:
                break
            inputs, targets = split_batch(batch, active_inputs, active_outputs, accelerator.device)
            model_outputs = model(inputs, targets=targets, loss_manager=loss_manager)
            noise_dict = {k: v for k, v in model_outputs.items() if k.endswith("_noise")}
            losses = loss_manager(model_outputs, targets, noise_dict)
            for k, v in losses.items():
                sums[k] += v.detach()
            n_batches += 1

            if log_steps and accelerator.is_main_process:
                if i % log_interval == 0 and wandb.run is not None:
                    step_log = {f"val_step/{k}": v.item() for k, v in losses.items()}
                    step_log.update({"epoch": epoch, "val_global_iter": val_global_iter})
                    wandb.log(step_log)
                val_global_iter += 1

    n_batches = accelerator.reduce(n_batches, reduction="sum")
    if n_batches.item() == 0:
        raise RuntimeError("Validation produced zero batches")
    val_losses = {k: (accelerator.reduce(v, reduction="sum") / n_batches).item() for k, v in sums.items()}

    # per-timestep MSE buckets: every rank pops and all-reduces, the main rank logs
    stats = {tk: (accelerator.reduce(s, reduction="sum"), accelerator.reduce(c, reduction="sum"))
             for tk, (s, c) in loss_manager.pop_per_timestep_mse_stats().items()}
    if stats and accelerator.is_main_process and wandb.run is not None:
        log = {f"val/mse_at_t/{tk}/t{t}": m for tk, pairs in _mse_at_t_means(stats).items() for t, m in pairs}
        if log:
            wandb.log({**log, "epoch": epoch})
    return val_losses, val_global_iter


DATASET_CLASSES = {"LiberoDataset": LiberoDataset, "HDF5DemoDataset": HDF5DemoDataset, "VideoOnlyDataset": VideoOnlyDataset}


def active_keys(cfg):
    """(input keys, target keys) the model consumes, from the sequence layout and its loss mappings."""
    # the decoder conditions on the first frame's latents, which do not appear in the sequence
    active_inputs = [*cfg["sequence"]["prefix_order"], "video/history_frames"]
    active_outputs = [m["target"] for m in cfg["sequence"]["loss_mappings"].values()]
    return active_inputs, active_outputs


def action_inference_steps(wandb_cfg):
    """training.wandb.action_inference_steps, checked at startup so that a bad value does not crash after the first epoch."""
    steps = list(wandb_cfg["action_inference_steps"])
    if not steps or len(set(steps)) != len(steps) or any(type(n) is not int or n < 1 for n in steps):
        raise ValueError(f"training.wandb.action_inference_steps must be a non-empty list of unique ints >= 1, got {steps}")
    return steps


def check_rollout_video(cfg):
    """evaluation.save_rollout_video logs to wandb, checked at startup so that the conflict does not surface after the first evaluation."""
    ev = cfg["evaluation"]
    if ev["enabled"] and ev["save_rollout_video"] and not cfg["training"]["wandb"]["use_wandb"]:
        raise ValueError("evaluation.save_rollout_video logs rollout videos to wandb: set training.wandb.use_wandb=true "
                         "or evaluation.save_rollout_video=false")


def resolve_wandb_run(tracker, experiment_name):
    """(run id, run name) for wandb.init. resume=true continues the checkpoint's run unless experiment_name was changed to fork it."""
    if not tracker.is_resume:   # includes checkpoint=... with resume=false: restores the state but starts a new run
        return None, experiment_name
    ckpt = load_checkpoint_cpu(tracker.checkpoint_path)
    state = TrainingState.from_dict(ckpt["training_state"])
    # a checkpoint written with wandb disabled has no run name
    ckpt_name = state.wandb_run_name if state.wandb_run_name is not None else ckpt["config"]["experiment_name"]
    if ckpt_name != experiment_name:
        return None, experiment_name
    if state.wandb_run_id is None:
        raise ValueError(f"Checkpoint at {tracker.checkpoint_path} is missing wandb_run_id; cannot resume the wandb run")
    return state.wandb_run_id, experiment_name


def init_wandb(cfg, run_id, run_name):
    """Start (run_id None) or resume the wandb run and define the step metric of every logged namespace."""
    if run_id is None:
        os.environ.pop("WANDB_RUN_ID", None)   # an id left over from the launcher would resume that run instead
    wandb.init(
        project=cfg["training"]["wandb"]["project"],
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        name=run_name,
        id=run_id,
        resume="allow" if run_id is not None else None,
        allow_val_change=True,
        settings=wandb.Settings(init_timeout=600),
    )
    if run_id is not None and wandb.run.id != run_id:
        raise RuntimeError(f"Wandb resumed with unexpected run id {wandb.run.id} (expected {run_id})")
    if wandb.run.name != run_name:
        raise RuntimeError(f"Wandb resumed with unexpected run name '{wandb.run.name}' (expected '{run_name}')")
    print(f"Wandb run: {wandb.run.name} (id: {wandb.run.id})")
    wandb.define_metric("epoch")
    wandb.define_metric("train_step/*", step_metric="train_global_iter")
    wandb.define_metric("val_step/*", step_metric="val_global_iter")
    for namespace in ("train", "val", "eval", "metrics", "jitter", "action_mse"):
        wandb.define_metric(f"{namespace}/*", step_metric="epoch")
    if "SLURM_JOBID" in os.environ:
        wandb.config.update({"slurm_job_id": os.environ["SLURM_JOBID"]}, allow_val_change=True)


def build_datasets(datasets_cfg, is_validation, quick):
    """(WeightedConcatDataset, names) over the entries of datasets_cfg with a nonzero batch_weight; names are their config keys."""
    names = [name for name, ds_cfg in datasets_cfg.items() if ds_cfg["batch_weight"] != 0]
    if not names:
        raise ValueError(f"Every dataset in {list(datasets_cfg)} has batch_weight 0")
    datasets = []
    for name in names:
        if datasets_cfg[name]["is_validation"] != is_validation:
            raise ValueError(f"dataset '{name}' has is_validation={datasets_cfg[name]['is_validation']}, expected {is_validation}")
        ds = create_dataset_from_config(datasets_cfg[name], DATASET_CLASSES)
        if quick:
            ds.index_map = ds.index_map[:10]   # before WeightedConcatDataset, which sizes itself at construction
        datasets.append(ds)
    return WeightedConcatDataset(datasets, [datasets_cfg[name]["batch_weight"] for name in names]), names


def check_samples(dataset, names, accelerator):
    """Load one sample from every sub-dataset so that a broken dataset fails at startup instead of mid-epoch."""
    for name, ds in zip(names, dataset.datasets):
        shapes = {k: tuple(v.shape) for k, v in ds[0].items() if isinstance(v, torch.Tensor)}
        accelerator.print(f"  {name} ({type(ds).__name__}, {len(ds)} samples): {shapes}")


def build_loaders(train_set, val_set, training_cfg):
    """(train loader, val loader, [one val loader per sub-dataset]). The list is empty when there is a single val dataset."""
    bs = training_cfg["batch_size"]
    kw = dict(
        num_workers=training_cfg["num_workers"],
        persistent_workers=training_cfg["persistent_workers"],
        prefetch_factor=training_cfg["prefetch_factor"],
        pin_memory=True,
        collate_fn=custom_collate,
    )
    # Sub-datasets are drawn in proportion to their batch_weight, with replacement (validation too).
    # Validation keeps its partial batches: dropping them can leave a rank without a single batch
    train_kw = dict(batch_size=bs, sampler=train_set.create_sampler(), drop_last=training_cfg["drop_last"])
    val_kw = dict(batch_size=bs, sampler=val_set.create_sampler(), drop_last=False)
    ds_val = [DataLoader(ds, batch_size=bs, shuffle=False, drop_last=False, **kw) for ds in val_set.datasets]
    return DataLoader(train_set, **train_kw, **kw), DataLoader(val_set, **val_kw, **kw), ds_val if len(ds_val) > 1 else []


def build_optimizer(model, training_cfg):
    """AdamW. pretrained_learning_rate_multiplier scales the learning rate of the decoder named by pretrained_decoder_key."""
    lr, wd, mult = training_cfg["learning_rate"], training_cfg["weight_decay"], training_cfg["pretrained_learning_rate_multiplier"]
    if mult is None:
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    pretrained = list(model.decoders[training_cfg["pretrained_decoder_key"]].parameters())
    ids = {id(p) for p in pretrained}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return optim.AdamW([{"params": rest, "lr": lr}, {"params": pretrained, "lr": lr * mult}], weight_decay=wd)


def build_scheduler(optimizer, training_cfg):
    """Per-epoch learning-rate scheduler named by training.lr_scheduler, or None (the paper recipes train at a constant rate)."""
    kind = training_cfg["lr_scheduler"]
    if kind is None:
        return None
    if kind == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_cfg["num_epochs"])
    raise ValueError(f"Unknown training.lr_scheduler: {kind}")


def unscaled(losses, loss_manager, prefix):
    """wandb entries for epoch-average losses with the objective weights divided out; a zero-weight objective logs 0."""
    out = {k: v / loss_manager.losses[k].weight if loss_manager.losses[k].weight != 0 else 0.0
           for k, v in losses.items() if k != "total"}
    return {f"{prefix}/total_loss": sum(out.values()), **{f"{prefix}/{k}_loss": v for k, v in out.items()}}


def print_losses(title, losses):
    print(title)
    for k, v in losses.items():
        print(f"  {k:25s}: {v:.6f}")


def sample_batch(dataset, device):
    """One random sample from every sub-dataset of a WeightedConcatDataset, collated and moved to `device`."""
    batch = custom_collate([ds[np.random.randint(len(ds))] for ds in dataset.datasets])
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def rows_with(batch, key, names):
    """The rows of a collated batch that have `key`, with their names. custom_collate adds `<key>_mask` [B] when some lack it."""
    if f"{key}_mask" not in batch:
        return batch, names
    idx = torch.where(batch[f"{key}_mask"])[0].tolist()
    rows = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            rows[k] = v[idx]
        else:
            rows[k] = [v[i] for i in idx]
    return rows, [names[i] for i in idx]


def chunk_stats(act):
    """Speed and jitter of action chunks [B, T, D] from their first and second differences over time."""
    dv = torch.diff(act.float(), dim=1)   # [B, T-1, D]
    da = torch.diff(dv, dim=1)            # [B, T-2, D]
    speed, jitter = dv.norm(dim=2).mean().item(), da.norm(dim=2).mean().item()
    ratio = jitter / (speed + 1e-6)
    return {"jitter_l2": jitter, "jitter_ratio": ratio, "smoothness": 1 / (1 + ratio), "speed_l2": speed}


def log_videos(model, loss_manager, batch, names, mapping, wandb_cfg, lpips_fn, epoch, stage):
    """Generate the future frames of every row of `batch` and log them to wandb beside the ground truth, with PSNR / SSIM / LPIPS."""
    key = mapping["target"]
    batch, names = rows_with(batch, key, names)
    dec, loss = model.decoders[mapping["decoder"]], loss_manager.losses[key]
    seq_info, seq_tokens = model.encode(batch)
    cond = seq_info.get_tokens(seq_tokens, mapping["tokens"])   # [N, L, D]
    target = batch[key]                                         # [N, V, T, C, H, W]
    # custom_collate adds video/view_mask only when the rows have different numbers of views
    first = dec.encode_multiview_to_latents(target, view_mask=batch.get("video/view_mask"), get_single_frame=0)   # [N, C_lat, 1, h, V*w]
    N, V, T, C, H, W = target.shape
    for g in wandb_cfg["guidance_scales"]:
        gen = torch.cat([   # one row at a time, so that memory does not grow with the number of datasets
            dec.generate_multiview_video(cond[i:i + 1], first[i:i + 1] if loss.predict_future_frames else None, V, H, W, T,
                                         loss.noise_scheduler, wandb_cfg["video_inference_steps"], g)
            for i in range(N)
        ])   # [N, V, T, C, H, W]
        metrics = compute_video_metrics(gen[:, 0], target[:, 0], lpips_fn)   # on the first view, which is never a padded one
        log_videos_to_wandb(rearrange(gen, "n v t c h w -> n t c h (v w)"), rearrange(target, "n v t c h w -> n t c h (v w)"),
                            names, metrics, epoch + 1, stage, g)


def log_action_metrics(policy, scheduler, batch, names, key, steps, epoch, stage):
    """Log the jitter of ground-truth and predicted action chunks for every entry of `steps` (denoising steps), and the
    per-dataset MSE of the prediction at the last entry."""
    batch, names = rows_with(batch, key, names)
    gt = batch[key]   # [N, T, D]
    cond = policy.encode(batch)
    base = f"jitter/{stage}/action_chunk"
    log = {f"{base}/gt_{k}": v for k, v in chunk_stats(gt).items()}
    for n in steps:
        pred = policy(cond, scheduler, num_inference_steps=n)   # [N, T, D]
        log.update({f"{base}/pred_{k}_steps_{n}": v for k, v in chunk_stats(pred).items()})
        mse = ((pred - gt) ** 2).mean(dim=(1, 2))   # [N]
        log.update({f"action_mse/{stage}/{name}_steps_{n}": m for name, m in zip(names, mse.tolist())})
    wandb.log({**log, "epoch": epoch + 1})


def log_media(model, loss_manager, dataset, names, cfg, steps, lpips_fn, epoch, stage, device):
    """Log generated videos and action-chunk metrics for one random sample of every sub-dataset of `dataset`."""
    mappings = cfg["sequence"]["loss_mappings"]
    batch = sample_batch(dataset, device)
    model.eval()
    with torch.no_grad():
        if "video" in mappings and mappings["video"]["target"] in batch:
            log_videos(model, loss_manager, batch, names, mappings["video"], cfg["training"]["wandb"], lpips_fn, epoch, stage)
        if "action" in mappings and mappings["action"]["target"] in batch:
            key = mappings["action"]["target"]
            log_action_metrics(UnifiedModelWrapper(model), loss_manager.losses[key].noise_scheduler, batch, names, key,
                               steps, epoch, stage)


def save_epoch(tracker, model, optimizer, scheduler, epoch, train_global_iter, val_global_iter, train_losses, val_losses,
               save_numbered):
    """Write latest.pt (and epoch_<n>.pt when save_numbered), plus best_checkpoint.pt when the validation action loss
    improved (the total validation loss for a model without an action objective)."""
    state = tracker.state
    has_action = "action/future_frames" in val_losses
    metric = val_losses["action/future_frames"] if has_action else val_losses["total"]
    is_best = metric < (state.best_val_action_future_frames_loss if has_action else state.best_val_loss)
    best_action = min(state.best_val_action_future_frames_loss, metric) if has_action else state.best_val_action_future_frames_loss
    tracker.update_state(
        epoch=epoch + 1, epoch_batches=0, train_global_iter=train_global_iter, val_global_iter=val_global_iter,
        train_loss=train_losses["total"], val_loss=val_losses["total"],
        best_val_loss=min(state.best_val_loss, val_losses["total"]), best_val_action_future_frames_loss=best_action,
    )
    metrics = {"train_losses": train_losses, "val_losses": val_losses}
    if is_best:
        path = tracker.save(model, optimizer, scheduler, metrics, "best_checkpoint.pt")
        print(f"New best validation loss {metric:.6f}; saved {path}")
    names = ["latest.pt", f"epoch_{epoch + 1}.pt"] if save_numbered else ["latest.pt"]
    for name in names:
        print(f"Saved checkpoint to {tracker.save(model, optimizer, scheduler, metrics, name)}")


@hydra.main(config_path="cfg", version_base=None)
def main(cfg: DictConfig):
    tracker = ExperimentTracker(cfg)
    cfg = tracker.get_merged_config()   # on resume: the checkpoint's config with this launch's overrides on top
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    training_cfg, wandb_cfg, logging_cfg = cfg["training"], cfg["training"]["wandb"], cfg["logging"]
    steps = action_inference_steps(wandb_cfg)
    check_rollout_video(cfg)

    # Rank 0 downloads the text encoder before joining the process group, so the other ranks find it in the cache.
    # LOCAL_RANK is unset for a plain `python train.py`.
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        download_t5(cfg_dict["encoders"]["text"]["model_name_or_path"])
    accelerator = setup_accelerator(cfg_dict["training"]["mixed_precision"], cfg_dict["use_static_graph"])
    accelerator.wait_for_everyone()
    accelerator.print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, {accelerator.num_processes} process(es), "
                      f"mixed precision {accelerator.mixed_precision}")

    seed = cfg_dict["training"]["seed"] + accelerator.process_index   # every rank draws different noise and augmentations
    torch.manual_seed(seed)
    np.random.seed(seed)

    # its own container: the model keeps it as model.cfg
    model = Tapestry(OmegaConf.to_container(cfg, resolve=True), from_checkpoint=False)
    if cfg["init_from"] is not None:
        summary = load_init_weights(model, cfg["init_from"])
        accelerator.print(f"init_from {cfg['init_from']}: {summary}")
    accelerator.print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    active_inputs, active_outputs = active_keys(cfg)
    accelerator.print(f"Model inputs: {active_inputs}\nLoss targets: {active_outputs}")

    ckpt_dir = tracker.checkpoint_dir   # resolved on every rank: rank 0 picks the directory and broadcasts it
    accelerator.print(f"Saving checkpoints to {ckpt_dir}")
    lpips_fn = None
    if accelerator.is_main_process and wandb_cfg["use_wandb"]:
        init_wandb(cfg, *resolve_wandb_run(tracker, cfg["experiment_name"]))
        # for the logged video metrics; built here because it downloads AlexNet weights on first use, which should not fail after an epoch
        lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(accelerator.device)

    ds_cfg = training_cfg["dataset"]
    train_set, train_names = build_datasets(ds_cfg["datasets"], False, cfg["quick"])
    val_set, val_names = build_datasets(ds_cfg["val_datasets"], True, cfg["quick"])
    accelerator.print("Training datasets:")
    check_samples(train_set, train_names, accelerator)
    train_loader, val_loader, ds_val_loaders = build_loaders(train_set, val_set, training_cfg)

    loss_manager = ModalityLossManager(training_cfg["objectives"])
    optimizer = build_optimizer(model, training_cfg)
    scheduler = build_scheduler(optimizer, training_cfg)

    # The scheduler stays out of prepare(): accelerate would step it once per process on every scheduler.step()
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    ds_val_loaders = [accelerator.prepare(loader) for loader in ds_val_loaders]
    unwrapped = accelerator.unwrap_model(model)

    state = tracker.setup_training(unwrapped, optimizer, scheduler)   # restores weights, optimizer and state on resume
    train_global_iter, val_global_iter = state.train_global_iter, state.val_global_iter
    if accelerator.is_main_process:
        if wandb.run is not None:
            tracker.update_state(wandb_run_id=wandb.run.id, wandb_run_name=wandb.run.name)   # saved with every checkpoint
            if state.epoch > 0 or state.epoch_batches > 0:
                wandb.log({"epoch": state.epoch, "train_global_iter": train_global_iter, "val_global_iter": val_global_iter,
                           "resumed": True})
        if state.epoch > 0 or state.epoch_batches > 0:
            print(f"Resuming from epoch {state.epoch} batch {state.epoch_batches}, train_iter {train_global_iter}, "
                  f"val_iter {val_global_iter}")
        print(OmegaConf.to_yaml(cfg))
        os.makedirs(ckpt_dir, exist_ok=True)   # resuming from a checkpoint=... of another directory
        OmegaConf.save(cfg, os.path.join(ckpt_dir, "training_config.yaml"))
    accelerator.wait_for_everyone()

    log_interval = logging_cfg["log_every_n_steps"]
    for epoch in range(state.epoch, training_cfg["num_epochs"]):
        train_losses, train_global_iter = train_epoch(
            model, optimizer, train_loader, loss_manager, accelerator, epoch, state.epoch_batches if epoch == state.epoch else 0,
            train_global_iter, active_inputs, active_outputs,
            log_interval=log_interval, tracker=tracker, scheduler=scheduler, step_ckpt_interval=training_cfg["save_interval"],
        )
        val_losses, val_global_iter = validate(
            model, val_loader, loss_manager, accelerator, epoch, val_global_iter, active_inputs, active_outputs,
            log_interval=log_interval, log_steps=True, max_batches=None,
        )
        ds_val_losses = {
            name: validate(model, loader, loss_manager, accelerator, epoch, 0, active_inputs, active_outputs,
                           log_interval=log_interval, log_steps=False,
                           max_batches=training_cfg["validation"]["per_dataset_max_batches"])[0]
            for name, loader in zip(val_names, ds_val_loaders)
        }
        if scheduler is not None:
            scheduler.step()

        if accelerator.is_main_process:
            print(f"\nEpoch {epoch + 1}/{training_cfg['num_epochs']}\n" + "=" * 60)
            print_losses("TRAIN LOSSES (epoch average):", train_losses)
            print_losses("\nVALIDATION LOSSES (epoch average):", val_losses)
            for name, losses in ds_val_losses.items():
                print_losses(f"\nVALIDATION LOSSES on {name} (epoch average):", losses)
            print("=" * 60)
            if wandb.run is not None:
                log = {**unscaled(train_losses, loss_manager, "train"), **unscaled(val_losses, loss_manager, "val")}
                for name, losses in ds_val_losses.items():
                    log.update(unscaled(losses, loss_manager, f"val/{name}"))
                wandb.log({**log, "epoch": epoch + 1, "train_global_iter": train_global_iter, "val_global_iter": val_global_iter})

            # The checkpoint comes first, so that a failure while logging media or evaluating cannot lose the epoch
            save_epoch(tracker, unwrapped, optimizer, scheduler, epoch, train_global_iter, val_global_iter, train_losses,
                       val_losses, save_numbered=(epoch + 1) % logging_cfg["save_every_n_epochs"] == 0)
            if wandb.run is not None and epoch % wandb_cfg["log_videos_every"] == 0:
                log_media(unwrapped, loss_manager, val_set, val_names, cfg, steps, lpips_fn, epoch, "val", accelerator.device)
                log_media(unwrapped, loss_manager, train_set, train_names, cfg, steps, lpips_fn, epoch, "train", accelerator.device)

        if cfg["evaluation"]["enabled"] and (epoch + 1) % cfg["evaluation"]["eval_every"] == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:   # LIBERO rollouts run on rank 0 while the other ranks wait
                log_evaluation_results(run_direct_libero_evaluation(model, cfg_dict), epoch + 1)
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
