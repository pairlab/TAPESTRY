"""Turn a training checkpoint into a self-contained inference checkpoint for tapestry.models.policy.TapestryPolicy.

    python scripts/prepare_release_checkpoint.py --checkpoint checkpoints/tapestry_libero_taskgen/latest.pt --output my_libero.pt --verify
    python scripts/prepare_release_checkpoint.py --checkpoint epoch_6.pt --output tapestry.pt \
        --normalizer-stats normalizer_stats_eepose_6drot.json --verify

Drops optimizer/training state, keeps weights + resolved config, and embeds the action post-processing the run
trained with: `normalizer` (content of the stats JSON, or None when actions were used as stored, e.g. LIBERO) and
`relative_rotation_actions`. The stats JSON is read from the path in the config unless --normalizer-stats points
at a copy. Pretrained sources in the config are replaced by their public HuggingFace ids, every other
machine-specific path by its basename and experiment_name by the output file name, so the file can be shared.
The frozen T5 text encoder is never stored; it is downloaded from HuggingFace.
"""
import argparse
import json
import os

import torch

from tapestry.models.model_loading import load_checkpoint_cpu

HF_SOURCES = {"t5-v1_1-xxl": "google/t5-v1_1-xxl", "CogVideoX-2b": "THUDM/CogVideoX-2b"}


def action_spec(cfg):
    """(normalizer_stats_path or None, relative_rotation_actions) shared by every action dataset of a training config."""
    ds_cfg = cfg["training"]["dataset"]
    if "normalize_actions" in ds_cfg:  # one action space for the whole run (cfg/train_tapestry.yaml, cfg/finetune.yaml)
        return (ds_cfg["normalizer_stats_path"] if ds_cfg["normalize_actions"] else None), ds_cfg["relative_rotation_actions"]
    specs = set()
    for name, ds in ds_cfg["datasets"].items():
        if "action" not in ds["active_modalities"]:
            continue
        if ds["_class"] == "LiberoDataset":
            specs.add((None, False))  # LIBERO actions are consumed as stored (normalized deltas)
        elif ds["_class"] == "HDF5DemoDataset":
            specs.add((ds["normalizer_stats_path"] if ds["normalize_actions"] else None, ds["relative_rotation_actions"]))
        else:
            raise ValueError(f"dataset {name}: no action post-processing rule for {ds['_class']}")
    if len(specs) != 1:
        raise ValueError(f"action datasets disagree on (normalizer_stats_path, relative_rotation_actions): {specs}")
    ((stats_path, relative_rotation_actions),) = specs
    return stats_path, relative_rotation_actions


def load_normalizer(cfg, stats_path, override):
    """Stats dict for the run's normalizer; `override` replaces the (training-machine) path stored in the config."""
    if stats_path is None:
        if override is not None:
            raise ValueError("--normalizer-stats given but the run did not normalize actions")
        return None
    path = os.path.expanduser(stats_path if override is None else override)
    if not os.path.exists(path):
        raise FileNotFoundError(f"normalizer stats {path} not found; pass --normalizer-stats <copy of {os.path.basename(stats_path)}>")
    with open(path) as f:
        stats = json.load(f)
    action_space = cfg["training"]["dataset"]["action_space"]
    if stats["action_space"] != action_space:
        raise ValueError(f"{path} is for action space {stats['action_space']}, the run trained with {action_space}")
    return stats


def public_source(path):
    """HuggingFace id of a pretrained-weights directory (training machines use local copies)."""
    name = os.path.basename(os.path.normpath(path))
    if name not in HF_SOURCES:
        raise ValueError(f"no public source known for pretrained weights {path!r}; expected one of {sorted(HF_SOURCES)}")
    return HF_SOURCES[name]


def scrub(node, rename):
    """Config tree with absolute/S3/home paths cut to their basename (keeps file names, drops hosts) and every string
    equal to a `rename` key, alone or as the last path component, renamed (experiment_name, checkpoint directory)."""
    if isinstance(node, dict):
        return {k: scrub(v, rename) for k, v in node.items()}
    if isinstance(node, list):
        return [scrub(v, rename) for v in node]
    if isinstance(node, str):
        if node.startswith(("/", "~", "s3://")):
            node = os.path.basename(node.rstrip("/"))
        head, tail = os.path.split(node)
        if tail in rename:
            node = os.path.join(head, rename[tail])
    return node


def release_config(cfg, name):
    """Shareable copy of a training config: `name` replaces experiment_name, pretrained sources get their public ids
    and machine paths their basenames."""
    cfg = json.loads(json.dumps(cfg))  # plain dicts/lists, detached from the checkpoint
    text = cfg["encoders"]["text"]
    text["model_name_or_path"] = public_source(text["model_name_or_path"])
    for dec in cfg["decoders"].values():
        dec["pretrained_model_name"] = public_source(dec["pretrained_model_name"])
    return scrub(cfg, {cfg["experiment_name"]: name})


def verify(path, device, t5_override):
    import numpy as np
    from tapestry.models.policy import TapestryPolicy

    policy = TapestryPolicy(path, device, num_inference_steps=1, t5_override=t5_override, text_cache_dir=None)
    images = [np.zeros((policy.image_size, policy.image_size, 3), dtype=np.uint8) for _ in policy.cameras]
    ref = None
    if policy.normalizer is not None or policy.relative_rotation_actions:
        ref = np.zeros(policy.action_dim, dtype=np.float32)
        if policy.relative_rotation_actions:  # identity rotation as row-major rot6d [R00, R01, R10, R11, R20, R21] per arm
            ref[[3, 6, 13, 16]] = 1.0
    actions = policy.act(images, "verify", ref_action=ref)
    if actions.shape != (policy.action_horizon, policy.action_dim) or not np.isfinite(actions).all():
        raise ValueError(f"verify failed: actions {actions.shape}, finite={bool(np.isfinite(actions).all())}")
    print(f"verify ok: actions {actions.shape} for cameras {policy.cameras}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--normalizer-stats", default=None, help="stats JSON to embed when the path in the config is not available here")
    ap.add_argument("--verify", action="store_true", help="reload the output with TapestryPolicy and run one forward pass")
    ap.add_argument("--device", default="cuda", help="device for --verify")
    ap.add_argument("--t5-override", default=None, help="for --verify: T5 encoder to load instead of the checkpoint's, e.g. THUDM/CogVideoX-2b for its smaller fp16 copy")
    args = ap.parse_args()

    ck = load_checkpoint_cpu(args.checkpoint)
    stats_path, relative_rotation_actions = action_spec(ck["config"])
    name = os.path.splitext(os.path.basename(args.output))[0]
    out = {
        "model": ck["model"],
        "config": release_config(ck["config"], name),
        "epoch": ck["epoch"],
        "train_global_iter": ck["train_global_iter"],
        "normalizer": load_normalizer(ck["config"], stats_path, args.normalizer_stats),
        "relative_rotation_actions": relative_rotation_actions,
    }
    torch.save(out, args.output)
    normalizer = out["normalizer"]
    print(f"wrote {args.output} ({os.path.getsize(args.output) / 1e9:.2f} GB); normalizer="
          f"{None if normalizer is None else normalizer['action_space']}, relative_rotation_actions={relative_rotation_actions}")
    if args.verify:
        verify(args.output, args.device, args.t5_override)


if __name__ == "__main__":
    main()
