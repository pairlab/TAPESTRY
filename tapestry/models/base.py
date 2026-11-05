"""TAPESTRY model: text tokens condition a video-diffusion decoder that denoises video latents and action chunks."""
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from tapestry.losses import ModalityLossManager
from tapestry.models.decoders import CogVideoXDecoder
from tapestry.models.encoders import T5TextEncoder

ACTION, VIDEO = "action/future_frames", "video/future_frames"


class SequenceInfo:
    """Where the tokens of every modality sit in the conditioning sequence."""

    def __init__(self):
        self.total_length = 0
        self.token_indices: Dict[str, List[int]] = {}

    def register_tokens(self, modality: str, num_tokens: int):
        self.token_indices.setdefault(modality, []).extend(range(self.total_length, self.total_length + num_tokens))
        self.total_length += num_tokens

    def get_tokens(self, h: torch.Tensor, modality: str) -> torch.Tensor:
        return h[:, self.token_indices[modality], :]  # [B, L, D] -> [B, L_modality, D]


def build(kind: str, cfg: Dict[str, Any]) -> nn.Module:
    """Instantiate cfg['_class'] (kind: 'encoders' or 'decoders') with the remaining keys as constructor arguments."""
    classes = {"encoders": {"T5TextEncoder": T5TextEncoder}, "decoders": {"CogVideoXDecoder": CogVideoXDecoder}}
    cfg = dict(cfg)
    return classes[kind][cfg.pop("_class")](**cfg)


def _present(batch: Dict[str, Any], key: str, B: int, device) -> torch.Tensor:
    """[B] bool, the samples that have `key`. custom_collate adds '<key>_mask' only when some samples lack the key."""
    mask = f"{key}_mask"
    return batch[mask] if mask in batch else torch.ones(B, dtype=torch.bool, device=device)


def _rows(batch: Dict[str, Any], keep: torch.Tensor) -> Dict[str, Any]:
    """`batch` with every per-sample tensor cut to the samples where keep [B] is True."""
    B = keep.shape[0]
    return {k: v[keep] if isinstance(v, torch.Tensor) and v.shape[0] == B else v for k, v in batch.items()}


class Tapestry(nn.Module):
    """TAPESTRY: the inputs in sequence.prefix_order become conditioning tokens, and every entry of sequence.loss_mappings sends
    its `tokens` to the `decoder` that denoises its `target`."""

    def __init__(self, cfg: Dict[str, Any], from_checkpoint: bool):
        super().__init__()
        self.cfg = cfg
        self.sequence_config = cfg["sequence"]
        self.encoders = nn.ModuleDict({name: build("encoders", ecfg) for name, ecfg in cfg["encoders"].items()})
        # from_checkpoint: build a cheap fp16 backbone before loading the full checkpoint.
        self.decoders = nn.ModuleDict({
            name: build("decoders", {**dcfg, "from_checkpoint": from_checkpoint})
            for name, dcfg in cfg["decoders"].items()
        })
        for name, dec in self.decoders.items():
            maps = {m["target"]: m["tokens"] for m in self.sequence_config["loss_mappings"].values() if m["decoder"] == name}
            if dec.joint_diffusion and (set(maps) != {ACTION, VIDEO} or maps[ACTION] != maps[VIDEO]):
                raise ValueError(f"joint_diffusion needs loss mappings for {ACTION} and {VIDEO} on the same tokens, got {maps}")

    def encode(self, batch: Dict[str, Any]) -> Tuple[SequenceInfo, torch.Tensor]:
        """Conditioning tokens [B, L, D] of the inputs in sequence.prefix_order, and where each modality sits in them."""
        seq_info, seq = SequenceInfo(), []
        for key in self.sequence_config["prefix_order"]:
            modality = key.split("/")[0]
            h = batch[key]  # B raw strings, or their precomputed embeddings [B, L, D]
            if not isinstance(h, torch.Tensor):
                h = self.encoders[modality](h)  # [B, L, D]
            seq_info.register_tokens(modality, h.shape[1])
            seq.append(h)
        return seq_info, torch.cat(seq, dim=1)

    def forward(self, inputs: Dict[str, Any], targets: Dict[str, Any], loss_manager: ModalityLossManager) -> Dict[str, Any]:
        """Noise every target of the batch and denoise it. Returns what ModalityLossManager.forward consumes: the
        prediction under the target key with '<modality>_noise', '_timesteps' and '_latent_target', whose rows are the
        samples that have the target."""
        seq_info, seq_tokens = self.encode(inputs)
        return self._forward_with_loss_mappings(seq_info, seq_tokens, {**inputs, **targets}, loss_manager)

    def _run_joint_diffusion(self, seq_info: SequenceInfo, seq_tokens: torch.Tensor, batch: Dict[str, Any],
                             loss_manager: ModalityLossManager) -> Dict[str, Dict[str, Any]]:
        """One joint video + action denoising pass per joint_diffusion decoder, over the samples that have both targets
        (each taken with probability joint_diffusion_mix_prob).

        Returns {target: {"mask": [B] bool, the jointly denoised samples, "outputs": the decoder outputs of this target}}.
        """
        B, device = seq_tokens.shape[0], seq_tokens.device
        joint = {}
        for name, dec in self.decoders.items():
            if not dec.joint_diffusion or ACTION not in batch or VIDEO not in batch:
                continue
            mask = _present(batch, ACTION, B, device) & _present(batch, VIDEO, B, device)  # [B]
            if not mask.any():
                continue
            if dec.joint_diffusion_mix_prob < 1.0:
                both_idx = torch.where(mask)[0]
                pick = torch.rand(both_idx.shape[0], device=device) < dec.joint_diffusion_mix_prob
                mask = torch.zeros_like(mask)
                mask[both_idx[pick]] = True
                if not mask.any():
                    continue
            maps = {m["target"]: m for m in self.sequence_config["loss_mappings"].values() if m["decoder"] == name}
            out = dec.forward_diffusion_training_joint(
                seq_info.get_tokens(seq_tokens, maps[ACTION]["tokens"])[mask], batch[VIDEO][mask], batch[ACTION][mask],
                loss_manager.losses[VIDEO], loss_manager.losses[ACTION], batch=_rows(batch, mask),
            )
            for kind, target in (("action", ACTION), ("video", VIDEO)):
                joint[target] = {"mask": mask, "outputs": {k: v for k, v in out.items() if k == kind or k.startswith(f"{kind}_")}}
        return joint

    def _forward_with_loss_mappings(self, seq_info: SequenceInfo, seq_tokens: torch.Tensor, batch: Dict[str, Any],
                                    loss_manager: ModalityLossManager) -> Dict[str, Any]:
        """Decoder outputs of every loss mapping whose target is in the batch. The samples picked by _run_joint_diffusion
        come from the joint pass and the others from a pass of their own, merged back into sample order."""
        B, device = seq_tokens.shape[0], seq_tokens.device
        joint = self._run_joint_diffusion(seq_info, seq_tokens, batch, loss_manager)
        outputs = {}
        for mapping in self.sequence_config["loss_mappings"].values():
            target = mapping["target"]
            if target not in batch:  # e.g. video-only datasets have no actions
                continue
            has = _present(batch, target, B, device)  # [B]
            if not has.any():
                continue
            sep = has & ~joint[target]["mask"] if target in joint else has  # [B] samples denoised on their own
            out = joint[target]["outputs"] if target in joint else None
            if sep.any():
                cond = seq_info.get_tokens(seq_tokens, mapping["tokens"])  # [B, L, D]
                out = self.decoders[mapping["decoder"]].forward_diffusion_training(
                    cond[sep], batch[target][sep], loss_manager.losses[target], batch=_rows(batch, sep))
                if target in joint:
                    # rows: the separate samples, then the joint ones. argsort of their batch indices restores sample order
                    order = torch.argsort(torch.cat([torch.where(sep)[0], torch.where(joint[target]["mask"])[0]]))
                    for k, v in joint[target]["outputs"].items():
                        out[k] = torch.cat([out[k], v])[order] if isinstance(v, torch.Tensor) else v
            for k, v in out.items():
                outputs[target if k in ("video", "action") else k] = v
        return outputs
