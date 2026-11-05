"""Policy interface around a trained Tapestry model."""
import torch
import torch.nn as nn

from tapestry.models.base import Tapestry


class UnifiedModelWrapper(nn.Module):
    """Turns a Tapestry model into an action-chunk policy:

        policy = UnifiedModelWrapper.from_checkpoint(path, "cuda", t5_override=None)
        text = policy.get_task_emb(["put the bowl in the drawer"])                     # [B, L, D] T5 tokens
        cond = policy.get_conditioning_dict(video, text)
        actions = policy(cond, noise_scheduler, num_inference_steps=1)                   # [B, action_horizon, action_dim]

    video: [B, V, 1, C, H, W] in [0, 1] (the current frame of each conditioning camera, in training.dataset.cond_cameraviews
    order). Actions come out in the model's normalized space (tapestry.utils.action_normalizer).
    """

    def __init__(self, model: Tapestry):
        super().__init__()
        cfg = model.cfg
        action_mapping = cfg["sequence"]["loss_mappings"]["action"]
        self.model = model
        self.action_decoder_name = action_mapping["decoder"]
        self.action_token_type = action_mapping["tokens"]
        self.action_dim = model.decoders[self.action_decoder_name].action_dim
        print(f"[UnifiedModelWrapper] decoder={self.action_decoder_name} tokens={self.action_token_type} action_dim={self.action_dim}")

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device, t5_override):
        from tapestry.models.model_loading import load_unified_model

        return cls(load_unified_model(checkpoint_path, device, skip_text_encoder=False, t5_override=t5_override))

    def get_task_emb(self, text):
        """[B, L, D] token embeddings of a list of B task strings from the model's frozen text encoder."""
        return self.model.encoders["text"](text)

    def get_conditioning_dict(self, video, text):
        """Encode the observations into the conditioning tokens the action decoder denoises against."""
        b, v = video.shape[:2]
        batch = {"video/history_frames": video, "text/embeddings": text,
                 "video/view_mask": torch.ones((b, v), device=video.device, dtype=torch.bool)}
        return self.encode(batch)

    def encode(self, batch):
        """cond for forward() from a batch that already holds the model's input keys, such as a collated training batch."""
        seq_info, seq_tokens = self.model.encode(batch)
        return {"h": seq_info.get_tokens(seq_tokens, self.action_token_type), "batch": batch}

    def forward(self, cond, noise_scheduler, num_inference_steps):
        """Denoise an action chunk [B, action_horizon, action_dim] (normalized) given cond from get_conditioning_dict or encode."""
        dec = self.model.decoders[self.action_decoder_name]
        kwargs = dict(batch=cond["batch"], num_inference_steps=num_inference_steps)
        if dec.joint_diffusion and dec.joint_diffusion_mix_prob == 1.0:
            predict_future_frames = self.model.cfg["training"]["objectives"]["video/future_frames"]["predict_future_frames"]
            return dec.decode_actions_with_joint_diffusion(cond["h"], noise_scheduler, predict_future_frames=predict_future_frames, **kwargs)
        # joint_diffusion_mix_prob < 1: trained on a mix of joint and action-only denoising; deployed action-only
        return dec.decode_actions_with_diffusion(cond["h"], noise_scheduler, **kwargs)
