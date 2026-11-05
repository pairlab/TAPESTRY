"""Deployment interface: camera images + task text -> a chunk of actions in the robot's own action space."""
import numpy as np
import torch
from einops import rearrange

from tapestry.eval_utils import action_noise_scheduler
from tapestry.models.model_loading import encode_task_text_with_caching, load_checkpoint_cpu, load_unified_model
from tapestry.models.unified_model_wrapper import UnifiedModelWrapper
from tapestry.utils.action_normalizer import ActionNormalizer
from tapestry.utils.action_preprocess import from_relative_rot6d_eepose20


class TapestryPolicy:
    """Runs a release checkpoint (scripts/prepare_release_checkpoint.py) as a closed-loop policy:

        policy = TapestryPolicy("tapestry.pt", device="cuda", num_inference_steps=1, t5_override=None, text_cache_dir="text_cache")
        actions = policy.act([front, left_wrist, right_wrist], "Pick up the corn and put it in the bowl", ref_action=pose)   # [action_horizon, action_dim]

    images: one [image_size, image_size, 3] uint8 RGB array per camera, in policy.cameras order (full frame, resized without cropping).
    ref_action: the robot's current pose in the action space, required for absolute action spaces (e.g. eepose_6drot):
    the predicted positions (and rotations, if trained with relative_rotation_actions) are relative to it.
    Execute the first few actions of each chunk (e.g. 8 of 16) and re-plan.
    t5_override: None loads the T5 encoder the checkpoint was trained with, and "THUDM/CogVideoX-2b" its smaller fp16 copy.
    text_cache_dir: None keeps T5 on `device` next to the policy. A directory keeps T5 off the GPU: each new instruction is
    encoded once with T5 on the CPU and saved there, and later calls and runs read it back.
    """

    def __init__(self, checkpoint, device, num_inference_steps, t5_override, text_cache_dir):
        ck = load_checkpoint_cpu(checkpoint)
        if "normalizer" not in ck:
            raise KeyError(f"{checkpoint} is a raw training checkpoint; convert it with scripts/prepare_release_checkpoint.py")
        self.wrapper = UnifiedModelWrapper(load_unified_model(checkpoint, device, skip_text_encoder=text_cache_dir is not None,
                                                              t5_override=t5_override))
        self.checkpoint, self.t5_override, self.text_cache_dir = checkpoint, t5_override, text_cache_dir
        cfg = self.wrapper.model.cfg
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.scheduler = action_noise_scheduler(cfg)
        self.cameras = list(cfg["training"]["dataset"]["cond_cameraviews"])
        self.image_size = cfg["model"]["image_size"]
        self.action_horizon = cfg["training"]["true_horizon"]
        self.action_dim = self.wrapper.action_dim
        self.normalizer = None if ck["normalizer"] is None else ActionNormalizer.from_dict(ck["normalizer"])
        self.relative_rotation_actions = ck["relative_rotation_actions"]
        self._text_cache = {}

    @torch.no_grad()
    def text_embedding(self, text):
        """[1, L, D] T5 embedding of the instruction `text`, computed on its first use and then kept in memory."""
        if text not in self._text_cache:
            if self.text_cache_dir is None:
                self._text_cache[text] = self.wrapper.get_task_emb([text])
            else:
                emb = encode_task_text_with_caching(self.checkpoint, text, self.text_cache_dir, use_cache=True, device="cpu",
                                                    t5_override=self.t5_override)
                self._text_cache[text] = emb.to(self.device)
        return self._text_cache[text]

    @torch.no_grad()
    def act(self, images, text, ref_action=None):
        if len(images) != len(self.cameras):
            raise ValueError(f"expected {len(self.cameras)} images ({self.cameras}), got {len(images)}")
        for cam, img in zip(self.cameras, images):
            if img.shape != (self.image_size, self.image_size, 3) or img.dtype != np.uint8:
                raise ValueError(f"{cam}: expected a ({self.image_size}, {self.image_size}, 3) uint8 image, got {img.shape} {img.dtype}")
        video = rearrange(torch.from_numpy(np.stack(images)), "v h w c -> 1 v 1 c h w").to(self.device, torch.float32) / 255.0
        cond = self.wrapper.get_conditioning_dict(video, self.text_embedding(text))
        actions = self.wrapper(cond, self.scheduler, num_inference_steps=self.num_inference_steps)
        actions = rearrange(actions, "1 t d -> t d").float().cpu().numpy()
        if self.normalizer is not None:
            actions = self.normalizer.denormalize(actions, ref_action)  # raises if ref_action is needed and missing
        if self.relative_rotation_actions:
            if ref_action is None:
                raise ValueError("relative_rotation_actions: pass ref_action (current absolute pose)")
            actions = from_relative_rot6d_eepose20(actions, np.asarray(ref_action, dtype=np.float32))
        return actions
