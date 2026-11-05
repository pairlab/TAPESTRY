from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from einops import rearrange
from torch.utils.data import Dataset

# Raw data to load per modality. Action prediction is conditioned on images.
RAW_KEYS = {"video": ["images"], "action": ["actions", "images"], "text": ["text"]}


class BaseDataset(Dataset, ABC):
    """Robot demos cut into chunks: create_index_map lists the chunks, load_data reads one, process_data makes the sample.
    Subclasses set self.index_map = self.create_index_map(). It must be a list: quick mode slices it (train.build_datasets)."""

    def __init__(
        self,
        active_modalities: List[str],   # of "video", "action", "text"
        data_percentage: float,         # share of the demos: the first ones for training, the last ones for validation
        cond_cameraviews: List[str],    # camera views to load, in this order
        img_shape: Tuple[int, int],     # (H, W) of the returned frames; stored frames of another size are resized
        true_horizon: int,              # timesteps per chunk
        pred_horizon: int,              # frames per chunk: every (true_horizon // pred_horizon)-th timestep
        is_validation: bool,
    ):
        super().__init__()
        if true_horizon % pred_horizon != 0:
            raise ValueError(f"true_horizon ({true_horizon}) must be a multiple of pred_horizon ({pred_horizon})")
        self.active_modalities = list(active_modalities)
        self.keys_to_load = sorted({key for modality in self.active_modalities for key in RAW_KEYS[modality]})
        self.data_percentage = data_percentage
        self.cond_cameraviews = list(cond_cameraviews)
        self.img_shape = tuple(img_shape)
        self.true_horizon = true_horizon
        self.pred_horizon = pred_horizon
        self.skip_every = true_horizon // pred_horizon
        self.is_validation = is_validation

    @abstractmethod
    def create_index_map(self) -> List[Dict]:
        """One dict per chunk with what load_data needs to read it."""
        raise NotImplementedError

    @abstractmethod
    def load_data(self, idx_dict: Dict) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def process_data(self, data: Dict) -> Dict:
        raise NotImplementedError

    def split(self, demos: List) -> List:
        """This dataset's share of the demos."""
        n = round(len(demos) * self.data_percentage)
        if n == 0:
            raise ValueError(f"data_percentage={self.data_percentage} selects 0 of {len(demos)} demos")
        return demos[len(demos) - n:] if self.is_validation else demos[:n]

    def to_video(self, images: np.ndarray) -> torch.Tensor:
        """Frames [V, T, H, W, C] in [0, 255] -> video [V, pred_horizon, C, *img_shape] in [0, 1]. A chunk cut short by the end
        of its demo repeats the last frame."""
        if images.shape[1] < self.pred_horizon:
            pad = np.repeat(images[:, images.shape[1] - 1:], self.pred_horizon - images.shape[1], axis=1)
            images = np.concatenate([images, pad], axis=1)
        video = torch.from_numpy(rearrange(images / 255.0, "v t h w c -> v t c h w")).float()
        if tuple(video.shape[3:]) != self.img_shape:
            v, t = video.shape[0], video.shape[1]
            video = T.Resize(self.img_shape)(rearrange(video, "v t c h w -> (v t) c h w"))
            video = rearrange(video, "(v t) c h w -> v t c h w", v=v, t=t)
        return video

    def apply_augmentation(self, images: torch.Tensor, aug_cfg: Dict) -> torch.Tensor:
        """Random crop (resized to img_shape), color jitter and blur of images [V, T, C, H, W] in [0, 1], as enabled in aug_cfg.
        Every view draws its own random parameters, shared by all of its frames."""
        if aug_cfg['random_crop']:
            crop = T.RandomResizedCrop(self.img_shape, scale=tuple(aug_cfg['crop_scale']), ratio=tuple(aug_cfg['crop_ratio']),
                                       interpolation=T.InterpolationMode.BICUBIC)
            images = torch.stack([crop(view) for view in images])
        if aug_cfg['color_jitter']:
            jitter = T.ColorJitter(aug_cfg['brightness_strength'], aug_cfg['contrast_strength'],
                                   aug_cfg['saturation_strength'], aug_cfg['hue_strength'])
            images = torch.stack([jitter(view) for view in images])
        if aug_cfg['gaussian_blur']:
            blur = T.GaussianBlur(aug_cfg['blur_kernel_size'], sigma=tuple(aug_cfg['blur_sigma']))
            images = torch.stack([blur(view) for view in images])
        return images.clamp(0, 1)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        return self.process_data(self.load_data(self.index_map[idx]))
