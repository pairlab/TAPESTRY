"""
VideoOnlyDataset: video + caption clips from large-scale video datasets (droid, ego4d, epic_kitchens, pandas, ssv2, bridge, agibot).

Each dataset has a metadata CSV under root_dir with one row per clip:
    video_path                              the clip: an MP4 or a folder of frames (jpg/png). With translate_paths it is mapped to the
                                            on-disk layout by tapestry/loaders/path_translator.py, otherwise it is used as is, relative to root_dir.
    n_frames, fps, width, height            clip properties, used for filtering and temporal sampling
    <caption_key>, <fallback_caption_key>   captions; the fallback is used when the first one is empty
    optional columns                        trim_start, trim_end (frame range to use), crop_left, crop_right, crop_top, crop_bottom
                                            (pixel box), stable_background, stable_brightness, detected_hand_in_first_frame
                                            (rows with a false value are dropped)

Sample keys:
    video/history_frames              [V, 1, C, H, W] first frame, float in [0, 1]
    video/future_frames               [V, T, C, H, W] T = pred_horizon frames, starting at the first frame
    text/embeddings                   caption string (encoded by the model's text encoder)
    active_modalities                 the configured modality list

Camera views: droid, bridge and agibot have three cameras (path_translator.MULTIVIEW_DATASETS). cond_cameraviews lists the views
to load, V = len(cond_cameraviews). With null one view is loaded (V = 1), for droid and bridge a random one per sample.

Temporal alignment (following Large Video Planner, LVP): every clip is resampled to target_fps at roughly human speed. A sample
of T frames spans round(T / target_fps * effective_fps) source frames. Human videos use effective_fps=null (the clip's own fps,
real time); robot videos set effective_fps so that the motion plays at human speed (a value above the recording fps speeds the
clip up). Longer clips are cut (trim_mode=random_cut) or sped up (speedup); shorter ones are slowed down (pad_mode=slowdown),
padded with their last frame (pad_last) or dropped (discard).
"""

import random
import re
from collections import Counter
from glob import glob
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from einops import rearrange, repeat
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms

from tapestry.loaders.path_translator import MULTIVIEW_DATASETS, translate_path

MAX_HEIGHT = 4096  # clips outside these bounds are dropped
MAX_WIDTH = 4096
MAX_FPS = 120
CROP_SCALE = (0.9, 1.0)  # training crop: area fraction, and aspect ratio relative to img_shape's
CROP_RATIO = (0.98, 1.02)
MAX_SAMPLE_RETRIES = 100

VALID_MP4_DECODERS = ("decord", "imageio")
VALID_TRIM_MODES = ("speedup", "random_cut")
VALID_PAD_MODES = ("slowdown", "pad_last", "discard")
FRAME_IDX_PATTERNS = (r"_idx=(\d+)", r"_(\d+)\.", r"(\d+)\.")  # 78687_idx=0.jpg, frame_000.jpg or im_0.jpg, 000.jpg


def _decord_reader(path):
    import decord  # Optional backend; importing the core loaders must not require it.

    decord.bridge.set_bridge("torch")
    return decord.VideoReader(uri=str(path))


class VideoOnlyDataset(Dataset):
    """Video + caption clips described by a metadata CSV (see the module docstring)."""

    def __init__(
        self,
        root_dir: str,
        dataset_name: str,
        metadata_path: str,  # CSV, relative to root_dir
        active_modalities: List[str],
        data_percentage: float,  # share of the (seeded-shuffle) records: the first ones for training, the last ones for validation
        img_shape: Tuple[int, int],
        pred_horizon: int,  # T, frames per sample
        target_fps: int,
        effective_fps: Optional[int],  # null: the clip's own fps
        caption_key: str,
        fallback_caption_key: str,
        trim_mode: str,
        pad_mode: str,
        filter_n_frames: Tuple[int, int],  # keep clips whose frame count lies in this range
        translate_paths: bool,
        cond_cameraviews: Optional[List[str]],
        mp4_decoder: str,
        is_validation: bool,
    ):
        super().__init__()
        if "action" in active_modalities:
            raise ValueError("VideoOnlyDataset is video+text only; use HDF5DemoDataset or LiberoDataset for action data")
        if trim_mode not in VALID_TRIM_MODES:
            raise ValueError(f"Invalid trim_mode: {trim_mode}. Must be one of {list(VALID_TRIM_MODES)}")
        if pad_mode not in VALID_PAD_MODES:
            raise ValueError(f"Invalid pad_mode: {pad_mode}. Must be one of {list(VALID_PAD_MODES)}")
        if cond_cameraviews is not None:
            views = MULTIVIEW_DATASETS[dataset_name.lower()] if dataset_name.lower() in MULTIVIEW_DATASETS else []
            if not cond_cameraviews or any(view not in views for view in cond_cameraviews):
                raise ValueError(f"cond_cameraviews={cond_cameraviews} for '{dataset_name}', whose camera views are {views}; use null for a single view")
            if not translate_paths:
                raise ValueError("cond_cameraviews needs translate_paths=true: the path translators select the camera")

        self.root_dir = Path(root_dir).expanduser()
        self.dataset_name = dataset_name
        self.metadata_path = metadata_path
        self.active_modalities = active_modalities
        self.data_percentage = data_percentage
        self.img_shape = tuple(img_shape)
        self.n_frames = pred_horizon
        self.target_fps = target_fps
        self.effective_fps = effective_fps
        self.caption_key = caption_key
        self.fallback_caption_key = fallback_caption_key
        self.trim_mode = trim_mode
        self.pad_mode = pad_mode
        self.filter_n_frames = tuple(filter_n_frames)
        self.translate_paths = translate_paths
        self.cond_cameraviews = cond_cameraviews
        self.mp4_decoder = mp4_decoder
        self.is_validation = is_validation
        self._validate_decoder_config()

        h, w = self.img_shape
        scale, ratio = ((1.0, 1.0), (1.0, 1.0)) if is_validation else (CROP_SCALE, CROP_RATIO)
        self.crop = transforms.RandomResizedCrop(
            size=self.img_shape, scale=scale, ratio=[w / h * ratio[0], w / h * ratio[1]],
            interpolation=transforms.InterpolationMode.BICUBIC,
        )

        records = self._load_records()
        n = round(len(records) * data_percentage)
        if not 0 < n <= len(records):
            raise ValueError(f"[{dataset_name}] data_percentage={data_percentage} selects {n} of {len(records)} records")
        self.index_map = records[len(records) - n:] if is_validation else records[:n]
        print(f"VideoOnlyDataset [{dataset_name}]: {len(self.index_map)} {'validation' if is_validation else 'training'} samples, "
              f"effective_fps={effective_fps}, views={cond_cameraviews}")

    def _validate_decoder_config(self) -> None:
        if self.mp4_decoder not in VALID_MP4_DECODERS:
            raise ValueError(f"mp4_decoder must be one of {list(VALID_MP4_DECODERS)}, got '{self.mp4_decoder}'")
        if self.mp4_decoder == "decord":
            import decord  # noqa: F401  # a missing backend fails here, not inside __getitem__'s retry loop

    def _n_frames_in_src(self, src_fps: int) -> int:
        """Source frames spanned by one sample, using effective_fps when set."""
        fps = self.effective_fps if self.effective_fps is not None else src_fps
        return round(self.n_frames / self.target_fps * fps)

    def _load_records(self) -> List[Dict[str, Any]]:
        """Read the metadata CSV, drop the filtered clips and shuffle with a fixed seed."""
        meta_path = self.root_dir / self.metadata_path
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path} (root_dir must contain {self.metadata_path})")
        df = pd.read_csv(meta_path, na_filter=False)
        missing = {"video_path", "n_frames", "fps", "width", "height", self.caption_key, self.fallback_caption_key} - set(df.columns)
        if missing:
            raise ValueError(f"{meta_path} lacks the columns {sorted(missing)}")

        records, rejections = [], Counter()
        for rec in df.to_dict("records"):
            reason = self._reject_reason(rec)
            if reason is None:
                records.append(rec)
            else:
                rejections[reason] += 1
        print(f"[{self.dataset_name}] {meta_path}: kept {len(records)} of {len(df)} clips, rejected {dict(rejections)}", flush=True)
        random.Random(42).shuffle(records)
        return records

    def _reject_reason(self, rec: Dict[str, Any]) -> Optional[str]:
        """Why a clip is filtered out, or None to keep it."""
        h, w = rec["height"], rec["width"]
        if "crop_left" in rec and "crop_right" in rec:
            w = rec["crop_right"] - rec["crop_left"]
        if "crop_top" in rec and "crop_bottom" in rec:
            h = rec["crop_bottom"] - rec["crop_top"]
        n_frames = rec["trim_end"] - rec["trim_start"] if "trim_start" in rec and "trim_end" in rec else rec["n_frames"]

        if not (0 <= h <= MAX_HEIGHT):
            return "height"
        if not (0 <= w <= MAX_WIDTH):
            return "width"
        if not (self.filter_n_frames[0] <= n_frames <= self.filter_n_frames[1]):
            return "n_frames"
        if not (0 <= rec["fps"] <= MAX_FPS):
            return "fps"
        if n_frames < self._n_frames_in_src(rec["fps"]) and self.pad_mode == "discard":
            return "short"
        # Stability and hand-detection flags, following LVP (videos without hands are ambiguous about the embodiment).
        if "stable_background" in rec and not rec["stable_background"]:
            return "stability"
        if "stable_brightness" in rec and not rec["stable_brightness"]:
            return "stability"
        if "detected_hand_in_first_frame" in rec and not rec["detected_hand_in_first_frame"]:
            return "no_hand"
        return None

    def _temporal_sample(self, n_frames: int, fps: int) -> np.ndarray:
        """Indices of the self.n_frames output frames within a clip of n_frames source frames."""
        target_len = self._n_frames_in_src(fps)

        if n_frames < target_len:
            # Video is too short
            if self.pad_mode == "pad_last":
                indices = np.linspace(0, target_len - 1, self.n_frames)
                indices = np.clip(indices, 0, n_frames - 1)
            elif self.pad_mode == "slowdown":
                indices = np.linspace(0, n_frames - 1, self.n_frames)
            else:
                raise ValueError(f"Short video with pad_mode={self.pad_mode} should be filtered")
        elif n_frames > target_len and self.trim_mode == "random_cut":
            start = np.random.randint(0, n_frames - target_len + 1)
            indices = start + np.linspace(0, target_len - 1, self.n_frames)
        else:  # the right length, or trim_mode == "speedup"
            indices = np.linspace(0, n_frames - 1, self.n_frames)

        return np.round(indices).astype(int)

    @staticmethod
    def _trim_range(rec: Dict[str, Any], total: int) -> Tuple[int, int]:
        """Optional trim_start/trim_end columns; the whole clip otherwise."""
        return (rec["trim_start"], rec["trim_end"]) if "trim_start" in rec else (0, total)

    def _resolve_sampling_total(self, rec: Dict[str, Any], probe: Callable[[], int]) -> int:
        """Clip length for index sampling: metadata n_frames, else trim_end, else probe(), which opens the clip."""
        total = rec["n_frames"]
        if total == 0 and "trim_end" in rec:
            total = int(rec["trim_end"])
        if total <= 0:
            total = int(probe())
        if total <= 0:
            raise ValueError(f"Unable to resolve total frames for record: {rec}")
        return total

    def _compute_sample_indices(self, rec: Dict[str, Any], total: int) -> List[int]:
        start, end = self._trim_range(rec, total)
        indices = start + self._temporal_sample(end - start, rec["fps"])
        return [min(max(0, i), total - 1) for i in indices]

    @staticmethod
    def _extract_frame_idx(filepath: str) -> int:
        name = Path(filepath).name
        for pattern in FRAME_IDX_PATTERNS:
            match = re.search(pattern, name)
            if match:
                return int(match.group(1))
        raise ValueError(f"No frame index in {filepath}")

    def _frame_files(self, folder: Path) -> List[str]:
        """Image files of a frame folder, in frame order."""
        files = []
        for pattern in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            files.extend(glob(str(folder / pattern)))
        if not files:
            raise FileNotFoundError(f"No frames in {folder}")
        return sorted(files, key=self._extract_frame_idx)

    def _load_frames_from_folder(self, files: List[str], indices: List[int]) -> torch.Tensor:
        """[T, H, W, C] uint8; indices past the end repeat the last frame."""
        frames = [np.array(Image.open(files[min(max(0, i), len(files) - 1)]).convert("RGB")) for i in indices]
        return torch.from_numpy(np.stack(frames))

    def _load_local_mp4_at_indices(self, video_path: Path, indices: List[int]) -> torch.Tensor:
        """[T, H, W, C] uint8"""
        if self.mp4_decoder == "decord":
            return _decord_reader(video_path).get_batch(indices)

        import imageio

        rd = imageio.get_reader(str(video_path), format="ffmpeg")
        try:
            return torch.stack([torch.from_numpy(np.asarray(rd.get_data(i))) for i in indices], dim=0)
        finally:
            rd.close()

    def _probe_mp4_total(self, video_path: Path) -> int:
        """Frame count according to the configured decoder."""
        if self.mp4_decoder == "decord":
            return len(_decord_reader(video_path))

        import imageio

        rd = imageio.get_reader(str(video_path), format="ffmpeg")
        try:
            return int(rd.count_frames())
        finally:
            rd.close()

    def _load_single_view(self, rec: Dict[str, Any]) -> torch.Tensor:
        """[T, H, W, C] uint8 from an MP4 or a frame folder."""
        meta_path = rec["video_path"]
        path = self.root_dir / (translate_path(self.dataset_name, meta_path, None) if self.translate_paths else meta_path)
        if not path.exists():  # fail before any sampling
            raise FileNotFoundError(f"No such file: '{path.absolute()}'")
        if not path.is_dir():
            total = self._resolve_sampling_total(rec, lambda: self._probe_mp4_total(path))
            return self._load_local_mp4_at_indices(path, self._compute_sample_indices(rec, total))
        files = self._frame_files(path)
        return self._load_frames_from_folder(files, self._compute_sample_indices(rec, len(files)))  # sampled over the actual length

    def _load_multiview(self, rec: Dict[str, Any]) -> torch.Tensor:
        """[V, T, H, W, C] uint8; every view is read at the indices sampled for the first one."""
        views = MULTIVIEW_DATASETS[self.dataset_name.lower()]
        indices, frames = None, []
        for view in self.cond_cameraviews:
            path = self.root_dir / translate_path(self.dataset_name, rec["video_path"], views.index(view))
            files = self._frame_files(path) if path.is_dir() else None
            if indices is None:
                probe = (lambda: self._probe_mp4_total(path)) if files is None else (lambda: len(files))
                indices = self._compute_sample_indices(rec, self._resolve_sampling_total(rec, probe))
            frames.append(self._load_local_mp4_at_indices(path, indices) if files is None else self._load_frames_from_folder(files, indices))
        return torch.stack(frames, dim=0)

    def _load_video(self, rec: Dict[str, Any]) -> torch.Tensor:
        """Load, crop and resize every view: [V, T, C, H, W] in [0, 1], padded to T = self.n_frames with the last frame."""
        if self.cond_cameraviews is None:
            frames = rearrange(self._load_single_view(rec), "t h w c -> 1 t h w c")
        else:
            frames = self._load_multiview(rec)
        if "crop_top" in rec and "crop_bottom" in rec:
            frames = frames[:, :, rec["crop_top"]:rec["crop_bottom"]]
        if "crop_left" in rec and "crop_right" in rec:
            frames = frames[:, :, :, rec["crop_left"]:rec["crop_right"]]

        frames = rearrange(frames.float(), "v t h w c -> v t c h w").contiguous() / 255.0
        frames = torch.stack([self.crop(view) for view in frames], dim=0)  # one random crop per view, shared by its frames
        t = frames.shape[1]
        if t < self.n_frames:
            pad = repeat(frames[:, t - 1:t], "v 1 c h w -> v t c h w", t=self.n_frames - t)
            frames = torch.cat([frames, pad], dim=1)
        return frames

    def _get_caption(self, rec: Dict[str, Any]) -> str:
        for key in (self.caption_key, self.fallback_caption_key):
            caption = str(rec[key]).strip()
            if caption:
                return caption
        raise ValueError(f"no caption for {rec['video_path']}")

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """One sample. A clip that fails to load is replaced by a random one: web-scale video always has a few bad files."""
        failures = []
        for attempt in range(MAX_SAMPLE_RETRIES):
            rec = self.index_map[idx]
            try:
                frames = self._load_video(rec)  # [V, T, C, H, W]
                caption = self._get_caption(rec)
                return {
                    "video/history_frames": frames[:, 0:1],
                    "video/future_frames": frames,
                    "active_modalities": list(self.active_modalities),
                    "text/embeddings": caption,
                }
            except Exception as e:
                failures.append(f"idx={idx} path={rec['video_path']}: {type(e).__name__}: {e}")
                print(f"[{self.dataset_name}] sample failed (attempt {attempt}), skipping {failures[len(failures) - 1]}", flush=True)
                idx = random.randint(0, len(self) - 1)

        raise RuntimeError(
            f"[{self.dataset_name}] {MAX_SAMPLE_RETRIES} consecutive sample failures. Examples: {' | '.join(failures[:3])}"
        )
