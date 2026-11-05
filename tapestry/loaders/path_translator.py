"""Map the video paths of the metadata CSVs to the on-disk layout (frame folders or MP4s, relative to root_dir)."""

import os
import random
from typing import Optional

# Camera views of the multi-view datasets, in view_idx order.
MULTIVIEW_DATASETS = {
    "droid": ["camera_0", "camera_1", "camera_2"],
    "bridge": ["images0", "images1", "images2"],
    "agibot": ["front_img_1", "left_wrist_img", "right_wrist_img"],
}
AGIBOT_VIDEO_STEMS = ["head_color", "hand_left_color", "hand_right_color"]  # file stems, in view_idx order


def translate_droid(meta_path: str, view_idx: Optional[int]) -> str:
    """droid_raw/1.0.1/<lab>/success/<date>/<session>/recordings/MP4/<camera>.mp4
    -> DROID/<lab>/success/<date>/<session>/frames/camera_<view_idx>/left_rgb_<n>.jpg; None picks a random view."""
    parts = meta_path.replace("\\", "/").split("/")
    if len(parts) < 6:
        raise ValueError(f"Cannot parse DROID path: {meta_path}")
    lab, status, date, session = parts[2:6]
    if view_idx is None:
        view_idx = random.choice([0, 1, 2])
    return f"DROID/{lab}/{status}/{date}/{session}/frames/camera_{view_idx}"


def translate_ego4d(meta_path: str) -> str:
    """raw/v2/clips/<uuid>.mp4 -> Ego4D_mp4/<uuid>.mp4"""
    uuid = os.path.basename(meta_path).replace(".mp4", "")
    return f"Ego4D_mp4/{uuid}.mp4"


def translate_epic_kitchens(meta_path: str) -> str:
    """EPIC-KITCHENS/P01/videos/P01_01.MP4 -> epic_kitchens/P01/P01_01/frame_<n>.jpg"""
    parts = meta_path.replace("\\", "/").split("/")
    video_id = parts[3].replace(".MP4", "").replace(".mp4", "")
    return f"epic_kitchens/{parts[1]}/{video_id}"


def translate_ssv2(meta_path: str) -> str:
    """20bn-something-something-v2/78687.mp4 -> ssv2/78687/78687_idx=<n>.jpg"""
    video_id = os.path.basename(meta_path).replace(".mp4", "")
    return f"ssv2/{video_id}"


def translate_bridge(meta_path: str, view_idx: Optional[int]) -> str:
    """<trajectory>/images0 -> <trajectory>/images<view_idx>/im_<n>.jpg; None picks a random view.
    Most Bridge subsets only have images0."""
    if view_idx is None:
        view_idx = random.choice([0, 1, 2])
    return meta_path.replace("images0", f"images{view_idx}")


def translate_pandas(meta_path: str) -> str:
    """Panda-70M clips are stored at their metadata paths."""
    return meta_path


def translate_agibot(meta_path: str, view_idx: Optional[int]) -> str:
    """observations/<task>/<episode>/videos/head_color[_reencoded].mp4 -> the video of camera view_idx in the same folder;
    None keeps the metadata camera."""
    path = meta_path.replace("\\", "/")
    if view_idx is None:
        return path
    if not path.endswith(".mp4"):
        raise ValueError(f"AgiBot metadata path must end with .mp4, got '{meta_path}'")
    suffix = "_reencoded.mp4" if path.endswith("_reencoded.mp4") else ".mp4"
    return f"{path.rsplit('/', 1)[0]}/{AGIBOT_VIDEO_STEMS[view_idx]}{suffix}"


TRANSLATORS = {
    "droid": translate_droid,
    "ego4d": translate_ego4d,
    "epic_kitchens": translate_epic_kitchens,
    "ssv2": translate_ssv2,
    "something_something_v2": translate_ssv2,
    "bridge": translate_bridge,
    "pandas": translate_pandas,
    "agibot": translate_agibot,
}


def translate_path(dataset_name: str, meta_path: str, view_idx: Optional[int]) -> str:
    """Path of a clip relative to root_dir; view_idx selects the camera of a multi-view dataset."""
    name = dataset_name.lower()
    if name not in TRANSLATORS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Known: {list(TRANSLATORS)}")
    if name in MULTIVIEW_DATASETS:
        return TRANSLATORS[name](meta_path, view_idx)
    return TRANSLATORS[name](meta_path)
