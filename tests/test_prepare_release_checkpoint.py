"""scripts/prepare_release_checkpoint.py: action post-processing rules and machine-path scrubbing."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("prep", Path(__file__).resolve().parents[1] / "scripts" / "prepare_release_checkpoint.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def libero_cfg():
    return {"training": {"dataset": {"datasets": {
        "libero_10": {"_class": "LiberoDataset", "active_modalities": ["video", "action", "text"]},
        "clips": {"_class": "VideoOnlyDataset", "active_modalities": ["video", "text"]},
    }}}}


def robot_cfg(stats_path):
    return {
        "experiment_name": "internal-run",
        "checkpoint": "s3://bucket/run/epoch_5.pt",
        "encoders": {"text": {"model_name_or_path": "/data/pretrained/t5-v1_1-xxl"}},
        "decoders": {"cogvideox": {"_class": "CogVideoXDecoder", "pretrained_model_name": "/data/pretrained/CogVideoX-2b/"}},
        "training": {
            "logging": {"save_dir": "checkpoints/internal-run"},
            "dataset": {
                "action_space": "eepose_6drot", "normalize_actions": True, "normalizer_stats_path": stats_path,
                "relative_rotation_actions": True, "root_dir": "/data/robot/training",
                "datasets": {"robot": {"_class": "RobotDataset", "active_modalities": ["video", "action", "text"],
                                       "normalizer_stats_path": "s3://bucket/other_stats.json", "batch_weight": 0}},
            },
        },
    }


def write_stats(path, action_space):
    path.write_text(json.dumps({"action_space": action_space, "normalize_dims": [0, 1, 2], "stats": {}}))
    return str(path)


def test_libero_actions_are_used_as_stored():
    cfg = libero_cfg()
    assert prep.action_spec(cfg) == (None, False)
    assert prep.load_normalizer(cfg, None, None) is None
    with pytest.raises(ValueError, match="did not normalize"):
        prep.load_normalizer(cfg, None, "stats.json")


def test_mixed_action_rules_fail():
    cfg = libero_cfg()
    cfg["training"]["dataset"]["datasets"]["demos"] = {"_class": "HDF5DemoDataset", "active_modalities": ["video", "action"],
                                                       "normalize_actions": True, "normalizer_stats_path": "s.json", "relative_rotation_actions": True}
    with pytest.raises(ValueError, match="disagree"):
        prep.action_spec(cfg)


def test_shared_action_space_embeds_the_run_normalizer(tmp_path):
    stats = write_stats(tmp_path / "normalizer_stats_eepose_6drot.json", "eepose_6drot")
    cfg = robot_cfg(stats)
    assert prep.action_spec(cfg) == (stats, True)
    assert prep.load_normalizer(cfg, stats, None)["action_space"] == "eepose_6drot"


def test_missing_stats_need_an_override(tmp_path):
    cfg = robot_cfg("/gone/normalizer_stats_eepose_6drot.json")
    with pytest.raises(FileNotFoundError, match="--normalizer-stats"):
        prep.load_normalizer(cfg, cfg["training"]["dataset"]["normalizer_stats_path"], None)
    good = write_stats(tmp_path / "copy.json", "eepose_6drot")
    assert prep.load_normalizer(cfg, "/gone/x.json", good)["action_space"] == "eepose_6drot"
    with pytest.raises(ValueError, match="action space"):
        prep.load_normalizer(cfg, "/gone/x.json", write_stats(tmp_path / "wrong.json", "libero"))


def test_release_config_has_public_sources_and_no_machine_paths():
    cfg = robot_cfg("/data/robot/normalizer_stats_eepose_6drot.json")
    frozen = json.dumps(cfg)
    rel = prep.release_config(cfg, "tapestry_test")
    assert json.dumps(cfg) == frozen  # input untouched
    assert rel["encoders"]["text"]["model_name_or_path"] == "google/t5-v1_1-xxl"
    assert rel["decoders"]["cogvideox"]["pretrained_model_name"] == "THUDM/CogVideoX-2b"
    assert rel["checkpoint"] == "epoch_5.pt"
    assert rel["training"]["dataset"]["root_dir"] == "training"
    assert rel["training"]["dataset"]["datasets"]["robot"]["normalizer_stats_path"] == "other_stats.json"
    assert not [s for s in json.dumps(rel).split('"') if s.startswith(("/", "~", "s3://"))]
    with pytest.raises(ValueError, match="no public source"):
        prep.public_source("/data/pretrained/some-other-model")


def test_release_config_replaces_the_experiment_name():
    rel = prep.release_config(robot_cfg("stats.json"), "tapestry_test")
    assert rel["experiment_name"] == "tapestry_test"
    assert rel["training"]["logging"]["save_dir"] == "checkpoints/tapestry_test"
    assert "internal" not in json.dumps(rel)
