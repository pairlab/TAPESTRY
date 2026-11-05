"""Checkpoint-only inference: the text encoder can be left out, and weights load strictly."""
import pytest
import torch
from torch import nn

from tapestry.models import model_loading


class _Model(nn.Module):
    def __init__(self, cfg, from_checkpoint):
        super().__init__()
        assert from_checkpoint
        self.cfg = cfg
        self.encoders = nn.ModuleDict({name: nn.Linear(1, 1).requires_grad_(False) for name in cfg["encoders"]})
        self.head = nn.Linear(4, 2)


def _checkpoint(tmp_path, monkeypatch, extra):
    cfg = {"encoders": {"text": {"model_name_or_path": "google/t5-v1_1-xxl", "max_seq_len": 6}}, "decoders": {}}
    source = _Model(cfg, True)
    sd = {k: v for k, v in source.state_dict().items() if not k.startswith("encoders.")}  # frozen encoders are not stored
    path = tmp_path / "checkpoint.pt"
    torch.save({"config": cfg, "model": {**sd, **extra}}, path)
    monkeypatch.setattr(model_loading, "Tapestry", _Model)
    return cfg, source, path


@pytest.mark.parametrize("skip_text_encoder", [False, True])
def test_load_unified_model(monkeypatch, tmp_path, skip_text_encoder):
    cfg, source, path = _checkpoint(tmp_path, monkeypatch, {})
    loaded = model_loading.load_unified_model(path, "cpu", skip_text_encoder=skip_text_encoder, t5_override=None)
    assert list(loaded.encoders) == ([] if skip_text_encoder else ["text"])
    assert list(loaded.cfg["encoders"]) == list(loaded.encoders)
    assert not loaded.training
    torch.testing.assert_close(loaded.head.weight, source.head.weight, rtol=0, atol=0)


def test_unexpected_key_fails(monkeypatch, tmp_path):
    _, _, path = _checkpoint(tmp_path, monkeypatch, {"proj.text.bias": torch.zeros(4)})
    with pytest.raises(RuntimeError, match="proj.text.bias"):
        model_loading.load_unified_model(path, "cpu", skip_text_encoder=True, t5_override=None)


def test_t5_override(monkeypatch, tmp_path):
    cfg, _, path = _checkpoint(tmp_path, monkeypatch, {})
    loaded = model_loading.load_unified_model(path, "cpu", skip_text_encoder=False, t5_override="THUDM/CogVideoX-2b")
    assert loaded.cfg["encoders"]["text"] == {"model_name_or_path": "THUDM/CogVideoX-2b", "max_seq_len": 6}
    ck = {"config": cfg}
    assert model_loading.checkpoint_config(ck, None) is cfg
    model_loading.checkpoint_config(ck, "THUDM/CogVideoX-2b")
    assert cfg["encoders"]["text"]["model_name_or_path"] == "google/t5-v1_1-xxl"  # the checkpoint's config is not mutated
