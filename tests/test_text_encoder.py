"""download_t5 fetches only the tokenizer and encoder files, and T5TextEncoder loads each from its own directory."""
from types import SimpleNamespace

import pytest
import torch

from tapestry.models.encoders import text


@pytest.mark.parametrize("name, tok_dir, enc_dir, patterns", [
    ("google/t5-v1_1-xxl", "snap", "snap", ["*.json", "*.model", "*.safetensors", "pytorch_model*.bin"]),
    ("THUDM/CogVideoX-2b", "snap/tokenizer", "snap/text_encoder", ["tokenizer/*", "text_encoder/*"]),
    ("zai-org/CogVideoX-2b", "snap/tokenizer", "snap/text_encoder", ["tokenizer/*", "text_encoder/*"]),
])
def test_download_t5(monkeypatch, name, tok_dir, enc_dir, patterns):
    calls = []
    monkeypatch.setattr(text, "snapshot_download", lambda repo_id, allow_patterns: calls.append((repo_id, allow_patterns)) or "snap")
    assert text.download_t5(name) == (tok_dir, enc_dir)
    assert calls == [(name, patterns)]


def test_local_directory_is_not_downloaded(monkeypatch, tmp_path):
    monkeypatch.setattr(text, "snapshot_download", None)
    assert text.download_t5(str(tmp_path)) == (str(tmp_path), str(tmp_path))


def test_encoder_loads_from_split_directories(monkeypatch):
    loads = []
    monkeypatch.setattr(text, "download_t5", lambda name: ("tok", "enc"))
    monkeypatch.setattr(text.T5Tokenizer, "from_pretrained", lambda path, legacy: loads.append(path))
    monkeypatch.setattr(text.T5EncoderModel, "from_pretrained", lambda path, torch_dtype: loads.append((path, torch_dtype)) or SimpleNamespace(encoder=torch.nn.Linear(1, 1)))
    enc = text.T5TextEncoder("THUDM/CogVideoX-2b", max_seq_len=4, torch_dtype="bfloat16")
    assert loads == ["tok", ("enc", torch.bfloat16)]
    assert not enc.training and not any(p.requires_grad for p in enc.parameters())
