"""Frozen T5 text encoder."""
import os
from typing import List, Tuple

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import T5EncoderModel, T5Tokenizer


def download_t5(name: str) -> Tuple[str, str]:
    """(tokenizer dir, encoder dir) of `name`: a local directory or HuggingFace repo holding both, such as
    google/t5-v1_1-xxl (45 GB, the encoder the checkpoints were trained with), or THUDM/CogVideoX-2b, whose fp16 copy
    of that encoder is a 9.5 GB download with slightly different embeddings. The main branch of google/t5-v1_1-xxl
    only has pytorch_model.bin, and from_pretrained on the repo id would also fetch a 45 GB safetensors copy in the
    background."""
    if os.path.isdir(name):
        return name, name
    if name in ("THUDM/CogVideoX-2b", "zai-org/CogVideoX-2b"):  # the same repo, renamed on HuggingFace
        path = snapshot_download(repo_id=name, allow_patterns=["tokenizer/*", "text_encoder/*"])
        return os.path.join(path, "tokenizer"), os.path.join(path, "text_encoder")
    path = snapshot_download(repo_id=name, allow_patterns=["*.json", "*.model", "*.safetensors", "pytorch_model*.bin"])
    return path, path


class T5TextEncoder(nn.Module):
    """B strings -> token embeddings [B, max_seq_len, D]. Frozen, so checkpoints do not store it."""

    def __init__(self, model_name_or_path: str, max_seq_len: int, torch_dtype: str):
        super().__init__()
        print(f"Loading T5 encoder from {model_name_or_path}")
        tok_dir, enc_dir = download_t5(model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(tok_dir, legacy=True)  # the default; passing it skips a warning
        self.model = T5EncoderModel.from_pretrained(enc_dir, torch_dtype=getattr(torch, torch_dtype)).encoder
        self.max_seq_len = max_seq_len
        self.requires_grad_(False)
        self.eval()

    def forward(self, text: List[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        tok = self.tokenizer(text, padding="max_length", max_length=self.max_seq_len, truncation=True, return_tensors="pt")
        out = self.model(input_ids=tok["input_ids"].to(device), attention_mask=tok["attention_mask"].to(device))
        return out.last_hidden_state  # [B, max_seq_len, D]
