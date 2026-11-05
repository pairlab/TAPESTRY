"""Build a Tapestry model from a training checkpoint ({'model': state_dict, 'config': resolved train config, ...})."""
import gc
import hashlib
import json
import os

import torch

from tapestry.models.base import Tapestry, build
from tapestry.utils.legacy_checkpoint import upgrade_legacy


def load_checkpoint_cpu(checkpoint_path):
    """Memory-mapped CPU load: no host-RAM spike for multi-GB checkpoints. Older training checkpoints are upgraded."""
    return upgrade_legacy(torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True))


def frozen_encoder_prefixes(model):
    """Fully frozen encoder prefixes omitted from checkpoints; these weights come from pretrained downloads."""
    prefixes = []
    for name, enc in model.encoders.items():
        params = list(enc.parameters())
        if params and all(not p.requires_grad for p in params):
            prefixes.append(f"encoders.{name}.")
    return prefixes


def load_state_dict_except_frozen_encoders(model, sd):
    """Strict load, except that frozen encoders may be absent from sd. assign=True: the checkpoint's tensors replace
    the fp16 skeleton's, so the model keeps the checkpoint's dtype."""
    result = model.load_state_dict(sd, strict=False, assign=True)
    prefixes = frozen_encoder_prefixes(model)
    missing = [k for k in result.missing_keys if not any(k.startswith(p) for p in prefixes)]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"state dict mismatch for {type(model).__name__}: missing={missing}, unexpected={list(result.unexpected_keys)}"
        )


def checkpoint_config(ck, t5_override):
    """ck['config'], with the frozen T5 encoder loaded from t5_override instead unless it is None (e.g.
    THUDM/CogVideoX-2b for its fp16 copy, see download_t5)."""
    cfg = ck["config"]
    if t5_override is None:
        return cfg
    text = {**cfg["encoders"]["text"], "model_name_or_path": t5_override}
    return {**cfg, "encoders": {**cfg["encoders"], "text": text}}


def load_unified_model(checkpoint_path, device, skip_text_encoder, t5_override):
    """Tapestry model in eval mode on `device`, built from checkpoint['config'] with weights from checkpoint['model'].

    skip_text_encoder=True omits T5; feed precomputed embeddings (encode_task_text_with_caching) as 'text/embeddings'.
    t5_override: see checkpoint_config.
    """
    ck = load_checkpoint_cpu(checkpoint_path)
    cfg = checkpoint_config(ck, t5_override)
    if skip_text_encoder:
        cfg = {**cfg, "encoders": {k: v for k, v in cfg["encoders"].items() if k != "text"}}
    model = Tapestry(cfg, from_checkpoint=True)
    load_state_dict_except_frozen_encoders(model, ck["model"])
    del ck
    gc.collect()
    return model.to(device).eval()


def encode_task_text_with_caching(checkpoint_path, text, cache_dir, use_cache, device, t5_override):
    """[1, L, D] embedding of `text` from the checkpoint's text encoder, cached on disk under cache_dir.

    The encoder is built and freed here, so it never shares GPU memory with the main model. t5_override: see
    checkpoint_config.
    """
    text_cfg = checkpoint_config(load_checkpoint_cpu(checkpoint_path), t5_override)["encoders"]["text"]
    identity = {k: text_cfg[k] for k in ("model_name_or_path", "max_seq_len", "torch_dtype")}
    key = hashlib.sha256(json.dumps({"text": text, **identity}, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    cache_path = os.path.join(cache_dir, f"{key}.pt")
    if use_cache and os.path.isfile(cache_path):
        print(f"Text embedding cache hit: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=True)["embedding"].to(device)

    encoder = build("encoders", text_cfg).to(device)  # in the checkpoint's T5 dtype, as when the policy holds T5
    with torch.no_grad():
        emb = encoder([text]).detach().float().cpu()  # [1, L, D]
    del encoder
    gc.collect()
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save({"text": text, "embedding": emb, **identity}, cache_path)
        print(f"Saved text embedding cache: {cache_path}")
    return emb.to(device)
