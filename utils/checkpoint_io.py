"""
utils/checkpoint_io.py
Shared, secure, and auditable checkpoint I/O utilities for CoTOP and baseline agents.
Ensures strict validation, SHA-256 calculation, and eliminates silent fallbacks.
"""

import os
import hashlib
import json
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn

def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file on disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot compute hash; file does not exist: {filepath}")
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_model_param_hash(model: nn.Module) -> str:
    """Computes deterministic SHA-256 hash of model parameters."""
    hasher = hashlib.sha256()
    for p in model.parameters():
        hasher.update(p.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()

def load_checkpoint_strict(
    checkpoint_path: str,
    model: nn.Module,
    expected_algorithm: Optional[str] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Loads a checkpoint strictly from disk into the provided model.
    Terminates with hard exceptions if:
      - File does not exist
      - File is corrupted or unreadable
      - State dict keys are incompatible
      - Algorithm mismatch occurs
    
    NEVER silently falls back to random weights or default policies.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"[FATAL ERROR] Checkpoint not found at '{checkpoint_path}'. "
            f"Strict checkpoint loading forbids falling back to unverified or random weights."
        )

    file_size = os.path.getsize(checkpoint_path)
    file_sha256 = compute_file_sha256(checkpoint_path)

    try:
        ckpt_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(
            f"[FATAL ERROR] Failed to parse checkpoint at '{checkpoint_path}': {e}"
        ) from e

    # Extract state dict
    state_dict = None
    if isinstance(ckpt_data, dict):
        if "model_state_dict" in ckpt_data:
            state_dict = ckpt_data["model_state_dict"]
        elif "online_net_state_dict" in ckpt_data:
            state_dict = ckpt_data["online_net_state_dict"]
        elif "state_dict" in ckpt_data:
            state_dict = ckpt_data["state_dict"]
        elif all(isinstance(v, torch.Tensor) for v in ckpt_data.values()):
            state_dict = ckpt_data
        else:
            raise KeyError(
                f"[FATAL ERROR] Checkpoint dictionary at '{checkpoint_path}' does not contain valid state dict keys "
                f"(found keys: {list(ckpt_data.keys())})."
            )
    elif isinstance(ckpt_data, nn.Module):
        state_dict = ckpt_data.state_dict()
    else:
        raise TypeError(
            f"[FATAL ERROR] Unrecognized checkpoint payload type '{type(ckpt_data)}' in '{checkpoint_path}'."
        )

    # Validate algorithm if specified
    if expected_algorithm and isinstance(ckpt_data, dict) and "algorithm" in ckpt_data:
        saved_algo = ckpt_data["algorithm"]
        if saved_algo.lower() != expected_algorithm.lower():
            raise ValueError(
                f"[FATAL ERROR] Checkpoint algorithm mismatch: expected '{expected_algorithm}', "
                f"but checkpoint contains '{saved_algo}'."
            )

    # Ingest weights strictly
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        raise RuntimeError(
            f"[FATAL ERROR] Incompatible model architecture or state dict keys for '{checkpoint_path}': {e}"
        ) from e

    model.eval()

    metadata = {
        "checkpoint_path": checkpoint_path,
        "file_size_bytes": file_size,
        "checkpoint_sha256": file_sha256,
        "model_param_hash": compute_model_param_hash(model),
        "saved_metadata": {k: v for k, v in ckpt_data.items() if k not in ["model_state_dict", "online_net_state_dict", "target_net_state_dict", "optimizer_state_dict"]} if isinstance(ckpt_data, dict) else {}
    }
    return metadata
