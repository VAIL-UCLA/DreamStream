"""Scene-context feature taps for the FDπ policy panel.

For every policy we tap the feature the policy's trajectory / scoring head
consumes (after any BEV, 3D-anchor or cross-attention step, before the
trajectory module) with a forward (pre-)hook, and global-average-pool it over
all non-batch, non-channel axes into one D-dim vector per sample. No policy
code is modified.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _pool_last_dim(t: torch.Tensor) -> torch.Tensor:
    """(B, ..., D) -> (B, D)"""
    return t.flatten(start_dim=1, end_dim=-2).mean(dim=1)


def _pool_spatial(t: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, C)"""
    return t.flatten(start_dim=2).mean(dim=2)


def _pool_token(t: torch.Tensor) -> torch.Tensor:
    """(B, N, D) -> (B, D)"""
    return t.mean(dim=1)


POOL_FNS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "last_dim": _pool_last_dim,
    "spatial": _pool_spatial,
    "token": _pool_token,
}

# Keyed by agent class name.
#   model_attr:  attribute of the agent holding the nn.Module
#   module_path: dotted path (integer parts index into containers) to the tapped module
#   mode:        "output" = forward hook on the module's return value,
#                "input"  = forward pre-hook on positional arg `capture_idx`
HOOK_SPECS: Dict[str, Dict[str, Any]] = {
    "DrivoRAgent": {
        "model_attr": "_drivor_model",
        "module_path": "image_backbone",
        "mode": "output",
        "pool_name": "last_dim",
        "hook_desc": "drivor_model.image_backbone:output (image_scene_tokens)",
    },
    "DiffusionDriveFrontOnlyAgent": {
        "model_attr": "_transfuser_model",
        "module_path": "_trajectory_head",
        "mode": "input",
        "capture_idx": 2,
        "pool_name": "spatial",
        "hook_desc": "transfuser_model._trajectory_head:input[2] (cross_bev_feature)",
    },
    "LatentTransfuserFrontOnlyAgent": {
        "model_attr": "_transfuser_model",
        "module_path": "_trajectory_head",
        "mode": "input",
        "capture_idx": 0,
        "pool_name": "token",
        "hook_desc": "transfuser_model._trajectory_head:input[0] (trajectory_query)",
    },
    "RAPAgent": {
        "model_attr": "_rap_model",
        "module_path": "scorer",
        "mode": "input",
        "capture_idx": 1,
        "pool_name": "token",
        "hook_desc": "rap_model.scorer:input[1] (post-refiner bev_feature)",
    },
    "SparseDriveAgent": {
        "model_attr": "_sparsedrive_model",
        "module_path": "_trajectory_head.decoder.layers.0.p_deform_model",
        "mode": "output",
        "pool_name": "token",
        "hook_desc": "sparsedrive_model._trajectory_head.decoder.layers[0].p_deform_model:output",
    },
}


class FeatureCapture:
    """Holds the pooled (B, D) tensor captured during the latest forward pass."""

    def __init__(self):
        self.pooled: Optional[torch.Tensor] = None
        self.raw_shape: Optional[Tuple[int, ...]] = None

    def reset(self):
        self.pooled = None
        self.raw_shape = None

    def take(self, batch_size: int) -> np.ndarray:
        if self.pooled is None:
            raise RuntimeError("Feature hook did not fire during the agent forward pass.")
        if self.pooled.shape[0] != batch_size:
            raise RuntimeError(
                f"Hook captured {self.pooled.shape[0]} rows for a batch of {batch_size} "
                f"(raw shape {self.raw_shape}); the tapped module ran more than once per forward."
            )
        out = self.pooled.numpy().astype(np.float32)
        self.reset()
        return out


def _resolve_module(root: torch.nn.Module, path: str) -> torch.nn.Module:
    cur: Any = root
    for part in path.split("."):
        cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
    return cur


def install_hook(agent) -> Tuple[Dict[str, Any], FeatureCapture, Any]:
    """Attach the feature tap to `agent`. Returns (spec, capture, handle)."""
    name = type(agent).__name__
    if name not in HOOK_SPECS:
        raise ValueError(f"No feature tap defined for agent class {name!r}")
    spec = HOOK_SPECS[name]
    pool_fn = POOL_FNS[spec["pool_name"]]
    target = _resolve_module(getattr(agent, spec["model_attr"]), spec["module_path"])
    capture = FeatureCapture()

    def _store(t):
        assert isinstance(t, torch.Tensor), f"{spec['hook_desc']}: expected a Tensor, got {type(t)}"
        capture.raw_shape = tuple(t.shape)
        capture.pooled = pool_fn(t.detach()).float().cpu()

    if spec["mode"] == "output":
        handle = target.register_forward_hook(lambda module, args, output: _store(output))
    else:
        idx = spec["capture_idx"]
        handle = target.register_forward_pre_hook(
            lambda module, args, kwargs: _store(args[idx]), with_kwargs=True
        )
    return spec, capture, handle


def save_features(
    path,
    tokens: Sequence[str],
    features: Dict[str, np.ndarray],
    skip_reasons: Dict[str, str],
    agent: str,
    frame_source: str,
    spec: Dict[str, Any],
) -> int:
    """Write the per-sample features as an npz. Row i <-> tokens[i]; rows that
    could not be built are NaN with a non-empty `skip_reason`. Returns #valid rows."""
    if not features:
        raise RuntimeError("No features were captured.")
    dim = int(next(iter(features.values())).shape[0])
    feats = np.full((len(tokens), dim), np.nan, dtype=np.float32)
    reasons: List[str] = []
    for i, tok in enumerate(tokens):
        if tok in features:
            feats[i] = features[tok]
            reasons.append("")
        else:
            reasons.append(skip_reasons.get(tok, "missing"))
    np.savez(
        path,
        features=feats,
        sample_tokens=np.asarray(tokens, dtype="<U32"),
        skip_reason=np.asarray(reasons, dtype="<U32"),
        agent=np.asarray(agent),
        frame_source=np.asarray(frame_source),
        hook_module=np.asarray(spec["hook_desc"]),
    )
    return len(features)
