"""FD_k and FDπ_k = FD_k / Tr(Σ_real) for one policy, on the samples valid in
both the real and the generated feature file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy import linalg


def _load(path: Path) -> Dict:
    z = np.load(path, allow_pickle=False)
    return {
        "features": z["features"],
        "sample_tokens": z["sample_tokens"],
        "skip_reason": z["skip_reason"],
        "agent": str(z["agent"]),
        "hook_module": str(z["hook_module"]),
        "frame_source": str(z["frame_source"]),
    }


def _paired_rows(a: Dict, b: Dict) -> Tuple[np.ndarray, np.ndarray]:
    a_idx = {t: i for i, t in enumerate(a["sample_tokens"]) if a["skip_reason"][i] == ""}
    b_idx = {t: i for i, t in enumerate(b["sample_tokens"]) if b["skip_reason"][i] == ""}
    common = sorted(set(a_idx) & set(b_idx))
    if not common:
        raise ValueError("No token is valid in both feature files.")
    a_rows = np.stack([a["features"][a_idx[t]] for t in common]).astype(np.float64)
    b_rows = np.stack([b["features"][b_idx[t]] for t in common]).astype(np.float64)
    return a_rows, b_rows


def _frechet(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> Tuple[float, bool]:
    """Fréchet distance between two Gaussians, with the usual FID epsilon fallback."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    epsilon_applied = not np.isfinite(covmean).all()
    if epsilon_applied:
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        rel = float(np.max(np.abs(covmean.imag))) / max(float(np.max(np.abs(covmean.real))), 1e-12)
        if rel > 1e-3:
            raise RuntimeError(f"sqrtm has a large imaginary component (max|imag|/max|real| = {rel:.3e})")
        covmean = covmean.real
    fd = float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))
    return fd, epsilon_applied


def frechet_distance(real_npz: Path, gen_npz: Path) -> Dict:
    """FD and normalized FDπ_k for one policy from its real / generated feature files."""
    a, b = _load(real_npz), _load(gen_npz)
    if (a["frame_source"], b["frame_source"]) != ("original", "rendered"):
        raise ValueError(f"Expected real then generated features, got {a['frame_source']!r} and {b['frame_source']!r}")
    if a["hook_module"] != b["hook_module"]:
        raise ValueError(f"Feature files come from different taps: {a['hook_module']!r} vs {b['hook_module']!r}")
    real, gen = _paired_rows(a, b)
    n, d = real.shape
    if n < 2:
        raise ValueError(f"Need at least 2 paired samples, got {n}")
    if not (np.isfinite(real).all() and np.isfinite(gen).all()):
        raise ValueError("Non-finite values in the extracted features")
    sigma_real = np.cov(real, rowvar=False, ddof=1)
    fd, epsilon_applied = _frechet(real.mean(axis=0), sigma_real, gen.mean(axis=0), np.cov(gen, rowvar=False, ddof=1))
    tr_sigma_real = float(np.trace(sigma_real))
    return {
        "agent": a["agent"],
        "n_paired": int(n),
        "feature_dim": int(d),
        "n_real_valid": int((a["skip_reason"] == "").sum()),
        "n_gen_valid": int((b["skip_reason"] == "").sum()),
        "fd": fd,
        "fdpi_k": fd / tr_sigma_real,
        "tr_sigma_real": tr_sigma_real,
        "epsilon_applied": epsilon_applied,
    }


def main():
    p = argparse.ArgumentParser(description="FD / FDπ_k for one policy from two feature files.")
    p.add_argument("real_npz", type=Path)
    p.add_argument("gen_npz", type=Path)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    res = frechet_distance(args.real_npz, args.gen_npz)
    print(json.dumps(res, indent=2))
    if args.json:
        args.json.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
