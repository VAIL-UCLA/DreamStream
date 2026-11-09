"""Compute FDπ for one set of generated frames.

For each panel policy, extract features on real and generated frames and
compute FDπ_k; FDπ is their mean. Feature files already in --output-dir are
reused, so use one --output-dir per set of generated frames.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from fdpi.agents import PANEL
from fdpi.frechet import frechet_distance


def _dump(args, agent: str, frame_source: str, out: Path) -> None:
    if out.is_file():
        print(f"[fdpi] reusing {out}", flush=True)
        return
    tmp = out.with_name(out.stem + ".partial.npz")
    cmd = [sys.executable, "-m", f"fdpi.dump_{args.dataset}", "--agent", agent,
           "--frame-source", frame_source, "--render-root", str(args.render_root),
           "--ckpt-dir", str(args.ckpt_dir), "--output", str(tmp),
           "--num-workers", str(args.num_workers), "--device", args.device]
    if args.dataset == "navsim":
        cmd += ["--navsim-root", str(args.navsim_root)]
    else:
        cmd += ["--dataroot", str(args.dataroot), "--render-image-name", args.render_image_name]
    print(f"[fdpi] {agent} / {frame_source}", flush=True)
    subprocess.run(cmd, check=True)
    tmp.rename(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["navsim", "nuscenes"])
    p.add_argument("--render-root", required=True, type=Path)
    p.add_argument("--ckpt-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--navsim-root", type=Path, help="NAVSIM data root (navsim_logs/test, sensor_blobs/test)")
    p.add_argument("--dataroot", type=Path, help="nuScenes root (v1.0-trainval, samples, can_bus)")
    p.add_argument("--render-image-name", default="output.png", help="nuScenes only")
    p.add_argument("--agents", default=",".join(PANEL), help="Comma-separated subset of the panel")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.dataset == "navsim" and args.navsim_root is None:
        p.error("--dataset navsim needs --navsim-root")
    if args.dataset == "nuscenes" and args.dataroot is None:
        p.error("--dataset nuscenes needs --dataroot")
    agents = list(dict.fromkeys(a for a in args.agents.split(",") if a))
    unknown = sorted(set(agents) - set(PANEL))
    if unknown or not agents:
        p.error(f"--agents must be a non-empty subset of {list(PANEL)}")

    # Feature files in --output-dir are reused, so refuse to mix inputs.
    run_config = {"dataset": args.dataset, "render_root": str(args.render_root.resolve()),
                  "render_image_name": args.render_image_name if args.dataset == "nuscenes" else None}
    config_file = args.output_dir / "run_config.json"
    if config_file.is_file() and json.loads(config_file.read_text()) != run_config:
        p.error(f"{args.output_dir} holds results for different inputs ({config_file}); use a new --output-dir")
    for d in ("real", "generated"):
        (args.output_dir / d).mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps(run_config, indent=2))

    results = {}
    for agent in agents:
        real = args.output_dir / "real" / f"{agent}.npz"
        gen = args.output_dir / "generated" / f"{agent}.npz"
        _dump(args, agent, "original", real)
        _dump(args, agent, "rendered", gen)
        results[agent] = frechet_distance(real, gen)

    fdpi = sum(r["fdpi_k"] for r in results.values()) / len(results)
    summary = {
        "dataset": args.dataset,
        "render_root": str(args.render_root),
        "agents": agents,
        "fdpi_x100": 100 * fdpi,
        "per_policy": results,
    }
    (args.output_dir / "fdpi.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'policy':16s} {'N':>7s} {'D':>6s} {'FDpi_k x100':>12s}")
    for agent, r in results.items():
        flag = "  (N < D: rank-deficient covariance)" if r["n_paired"] < r["feature_dim"] else ""
        print(f"{PANEL[agent]['name']:16s} {r['n_paired']:7d} {r['feature_dim']:6d} {100 * r['fdpi_k']:12.2f}{flag}")
    label = "FDpi x100" if len(agents) == len(PANEL) else f"FDpi x100 ({len(agents)}/{len(PANEL)} policies)"
    print(f"{label:31s} {100 * fdpi:12.2f}")
    print(f"\nWrote {args.output_dir / 'fdpi.json'}")


if __name__ == "__main__":
    main()
