"""Extract one policy's features on nuScenes val keyframes with at least three
preceding keyframes. With --frame-source rendered, the current CAM_FRONT image
is replaced by <render_root>/<scene>/frame-<5*keyframe_idx>/<image_name>."""
from __future__ import annotations

# nuscenes-devkit uses matplotlib style names that newer matplotlib renamed.
import matplotlib.style as _mpl_style

_mpl_style_use = _mpl_style.use


def _mpl_use_compat(style, *args, **kwargs):
    if isinstance(style, str) and style.startswith("seaborn-") and not style.startswith("seaborn-v0_8-"):
        style = "seaborn-v0_8-" + style[len("seaborn-"):]
    return _mpl_style_use(style, *args, **kwargs)


_mpl_style.use = _mpl_use_compat

import argparse
import logging
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import torch
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from navsim.nuscenes_eval.adapter.agent_input import build_agent_input
from navsim.nuscenes_eval.adapter.render_camera import RenderProvider, build_front_camera_rendered
from navsim.nuscenes_eval.val_split import get_val_sample_metadata

from fdpi.agents import PANEL, build_agent
from fdpi.dump_navsim import _to_device
from fdpi.hooks import install_hook, save_features

logger = logging.getLogger(__name__)

NUSCENES_CAMERA_HW = (900, 1600)


def _resize_camera_in_lockstep(cam, target_hw: Tuple[int, int]) -> None:
    """Resize cam.image to target_hw and rescale cam.intrinsics by the same factors."""
    h, w = cam.image.shape[:2]
    th, tw = target_hw
    if (h, w) == (th, tw):
        return
    cam.image = cv2.resize(cam.image, (tw, th), interpolation=cv2.INTER_LANCZOS4)
    K = cam.intrinsics.copy()
    K[0, 0] *= tw / float(w)
    K[0, 2] *= tw / float(w)
    K[1, 1] *= th / float(h)
    K[1, 2] *= th / float(h)
    cam.intrinsics = K


def _prepare(meta, nusc, can_bus, dataroot, feature_builders,
             provider: Optional[RenderProvider], resize_hw: Optional[Tuple[int, int]]):
    """Build the policy features for one keyframe. Returns (features, skip_reason)."""
    try:
        front_cam = None
        if provider is not None:
            front_cam = build_front_camera_rendered(nusc, meta.sample_token, meta.scene_name,
                                                    meta.keyframe_idx, provider)
            if front_cam is None:
                return None, "missing_render"
            if resize_hw is not None:
                _resize_camera_in_lockstep(front_cam, resize_hw)
        agent_input = build_agent_input(nusc=nusc, can_bus_api=can_bus, dataroot=dataroot,
                                        sample_token=meta.sample_token,
                                        front_camera_override=front_cam)
        if agent_input is None:
            return None, "insufficient_history"
        features: Dict = {}
        for builder in feature_builders:
            features.update(builder.compute_features(agent_input))
        # SparseDriveV2 loads and preprocesses its images in a per-sample pipeline.
        if hasattr(feature_builders[0], "pipeline"):
            features, _, _ = feature_builders[0].pipeline(features, {}, meta.sample_token, test_mode=True)
        return features, ""
    except Exception:
        logger.warning("Failed to prepare %s:\n%s", meta.sample_token, traceback.format_exc())
        return None, "prep_exception"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", required=True, choices=sorted(PANEL))
    p.add_argument("--frame-source", required=True, choices=["original", "rendered"])
    p.add_argument("--dataroot", required=True, type=Path,
                   help="nuScenes root containing v1.0-trainval, samples and can_bus")
    p.add_argument("--version", default="v1.0-trainval")
    p.add_argument("--render-root", type=Path, default=None)
    p.add_argument("--render-image-name", default="output.png")
    p.add_argument("--ckpt-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=None, help="Default: per-policy value in fdpi.agents.PANEL")
    p.add_argument("--num-workers", type=int, default=8, help="Threads preparing samples")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    provider = None
    if args.frame_source == "rendered":
        if args.render_root is None or not args.render_root.is_dir():
            raise SystemExit(f"--frame-source rendered needs an existing --render-root (got {args.render_root})")
        provider = RenderProvider(root=args.render_root, image_name=args.render_image_name)

    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=False)
    can_bus = NuScenesCanBus(dataroot=str(args.dataroot))
    samples = get_val_sample_metadata(nusc)
    logger.info("nuScenes val keyframes: %d", len(samples))

    spec = PANEL[args.agent]
    resize_hw = NUSCENES_CAMERA_HW if spec["resize"] else None
    agent = build_agent(args.agent, args.ckpt_dir, args.device, image_resize_hw=resize_hw)
    hook_spec, capture, handle = install_hook(agent)
    feature_builders = list(agent.get_feature_builders())
    batch_size = args.batch_size or spec["batch_size"]

    features, skip_reasons = {}, {}
    pending_tokens, pending_feats = [], []

    def flush():
        if not pending_feats:
            return
        capture.reset()
        with torch.inference_mode():
            agent(_to_device(default_collate(pending_feats), args.device))
        for tok, vec in zip(pending_tokens, capture.take(len(pending_tokens))):
            features[tok] = vec
        pending_tokens.clear()
        pending_feats.clear()

    # Prepare samples in parallel but batch them in val order.
    window = max(4 * batch_size, 32)
    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as pool:
        queue = deque()
        it = iter(samples)
        for meta in it:
            queue.append((meta, pool.submit(_prepare, meta, nusc, can_bus, args.dataroot,
                                            feature_builders, provider, resize_hw)))
            if len(queue) >= window:
                break
        with tqdm(total=len(samples), desc=f"{args.agent}/{args.frame_source}") as bar:
            while queue:
                meta, fut = queue.popleft()
                nxt = next(it, None)
                if nxt is not None:
                    queue.append((nxt, pool.submit(_prepare, nxt, nusc, can_bus, args.dataroot,
                                                   feature_builders, provider, resize_hw)))
                feats, reason = fut.result()
                if feats is None:
                    skip_reasons[meta.sample_token] = reason
                else:
                    pending_tokens.append(meta.sample_token)
                    pending_feats.append(feats)
                    if len(pending_feats) >= batch_size:
                        flush()
                bar.update(1)
    flush()
    handle.remove()

    counts = {}
    for r in skip_reasons.values():
        counts[r] = counts.get(r, 0) + 1
    logger.info("Captured %d samples; skipped %s", len(features), counts or "none")
    if counts.get("prep_exception"):
        logger.warning("%d samples failed to prepare (see warnings above)", counts["prep_exception"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_features(args.output, [m.sample_token for m in samples], features, skip_reasons,
                  args.agent, args.frame_source, hook_spec)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
