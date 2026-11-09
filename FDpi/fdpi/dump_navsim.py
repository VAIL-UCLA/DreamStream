"""Extract one policy's features on NAVSIM navtest, for every scene whose
current front-camera frame exists under --render-root."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader

from fdpi.agents import PANEL, build_agent
from fdpi.hooks import install_hook, save_features

logger = logging.getLogger(__name__)

SCENE_FILTER_YAML = Path(__file__).resolve().parent / "configs" / "navtest_scene_filter.yaml"
NAVSIM_CAMERA_HW = (1080, 1920)


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, scene_loader: SceneLoader, tokens: List[str], feature_builders, render_root=None):
        self._scene_loader = scene_loader
        self._tokens = tokens
        self._feature_builders = feature_builders
        self._render_root = render_root

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, idx):
        token = self._tokens[idx]
        agent_input = self._scene_loader.get_agent_input_from_token(token)
        if self._render_root is not None:
            # The loader silently falls back to the real image when a generated one is missing.
            image_path = Path(agent_input.cameras[-1].cam_f0.image_path)
            if not image_path.is_relative_to(self._render_root):
                raise RuntimeError(f"{token}: front camera was not loaded from the render root ({image_path})")
        features: Dict = {}
        for builder in self._feature_builders:
            features.update(builder.compute_features(agent_input))
        features["scenario_token"] = token
        # SparseDriveV2 loads and preprocesses its images in a per-sample pipeline.
        if hasattr(self._feature_builders[0], "pipeline"):
            features, _, _ = self._feature_builders[0].pipeline(features, {}, token, test_mode=True)
        return features, token


def _to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def rendered_tokens(scene_loader: SceneLoader, render_root: Path, scene_filter) -> List[str]:
    """navtest tokens whose current CAM_F0 frame was rendered, in scene-filter order."""
    frames_by_token = scene_loader.scene_frames_dicts
    kept = []
    for token in scene_filter.tokens:
        frames = frames_by_token.get(token)
        if frames is None:
            continue
        rel = frames[scene_filter.num_history_frames - 1]["cams"]["CAM_F0"]["data_path"]
        if (render_root / rel).is_file():
            kept.append(token)
    return kept


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", required=True, choices=sorted(PANEL))
    p.add_argument("--frame-source", required=True, choices=["original", "rendered"])
    p.add_argument("--navsim-root", required=True, type=Path,
                   help="OpenScene/NAVSIM data root containing navsim_logs/test and sensor_blobs/test")
    p.add_argument("--render-root", required=True, type=Path,
                   help="Generated frames; also defines the evaluated token set for --frame-source original")
    p.add_argument("--ckpt-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=None, help="Default: per-policy value in fdpi.agents.PANEL")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    spec = PANEL[args.agent]
    agent = build_agent(args.agent, args.ckpt_dir, args.device,
                        image_resize_hw=NAVSIM_CAMERA_HW if spec["resize"] else None)
    hook_spec, capture, handle = install_hook(agent)

    sensor_config = agent.get_sensor_config()
    if args.agent == "sd":
        # SparseDriveV2 only reads the current front camera; skip loading the rest.
        sensor_config = SensorConfig.build_no_sensors()
        sensor_config.cam_f0 = [3]

    scene_filter = instantiate(OmegaConf.load(SCENE_FILTER_YAML))
    rendered = args.frame_source == "rendered"
    scene_loader = SceneLoader(
        data_path=args.navsim_root / "navsim_logs" / "test",
        sensor_blobs_path=args.navsim_root / "sensor_blobs" / "test",
        scene_filter=scene_filter,
        sensor_config=sensor_config,
        render_root=args.render_root if rendered else None,
    )
    tokens = rendered_tokens(scene_loader, args.render_root, scene_filter)
    logger.info("%d / %d navtest tokens have a rendered frame", len(tokens), len(scene_loader.tokens))
    if not tokens:
        raise SystemExit(f"No rendered frames found under {args.render_root} "
                         "(expected <render_root>/<log_name>/CAM_F0/<image_token>.jpg)")

    dataset = FeatureDataset(scene_loader, tokens, agent.get_feature_builders(),
                             render_root=args.render_root if rendered else None)
    loader = DataLoader(dataset, batch_size=args.batch_size or spec["batch_size"], shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        prefetch_factor=1 if args.num_workers > 0 else None)

    features = {}
    with torch.no_grad():
        for batch_features, batch_tokens in tqdm(loader, desc=f"{args.agent}/{args.frame_source}"):
            capture.reset()
            agent.forward(_to_device(batch_features, args.device))
            for tok, vec in zip(batch_tokens, capture.take(len(batch_tokens))):
                features[tok] = vec
    handle.remove()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = save_features(args.output, tokens, features, {}, args.agent, args.frame_source, hook_spec)
    logger.info("Wrote %s (%d samples)", args.output, n)


if __name__ == "__main__":
    main()
