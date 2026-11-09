"""The FDπ policy panel."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from hydra.utils import instantiate
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).resolve().parent / "configs" / "agents"

# resize: the policy takes a fixed input resolution (image_resize_hw).
PANEL = {
    "drivor": dict(name="DrivoR", ckpt="drivor_Nav1_f0_25epochs.pth", batch_size=64, resize=False),
    "dd": dict(name="DiffusionDrive", ckpt="diffusiondrive_front_only_epoch_99_step_33300.ckpt", batch_size=64, resize=False),
    "ltf": dict(name="LTF", ckpt="ltf_epoch99_step_11100.ckpt", batch_size=64, resize=False),
    "rap": dict(name="RAP", ckpt="RAP_DINO_navsimv1.ckpt", batch_size=64, resize=True),
    "sd": dict(name="SparseDriveV2", ckpt="sparsedrivev2_epoch_9_step_6650.ckpt", batch_size=16, resize=True),
}


def build_agent(key: str, ckpt_dir: Path, device: str, image_resize_hw: Optional[Tuple[int, int]] = None):
    """Instantiate a panel policy from its config and load its checkpoint."""
    spec = PANEL[key]
    cfg = OmegaConf.load(CONFIG_DIR / f"{key}.yaml")
    cfg.ckpt_dir = str(Path(ckpt_dir).resolve())
    OmegaConf.resolve(cfg)
    del cfg["ckpt_dir"]
    cfg.checkpoint_path = str(Path(ckpt_dir).resolve() / spec["ckpt"])
    if not Path(cfg.checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_path}")
    if spec["resize"]:
        cfg.config.image_resize_hw = list(image_resize_hw) if image_resize_hw else None
    agent = instantiate(cfg)
    agent.initialize()
    return agent.eval().to(device)
