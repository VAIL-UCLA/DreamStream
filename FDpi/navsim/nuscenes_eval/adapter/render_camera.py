"""Substitute the nuScenes CAM_FRONT image with a generated frame.

The policies only see a camera image on the current (last) history frame; ego
pose, velocity and driving command come from nuScenes. Substitution therefore
happens at one seam: the `Camera` built for the current keyframe.

Expected layout under the render root (N is the 10 Hz camera frame index;
keyframes are every 5th frame, so N = 5 * keyframe_idx):

    <render_root>/scene-XXXX/frame-N/output.png

Intrinsics: the real CAM_FRONT K, rescaled from 1600x900 to the generated
image's pixel size. Extrinsics: the real CAM_FRONT mount, unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
from PIL import Image

from navsim.common.dataclasses import Camera

from .agent_input import _cam_to_lidar_rt


# nuScenes CAM_FRONT native dimensions (uniform across all logs in v1.0-trainval).
_NUSCENES_NATIVE_W = 1600
_NUSCENES_NATIVE_H = 900


@dataclass(frozen=True)
class RenderProvider:
    """Locates and loads rendered front-camera frames (any resolution)."""

    root: Path
    image_name: str = "output.png"

    def path_for(self, scene_name: str, keyframe_idx: int) -> Path:
        frame_n = int(keyframe_idx) * 5
        return Path(self.root) / scene_name / f"frame-{frame_n}" / self.image_name

    def load(self, scene_name: str, keyframe_idx: int) -> Optional[npt.NDArray[np.uint8]]:
        p = self.path_for(scene_name, keyframe_idx)
        if not p.exists():
            return None
        return np.array(Image.open(p).convert("RGB"))


def _scale_intrinsics_to_render(
    intrinsics_native: npt.NDArray[np.float64],
    render_h: int,
    render_w: int,
) -> npt.NDArray[np.float64]:
    """Rescale nuScenes-native K (defined for 1600x900) to the rendered image's
    pixel size. Non-uniform when the renderer's aspect != 1.7778."""
    K = intrinsics_native.copy()
    sx = render_w / _NUSCENES_NATIVE_W
    sy = render_h / _NUSCENES_NATIVE_H
    K[0, 0] *= sx  # fx
    K[1, 1] *= sy  # fy
    K[0, 2] *= sx  # cx
    K[1, 2] *= sy  # cy
    return K


def build_front_camera_rendered(
    nusc,
    sample_token: str,
    scene_name: str,
    keyframe_idx: int,
    provider: RenderProvider,
) -> Optional[Camera]:
    """Like `build_front_camera`, but pulls the image from `provider` instead
    of nuScenes. Returns None if the render is missing (caller should skip).
    """
    image = provider.load(scene_name, keyframe_idx)
    if image is None:
        return None

    sample = nusc.get("sample", sample_token)
    sd_cam = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    sd_lid = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    cs_cam = nusc.get("calibrated_sensor", sd_cam["calibrated_sensor_token"])
    intrinsics_native = np.asarray(cs_cam["camera_intrinsic"], dtype=np.float64)

    H_gen, W_gen = image.shape[:2]
    intrinsics = _scale_intrinsics_to_render(intrinsics_native, H_gen, W_gen)

    R_cl, t_cl = _cam_to_lidar_rt(nusc, sd_cam["token"], sd_lid["token"])

    # SparseDriveV2 re-reads the image from `image_path` instead of using
    # `image`; both refer to the same rendered file.
    return Camera(
        image=image,
        image_path=provider.path_for(scene_name, keyframe_idx),
        sensor2lidar_rotation=R_cl,
        sensor2lidar_translation=t_cl,
        intrinsics=intrinsics,
        distortion=np.zeros(5, dtype=np.float64),
    )
