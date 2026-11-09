"""Build a navsim AgentInput from a nuScenes sample.

Front-camera-only configuration. Loads the last `num_history_frames` keyframes
(at 2 Hz, ~0.5 s spacing) ending at the target sample.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from PIL import Image
from pyquaternion import Quaternion

from nuplan.common.actor_state.state_representation import StateSE2

from navsim.common.dataclasses import (
    AgentInput,
    Camera,
    Cameras,
    EgoStatus,
    Lidar,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)

from .driving_command import driving_command_from_future
from .ego_state import global_pose_from_nuscenes


def _quat_to_matrix(q_wxyz) -> npt.NDArray[np.float64]:
    """nuScenes (w,x,y,z) quaternion -> 3x3 rotation matrix."""
    return Quaternion(q_wxyz).rotation_matrix


def _make_4x4(R: npt.NDArray[np.float64], t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _cam_to_lidar_rt(
    nusc, sample_data_token_cam: str, sample_data_token_lidar: str
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Build the cam → "lidar frame" transform that RAP / SparseDriveV2 / DrivoR
    expect as `Camera.sensor2lidar_*`.

    Important: navsim's "lidar frame" is actually the **vehicle / rear-axle**
    frame (cam_f0.sensor2lidar_translation ≈ [1.65, -0.01, 1.49] — i.e. cam
    1.65 m forward of rear axle, 1.49 m above ground; axes x=forward, y=left,
    z=up). It is NOT the LIDAR_TOP sensor frame. nuScenes' **ego** frame uses
    exactly this convention (rear axle, x=forward, y=left, z=up), whereas
    nuScenes' LIDAR_TOP frame is x=right, y=forward, z=up and sits at the
    physical lidar mount (z≈1.84 m).

    So we return cam → ego directly from `cs_cam`.

    The `sample_data_token_lidar` argument is unused; kept for call-site
    symmetry.
    """
    del sample_data_token_lidar  # not needed once we return cam → ego
    sd_cam = nusc.get("sample_data", sample_data_token_cam)
    cs_cam = nusc.get("calibrated_sensor", sd_cam["calibrated_sensor_token"])

    R_ce = _quat_to_matrix(cs_cam["rotation"])
    t_ce = np.asarray(cs_cam["translation"], dtype=np.float64)
    return R_ce, t_ce


def build_front_camera(
    nusc,
    dataroot: Path,
    sample_token: str,
) -> Camera:
    """Load CAM_FRONT image + intrinsics + cam->lidar transform."""
    sample = nusc.get("sample", sample_token)
    sd_cam = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    sd_lid = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])

    cs_cam = nusc.get("calibrated_sensor", sd_cam["calibrated_sensor_token"])
    intrinsics = np.asarray(cs_cam["camera_intrinsic"], dtype=np.float64)

    R_cl, t_cl = _cam_to_lidar_rt(nusc, sd_cam["token"], sd_lid["token"])

    image_path = Path(dataroot) / sd_cam["filename"]
    image = np.array(Image.open(image_path))

    return Camera(
        image=image,
        image_path=image_path,
        sensor2lidar_rotation=R_cl,
        sensor2lidar_translation=t_cl,
        intrinsics=intrinsics,
        distortion=np.zeros(5, dtype=np.float64),
    )


def _empty_camera() -> Camera:
    return Camera()


def _cameras_with_only_front(front: Camera) -> Cameras:
    e = _empty_camera
    return Cameras(
        cam_f0=front,
        cam_l0=e(), cam_l1=e(), cam_l2=e(),
        cam_r0=e(), cam_r1=e(), cam_r2=e(),
        cam_b0=e(),
    )


def collect_history_sample_tokens(
    nusc, sample_token: str, num_history_frames: int
) -> Optional[List[str]]:
    """
    Walk `sample.prev` to gather num_history_frames samples ending at sample_token.
    Returns None if there are not enough preceding samples.
    """
    tokens = [sample_token]
    cur = nusc.get("sample", sample_token)
    while len(tokens) < num_history_frames:
        if cur["prev"] == "":
            return None
        cur = nusc.get("sample", cur["prev"])
        tokens.append(cur["token"])
    tokens.reverse()
    return tokens


def collect_future_sample_tokens(
    nusc, sample_token: str, num_future_frames: int
) -> List[str]:
    """Walk `sample.next` to gather up to num_future_frames samples after sample_token."""
    tokens: List[str] = []
    cur = nusc.get("sample", sample_token)
    for _ in range(num_future_frames):
        if cur["next"] == "":
            break
        cur = nusc.get("sample", cur["next"])
        tokens.append(cur["token"])
    return tokens


def _ego_pose_for_sample(nusc, sample_token: str):
    """Return the ego_pose record associated with the LIDAR_TOP sample_data of a sample."""
    sample = nusc.get("sample", sample_token)
    sd_lid = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    return nusc.get("ego_pose", sd_lid["ego_pose_token"])


def _can_bus_velocity_acceleration(
    can_bus_api, scene_name: str, timestamp_us: int
) -> Tuple[Optional[npt.NDArray[np.float32]], Optional[npt.NDArray[np.float32]]]:
    """
    Look up linear velocity and acceleration in the ego frame from the CAN-bus
    'pose' messages at the given timestamp.

    :return: (velocity_xy [m/s], acceleration_xy [m/s^2]) or (None, None) on failure.
    """
    try:
        pose_msgs = can_bus_api.get_messages(scene_name, "pose")
    except Exception:
        return None, None
    if not pose_msgs:
        return None, None

    times = np.asarray([m["utime"] for m in pose_msgs], dtype=np.int64)
    idx = int(np.argmin(np.abs(times - timestamp_us)))
    msg = pose_msgs[idx]
    vel = np.asarray(msg.get("vel", [0.0, 0.0, 0.0]), dtype=np.float32)[:2]
    accel = np.asarray(msg.get("accel", [0.0, 0.0, 0.0]), dtype=np.float32)[:2]
    return vel, accel


def _ego_status_for_frame(
    nusc,
    can_bus_api,
    scene_name: str,
    sample_token: str,
    local_pose_3: npt.NDArray[np.float32],
    driving_command_4: npt.NDArray[np.int_],
) -> EgoStatus:
    pose_rec = _ego_pose_for_sample(nusc, sample_token)
    vel, accel = (None, None)
    if can_bus_api is not None:
        vel, accel = _can_bus_velocity_acceleration(
            can_bus_api, scene_name, int(pose_rec["timestamp"])
        )
    if vel is None:
        vel = np.zeros(2, dtype=np.float32)
    if accel is None:
        accel = np.zeros(2, dtype=np.float32)

    return EgoStatus(
        ego_pose=local_pose_3.astype(np.float32),
        ego_velocity=vel,
        ego_acceleration=accel,
        driving_command=driving_command_4.astype(int),
        in_global_frame=False,
    )


def build_agent_input(
    nusc,
    can_bus_api,
    dataroot: Path,
    sample_token: str,
    num_history_frames: int = 4,
    num_future_frames_for_command: int = 8,
    front_camera_override: Optional[Camera] = None,
) -> Optional[AgentInput]:
    """
    Build a navsim AgentInput from a nuScenes sample (front-camera-only).

    :param nusc: NuScenes(...) object
    :param can_bus_api: NuScenesCanBus(...) object
    :param dataroot: nuScenes dataset root (where sample_data filenames resolve)
    :param sample_token: target sample token
    :param num_history_frames: history length to load (default matches navsim)
    :param num_future_frames_for_command: how many future keyframes to use for
        the OpenScene-style driving command (8 frames * 0.5 s = 4 s lookahead)
    :param front_camera_override: use this Camera (e.g. a generated frame)
        instead of the real CAM_FRONT image on the current frame
    :return: AgentInput, or None if there is not enough history.
    """
    history_tokens = collect_history_sample_tokens(
        nusc, sample_token, num_history_frames
    )
    if history_tokens is None:
        return None

    # Global poses for all history frames.
    global_poses_history = []
    for tok in history_tokens:
        rec = _ego_pose_for_sample(nusc, tok)
        x, y, yaw = global_pose_from_nuscenes(rec["translation"], rec["rotation"])
        global_poses_history.append([x, y, yaw])
    global_poses_history = np.asarray(global_poses_history, dtype=np.float64)

    # Convert to ego-local frame relative to current (last history) frame.
    current_origin = StateSE2(*global_poses_history[-1])
    local_poses_history = convert_absolute_to_relative_se2_array(
        current_origin, global_poses_history
    )

    # Driving command from future trajectory in global frame.
    future_tokens = collect_future_sample_tokens(
        nusc, sample_token, num_future_frames_for_command
    )
    future_global = []
    for tok in future_tokens:
        rec = _ego_pose_for_sample(nusc, tok)
        x, y, yaw = global_pose_from_nuscenes(rec["translation"], rec["rotation"])
        future_global.append([x, y, yaw])
    future_global_arr = np.asarray(future_global, dtype=np.float64) if future_global else np.zeros((0, 3))
    cmd = driving_command_from_future(current_origin, future_global_arr)

    scene_token = nusc.get("sample", sample_token)["scene_token"]
    scene_name = nusc.get("scene", scene_token)["name"]

    ego_statuses: List[EgoStatus] = []
    cameras_list: List[Cameras] = []
    lidars_list: List[Lidar] = []
    for i, tok in enumerate(history_tokens):
        es = _ego_status_for_frame(
            nusc, can_bus_api, scene_name, tok,
            local_poses_history[i].astype(np.float32),
            cmd,
        )
        ego_statuses.append(es)
        # Front-camera image only on the last history frame, matching
        # the navsim cam_f0:[3] sensor config.
        if i == len(history_tokens) - 1:
            if front_camera_override is not None:
                front_cam = front_camera_override
            else:
                front_cam = build_front_camera(nusc, dataroot, tok)
            cameras_list.append(_cameras_with_only_front(front_cam))
        else:
            cameras_list.append(_cameras_with_only_front(_empty_camera()))
        lidars_list.append(Lidar())

    return AgentInput(ego_statuses=ego_statuses, cameras=cameras_list, lidars=lidars_list)
