from typing import Optional

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import StateSE2

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)


LOOKAHEAD_DISTANCE_M: float = 20.0
LATERAL_THRESHOLD_M: float = 2.0


def driving_command_from_future(
    current_pose_global: StateSE2,
    future_poses_global: npt.NDArray[np.float64],
    distance: float = LOOKAHEAD_DISTANCE_M,
    lateral_offset: float = LATERAL_THRESHOLD_M,
) -> npt.NDArray[np.int_]:
    """
    Reproduce OpenScene's driving-command rule using the future ego trajectory
    as the lookahead path (since nuScenes has no route centerline).

    OpenScene reference: DriveEngine/process_data/helpers/driving_command.py:40-100
    Walk `distance` m along the path, express the target pose in the current
    ego SE(2) frame, and threshold lateral y by `lateral_offset`.

    :param current_pose_global: current global ego pose (x, y, heading)
    :param future_poses_global: (N, 3) array of future ego poses (x, y, heading)
        in the same global frame, ordered by time
    :param distance: longitudinal lookahead in meters
    :param lateral_offset: lateral threshold for left/right
    :return: one-hot (left, forward, right, unknown) length-4 int array
    """
    cmd = np.zeros((4,), dtype=int)

    if future_poses_global is None or len(future_poses_global) == 0:
        cmd[3] = 1
        return cmd

    target_xy = _interpolate_along_path(
        np.asarray(future_poses_global, dtype=np.float64),
        target_distance=distance,
    )
    if target_xy is None:
        cmd[3] = 1
        return cmd

    target_global = np.array([target_xy[0], target_xy[1], 0.0], dtype=np.float64)
    target_local = convert_absolute_to_relative_se2_array(
        current_pose_global, target_global[None, ...]
    )[0]

    y_local = target_local[1]
    if y_local >= lateral_offset:
        cmd[0] = 1   # left
    elif y_local <= -lateral_offset:
        cmd[2] = 1   # right
    else:
        cmd[1] = 1   # forward

    return cmd


def _interpolate_along_path(
    poses_xyh: npt.NDArray[np.float64],
    target_distance: float,
) -> Optional[npt.NDArray[np.float64]]:
    """
    Arc-length-interpolate a path of (x, y, heading) waypoints to `target_distance`.
    Falls back to the farthest waypoint if the path is shorter than `target_distance`.
    Returns None for degenerate inputs.
    """
    if poses_xyh.shape[0] < 1:
        return None

    xy = poses_xyh[:, :2]
    if poses_xyh.shape[0] == 1:
        return xy[0]

    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0.0:
        return xy[0]

    t = min(target_distance, total)
    idx = int(np.searchsorted(cum, t, side="right")) - 1
    idx = max(0, min(idx, len(seg) - 1))
    seg_len = seg[idx]
    if seg_len <= 0.0:
        return xy[idx]
    alpha = (t - cum[idx]) / seg_len
    return xy[idx] + alpha * (xy[idx + 1] - xy[idx])
