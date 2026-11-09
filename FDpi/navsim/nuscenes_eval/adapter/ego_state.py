"""nuScenes -> nuPlan EgoState construction."""
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion

from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import (
    StateSE2,
    StateVector2D,
    TimePoint,
)
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters


def _renault_zoe_parameters() -> VehicleParameters:
    """
    Approximate Renault Zoe ZE40 parameters (the nuScenes ego rig).
    nuPlan's default Pacifica is 5.176 x 2.297 m -- ~25% larger than the Zoe
    in both dimensions, which makes the simulated ego footprint clip into
    walkways at lane edges and inflates DAC failures.
    """
    front_overhang = 0.800     # [m] rear axle -> front bumper minus wheelbase
    rear_overhang = 0.696      # [m] rear axle -> rear bumper
    wheel_base = 2.588         # [m]
    return VehicleParameters(
        vehicle_name="renault_zoe",
        vehicle_type="ev",
        width=1.730,
        front_length=wheel_base + front_overhang,   # 3.388
        rear_length=rear_overhang,                  # 0.696 (total length 4.084)
        wheel_base=wheel_base,
        cog_position_from_rear_axle=wheel_base / 2.0,  # ~1.294
        height=1.562,
    )


DEFAULT_VEHICLE_PARAMS: VehicleParameters = _renault_zoe_parameters()

# nuScenes ego_pose translation is at the ego frame origin. The widely-used
# convention in nuScenes is that the ego frame is at the rear-axle center,
# so we leave the offset at 0. Set this to a small positive number if you
# observe the ego footprint biased forward of the lane in visualizations.
EGO_TO_REAR_AXLE_OFFSET_M: float = 0.0


def yaw_from_quaternion(rotation_wxyz) -> float:
    """nuScenes stores quaternions as (w, x, y, z). Returns yaw around z."""
    return Quaternion(rotation_wxyz).yaw_pitch_roll[0]


def global_pose_from_nuscenes(
    translation_xyz, rotation_wxyz
) -> Tuple[float, float, float]:
    """Convert nuScenes ego_pose record fields into (x, y, yaw)."""
    yaw = yaw_from_quaternion(rotation_wxyz)
    x, y = float(translation_xyz[0]), float(translation_xyz[1])
    if EGO_TO_REAR_AXLE_OFFSET_M != 0.0:
        x -= EGO_TO_REAR_AXLE_OFFSET_M * np.cos(yaw)
        y -= EGO_TO_REAR_AXLE_OFFSET_M * np.sin(yaw)
    return x, y, yaw


def build_ego_state(
    translation_xyz,
    rotation_wxyz,
    timestamp_us: int,
    velocity_xy_local: Optional[npt.NDArray[np.float32]] = None,
    acceleration_xy_local: Optional[npt.NDArray[np.float32]] = None,
    angular_velocity: float = 0.0,
    angular_acceleration: float = 0.0,
    tire_steering_angle: float = 0.0,
    vehicle: VehicleParameters = DEFAULT_VEHICLE_PARAMS,
) -> EgoState:
    """
    Build a nuPlan EgoState from a nuScenes ego_pose record.

    :param translation_xyz: nuScenes 'translation' (length-3, meters, global)
    :param rotation_wxyz: nuScenes 'rotation' (length-4 quaternion w,x,y,z)
    :param timestamp_us: nuScenes 'timestamp' in microseconds
    :param velocity_xy_local: (vx, vy) in ego frame (m/s); 0 if unknown
    :param acceleration_xy_local: (ax, ay) in ego frame (m/s^2); 0 if unknown
    :return: EgoState representing the rear axle at this timestamp
    """
    x, y, yaw = global_pose_from_nuscenes(translation_xyz, rotation_wxyz)
    rear_axle = StateSE2(x, y, yaw)

    if velocity_xy_local is None:
        velocity_xy_local = np.zeros(2, dtype=np.float32)
    if acceleration_xy_local is None:
        acceleration_xy_local = np.zeros(2, dtype=np.float32)

    return EgoState.build_from_rear_axle(
        rear_axle_pose=rear_axle,
        rear_axle_velocity_2d=StateVector2D(
            float(velocity_xy_local[0]), float(velocity_xy_local[1])
        ),
        rear_axle_acceleration_2d=StateVector2D(
            float(acceleration_xy_local[0]), float(acceleration_xy_local[1])
        ),
        tire_steering_angle=float(tire_steering_angle),
        time_point=TimePoint(int(timestamp_us)),
        vehicle_parameters=vehicle,
        is_in_auto_mode=True,
        angular_vel=float(angular_velocity),
        angular_accel=float(angular_acceleration),
    )
