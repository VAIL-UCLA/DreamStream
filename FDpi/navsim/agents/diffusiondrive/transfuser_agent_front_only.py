from typing import List

import torch.multiprocessing

# Front-only camera tensors are ~1.9x larger than the original 3-cam stitched
# tensor; the default file_descriptor strategy quickly exhausts /dev/shm under
# DDP with multiple DataLoader workers. file_system uses $TMPDIR-backed files
# instead, which are bounded by disk rather than tmpfs size.
torch.multiprocessing.set_sharing_strategy("file_system")

from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
from navsim.agents.diffusiondrive.transfuser_features_front_only import (
    TransfuserFrontOnlyFeatureBuilder,
)
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder


class DiffusionDriveFrontOnlyAgent(TransfuserAgent):
    """DiffusionDrive variant that uses only the front camera (cam_f0).

    Inherits the full model, loss, and optimizer setup from TransfuserAgent.
    Differences:
      - Sensor config loads only cam_f0 (no L/R/back cameras, no lidar).
      - Feature builder produces only the front view and a zero lidar
        placeholder (overwritten by the learned latent in the backbone
        when config.latent=True).
    """

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig(
            cam_f0=[3],
            cam_l0=False,
            cam_l1=False,
            cam_l2=False,
            cam_r0=False,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=False,
        )

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [TransfuserFrontOnlyFeatureBuilder(config=self._config)]
