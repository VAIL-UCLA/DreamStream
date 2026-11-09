from typing import Dict

import cv2
import numpy as np
import torch
from torchvision import transforms

from navsim.agents.diffusiondrive.transfuser_features import TransfuserFeatureBuilder
from navsim.common.dataclasses import AgentInput


class TransfuserFrontOnlyFeatureBuilder(TransfuserFeatureBuilder):
    """Feature builder that uses only the front camera (cam_f0).

    Lidar is replaced by a learned latent inside the backbone when
    ``config.latent=True``, so this builder returns a zero placeholder for
    the lidar tensor and never reads any lidar point cloud from disk.
    """

    def get_unique_name(self) -> str:
        return "transfuser_front_only_feature"

    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        cameras = agent_input.cameras[-1]
        f0 = cameras.cam_f0.image[28:-28]
        resized_image = cv2.resize(
            f0, (self._config.camera_width, self._config.camera_height)
        )
        return transforms.ToTensor()(resized_image)

    def _get_lidar_feature(self, agent_input: AgentInput) -> torch.Tensor:
        # The backbone replaces this tensor with a learned latent when
        # config.latent=True; only the batch dimension is read.
        in_channels = 2 if self._config.use_ground_plane else 1
        return torch.zeros(
            in_channels,
            self._config.lidar_resolution_height,
            self._config.lidar_resolution_width,
            dtype=torch.float32,
        )
