from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch

from navsim.agents.diffusiondrive.transfuser_features_front_only import (
    TransfuserFrontOnlyFeatureBuilder,
)
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.agents.transfuser.transfuser_loss import transfuser_loss
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder

# Front-only camera tensors are ~1.9x larger than the original 3-cam stitched
# tensor; the default file_descriptor strategy quickly exhausts /dev/shm under
# DDP with multiple DataLoader workers. file_system uses $TMPDIR-backed files
# instead, which are bounded by disk rather than tmpfs size.
torch.multiprocessing.set_sharing_strategy("file_system")


class LatentTransfuserFrontOnlyAgent(TransfuserAgent):
    """Latent TransFuser variant that uses only the front camera (cam_f0).

    Inherits the upstream TransFuser model, loss, and optimizer (with
    ``config.latent=True`` the backbone replaces the LiDAR input with a
    learned latent parameter and the loss filters agents outside the
    front camera FoV).

    Reuses the DiffusionDrive front-only feature builder so the cached
    features produced by the DiffusionDrive front-only run are picked
    up unchanged: same ``get_unique_name()`` and same tensor shapes.

    Overrides ``forward`` and ``compute_loss`` to match the
    DiffusionDrive AgentLightningModule contract (which calls
    ``forward(features, targets)`` and expects a dict from
    ``compute_loss`` with a ``loss`` key).
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

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        # Upstream TransfuserModel only consumes features; targets are
        # accepted (and ignored) so this agent works with DiffusionDrive's
        # AgentLightningModule, which always passes (features, targets).
        return self._transfuser_model(features)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # DiffusionDrive's lightning module expects a dict and reads
        # ``loss_dict['loss']``; the upstream transfuser_loss returns a
        # scalar tensor, so wrap it.
        loss = transfuser_loss(targets, predictions, self._config)
        return {"loss": loss}

    def get_training_callbacks(self) -> List[pl.Callback]:
        # The upstream TransfuserCallback hard-codes the 3-cam stitched
        # canvas (256x768 tile with a 128x512 camera region) and crashes
        # on the front-only (3, 512, 960) tensor. Disable visualization;
        # loss curves and checkpoints are unaffected.
        return []
