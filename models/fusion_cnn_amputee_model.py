# fusion_cnn_amputee_model.py
# End-to-end single-window FusionCNN for amputee data.
#
# Architecture:
#   GAP(IntentCNN backbone) + GAP(LocomotionMMGCNN backbone)
#       -> Concatenate  (batch, C1+C2)
#       -> FC Layer     (Optuna: hidden units)
#       -> ReLU + Dropout
#       -> Linear       (batch, num_classes)
#
# Both backbones are built inside the model and trained from scratch with the
# head. Optuna jointly searches both backbones, the head, and the optimiser.

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from device_utils import resolve_device
from fusion_amputee_common import (
    AMPUTEE_NUM_CLASSES,
    OPTIMISER_SEARCH,
    PLATEAU_SCHEDULE_SEARCH,
    SINGLE_WINDOW_BACKBONE_SEARCH,
    AmputeeFusionBackbones,
    AmputeeFusionTrainer,
    AmputeeFusionTuner,
    default_backbone_configs,
    init_linear_layers,
    load_from_checkpoint,
)

MODEL_KEY = "fusion_cnn_amputee"


class FusionCNNAmputee(nn.Module):
    """
    Inputs:
        x_imu : (batch, 6, 125)
        x_cwt : (batch, 5, 40, 125)
    """

    def __init__(
        self,
        num_classes:      int                   = AMPUTEE_NUM_CLASSES,
        backbone_configs: dict | None           = None,
        fc_hidden:        int                   = 128,
        dropout:          float                 = 0.5,
        device:           torch.device | str | None = None,
    ):
        super().__init__()
        self.backbones = AmputeeFusionBackbones(
            backbone_configs or default_backbone_configs(windowed=False), windowed=False
        )
        feat_dim = self.backbones.feature_dim
        self.head = nn.Sequential(
            nn.Linear(feat_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, num_classes),
        )
        self.config: dict[str, Any] = dict(
            model_type       = "FusionCNNAmputee",
            num_classes      = num_classes,
            fc_hidden        = fc_hidden,
            dropout          = dropout,
            feature_dim      = feat_dim,
            backbone_configs = self.backbones.backbone_configs,
        )
        init_linear_layers(self.head)
        if device is not None:
            self.to(resolve_device(device))

    @classmethod
    def from_config(
        cls, config: dict, device: torch.device | str | None = None
    ) -> "FusionCNNAmputee":
        """Rebuild the architecture from a saved ``model_config`` (random weights)."""
        return cls(
            num_classes      = config["num_classes"],
            backbone_configs = config["backbone_configs"],
            fc_hidden        = config["fc_hidden"],
            dropout          = config["dropout"],
            device           = device,
        )

    @classmethod
    def from_checkpoint(
        cls, path: str, device: torch.device | str | None = None
    ) -> "FusionCNNAmputee":
        return load_from_checkpoint(cls, path, device)

    def forward(self, x_imu: torch.Tensor, x_cwt: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbones(x_imu, x_cwt))


class FusionCNNAmputeeTrainer(AmputeeFusionTrainer):
    """DataLoader yields (X_imu, X_cwt, y). Uses ReduceLROnPlateau."""

    NAME = "FusionCNNAmputeeTrainer"
    SCHEDULER = "plateau"


class FusionCNNAmputeeTuner(AmputeeFusionTuner):
    """Searches IMU backbone + MMG backbone + FC head + optimiser + LR plateau."""

    NAME = "FusionCNNAmputeeTuner"
    MODEL_CLS = FusionCNNAmputee
    TRAINER_CLS = FusionCNNAmputeeTrainer
    WINDOWED = False
    _SEARCH = dict(
        **SINGLE_WINDOW_BACKBONE_SEARCH,
        fc_hidden  = [64, 128, 256, 512],
        dropout    = (0.1, 0.5),
        batch_size = [64, 128, 200],
        epochs     = 50,
        **OPTIMISER_SEARCH,
        **PLATEAU_SCHEDULE_SEARCH,
    )

    def _suggest_head(self, trial) -> dict[str, Any]:
        return dict(
            fc_hidden = trial.suggest_categorical("fc_hidden", self.search["fc_hidden"]),
            dropout   = trial.suggest_float("dropout", *self.search["dropout"]),
        )
