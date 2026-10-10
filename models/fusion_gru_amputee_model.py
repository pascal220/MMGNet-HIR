# fusion_gru_amputee_model.py
# End-to-end single-window FusionGRU for amputee data.
#
# Architecture:
#   GAP(IntentCNN backbone) + GAP(LocomotionMMGCNN backbone)
#       -> Concatenate      (batch, C1+C2)
#       -> Unsqueeze        (batch, 1, C1+C2)
#       -> GRU (1 layer)    (batch, 1, gru_hidden)
#       -> FC + ReLU + Dropout -> Linear (batch, num_classes)
#
# Both backbones are built inside the model and trained from scratch with the
# GRU and head. Optuna jointly searches both backbones, the GRU, the head, and
# the optimiser.

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

MODEL_KEY = "fusion_gru_amputee"


class FusionGRUAmputee(nn.Module):
    """
    Inputs:
        x_imu : (batch, 6, 125)
        x_cwt : (batch, 5, 40, 125)
    """

    def __init__(
        self,
        num_classes:      int                   = AMPUTEE_NUM_CLASSES,
        backbone_configs: dict | None           = None,
        gru_hidden:       int                   = 128,
        fc_hidden:        int                   = 128,
        fc_dropout:       float                 = 0.3,
        device:           torch.device | str | None = None,
    ):
        super().__init__()
        self.backbones = AmputeeFusionBackbones(
            backbone_configs or default_backbone_configs(windowed=False), windowed=False
        )
        feat_dim = self.backbones.feature_dim
        self.gru = nn.GRU(
            input_size  = feat_dim,
            hidden_size = gru_hidden,
            num_layers  = 1,
            batch_first = True,
        )
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=fc_dropout),
            nn.Linear(fc_hidden, num_classes),
        )
        self.config: dict[str, Any] = dict(
            model_type       = "FusionGRUAmputee",
            num_classes      = num_classes,
            gru_hidden       = gru_hidden,
            fc_hidden        = fc_hidden,
            fc_dropout       = fc_dropout,
            feature_dim      = feat_dim,
            backbone_configs = self.backbones.backbone_configs,
        )
        init_linear_layers(self.head)
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
        if device is not None:
            self.to(resolve_device(device))

    @classmethod
    def from_config(
        cls, config: dict, device: torch.device | str | None = None
    ) -> "FusionGRUAmputee":
        """Rebuild the architecture from a saved ``model_config`` (random weights)."""
        return cls(
            num_classes      = config["num_classes"],
            backbone_configs = config["backbone_configs"],
            gru_hidden       = config["gru_hidden"],
            fc_hidden        = config["fc_hidden"],
            fc_dropout       = config["fc_dropout"],
            device           = device,
        )

    @classmethod
    def from_checkpoint(
        cls, path: str, device: torch.device | str | None = None
    ) -> "FusionGRUAmputee":
        return load_from_checkpoint(cls, path, device)

    def forward(self, x_imu: torch.Tensor, x_cwt: torch.Tensor) -> torch.Tensor:
        features = self.backbones(x_imu, x_cwt)
        gru_out, _ = self.gru(features.unsqueeze(1))
        return self.head(gru_out.squeeze(1))


class FusionGRUAmputeeTrainer(AmputeeFusionTrainer):
    """DataLoader yields (X_imu, X_cwt, y). Uses ReduceLROnPlateau."""

    NAME = "FusionGRUAmputeeTrainer"
    SCHEDULER = "plateau"


class FusionGRUAmputeeTuner(AmputeeFusionTuner):
    """Searches IMU backbone + MMG backbone + GRU + FC head + optimiser + LR plateau."""

    NAME = "FusionGRUAmputeeTuner"
    MODEL_CLS = FusionGRUAmputee
    TRAINER_CLS = FusionGRUAmputeeTrainer
    WINDOWED = False
    _SEARCH = dict(
        **SINGLE_WINDOW_BACKBONE_SEARCH,
        gru_hidden = [64, 128, 256, 512],
        fc_hidden  = [64, 128, 256, 512],
        fc_dropout = (0.1, 0.5),
        batch_size = [64, 128, 200],
        epochs     = 50,
        **OPTIMISER_SEARCH,
        **PLATEAU_SCHEDULE_SEARCH,
    )

    def _suggest_head(self, trial) -> dict[str, Any]:
        return dict(
            gru_hidden = trial.suggest_categorical("gru_hidden", self.search["gru_hidden"]),
            fc_hidden  = trial.suggest_categorical("fc_hidden", self.search["fc_hidden"]),
            fc_dropout = trial.suggest_float("fc_dropout", *self.search["fc_dropout"]),
        )
