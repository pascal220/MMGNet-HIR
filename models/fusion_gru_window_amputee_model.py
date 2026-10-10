# fusion_gru_window_amputee_model.py
# End-to-end windowed FusionGRU for amputee data.
#
# Architecture:
#   IntentCNNWindow features + LocomotionMMGCNNWindow features
#       -> Concatenate          (batch, C1+C2)
#       -> Sequence of length 1 (batch, 1, C1+C2)
#       -> GRU (1 layer), last step (batch, gru_hidden_dim)
#       -> [Linear -> ReLU -> Dropout] per hidden layer -> Linear (batch, num_classes)
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
    STEP_SCHEDULE_SEARCH,
    WINDOWED_BACKBONE_SEARCH,
    AmputeeFusionBackbones,
    AmputeeFusionTrainer,
    AmputeeFusionTuner,
    default_backbone_configs,
    init_linear_layers,
    load_from_checkpoint,
)

MODEL_KEY = "fusion_gru_windowed_amputee"


class FusionGRUWindowAmputee(nn.Module):
    """
    Inputs:
        x_imu : (batch, 6, 125, 4)
        x_cwt : (batch, 5, 40, 125, 4)
    """

    def __init__(
        self,
        num_classes:      int                   = AMPUTEE_NUM_CLASSES,
        backbone_configs: dict | None           = None,
        gru_hidden_dim:   int                   = 128,
        fc_hidden_dims:   list[int] | None      = None,
        fc_dropout:       float                 = 0.3,
        device:           torch.device | str | None = None,
    ):
        super().__init__()
        self.backbones = AmputeeFusionBackbones(
            backbone_configs or default_backbone_configs(windowed=True), windowed=True
        )
        self.gru = nn.GRU(
            input_size  = self.backbones.feature_dim,
            hidden_size = gru_hidden_dim,
            num_layers  = 1,
            batch_first = True,
        )
        fc_hidden_dims = list(fc_hidden_dims) if fc_hidden_dims is not None else [128]
        layers: list[nn.Module] = []
        current_dim = gru_hidden_dim
        for hidden_dim in fc_hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=fc_dropout),
            ])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_classes))
        self.fc_head = nn.Sequential(*layers)

        self.config: dict[str, Any] = dict(
            model_type       = "FusionGRUWindowAmputee",
            num_classes      = num_classes,
            gru_hidden_dim   = gru_hidden_dim,
            fc_hidden_dims   = fc_hidden_dims,
            fc_dropout       = fc_dropout,
            feature_dim      = self.backbones.feature_dim,
            backbone_configs = self.backbones.backbone_configs,
        )
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
        init_linear_layers(self.fc_head)
        if device is not None:
            self.to(resolve_device(device))

    @classmethod
    def from_config(
        cls, config: dict, device: torch.device | str | None = None
    ) -> "FusionGRUWindowAmputee":
        """Rebuild the architecture from a saved ``model_config`` (random weights)."""
        return cls(
            num_classes      = config["num_classes"],
            backbone_configs = config["backbone_configs"],
            gru_hidden_dim   = config["gru_hidden_dim"],
            fc_hidden_dims   = config["fc_hidden_dims"],
            fc_dropout       = config["fc_dropout"],
            device           = device,
        )

    @classmethod
    def from_checkpoint(
        cls, path: str, device: torch.device | str | None = None
    ) -> "FusionGRUWindowAmputee":
        return load_from_checkpoint(cls, path, device)

    def forward(self, x_imu: torch.Tensor, x_cwt: torch.Tensor) -> torch.Tensor:
        features = self.backbones(x_imu, x_cwt)
        gru_out, _ = self.gru(features.unsqueeze(1))
        return self.fc_head(gru_out[:, -1, :])


class FusionGRUWindowAmputeeTrainer(AmputeeFusionTrainer):
    """DataLoader yields ((X_imu, X_cwt), y). Uses StepLR."""

    NAME = "FusionGRUWindowAmputeeTrainer"
    SCHEDULER = "step"


class FusionGRUWindowAmputeeTuner(AmputeeFusionTuner):
    """Searches IMU backbone + MMG backbone + GRU + FC head + optimiser + StepLR."""

    NAME = "FusionGRUWindowAmputeeTuner"
    MODEL_CLS = FusionGRUWindowAmputee
    TRAINER_CLS = FusionGRUWindowAmputeeTrainer
    WINDOWED = True
    _SEARCH = dict(
        **WINDOWED_BACKBONE_SEARCH,
        gru_hidden_dim = [64, 128, 256, 512],
        fc_hidden_dim  = [64, 128, 256, 512],
        fc_dropout     = (0.1, 0.5),
        # The standalone windowed MMG tuner's batch sizes: the trainable Conv3D
        # first layer dominates GPU memory.
        batch_size     = [64, 128],
        epochs         = 50,
        **OPTIMISER_SEARCH,
        **STEP_SCHEDULE_SEARCH,
    )

    def _suggest_head(self, trial) -> dict[str, Any]:
        return dict(
            gru_hidden_dim = trial.suggest_categorical(
                "gru_hidden_dim", self.search["gru_hidden_dim"]
            ),
            fc_hidden_dims = [
                trial.suggest_categorical("fc_hidden_dim", self.search["fc_hidden_dim"])
            ],
            fc_dropout     = trial.suggest_float("fc_dropout", *self.search["fc_dropout"]),
        )
