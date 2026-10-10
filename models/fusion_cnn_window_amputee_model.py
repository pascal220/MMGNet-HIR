# fusion_cnn_window_amputee_model.py
# End-to-end windowed FusionCNN for amputee data.
#
# Architecture:
#   IntentCNNWindow features + LocomotionMMGCNNWindow features
#       -> Concatenate  (batch, C1+C2)
#       -> [Linear -> ReLU -> Dropout] per hidden layer
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
    STEP_SCHEDULE_SEARCH,
    WINDOWED_BACKBONE_SEARCH,
    AmputeeFusionBackbones,
    AmputeeFusionTrainer,
    AmputeeFusionTuner,
    default_backbone_configs,
    init_linear_layers,
    load_from_checkpoint,
)

MODEL_KEY = "fusion_cnn_windowed_amputee"


class FusionCNNWindowAmputee(nn.Module):
    """
    Inputs:
        x_imu : (batch, 6, 125, 4)
        x_cwt : (batch, 5, 40, 125, 4)
    """

    def __init__(
        self,
        num_classes:      int                   = AMPUTEE_NUM_CLASSES,
        backbone_configs: dict | None           = None,
        hidden_dims:      list[int] | None      = None,
        dropout_rate:     float                 = 0.3,
        device:           torch.device | str | None = None,
    ):
        super().__init__()
        self.backbones = AmputeeFusionBackbones(
            backbone_configs or default_backbone_configs(windowed=True), windowed=True
        )
        hidden_dims = list(hidden_dims) if hidden_dims is not None else [128]
        layers: list[nn.Module] = []
        current_dim = self.backbones.feature_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
            ])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_classes))
        self.head = nn.Sequential(*layers)

        self.config: dict[str, Any] = dict(
            model_type       = "FusionCNNWindowAmputee",
            num_classes      = num_classes,
            hidden_dims      = hidden_dims,
            dropout_rate     = dropout_rate,
            feature_dim      = self.backbones.feature_dim,
            backbone_configs = self.backbones.backbone_configs,
        )
        init_linear_layers(self.head)
        if device is not None:
            self.to(resolve_device(device))

    @classmethod
    def from_config(
        cls, config: dict, device: torch.device | str | None = None
    ) -> "FusionCNNWindowAmputee":
        """Rebuild the architecture from a saved ``model_config`` (random weights)."""
        return cls(
            num_classes      = config["num_classes"],
            backbone_configs = config["backbone_configs"],
            hidden_dims      = config["hidden_dims"],
            dropout_rate     = config["dropout_rate"],
            device           = device,
        )

    @classmethod
    def from_checkpoint(
        cls, path: str, device: torch.device | str | None = None
    ) -> "FusionCNNWindowAmputee":
        return load_from_checkpoint(cls, path, device)

    def forward(self, x_imu: torch.Tensor, x_cwt: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbones(x_imu, x_cwt))


class FusionCNNWindowAmputeeTrainer(AmputeeFusionTrainer):
    """DataLoader yields ((X_imu, X_cwt), y). Uses StepLR."""

    NAME = "FusionCNNWindowAmputeeTrainer"
    SCHEDULER = "step"


class FusionCNNWindowAmputeeTuner(AmputeeFusionTuner):
    """Searches IMU backbone + MMG backbone + FC head + optimiser + StepLR."""

    NAME = "FusionCNNWindowAmputeeTuner"
    MODEL_CLS = FusionCNNWindowAmputee
    TRAINER_CLS = FusionCNNWindowAmputeeTrainer
    WINDOWED = True
    _SEARCH = dict(
        **WINDOWED_BACKBONE_SEARCH,
        hidden_dim   = [64, 128, 256, 512],
        dropout_rate = (0.1, 0.5),
        # The standalone windowed MMG tuner's batch sizes: the trainable Conv3D
        # first layer dominates GPU memory.
        batch_size   = [64, 128],
        epochs       = 50,
        **OPTIMISER_SEARCH,
        **STEP_SCHEDULE_SEARCH,
    )

    def _suggest_head(self, trial) -> dict[str, Any]:
        return dict(
            hidden_dims  = [trial.suggest_categorical("hidden_dim", self.search["hidden_dim"])],
            dropout_rate = trial.suggest_float("dropout_rate", *self.search["dropout_rate"]),
        )
