# backbone_finetune.py
# Shared fine-tuning helpers for the healthy-volunteer fusion models.
#
# The IMU and MMG backbones start from their pre-trained weights and are
# trained together with the fusion head. Their learning rate is the head
# learning rate multiplied by ``backbone_lr_scale`` (discriminative learning
# rates), so the generic features learned by the stand-alone models are only
# nudged while the freshly initialised head learns at full speed.

from typing import Any

import torch.nn as nn

DEFAULT_BACKBONE_LR_SCALE = 0.1
# Decay would pull pre-trained weights towards zero rather than towards their
# starting values, so the backbones are not decayed by default.
DEFAULT_BACKBONE_WEIGHT_DECAY = 0.0
BACKBONE_LR_SCALE_RANGE = (0.01, 0.1)


def build_param_groups(
    model: nn.Module,
    lr: float,
    backbone_lr_scale: float = DEFAULT_BACKBONE_LR_SCALE,
    backbone_weight_decay: float = DEFAULT_BACKBONE_WEIGHT_DECAY,
) -> list[dict[str, Any]]:
    """Split trainable parameters into a head group and a backbone group.

    ``model`` must expose its pre-trained feature extractors as ``model.backbones``.
    Parameters that take no part in the forward pass stay frozen by the caller
    and are left out because they do not require gradients. The backbone group
    uses its own fixed weight decay; the head keeps the optimiser's setting.
    """
    if backbone_lr_scale <= 0:
        raise ValueError("backbone_lr_scale must be positive.")
    if backbone_weight_decay < 0:
        raise ValueError("backbone_weight_decay must not be negative.")

    backbone_params = [p for p in model.backbones.parameters() if p.requires_grad]
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in backbone_ids
    ]
    groups: list[dict[str, Any]] = [{"params": head_params, "lr": lr}]
    if backbone_params:
        groups.append({
            "params": backbone_params,
            "lr": lr * backbone_lr_scale,
            "weight_decay": backbone_weight_decay,
        })
    return groups
