"""Shared train/validation splitting for the tests/*.py entry points.

``prepare_training_data`` only produces train/test tensors; the 10% validation
slice used by each test-script's Optuna tuner and final training run is carved
out of the training data here, at call time, rather than inside the loader.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import IMU_SOURCE_FILE, PreparedData
from datasets import SOURCE_SAMPLE_INDEX

GROUP_COLUMNS = [IMU_SOURCE_FILE, SOURCE_SAMPLE_INDEX]


def validate_prepared_data(
    prepared: PreparedData,
    expected_input_mode: Literal["single_window", "windowed"],
    expected_model_target: Literal["standalone", "fusion"],
) -> None:
    """Reject prepared data intended for a different test entry point."""
    if prepared.input_mode != expected_input_mode:
        raise ValueError(
            f"Expected input_mode={expected_input_mode!r}, "
            f"got {prepared.input_mode!r}."
        )
    if prepared.model_target != expected_model_target:
        raise ValueError(
            f"Expected model_target={expected_model_target!r}, "
            f"got {prepared.model_target!r}."
        )


def split_train_validation(
    y_train: torch.Tensor,
    train_metadata: pd.DataFrame,
    val_fraction: float = 0.10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) row positions for a stratified, grouped split.

    Rows sharing the same original sample (identical ``imu_source_file`` +
    ``source_sample_index``) always land on the same side of the split, so
    single_window mode's four expanded windows per sample never leak between
    train and validation. The split is stratified by class at the group
    level to keep the validation set's class balance close to training's.
    """
    if len(train_metadata) != len(y_train):
        raise ValueError("train_metadata and y_train must be row-aligned.")

    metadata = train_metadata.reset_index(drop=True)
    labels = y_train.numpy() if isinstance(y_train, torch.Tensor) else np.asarray(y_train)

    groups = metadata.groupby(GROUP_COLUMNS, sort=False).indices
    group_keys = list(groups.keys())
    group_labels = [labels[positions[0]] for positions in groups.values()]

    train_keys, val_keys = train_test_split(
        group_keys,
        test_size=val_fraction,
        random_state=seed,
        stratify=group_labels,
    )

    train_idx = np.concatenate([groups[key] for key in train_keys])
    val_idx = np.concatenate([groups[key] for key in val_keys])
    return np.sort(train_idx), np.sort(val_idx)
