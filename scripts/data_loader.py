"""Reusable experiment data loading utilities.

This module owns the file-scanning, volunteer split, memory planning, tensor
loading, and optional model-input preparation steps. ``main.py`` calls into this
module, and the returned tensors can also be passed directly to the test helper
functions in ``tests/``.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_registry import (
    DatasetRegistry,
    LABEL_TO_CLASS,
    RegistryColumns,
    build_sample_table,
    exclude_samples,
)
from datasets import SOURCE_FILE, ModalityTensors, SingleModalityDataset
from memory_manager import MemoryBudget, plan_resident_set

logger = logging.getLogger(__name__)

InputMode = Literal["single_window", "windowed"]
ModelTarget = Literal["standalone", "fusion"]
WINDOW_INDEX_COLUMN = "window_index"
ALL_WINDOWS = "all"
IMU_SOURCE_FILE = "imu_source_file"
MMG_SOURCE_FILE = "mmg_source_file"


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one reproducible volunteer-based experiment."""

    setup: str = "separate_volunteers"
    same_volunteer_id: int | str | None = None
    train_volunteer_count: int = 8
    test_volunteer_count: int = 2
    total_budget_gb: float = 24.0
    seed: int = 42
    test_fraction: float = 0.10
    just_states_ratio: float = 1.10
    batch_size: int = 32

    def validate(self) -> None:
        """Validate configuration values before scanning or loading data."""
        if self.setup not in {"same_volunteer", "separate_volunteers"}:
            raise ValueError(
                "setup must be 'same_volunteer' or 'separate_volunteers'."
            )
        if self.total_budget_gb <= 0:
            raise ValueError("total_budget_gb must be positive.")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1.")
        if self.just_states_ratio <= 0:
            raise ValueError("just_states_ratio must be positive.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if self.setup == "same_volunteer" and self.same_volunteer_id is None:
            raise ValueError("same_volunteer_id is required in same_volunteer mode.")


@dataclass(frozen=True)
class SelectedData:
    """The four sample tables the residency plan works with."""

    mandatory_train: pd.DataFrame
    mandatory_test: pd.DataFrame
    optional_train: pd.DataFrame
    optional_test: pd.DataFrame


@dataclass(frozen=True)
class ExperimentData:
    """Everything one experiment needs, keyed ``<split>_<modality>``.

    ``bundles`` holds the four resident tensors with their row-aligned labels and
    metadata; ``loaders`` wraps the same tensors for training.
    """

    config: ExperimentConfig
    bundles: dict[str, ModalityTensors]
    loaders: dict[str, DataLoader]

    @property
    def train_imu(self) -> ModalityTensors:
        return self.bundles["train_imu"]

    @property
    def train_mmg(self) -> ModalityTensors:
        return self.bundles["train_mmg"]

    @property
    def test_imu(self) -> ModalityTensors:
        return self.bundles["test_imu"]

    @property
    def test_mmg(self) -> ModalityTensors:
        return self.bundles["test_mmg"]


@dataclass(frozen=True)
class PreparedData:
    """Model-ready train/test tensors plus row-aligned metadata.

    The raw resident tensors and modality-specific metadata remain available in
    ``experiment``. The top-level tensors are transformed according to
    ``input_mode``:

        - ``windowed`` keeps all four windows per sample:
            IMU ``(N, 6, 125, 4)``, MMG/CWT ``(N, 5, 40, 125, 4)``.
    - ``single_window`` expands every window into a separate sample:
      IMU ``(N*4, 6, 125)``, MMG/CWT ``(N*4, 5, 40, 125)``.
    """

    experiment: ExperimentData
    input_mode: InputMode
    model_target: ModelTarget
    X_imu_train: torch.Tensor
    X_cwt_train: torch.Tensor
    y_train: torch.Tensor
    X_imu_test: torch.Tensor
    X_cwt_test: torch.Tensor
    y_test: torch.Tensor
    train_metadata: pd.DataFrame
    test_metadata: pd.DataFrame

    @property
    def imu_args(self) -> tuple[torch.Tensor, ...]:
        """Train/test arguments for standalone IMU test helpers."""
        return (
            self.X_imu_train,
            self.y_train,
            self.X_imu_test,
            self.y_test,
        )

    @property
    def mmg_args(self) -> tuple[torch.Tensor, ...]:
        """Train/test arguments for standalone MMG test helpers."""
        return (
            self.X_cwt_train,
            self.y_train,
            self.X_cwt_test,
            self.y_test,
        )

    @property
    def fusion_args(self) -> tuple[torch.Tensor, ...]:
        """Train/test arguments for fusion test helpers."""
        return (
            self.X_imu_train,
            self.X_cwt_train,
            self.y_train,
            self.X_imu_test,
            self.X_cwt_test,
            self.y_test,
        )


def _select_experiment_data(
    registry: DatasetRegistry,
    folder_1_df: pd.DataFrame,
    folder_2_df: pd.DataFrame,
    config: ExperimentConfig,
) -> SelectedData:
    """Choose the transition and just_states samples for each split."""
    combined = pd.concat([folder_1_df, folder_2_df], ignore_index=True)

    if config.setup == "same_volunteer":
        if config.same_volunteer_id is None:
            raise ValueError("same_volunteer_id is required in same_volunteer mode.")
        volunteer_id = registry.normalize_volunteer_id(config.same_volunteer_id)
        trans_train, trans_test = registry.split_transitions_by_fraction(
            combined,
            volunteer_id,
            test_fraction=config.test_fraction,
            seed=config.seed,
        )
        js_pool = build_sample_table(
            registry.filter_by_volunteer(folder_2_df, volunteer_id)
        )
        js_test = registry.match_just_states(
            trans_test,
            js_pool,
            ratio=config.just_states_ratio,
            seed=config.seed,
        )
        js_train = registry.match_just_states(
            trans_train,
            exclude_samples(js_pool, js_test),
            ratio=config.just_states_ratio,
            seed=config.seed,
        )
        return SelectedData(trans_train, trans_test, js_train, js_test)

    # Volunteer-level split: no within-volunteer test extraction.
    transitions_all = registry.get_valid_transitions(combined)
    trans_train_rows, trans_test_rows = registry.select_volunteers_split(
        transitions_all,
        config.train_volunteer_count,
        config.test_volunteer_count,
        seed=config.seed,
    )
    trans_train = build_sample_table(trans_train_rows)
    trans_test = build_sample_table(trans_test_rows)

    # Train and test volunteers are disjoint, so the pools cannot overlap.
    js_pool = build_sample_table(folder_2_df)
    js_train = registry.match_just_states(
        trans_train, js_pool, ratio=config.just_states_ratio, seed=config.seed
    )
    js_test = registry.match_just_states(
        trans_test, js_pool, ratio=config.just_states_ratio, seed=config.seed
    )
    return SelectedData(trans_train, trans_test, js_train, js_test)


def _build_bundles(
    train_samples: pd.DataFrame,
    test_samples: pd.DataFrame,
) -> dict[str, ModalityTensors]:
    """Read the four resident tensors: IMU/MMG for train and test."""
    frames = {"train": train_samples, "test": test_samples}
    bundles: dict[str, ModalityTensors] = {}
    for split, samples in frames.items():
        if samples.empty:
            raise ValueError(f"{split} selection is empty.")
        for modality in ("IMU", "MMG"):
            bundles[f"{split}_{modality.lower()}"] = ModalityTensors.from_samples(
                samples, modality
            )
    return bundles


def _build_loaders(
    bundles: dict[str, ModalityTensors],
    batch_size: int,
    seed: int,
) -> dict[str, DataLoader]:
    """Build one loader per split and modality.

    ``num_workers`` stays at zero: items are slices of an already resident
    tensor, and worker processes on Windows would copy the whole resident set
    into every worker.
    """
    loaders: dict[str, DataLoader] = {}
    for name, bundle in bundles.items():
        is_train = name.startswith("train")
        loaders[name] = DataLoader(
            SingleModalityDataset(bundle),
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=0,
            pin_memory=True,
            generator=torch.Generator().manual_seed(seed) if is_train else None,
        )
    return loaders


def _log_class_distribution(name: str, samples: pd.DataFrame) -> None:
    counts = samples.groupby(RegistryColumns.CLASS_LABEL).size().sort_index()
    logger.info(
        "%s samples per class: %s",
        name,
        {LABEL_TO_CLASS[cast(int, label)]: int(value) for label, value in counts.items()},
    )


def prepare_experiment_data(
    setup: str = "separate_volunteers",
    same_volunteer_id: int | str | None = None,
    train_volunteer_count: int = 8,
    test_volunteer_count: int = 2,
    total_budget_gb: float = 24.0,
    seed: int = 42,
    test_fraction: float = 0.10,
    just_states_ratio: float = 1.10,
    batch_size: int = 32,
) -> ExperimentData:
    """Load reproducible volunteer-based training and test data.

    Returns raw resident tensors in their on-disk sample geometry:
    ``train_imu``/``test_imu`` as ``(N, 4, 125, 6)`` and
    ``train_mmg``/``test_mmg`` as ``(N, 4, 40, 125, 5)``.
    """
    config = ExperimentConfig(
        setup=setup,
        same_volunteer_id=same_volunteer_id,
        train_volunteer_count=train_volunteer_count,
        test_volunteer_count=test_volunteer_count,
        total_budget_gb=total_budget_gb,
        seed=seed,
        test_fraction=test_fraction,
        just_states_ratio=just_states_ratio,
        batch_size=batch_size,
    )
    config.validate()

    registry = DatasetRegistry()
    folder_1_df, folder_2_df = registry.build_dual_folder(
        folder_1="data/transitions",
        folder_2="data/just_states",
    )
    selected = _select_experiment_data(registry, folder_1_df, folder_2_df, config)

    budget = MemoryBudget(total_budget_gb=config.total_budget_gb)
    logger.info("%s", budget.summary())
    plan = plan_resident_set(
        selected.mandatory_train,
        selected.mandatory_test,
        selected.optional_train,
        selected.optional_test,
        budget=budget,
        seed=config.seed,
    )

    train_samples = pd.concat(
        [selected.mandatory_train, plan.optional_train], ignore_index=True
    )
    test_samples = pd.concat(
        [selected.mandatory_test, plan.optional_test], ignore_index=True
    )

    bundles = _build_bundles(train_samples, test_samples)
    resident = sum(bundle.nbytes for bundle in bundles.values())
    if resident > budget.total_budget_bytes:
        raise MemoryError(
            f"Resident tensors need {resident / 1024 ** 3:.2f} GiB but the "
            f"budget is {config.total_budget_gb:.2f} GiB. The residency plan "
            "and the data on disk disagree."
        )

    loaders = _build_loaders(bundles, config.batch_size, config.seed)
    logger.info(
        "Prepared %s mode: train %d samples | test %d samples | %.2f GiB resident",
        config.setup, len(train_samples), len(test_samples),
        resident / 1024 ** 3,
    )
    _log_class_distribution("Train", train_samples)
    _log_class_distribution("Test", test_samples)
    for name in ("train_imu", "train_mmg"):
        bundle = bundles[name]
        logger.info(
            "%s tensor: %s (%s)",
            name, tuple(bundle.data.shape), bundle.shape_spec.describe(),
        )
    return ExperimentData(config=config, bundles=bundles, loaders=loaders)


def _validate_model_target(model_target: str) -> ModelTarget:
    """Validate and narrow a model-target string."""
    if model_target not in {"standalone", "fusion"}:
        raise ValueError("model_target must be 'standalone' or 'fusion'.")
    return cast(ModelTarget, model_target)


def _validate_input_mode(input_mode: str) -> InputMode:
    """Validate and narrow an input-mode string."""
    if input_mode not in {"single_window", "windowed"}:
        raise ValueError("input_mode must be 'single_window' or 'windowed'.")
    return cast(InputMode, input_mode)


def _format_imu_windowed(data: torch.Tensor) -> torch.Tensor:
    """Convert IMU from ``(N, 4, 125, 6)`` to ``(N, 6, 125, 4)``."""
    if data.dim() != 4:
        raise ValueError(
            f"Expected IMU data with shape (N, 4, 125, 6), got {tuple(data.shape)}."
        )
    return data.permute(0, 3, 2, 1).contiguous()


def _format_mmg_windowed(data: torch.Tensor) -> torch.Tensor:
    """Convert MMG from ``(N, 4, 40, 125, 5)`` to ``(N, 5, 40, 125, 4)``."""
    if data.dim() != 5:
        raise ValueError(
            f"Expected MMG data with shape (N, 4, 40, 125, 5), got {tuple(data.shape)}."
        )
    return data.permute(0, 4, 2, 3, 1).contiguous()


def _expand_imu_windows(data: torch.Tensor) -> torch.Tensor:
    """Convert windowed IMU from ``(N, 6, 125, 4)`` to ``(N*4, 6, 125)``."""
    if data.dim() != 4:
        raise ValueError(
            f"Expected windowed IMU data with shape (N, 6, 125, 4), got {tuple(data.shape)}."
        )
    n_samples, channels, time_steps, n_windows = data.shape
    return data.permute(0, 3, 1, 2).reshape(
        n_samples * n_windows, channels, time_steps
    ).contiguous()


def _expand_mmg_windows(data: torch.Tensor) -> torch.Tensor:
    """Convert windowed MMG from ``(N, 5, 40, 125, 4)`` to ``(N*4, 5, 40, 125)``."""
    if data.dim() != 5:
        raise ValueError(
            f"Expected windowed MMG data with shape (N, 5, 40, 125, 4), got {tuple(data.shape)}."
        )
    n_samples, channels, scales, time_steps, n_windows = data.shape
    return data.permute(0, 4, 1, 2, 3).reshape(
        n_samples * n_windows, channels, scales, time_steps
    ).contiguous()


def _expand_labels_for_windows(labels: torch.Tensor, n_windows: int) -> torch.Tensor:
    """Repeat one sample label for each of its windows."""
    return labels.repeat_interleave(n_windows).contiguous()


def _combined_metadata(
    imu: ModalityTensors,
    mmg: ModalityTensors,
    window_index: int | str,
) -> pd.DataFrame:
    """Build one metadata table containing both modality source files."""
    metadata = imu.metadata.copy().reset_index(drop=True)
    metadata = metadata.rename(columns={SOURCE_FILE: IMU_SOURCE_FILE})
    metadata[MMG_SOURCE_FILE] = mmg.metadata[SOURCE_FILE].reset_index(drop=True)
    metadata[WINDOW_INDEX_COLUMN] = window_index
    return metadata


def _expand_metadata_for_windows(metadata: pd.DataFrame, n_windows: int) -> pd.DataFrame:
    """Repeat metadata rows and add the concrete expanded window index."""
    expanded = metadata.drop(columns=[WINDOW_INDEX_COLUMN], errors="ignore")
    expanded = expanded.loc[expanded.index.repeat(n_windows)].reset_index(drop=True)
    expanded[WINDOW_INDEX_COLUMN] = [i for _ in range(len(metadata)) for i in range(n_windows)]
    return expanded


def prepare_windowed_inputs(
    experiment: ExperimentData,
    model_target: str = "standalone",
) -> PreparedData:
    """Prepare train/test tensors while preserving all four windows per sample."""
    target = _validate_model_target(model_target)
    train_imu = _format_imu_windowed(experiment.train_imu.data)
    test_imu = _format_imu_windowed(experiment.test_imu.data)
    train_mmg = _format_mmg_windowed(experiment.train_mmg.data)
    test_mmg = _format_mmg_windowed(experiment.test_mmg.data)

    return PreparedData(
        experiment=experiment,
        input_mode="windowed",
        model_target=target,
        X_imu_train=train_imu,
        X_cwt_train=train_mmg,
        y_train=experiment.train_imu.labels,
        X_imu_test=test_imu,
        X_cwt_test=test_mmg,
        y_test=experiment.test_imu.labels,
        train_metadata=_combined_metadata(
            experiment.train_imu, experiment.train_mmg, ALL_WINDOWS
        ),
        test_metadata=_combined_metadata(
            experiment.test_imu, experiment.test_mmg, ALL_WINDOWS
        ),
    )


def prepare_single_window_inputs(
    experiment: ExperimentData,
    model_target: str = "standalone",
) -> PreparedData:
    """Prepare train/test tensors by expanding every window into a sample."""
    target = _validate_model_target(model_target)
    train_imu = _format_imu_windowed(experiment.train_imu.data)
    test_imu = _format_imu_windowed(experiment.test_imu.data)
    train_mmg = _format_mmg_windowed(experiment.train_mmg.data)
    test_mmg = _format_mmg_windowed(experiment.test_mmg.data)
    n_windows = int(train_imu.shape[-1])

    return PreparedData(
        experiment=experiment,
        input_mode="single_window",
        model_target=target,
        X_imu_train=_expand_imu_windows(train_imu),
        X_cwt_train=_expand_mmg_windows(train_mmg),
        y_train=_expand_labels_for_windows(experiment.train_imu.labels, n_windows),
        X_imu_test=_expand_imu_windows(test_imu),
        X_cwt_test=_expand_mmg_windows(test_mmg),
        y_test=_expand_labels_for_windows(experiment.test_imu.labels, n_windows),
        train_metadata=_expand_metadata_for_windows(
            _combined_metadata(experiment.train_imu, experiment.train_mmg, ALL_WINDOWS),
            n_windows,
        ),
        test_metadata=_expand_metadata_for_windows(
            _combined_metadata(experiment.test_imu, experiment.test_mmg, ALL_WINDOWS),
            n_windows,
        ),
    )


def prepare_training_data(
    experiment: ExperimentData,
    input_mode: str = "windowed",
    model_target: str = "standalone",
) -> PreparedData:
    """Dispatch to the requested train/test input representation."""
    mode = _validate_input_mode(input_mode)
    _validate_model_target(model_target)
    if mode == "windowed":
        return prepare_windowed_inputs(experiment, model_target=model_target)
    return prepare_single_window_inputs(experiment, model_target=model_target)
