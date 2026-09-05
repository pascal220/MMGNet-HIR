"""Build volunteer-based train/test data and prepare memory-aware loaders."""

from __future__ import annotations

import logging
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from dataset_registry import (
    DatasetRegistry,
    LABEL_TO_CLASS,
    RegistryColumns,
    build_sample_table,
    exclude_samples,
)
from datasets import ModalityTensors, SingleModalityDataset
from memory_manager import MemoryBudget, plan_resident_set


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("session_registry.log"),
    ],
)
logger = logging.getLogger(__name__)


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

    ``bundles`` holds the four resident tensors with their row-aligned
    labels and metadata; ``loaders`` wraps the same tensors for training.
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
    tensor, and worker processes on Windows would copy the whole resident
    set into every worker.
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


def main(
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
    """Prepare reproducible volunteer-based training and test data."""
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare volunteer-based training and test loaders."
    )
    parser.add_argument(
        "--setup",
        choices=["separate_volunteers", "same_volunteer"],
        default="separate_volunteers",
    )
    parser.add_argument(
        "--same-volunteer-id",
        default=None,
        help="Volunteer ID for same_volunteer mode (e.g. 4 or N004).",
    )
    parser.add_argument("--train-volunteer-count", type=int, default=8)
    parser.add_argument("--test-volunteer-count", type=int, default=2)
    parser.add_argument("--total-budget-gb", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--just-states-ratio", type=float, default=1.10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    main(
        setup=args.setup,
        same_volunteer_id=args.same_volunteer_id,
        train_volunteer_count=args.train_volunteer_count,
        test_volunteer_count=args.test_volunteer_count,
        total_budget_gb=args.total_budget_gb,
        seed=args.seed,
        test_fraction=args.test_fraction,
        just_states_ratio=args.just_states_ratio,
        batch_size=args.batch_size,
    )
