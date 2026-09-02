"""Build volunteer-based train/test data and prepare memory-aware loaders."""

from __future__ import annotations

import logging
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from dataset_registry import DatasetRegistry, LABEL_TO_CLASS, RegistryColumns
from datasets import SingleModalityDataset
from memory_manager import MemoryBudget, TensorStore, plan_resident_set


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
    """The four groups the residency plan works with."""

    mandatory_train: pd.DataFrame
    mandatory_test: pd.DataFrame
    optional_train: pd.DataFrame
    optional_test: pd.DataFrame


def _select_experiment_data(
    registry: DatasetRegistry,
    folder_1_df: pd.DataFrame,
    folder_2_df: pd.DataFrame,
    config: ExperimentConfig,
) -> SelectedData:
    """Choose the transitions and just_states rows for each split."""
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
        js_pool = registry.filter_by_volunteer(folder_2_df, volunteer_id)
        js_test = registry.match_just_states(
            trans_test,
            js_pool,
            ratio=config.just_states_ratio,
            seed=config.seed,
        )
        # Keep the train and test just_states draws disjoint.
        remaining_pool = js_pool[
            ~js_pool[RegistryColumns.FILE_PATH].isin(
                set(js_test[RegistryColumns.FILE_PATH])
            )
        ]
        js_train = registry.match_just_states(
            trans_train,
            remaining_pool,
            ratio=config.just_states_ratio,
            seed=config.seed,
        )
        return SelectedData(trans_train, trans_test, js_train, js_test)

    # Volunteer-level split: no within-volunteer test extraction.
    transitions_all = registry.get_valid_transitions(combined)
    trans_train, trans_test = registry.select_volunteers_split(
        transitions_all,
        config.train_volunteer_count,
        config.test_volunteer_count,
        seed=config.seed,
    )
    js_train = registry.match_just_states(
        trans_train,
        folder_2_df,
        ratio=config.just_states_ratio,
        seed=config.seed,
    )
    js_test = registry.match_just_states(
        trans_test,
        folder_2_df,
        ratio=config.just_states_ratio,
        seed=config.seed,
    )
    return SelectedData(trans_train, trans_test, js_train, js_test)


def _build_loaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    registry: DatasetRegistry,
    store: TensorStore,
    batch_size: int,
) -> dict[str, DataLoader]:
    """Build one loader per split and modality.

    ``num_workers`` stays at zero: the arrays are already resident float32
    tensors, so an item is a slice, and worker processes on Windows would
    copy the whole resident set into every worker.
    """
    frames = {
        "train": train_df,
        "test": test_df,
    }
    loaders: dict[str, DataLoader] = {}

    for split, df in frames.items():
        for modality in ("imu", "mmg"):
            subset = registry.filter_by_modality(df, modality)
            if subset.empty:
                raise ValueError(
                    f"{split} data contains no {modality.upper()} records."
                )
            dataset = SingleModalityDataset(subset, store)
            loaders[f"{split}_{modality}"] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=split == "train",
                num_workers=0,
                pin_memory=True,
            )
    return loaders


def _log_class_distribution(name: str, df: pd.DataFrame) -> None:
    counts = (
        df[df[RegistryColumns.MODALITY] == "IMU"]
        .groupby(RegistryColumns.CLASS_LABEL)[RegistryColumns.SAMPLES]
        .sum()
        .sort_index()
    )
    logger.info(
        "%s examples per class: %s",
        name,
        {LABEL_TO_CLASS[int(label)]: int(value) for label, value in counts.items()},
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
) -> dict[str, DataLoader]:
    """Prepare reproducible volunteer-based training and test loaders."""
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

    train_df = pd.concat(
        [selected.mandatory_train, plan.optional_train], ignore_index=True
    )
    test_df = pd.concat(
        [selected.mandatory_test, plan.optional_test], ignore_index=True
    )

    store = TensorStore(budget=budget)
    store.load(
        train_df[RegistryColumns.FILE_PATH].tolist()
        + test_df[RegistryColumns.FILE_PATH].tolist()
    )

    loaders = _build_loaders(train_df, test_df, registry, store, config.batch_size)
    logger.info(
        "Prepared %s mode: train %d examples from %d recordings | "
        "test %d examples from %d recordings",
        config.setup,
        len(loaders["train_imu"].dataset),
        len(train_df) // 2,
        len(loaders["test_imu"].dataset),
        len(test_df) // 2,
    )
    _log_class_distribution("Train", train_df)
    _log_class_distribution("Test", test_df)
    for name in ("train_imu", "train_mmg"):
        dataset = loaders[name].dataset
        logger.info(
            "%s input shape: %s (%s)",
            name, dataset.item_shape, dataset.shape_spec.describe(),
        )
    return loaders


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
