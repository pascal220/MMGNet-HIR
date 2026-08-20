"""Build volunteer-based train/test data and prepare memory-aware loaders."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from class_balancer import ClassBalancer
from dataset_registry import DatasetRegistry, RegistryColumns
from datasets import FusedModalityDataset, SingleModalityDataset
from memory_manager import LRUArrayCache, MemoryBudget


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
    total_budget_gb: float = 21.0
    seed: int = 42
    allowed_class_imbalance: float = 0.10
    minimum_transition_infos: int = 5

    def validate(self) -> None:
        """Validate configuration values before scanning or loading data."""
        if self.setup not in {"same_volunteer", "separate_volunteers"}:
            raise ValueError(
                "setup must be 'same_volunteer' or 'separate_volunteers'."
            )
        if self.total_budget_gb <= 0:
            raise ValueError("total_budget_gb must be positive.")
        if not 0 <= self.allowed_class_imbalance:
            raise ValueError("allowed_class_imbalance must be non-negative.")
        if self.minimum_transition_infos < 1:
            raise ValueError("minimum_transition_infos must be positive.")
        if self.setup == "same_volunteer" and self.same_volunteer_id is None:
            raise ValueError("same_volunteer_id is required in same_volunteer mode.")


def _select_experiment_data(
    registry: DatasetRegistry,
    folder_1_df: pd.DataFrame,
    folder_2_df: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train data, test data, and all selected transition records."""
    combined = pd.concat([folder_1_df, folder_2_df], ignore_index=True)

    if config.setup == "same_volunteer":
        volunteer_id = registry.normalize_volunteer_id(config.same_volunteer_id)
        selected = registry.filter_by_volunteer(combined, volunteer_id)
        mandatory_test = registry.select_test_coverage(
            selected,
            [volunteer_id],
            minimum_transition_infos=config.minimum_transition_infos,
            seed=config.seed,
        )
        test_paths = set(mandatory_test[RegistryColumns.FILE_PATH])
        train = selected[
            ~selected[RegistryColumns.FILE_PATH].isin(test_paths)
        ].reset_index(drop=True)
        test = mandatory_test.reset_index(drop=True)
        selected_transitions = selected[
            selected[RegistryColumns.FOLDER] == "folder_1"
        ]
        return train, test, selected_transitions.reset_index(drop=True)

    train, test = registry.select_volunteers_split(
        combined,
        config.train_volunteer_count,
        config.test_volunteer_count,
        seed=config.seed,
    )
    test_volunteers = sorted(test[RegistryColumns.VOLUNTEER_ID].unique())
    mandatory_test = registry.select_test_coverage(
        test,
        test_volunteers,
        minimum_transition_infos=config.minimum_transition_infos,
        seed=config.seed,
    )
    logger.info("Mandatory test coverage: %d rows", len(mandatory_test))
    selected_paths = set(train[RegistryColumns.FILE_PATH]) | set(
        test[RegistryColumns.FILE_PATH]
    )
    selected_transitions = combined[
        (combined[RegistryColumns.FOLDER].eq("folder_1"))
        & (combined[RegistryColumns.FILE_PATH].isin(selected_paths))
    ]
    return train, test, selected_transitions.reset_index(drop=True)


def _build_loaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    registry: DatasetRegistry,
    balancer: ClassBalancer,
    cache: LRUArrayCache,
) -> dict[str, DataLoader]:
    """Build modality and fused loaders from finalized train/test registries."""
    train_mmg = registry.filter_by_modality(train_df, "MMG")
    train_imu = registry.filter_by_modality(train_df, "IMU")
    test_mmg = registry.filter_by_modality(test_df, "MMG")
    test_imu = registry.filter_by_modality(test_df, "IMU")

    if train_mmg.empty:
        raise ValueError("Training data contains no MMG records.")
    sample_weights = balancer.compute_sample_weights(train_mmg)
    mmg_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_mmg),
        replacement=True,
    )

    loaders = {
        "train_mmg": DataLoader(
            SingleModalityDataset(train_mmg, cache),
            batch_size=32,
            sampler=mmg_sampler,
            num_workers=0,
            pin_memory=True,
        ),
        "train_imu": DataLoader(
            SingleModalityDataset(train_imu, cache),
            batch_size=32,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        ),
        "test_mmg": DataLoader(
            SingleModalityDataset(test_mmg, cache),
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        ),
        "test_imu": DataLoader(
            SingleModalityDataset(test_imu, cache),
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        ),
    }

    if not train_mmg.empty and not train_imu.empty:
        loaders["train_fused"] = DataLoader(
            FusedModalityDataset(train_mmg, train_imu, cache),
            batch_size=32,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
    if not test_mmg.empty and not test_imu.empty:
        loaders["test_fused"] = DataLoader(
            FusedModalityDataset(test_mmg, test_imu, cache),
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
    return loaders


def main(
    setup: str = "separate_volunteers",
    same_volunteer_id: int | str | None = None,
    train_volunteer_count: int = 8,
    test_volunteer_count: int = 2,
    total_budget_gb: float = 21.0,
    seed: int = 42,
    allowed_class_imbalance: float = 0.10,
    minimum_transition_infos: int = 5,
) -> dict[str, DataLoader]:
    """Prepare reproducible volunteer-based training and test loaders."""
    config = ExperimentConfig(
        setup=setup,
        same_volunteer_id=same_volunteer_id,
        train_volunteer_count=train_volunteer_count,
        test_volunteer_count=test_volunteer_count,
        total_budget_gb=total_budget_gb,
        seed=seed,
        allowed_class_imbalance=allowed_class_imbalance,
        minimum_transition_infos=minimum_transition_infos,
    )
    config.validate()

    registry = DatasetRegistry()
    folder_1_df, folder_2_df = registry.build_dual_folder(
        folder_1="data/transitions",
        folder_2="data/just_states",
        load_shapes=True,
    )
    train_selected, test_selected, selected_transitions = _select_experiment_data(
        registry, folder_1_df, folder_2_df, config
    )

    balancer = ClassBalancer(strategy="undersample", random_state=config.seed)
    balanced_train = balancer.balance_with_tolerance(
        train_selected,
        allowed_imbalance=config.allowed_class_imbalance,
    )

    transition_size_gb = (
        selected_transitions["file_size_bytes"].sum() / (1024**3)
    )
    budget = MemoryBudget(
        total_budget_gb=config.total_budget_gb,
        folder_1_size_gb=transition_size_gb,
    )
    logger.info("Selected transitions: %.2f GB", transition_size_gb)
    logger.info("%s", budget.summary())

    cache = LRUArrayCache(budget=budget)
    cache.load_folder_1(
        selected_transitions[RegistryColumns.FILE_PATH].tolist()
    )
    loaders = _build_loaders(
        balanced_train,
        test_selected,
        registry,
        balancer,
        cache,
    )
    logger.info(
        "Prepared %d training rows and %d test rows in %s mode.",
        len(balanced_train),
        len(test_selected),
        config.setup,
    )
    return loaders


if __name__ == "__main__":
    main()
