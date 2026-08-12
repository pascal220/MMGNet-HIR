"""
run_registry.py 
Full dual-folder workflow with class balancing and memory-aware loading.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from dataset_registry import DatasetRegistry, RegistryColumns
from class_balancer import ClassBalancer
from memory_manager import LRUArrayCache, MemoryBudget
from datasets import SingleModalityDataset, FusedModalityDataset

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console
        logging.FileHandler('session_registry.log')  # File
    ]
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("Starting main workflow")
    logger.info("=" * 80)
    # ------------------------------------------------------------------ #
    # 1. Build separate registries per folder                             #
    # ------------------------------------------------------------------ #
    logger.info("Step 1: Building registries for both folders")
    registry = DatasetRegistry()

    df_folder_1, df_folder_2 = registry.build_dual_folder(
        folder_1="data/transitions",
        folder_2="data/just_states",
        load_shapes=True,
    )
    logger.info(f"Folder 1 summary:\n{registry.summary(df_folder_1)}")
    logger.info(f"Folder 2 summary:\n{registry.summary(df_folder_2)}")

    # ------------------------------------------------------------------ #
    # 2. Compute folder 1 actual size for memory budget                  #
    # ------------------------------------------------------------------ #
    logger.info("Step 2: Computing folder 1 actual size")
    folder_1_size_gb = (
        df_folder_1["file_size_bytes"].sum() / (1024 ** 3)
    )
    logger.info(f"Folder 1 actual size: {folder_1_size_gb:.2f} GB")

    # ------------------------------------------------------------------ #
    # 3. Configure memory budget  ← specifiable parameter                #
    # ------------------------------------------------------------------ #
    logger.info("Step 3: Configuring memory budget")
    budget = MemoryBudget(
        total_budget_gb = 24.0,           # ← set to about 24 GB 
        folder_1_size_gb = folder_1_size_gb,
    )
    logger.info(f"Memory budget configured: {budget.summary()}")

    # ------------------------------------------------------------------ #
    # 4. Balance classes before loading                                   #
    # ------------------------------------------------------------------ #
    logger.info("Step 4: Balancing classes across folders")
    balancer = ClassBalancer(strategy="undersample", random_state=42)

    balanced_folder_1, balanced_folder_2 = balancer.balance_to_available(
        primary_df=df_folder_1,
        secondary_df=df_folder_2,
        secondary_budget_bytes=budget.folder_2_budget_bytes,
    )
    logger.info(f"Balancing complete. Folder 1: {len(balanced_folder_1)} samples, Folder 2: {len(balanced_folder_2)} samples")

    # ------------------------------------------------------------------ #
    # 5. Build LRU cache and load folder 1 permanently                   #
    # ------------------------------------------------------------------ #
    logger.info("Step 5: Building LRU cache and loading folder 1")
    cache = LRUArrayCache(budget=budget)
    cache.load_folder_1(
        balanced_folder_1[RegistryColumns.FILE_PATH].tolist()
    )
    logger.info("Folder 1 loading complete")

    # ------------------------------------------------------------------ #
    # 6. Per-epoch: sample folder 2 proportional to folder 1 counts      #
    # ------------------------------------------------------------------ #
    logger.info("Step 6: Starting training epoch loop")
    n_folder_1 = len(balanced_folder_1)
    logger.info(f"Starting 10 epochs with {n_folder_1} folder 1 samples")

    for epoch in range(10):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/10")
        logger.info(f"{'='*60}")

        # Sample folder 2 up to the same count as folder 1
        logger.debug(f"Sampling folder 2 data for epoch {epoch + 1}")
        folder_2_sample = balanced_folder_2.sample(
            n=min(n_folder_1, len(balanced_folder_2)),
            replace=False,
            random_state=epoch,
        )
        logger.debug(f"Sampled {len(folder_2_sample)} folder 2 samples")

        # Combine and shuffle
        logger.debug("Combining and shuffling folder 1 and folder 2 data")
        epoch_df = pd.concat(
            [balanced_folder_1, folder_2_sample], ignore_index=True
        ).sample(frac=1, random_state=epoch)
        logger.debug(f"Combined epoch dataframe has {len(epoch_df)} samples")

        # ── Split by modality ─────────────────────────────────────── #
        logger.debug("Filtering data by modality")
        mmg_df = registry.filter_by_modality(epoch_df, "MMG")
        imu_df = registry.filter_by_modality(epoch_df, "IMU")
        logger.info(f"Epoch {epoch + 1}: MMG samples={len(mmg_df)}, IMU samples={len(imu_df)}")

        # ── Compute sample weights for WeightedRandomSampler ─────── #
        logger.debug("Computing sample weights")
        sample_weights = balancer.compute_sample_weights(mmg_df)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(mmg_df),
            replacement=True,
        )
        logger.debug(f"Sample weights computed, range: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")

        # ── Single modality datasets ──────────────────────────────── #
        logger.debug("Creating single modality datasets")
        mmg_dataset = SingleModalityDataset(mmg_df, cache=cache)
        imu_dataset = SingleModalityDataset(imu_df, cache=cache)
        logger.debug(f"MMG dataset: {len(mmg_dataset)} samples, IMU dataset: {len(imu_dataset)} samples")

        logger.debug("Creating MMG DataLoader")
        mmg_loader = DataLoader(
            mmg_dataset,
            batch_size=32,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
        )
        logger.debug(f"MMG DataLoader created with batch_size=32")

        # ── Fused modality dataset ────────────────────────────────── #
        logger.debug("Creating fused modality dataset")
        fused_dataset = FusedModalityDataset(
            mmg_registry=mmg_df,
            imu_registry=imu_df,
            cache=cache,
            fusion_strategy="early",
        )
        logger.debug(f"Fused dataset created: {len(fused_dataset)} samples")

        logger.debug("Creating fused DataLoader")
        fused_loader = DataLoader(
            fused_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        logger.debug(f"Fused DataLoader created with batch_size=32")

        # ── Training loop placeholder ─────────────────────────────── #
        logger.debug(f"Training on {len(mmg_loader)} MMG batches")
        batch_count = 0
        for data, labels in mmg_loader:
            batch_count += 1
            # model.train_step(data, labels)
        logger.debug(f"Processed {batch_count} batches")

        stats = cache.stats()
        logger.info(f"Epoch {epoch + 1} complete - {stats['budget_summary']}")

    logger.info("\n" + "=" * 80)
    logger.info("Training workflow completed successfully")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()