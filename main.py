"""
run_registry.py 
Full dual-folder workflow with class balancing and memory-aware loading.
"""

import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset_registry import DatasetRegistry, RegistryColumns
from class_balancer import ClassBalancer
from memory_manager import LRUArrayCache, MemoryBudget
from datasets import SingleModalityDataset, FusedModalityDataset


def main():
    # ------------------------------------------------------------------ #
    # 1. Build separate registries per folder                             #
    # ------------------------------------------------------------------ #
    registry = DatasetRegistry()

    df_folder_1, df_folder_2 = registry.build_dual_folder(
        folder_1="data/transitions",
        folder_2="data/just_states",
        load_shapes=True,
    )

    print(registry.summary(df_folder_1))
    print(registry.summary(df_folder_2))

    # ------------------------------------------------------------------ #
    # 2. Compute folder 1 actual size for memory budget                  #
    # ------------------------------------------------------------------ #
    folder_1_size_gb = (
        df_folder_1["file_size_bytes"].sum() / (1024 ** 3)
    )

    print(f"[Main] Folder 1 actual size: {folder_1_size_gb:.2f} GB")

    # ------------------------------------------------------------------ #
    # 3. Configure memory budget  ← specifiable parameter                #
    # ------------------------------------------------------------------ #
    budget = MemoryBudget(
        total_budget_gb = 24.0,           # ← set to about 24 GB 
        folder_1_size_gb = folder_1_size_gb,
    )

    print(budget.summary())

    # ------------------------------------------------------------------ #
    # 4. Balance classes before loading                                   #
    # ------------------------------------------------------------------ #
    balancer = ClassBalancer(strategy="undersample", random_state=42)

    balanced_folder_1, balanced_folder_2 = balancer.balance_to_available(
        primary_df=df_folder_1,
        secondary_df=df_folder_2,
        secondary_budget_bytes=budget.folder_2_budget_bytes,
    )

    # ------------------------------------------------------------------ #
    # 5. Build LRU cache and load folder 1 permanently                   #
    # ------------------------------------------------------------------ #
    cache = LRUArrayCache(budget=budget)
    cache.load_folder_1(
        balanced_folder_1[RegistryColumns.FILE_PATH].tolist()
    )

    # ------------------------------------------------------------------ #
    # 6. Per-epoch: sample folder 2 proportional to folder 1 counts      #
    # ------------------------------------------------------------------ #
    n_folder_1 = len(balanced_folder_1)

    for epoch in range(10):

        # Sample folder 2 up to the same count as folder 1
        folder_2_sample = balanced_folder_2.sample(
            n=min(n_folder_1, len(balanced_folder_2)),
            replace=False,
            random_state=epoch,
        )

        # Combine and shuffle
        epoch_df = pd.concat(
            [balanced_folder_1, folder_2_sample], ignore_index=True
        ).sample(frac=1, random_state=epoch)

        # ── Split by modality ─────────────────────────────────────── #
        mmg_df = registry.filter_by_modality(epoch_df, "MMG")
        imu_df = registry.filter_by_modality(epoch_df, "IMU")

        # ── Compute sample weights for WeightedRandomSampler ─────── #
        sample_weights = balancer.compute_sample_weights(mmg_df)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(mmg_df),
            replacement=True,
        )

        # ── Single modality datasets ──────────────────────────────── #
        mmg_dataset = SingleModalityDataset(mmg_df, cache=cache)
        imu_dataset = SingleModalityDataset(imu_df, cache=cache)

        mmg_loader = DataLoader(
            mmg_dataset,
            batch_size=32,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )

        # ── Fused modality dataset ────────────────────────────────── #
        fused_dataset = FusedModalityDataset(
            mmg_registry=mmg_df,
            imu_registry=imu_df,
            cache=cache,
            fusion_strategy="early",
        )

        fused_loader = DataLoader(
            fused_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        # ── Training loop placeholder ─────────────────────────────── #
        for data, labels in mmg_loader:
            pass  # model.train_step(data, labels)

        print(f"Epoch {epoch + 1} | {cache.stats()['budget_summary']}")


if __name__ == "__main__":
    main()