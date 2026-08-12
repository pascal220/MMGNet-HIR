"""
core_selector.py

Determines which files form the permanent 9.2 GB core cache and
proportionally samples the remaining files for dynamic loading.
The core is selected to be proportionally representative across
volunteers, modalities, and activity classes.
"""

from typing import Optional

import pandas as pd

from dataset_registry import RegistryColumns


BYTES_PER_GB = 1024 ** 3


class CoreDataSelector:
    """
    Selects a representative subset of files to form the permanent
    core cache, targeting a specified memory size.

    Selection is stratified across volunteer × modality × class to
    ensure the core is proportionally representative. The number of
    dynamic files loaded per epoch is then scaled to match the ratio
    of examples in the core.

    Parameters
    ----------
    registry_df : pd.DataFrame
        Full dataset registry from DatasetRegistry.
    core_size_gb : float
        Target size of the core cache in gigabytes. Default 9.2.
    """

    def __init__(
        self,
        registry_df: pd.DataFrame,
        core_size_gb: float = 9.2,
    ):
        self._df = registry_df.copy()
        self._core_size_bytes = int(core_size_gb * BYTES_PER_GB)
        self._core_df: Optional[pd.DataFrame] = None
        self._dynamic_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Partition the registry into core and dynamic DataFrames.

        Returns
        -------
        core_df : pd.DataFrame
            Files to be permanently loaded into memory.
        dynamic_df : pd.DataFrame
            Remaining files loaded on demand via LRU cache.
        """
        self._validate_size_column()

        core_rows = self._stratified_fill(self._core_size_bytes)

        core_indices = set(core_rows.index)
        dynamic_rows = self._df[~self._df.index.isin(core_indices)]

        self._core_df = core_rows.reset_index(drop=True)
        self._dynamic_df = dynamic_rows.reset_index(drop=True)

        actual_core_gb = (
            self._core_df["file_size_bytes"].sum() / BYTES_PER_GB
        )

        print(
            f"[CoreDataSelector] Core : {len(self._core_df)} files | "
            f"{actual_core_gb:.2f} GB"
        )
        print(
            f"[CoreDataSelector] Dynamic: {len(self._dynamic_df)} files | "
            f"{self._dynamic_df['file_size_bytes'].sum() / BYTES_PER_GB:.2f} GB"
        )
        print(
            f"[CoreDataSelector] Dynamic load ratio: "
            f"{self.dynamic_load_ratio:.4f} "
            f"(~{self.dynamic_files_per_core_example:.1f} dynamic files "
            f"per core example)"
        )

        return self._core_df, self._dynamic_df

    @property
    def core_df(self) -> pd.DataFrame:
        """Core DataFrame. Available after select() is called."""
        self._assert_selected()
        return self._core_df

    @property
    def dynamic_df(self) -> pd.DataFrame:
        """Dynamic DataFrame. Available after select() is called."""
        self._assert_selected()
        return self._dynamic_df

    @property
    def dynamic_load_ratio(self) -> float:
        """
        Ratio of dynamic examples to core examples.
        Used to scale how many dynamic files are loaded per training step.
        """
        self._assert_selected()
        n_core = len(self._core_df)
        if n_core == 0:
            return 0.0
        return len(self._dynamic_df) / n_core

    @property
    def dynamic_files_per_core_example(self) -> float:
        """
        Number of dynamic files to load per core example per epoch,
        ensuring the full dataset is seen proportionally.
        """
        return self.dynamic_load_ratio

    def get_dynamic_sample_for_epoch(
        self,
        random_state: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Sample dynamic files proportionally to core size so that
        each epoch sees a representative slice of the full dataset.

        The number of dynamic files sampled equals the number of
        core files, maintaining a 1:1 ratio per epoch while ensuring
        the full dynamic set is covered across multiple epochs.

        Parameters
        ----------
        random_state : int, optional
            Seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Sampled dynamic files for one epoch.
        """
        self._assert_selected()

        n_sample = min(len(self._core_df), len(self._dynamic_df))

        if n_sample == 0:
            return self._dynamic_df.copy()

        return self._dynamic_df.sample(
            n=n_sample,
            replace=False,
            random_state=random_state,
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _stratified_fill(self, budget_bytes: int) -> pd.DataFrame:
        """
        Greedily fill the core budget using stratified sampling across
        volunteer × modality × class groups.

        Files are sorted within each stratum by size (ascending) to
        maximise the number of files that fit within the budget.
        """
        strata_cols = [
            RegistryColumns.VOLUNTEER_ID,
            RegistryColumns.MODALITY,
            RegistryColumns.ACTIVITY_CLASS,
        ]

        groups = self._df.groupby(strata_cols, group_keys=False)

        # Interleave one file per stratum per round until budget is full
        stratum_iters = {
            key: iter(group.sort_values("file_size_bytes").itertuples())
            for key, group in groups
        }

        selected_indices = []
        accumulated_bytes = 0
        exhausted = set()

        while len(exhausted) < len(stratum_iters):
            for key, it in stratum_iters.items():
                if key in exhausted:
                    continue
                try:
                    row = next(it)
                    if accumulated_bytes + row.file_size_bytes <= budget_bytes:
                        selected_indices.append(row.Index)
                        accumulated_bytes += row.file_size_bytes
                    else:
                        # Budget exceeded — mark stratum as done
                        exhausted.add(key)
                except StopIteration:
                    exhausted.add(key)

        return self._df.loc[selected_indices]

    def _validate_size_column(self) -> None:
        """
        Ensure 'file_size_bytes' column exists. If not, compute it
        from the shape columns (width × height × channels × samples × 4 bytes).
        """
        if "file_size_bytes" in self._df.columns:
            return

        required = [
            RegistryColumns.WIDTH,
            RegistryColumns.HEIGHT,
            RegistryColumns.CHANNELS,
            RegistryColumns.SAMPLES,
        ]

        if all(col in self._df.columns for col in required):
            self._df["file_size_bytes"] = (
                self._df[RegistryColumns.WIDTH]
                * self._df[RegistryColumns.HEIGHT]
                * self._df[RegistryColumns.CHANNELS]
                * self._df[RegistryColumns.SAMPLES]
                * 4  # float32 = 4 bytes
            )
        else:
            # Fall back to reading file sizes from disk
            import os
            self._df["file_size_bytes"] = self._df[
                RegistryColumns.FILE_PATH
            ].apply(os.path.getsize)

    def _assert_selected(self) -> None:
        if self._core_df is None:
            raise RuntimeError("Call select() before accessing core/dynamic splits.")