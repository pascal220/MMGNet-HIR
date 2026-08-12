"""
class_balancer.py

Handles class imbalance by computing per-class weights and producing
a balanced sampling plan before any data is loaded into memory.
Balancing is performed across what is available in each folder
independently, then combined.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from dataset_registry import RegistryColumns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass — per-class balance statistics
# ---------------------------------------------------------------------------

@dataclass
class BalanceStats:
    """
    Holds per-class counts and computed sampling weights.

    Parameters
    ----------
    class_counts : pd.Series
        Raw count of samples per class label.
    weights : pd.Series
        Inverse-frequency weight per class label.
    target_count : int
        Target number of samples per class after balancing.
    """

    class_counts: pd.Series
    weights: pd.Series
    target_count: int

    def summary(self) -> str:
        lines = ["[BalanceStats] Per-class distribution:"]
        for label, count in self.class_counts.items():
            weight = self.weights[label]
            lines.append(
                f"  Class {label:>2} | Count: {count:>6} | "
                f"Weight: {weight:.4f}"
            )
        lines.append(f"  Target count per class: {self.target_count}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Class Balancer
# ---------------------------------------------------------------------------

class ClassBalancer:
    """
    Computes a balanced sampling plan for an imbalanced dataset registry.

    Balancing strategy:
        - Compute per-class sample counts.
        - Determine the target count as the minimum class count
          (undersampling) or a user-specified value.
        - Assign inverse-frequency sampling weights.
        - Return a balanced DataFrame by stratified undersampling.

    Parameters
    ----------
    strategy : str
        'undersample' — downsample all classes to the minority class count.
        'weighted'    — return sample weights for use with WeightedRandomSampler.
    random_state : int, optional
        Seed for reproducibility.
    """

    _VALID_STRATEGIES = {"undersample", "weighted"}

    def __init__(
        self,
        strategy: str = "undersample",
        random_state: Optional[int] = 42,
    ):
        if strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {self._VALID_STRATEGIES}, "
                f"got '{strategy}'."
            )
        self._strategy = strategy
        self._random_state = random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_stats(self, df: pd.DataFrame) -> BalanceStats:
        """
        Compute class distribution statistics for a registry DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Registry DataFrame containing a CLASS_LABEL column.

        Returns
        -------
        BalanceStats
            Per-class counts, weights, and target count.
        """
        logger.debug(f"Computing balance stats for {len(df)} samples")
        counts = df[RegistryColumns.CLASS_LABEL].value_counts().sort_index()
        target = int(counts.min())
        logger.debug(f"Class counts: {dict(counts)}")
        logger.debug(f"Minimum class count (target): {target}")

        # Inverse frequency weights — minority class gets weight 1.0
        weights = target / counts
        logger.debug(f"Computed inverse frequency weights")

        return BalanceStats(
            class_counts=counts,
            weights=weights,
            target_count=target,
        )

    def balance(
        self,
        df: pd.DataFrame,
        target_count: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Return a balanced DataFrame using the configured strategy.

        Parameters
        ----------
        df : pd.DataFrame
            Registry DataFrame to balance.
        target_count : int, optional
            Override the target samples per class. Defaults to the
            minority class count in df.

        Returns
        -------
        pd.DataFrame
            Balanced DataFrame (for 'undersample') or original DataFrame
            with an added 'sample_weight' column (for 'weighted').
        """
        logger.info(f"Balancing DataFrame with strategy='{self._strategy}'")
        logger.debug(f"Input DataFrame: {len(df)} samples")
        stats = self.compute_stats(df)
        n = target_count or stats.target_count
        logger.info(f"Target samples per class: {n}")

        if self._strategy == "undersample":
            logger.debug("Applying undersampling strategy")
            return self._undersample(df, n)

        logger.debug("Applying weighted strategy")
        return self._attach_weights(df, stats)

    def balance_to_available(
        self,
        primary_df: pd.DataFrame,
        secondary_df: pd.DataFrame,
        secondary_budget_bytes: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Balance primary (folder 1) and secondary (folder 2) DataFrames
        jointly, respecting the secondary memory budget.

        Strategy:
            1. Balance primary_df by undersampling to its minority class.
            2. Determine how many secondary bytes are available per class.
            3. Greedily fill secondary up to budget, stratified by class,
               then undersample secondary to match primary class counts.

        Parameters
        ----------
        primary_df : pd.DataFrame
            Registry for folder 1 (always fully loaded).
        secondary_df : pd.DataFrame
            Registry for folder 2 (loaded up to budget).
        secondary_budget_bytes : int
            Remaining RAM budget for folder 2 data.

        Returns
        -------
        balanced_primary : pd.DataFrame
        balanced_secondary : pd.DataFrame
        """
        logger.info("="*60)
        logger.info("Starting balance_to_available")
        logger.info(f"Primary folder (1): {len(primary_df)} samples")
        logger.info(f"Secondary folder (2): {len(secondary_df)} samples")
        logger.info(f"Secondary memory budget: {secondary_budget_bytes / (1024**3):.2f} GB")
        
        # Step 1 — balance primary independently
        logger.info("Step 1: Balancing primary folder")
        balanced_primary = self.balance(primary_df)
        primary_stats = self.compute_stats(balanced_primary)
        logger.info(f"Primary balanced to {len(balanced_primary)} samples")
        logger.debug(primary_stats.summary())

        # Step 2 — fill secondary up to budget, stratified by class
        logger.info("Step 2: Filling secondary folder up to budget")
        filled_secondary = self._fill_secondary_by_budget(
            secondary_df, secondary_budget_bytes
        )
        logger.info(f"Secondary filled to {len(filled_secondary)} samples")

        if filled_secondary.empty:
            logger.warning(
                "No secondary data fits within the remaining memory budget."
            )
            return balanced_primary, filled_secondary

        # Step 3 — balance secondary to match primary class counts
        secondary_stats = self.compute_stats(filled_secondary)

        # Target = min of what primary has and what secondary has per class
        joint_target = {
            cls: min(
                primary_stats.class_counts.get(cls, 0),
                secondary_stats.class_counts.get(cls, 0),
            )
            for cls in set(primary_stats.class_counts.index)
            | set(secondary_stats.class_counts.index)
        }

        balanced_secondary = self._undersample_per_class(
            filled_secondary, joint_target
        )

        print(
            f"[ClassBalancer] Balanced primary  : {len(balanced_primary)} samples"
        )
        print(
            f"[ClassBalancer] Balanced secondary: {len(balanced_secondary)} samples"
        )

        return balanced_primary, balanced_secondary

    def compute_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute a per-sample weight array for use with PyTorch's
        WeightedRandomSampler.

        Parameters
        ----------
        df : pd.DataFrame
            Registry DataFrame (may be balanced or raw).

        Returns
        -------
        np.ndarray
            Array of shape (len(df),) with per-sample weights.
        """
        logger.debug(f"Computing sample weights for {len(df)} samples")
        stats = self.compute_stats(df)
        weights = df[RegistryColumns.CLASS_LABEL].map(stats.weights).to_numpy(
            dtype=np.float32
        )
        logger.debug(f"Sample weights computed: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")
        return weights

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _undersample(self, df: pd.DataFrame, target: int) -> pd.DataFrame:
        """Undersample each class to exactly target samples."""
        return self._undersample_per_class(
            df,
            {cls: target for cls in df[RegistryColumns.CLASS_LABEL].unique()},
        )

    def _undersample_per_class(
        self,
        df: pd.DataFrame,
        target_per_class: dict[int, int],
    ) -> pd.DataFrame:
        """
        Undersample each class to its specified target count.
        Classes with fewer samples than the target are kept as-is.
        """
        parts = []

        for cls, target in target_per_class.items():
            subset = df[df[RegistryColumns.CLASS_LABEL] == cls]
            if len(subset) == 0 or target == 0:
                continue
            n = min(len(subset), target)
            parts.append(
                subset.sample(n=n, random_state=self._random_state)
            )

        if not parts:
            return pd.DataFrame(columns=df.columns)

        return pd.concat(parts, ignore_index=True).sample(
            frac=1, random_state=self._random_state
        )

    @staticmethod
    def _attach_weights(df: pd.DataFrame, stats: BalanceStats) -> pd.DataFrame:
        """Attach a 'sample_weight' column for WeightedRandomSampler."""
        df = df.copy()
        df["sample_weight"] = df[RegistryColumns.CLASS_LABEL].map(stats.weights)
        return df

    def _fill_secondary_by_budget(
        self,
        df: pd.DataFrame,
        budget_bytes: int,
    ) -> pd.DataFrame:
        """
        Greedily select secondary files up to budget_bytes,
        filling each class proportionally (round-robin across classes).
        """
        self._ensure_size_column(df)

        class_groups = {
            cls: iter(
                group.sort_values("file_size_bytes").itertuples()
            )
            for cls, group in df.groupby(RegistryColumns.CLASS_LABEL)
        }

        selected_indices = []
        accumulated = 0
        exhausted = set()

        while len(exhausted) < len(class_groups):
            for cls, it in class_groups.items():
                if cls in exhausted:
                    continue
                try:
                    row = next(it)
                    if accumulated + row.file_size_bytes <= budget_bytes:
                        selected_indices.append(row.Index)
                        accumulated += row.file_size_bytes
                    else:
                        exhausted.add(cls)
                except StopIteration:
                    exhausted.add(cls)

        print(
            f"[ClassBalancer] Secondary fill: {len(selected_indices)} files | "
            f"{accumulated / (1024 ** 3):.2f} GB"
        )

        return df.loc[selected_indices].reset_index(drop=True)

    @staticmethod
    def _ensure_size_column(df: pd.DataFrame) -> None:
        """
        Add 'file_size_bytes' column if absent, computed from shape
        columns or from disk.
        """
        if "file_size_bytes" in df.columns:
            return

        shape_cols = [
            RegistryColumns.WIDTH,
            RegistryColumns.HEIGHT,
            RegistryColumns.CHANNELS,
            RegistryColumns.SAMPLES,
        ]

        if all(c in df.columns for c in shape_cols):
            df["file_size_bytes"] = (
                df[RegistryColumns.WIDTH]
                * df[RegistryColumns.HEIGHT]
                * df[RegistryColumns.CHANNELS]
                * df[RegistryColumns.SAMPLES]
                * 4  # float32
            )
        else:
            import os
            df["file_size_bytes"] = df[
                RegistryColumns.FILE_PATH
            ].apply(os.path.getsize)