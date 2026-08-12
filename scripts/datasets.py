"""
datasets.py  (updated)

PyTorch Dataset classes updated to use the LRUArrayCache for
memory-efficient data loading.
"""

import logging
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import Optional, Callable

from dataset_registry import RegistryColumns, LABEL_TO_CLASS
from memory_manager import LRUArrayCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base Dataset — now cache-aware
# ---------------------------------------------------------------------------

class BaseActivityDataset(Dataset):
    """
    Abstract base class for activity recognition datasets.

    Parameters
    ----------
    registry_df : pd.DataFrame
        Filtered DataFrame from DatasetRegistry.
    cache : LRUArrayCache
        Shared cache instance for memory-efficient array loading.
    transform : Callable, optional
        Transform applied to the loaded tensor.
    """

    def __init__(
        self,
        registry_df: pd.DataFrame,
        cache: LRUArrayCache,
        transform: Optional[Callable] = None,
    ):
        logger.debug(f"Initializing {self.__class__.__name__} with {len(registry_df)} samples")
        self._df = registry_df.reset_index(drop=True)
        self._cache = cache
        self._transform = transform
        logger.debug(f"{self.__class__.__name__} initialized successfully")

    def __len__(self) -> int:
        return len(self._df)

    def get_class_name(self, label: int) -> str:
        return LABEL_TO_CLASS[label]

    def get_transition_info(self, index: int) -> Optional[str]:
        row = self._df.iloc[index]
        if row[RegistryColumns.HAS_TRANSITION_INFO]:
            return row[RegistryColumns.TRANSITION_POINT]
        return None

    def get_metadata(self, index: int) -> dict:
        row = self._df.iloc[index]
        return {
            "volunteer_id": row[RegistryColumns.VOLUNTEER_ID],
            "modality": row[RegistryColumns.MODALITY],
            "activity_class": row[RegistryColumns.ACTIVITY_CLASS],
            "class_label": row[RegistryColumns.CLASS_LABEL],
            "is_transition_class": row[RegistryColumns.IS_TRANSITION_CLASS],
            "transition_point": row.get(RegistryColumns.TRANSITION_POINT),
            "has_transition_info": row[RegistryColumns.HAS_TRANSITION_INFO],
        }

    # ------------------------------------------------------------------
    # Protected helpers
    # ------------------------------------------------------------------

    def _load_array(self, file_path: str) -> np.ndarray:
        """Load an array via the shared LRU cache."""
        logger.debug(f"Loading array from cache: {file_path}")
        return self._cache.get(file_path)

    @staticmethod
    def _to_tensor(array: np.ndarray) -> Tensor:
        """
        (width, height, channels, samples) → (samples, channels, width, height)
        """
        array = np.transpose(array, (3, 2, 0, 1)).astype(np.float32)
        return torch.from_numpy(array)

    def _prepare_tensor(self, file_path: str) -> Tensor:
        """Load, transpose, and squeeze/aggregate the sample dimension."""
        logger.debug(f"Preparing tensor from {file_path}")
        array = self._load_array(file_path)
        logger.debug(f"Array shape: {array.shape}")
        tensor = self._to_tensor(array)          # (samples, C, W, H)
        logger.debug(f"Tensor shape after transpose: {tensor.shape}")

        if tensor.shape[0] == 1:
            result = tensor.squeeze(0)             # (C, W, H)
            logger.debug(f"Squeezed tensor shape: {result.shape}")
            return result
        result = tensor.mean(dim=0)                # (C, W, H)
        logger.debug(f"Aggregated tensor shape: {result.shape}")
        return result

    def _apply_transform(self, tensor: Tensor) -> Tensor:
        if self._transform is not None:
            tensor = self._transform(tensor)
        return tensor


# ---------------------------------------------------------------------------
# Single Modality Dataset
# ---------------------------------------------------------------------------

class SingleModalityDataset(BaseActivityDataset):
    """
    Dataset for a single modality (MMG-only or IMU-only).

    Parameters
    ----------
    registry_df : pd.DataFrame
        DataFrame filtered to a single modality.
    cache : LRUArrayCache
        Shared LRU cache instance.
    transform : Callable, optional
        Optional transform applied to the data tensor.
    """

    def __init__(
        self,
        registry_df: pd.DataFrame,
        cache: LRUArrayCache,
        transform: Optional[Callable] = None,
    ):
        logger.debug(f"Initializing SingleModalityDataset with {len(registry_df)} samples")
        super().__init__(registry_df, cache, transform)
        logger.info(f"SingleModalityDataset created: {len(self)} samples")

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        logger.debug(f"Getting item {index}")
        row = self._df.iloc[index]

        tensor = self._prepare_tensor(row[RegistryColumns.FILE_PATH])
        tensor = self._apply_transform(tensor)

        label = torch.tensor(
            row[RegistryColumns.CLASS_LABEL], dtype=torch.long
        )

        return tensor, label


# ---------------------------------------------------------------------------
# Fused Modality Dataset
# ---------------------------------------------------------------------------

class FusedModalityDataset(BaseActivityDataset):
    """
    Dataset for fused MMG + IMU data.

    Parameters
    ----------
    mmg_registry : pd.DataFrame
        Registry DataFrame filtered to MMG modality.
    imu_registry : pd.DataFrame
        Registry DataFrame filtered to IMU modality.
    cache : LRUArrayCache
        Shared LRU cache instance.
    fusion_strategy : str
        'early' (channel concat) or 'late' (separate tensors).
    transform : Callable, optional
        Optional transform applied after fusion.
    """

    _VALID_STRATEGIES = {"early", "late"}

    def __init__(
        self,
        mmg_registry: pd.DataFrame,
        imu_registry: pd.DataFrame,
        cache: LRUArrayCache,
        fusion_strategy: str = "early",
        transform: Optional[Callable] = None,
    ):
        logger.debug(f"Initializing FusedModalityDataset with strategy='{fusion_strategy}'")
        logger.debug(f"MMG samples: {len(mmg_registry)}, IMU samples: {len(imu_registry)}")
        
        if fusion_strategy not in self._VALID_STRATEGIES:
            logger.error(f"Invalid fusion_strategy: {fusion_strategy}")
            raise ValueError(
                f"fusion_strategy must be one of {self._VALID_STRATEGIES}, "
                f"got '{fusion_strategy}'."
            )

        self._fusion_strategy = fusion_strategy
        self._mmg_df, self._imu_df = self._align_pairs(mmg_registry, imu_registry)
        logger.info(f"Aligned {len(self._mmg_df)} MMG-IMU pairs")

        super().__init__(self._mmg_df, cache, transform)
        logger.info(f"FusedModalityDataset created: {len(self)} samples, strategy={fusion_strategy}")

    def __len__(self) -> int:
        return len(self._mmg_df)

    def __getitem__(
        self, index: int
    ) -> tuple[Tensor | tuple[Tensor, Tensor], Tensor]:

        mmg_row = self._mmg_df.iloc[index]
        imu_row = self._imu_df.iloc[index]

        mmg_tensor = self._prepare_tensor(mmg_row[RegistryColumns.FILE_PATH])
        imu_tensor = self._prepare_tensor(imu_row[RegistryColumns.FILE_PATH])

        label = torch.tensor(
            mmg_row[RegistryColumns.CLASS_LABEL], dtype=torch.long
        )

        if self._fusion_strategy == "early":
            fused = torch.cat([mmg_tensor, imu_tensor], dim=0)
            return self._apply_transform(fused), label

        return (
            self._apply_transform(mmg_tensor),
            self._apply_transform(imu_tensor),
        ), label

    @staticmethod
    def _align_pairs(
        mmg_df: pd.DataFrame,
        imu_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        merge_keys = [
            RegistryColumns.VOLUNTEER_ID,
            RegistryColumns.ACTIVITY_CLASS,
            RegistryColumns.TRANSITION_POINT,
        ]

        merged = pd.merge(
            mmg_df, imu_df,
            on=merge_keys,
            suffixes=("_mmg", "_imu"),
        )

        if merged.empty:
            raise ValueError(
                "No matching MMG/IMU pairs found after alignment."
            )

        mmg_cols = {
            c: c.replace("_mmg", "")
            for c in merged.columns if c.endswith("_mmg")
        }
        imu_cols = {
            c: c.replace("_imu", "")
            for c in merged.columns if c.endswith("_imu")
        }

        aligned_mmg = merged[
            merge_keys + list(mmg_cols.keys())
        ].rename(columns=mmg_cols).reset_index(drop=True)

        aligned_imu = merged[
            merge_keys + list(imu_cols.keys())
        ].rename(columns=imu_cols).reset_index(drop=True)

        return aligned_mmg, aligned_imu