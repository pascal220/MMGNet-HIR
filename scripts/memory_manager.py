"""
memory_manager.py  

Folder 1 maps directly to the core cache (always resident).
Folder 2 is loaded dynamically up to the remaining RAM budget.
"""

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from dataset_registry import RegistryColumns

logger = logging.getLogger(__name__)

BYTES_PER_GB = 1024 ** 3


@dataclass
class CachedEntry:
    array: np.ndarray
    size_bytes: int
    is_core: bool = False

    @classmethod
    def from_array(cls, array: np.ndarray, is_core: bool = False) -> "CachedEntry":
        return cls(array=array, size_bytes=array.nbytes, is_core=is_core)


@dataclass
class MemoryBudget:
    """
    Tracks RAM usage for dual-folder loading.

    Parameters
    ----------
    total_budget_gb : float
        Total available RAM in gigabytes (32 or 64 — specifiable).
    folder_1_size_gb : float
        Actual size of folder 1 data in gigabytes (always loaded).
    """

    total_budget_gb: float
    folder_1_size_gb: float
    _used_dynamic_bytes: int = field(default=0, init=False)

    @property
    def total_budget_bytes(self) -> int:
        return int(self.total_budget_gb * BYTES_PER_GB)

    @property
    def folder_1_bytes(self) -> int:
        return int(self.folder_1_size_gb * BYTES_PER_GB)

    @property
    def folder_2_budget_bytes(self) -> int:
        """Remaining bytes available for folder 2 (dynamic) data."""
        return max(0, self.total_budget_bytes - self.folder_1_bytes)

    @property
    def used_dynamic_bytes(self) -> int:
        return self._used_dynamic_bytes

    @property
    def available_dynamic_bytes(self) -> int:
        return self.folder_2_budget_bytes - self._used_dynamic_bytes

    def can_fit(self, size_bytes: int) -> bool:
        return size_bytes <= self.available_dynamic_bytes

    def allocate(self, size_bytes: int) -> None:
        self._used_dynamic_bytes += size_bytes

    def release(self, size_bytes: int) -> None:
        self._used_dynamic_bytes = max(0, self._used_dynamic_bytes - size_bytes)

    def utilisation_pct(self) -> float:
        if self.folder_2_budget_bytes == 0:
            return 0.0
        return (self._used_dynamic_bytes / self.folder_2_budget_bytes) * 100.0

    def summary(self) -> str:
        summary_str = (
            f"MemoryBudget | Total: {self.total_budget_gb:.1f} GB | "
            f"Folder 1 (core): {self.folder_1_size_gb:.2f} GB | "
            f"Folder 2 budget: {self.folder_2_budget_bytes / BYTES_PER_GB:.2f} GB | "
            f"Folder 2 used: {self._used_dynamic_bytes / BYTES_PER_GB:.2f} GB "
            f"({self.utilisation_pct():.1f}%)"
        )
        logger.debug(summary_str)
        return summary_str


class LRUArrayCache:
    """
    Thread-safe LRU cache.

    Folder 1 files → permanent core (never evicted).
    Folder 2 files → dynamic LRU (evicted when budget exceeded).

    Parameters
    ----------
    budget : MemoryBudget
        Memory budget tracker.
    """

    def __init__(self, budget: MemoryBudget):
        self._budget = budget
        self._core: dict[str, CachedEntry] = {}
        self._dynamic: OrderedDict[str, CachedEntry] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_folder_1(self, file_paths: list[str]) -> None:
        """
        Load all folder 1 files into the permanent core cache.

        Parameters
        ----------
        file_paths : list[str]
            All file paths from folder 1 (post-balancing).
        """
        logger.info(f"Starting folder 1 core cache loading: {len(file_paths)} files")
        total = len(file_paths)
        loaded_bytes = 0
        skipped_count = 0

        for idx, path in enumerate(file_paths, start=1):
            if path in self._core:
                logger.debug(f"File already in core cache: {path}")
                skipped_count += 1
                continue

            logger.debug(f"Loading file {idx}/{total}: {path}")
            array = np.load(path, allow_pickle=False)
            entry = CachedEntry.from_array(array, is_core=True)
            self._core[path] = entry
            loaded_bytes += entry.size_bytes

            if idx % max(1, total // 10) == 0 or idx == total:
                pct = (idx / total) * 100
                gb_loaded = loaded_bytes / BYTES_PER_GB
                logger.info(
                    f"Folder 1 loading progress: {idx}/{total} ({pct:.0f}%) | "
                    f"{gb_loaded:.2f} GB loaded"
                )

        core_size_gb = loaded_bytes / BYTES_PER_GB
        logger.info(
            f"Folder 1 core cache ready: {len(self._core)} files | "
            f"{core_size_gb:.2f} GB | {skipped_count} files already cached"
        )

    def get(self, file_path: str) -> np.ndarray:
        """
        Retrieve an array by file path via core or LRU dynamic cache.

        Parameters
        ----------
        file_path : str
            Path to the .npy file.

        Returns
        -------
        np.ndarray
        """
        logger.debug(f"Getting array from cache: {file_path}")
        with self._lock:
            if file_path in self._core:
                logger.debug(f"Found in core cache")
                return self._core[file_path].array

            if file_path in self._dynamic:
                logger.debug(f"Found in dynamic cache, moving to end")
                self._dynamic.move_to_end(file_path)
                return self._dynamic[file_path].array

            logger.debug(f"Not in cache, loading dynamically")
            return self._load_dynamic(file_path)

    def evict_all_dynamic(self) -> None:
        """Evict all folder 2 dynamic cache entries."""
        logger.info(f"Evicting all dynamic cache entries")
        with self._lock:
            evicted_size = sum(entry.size_bytes for entry in self._dynamic.values())
            for entry in self._dynamic.values():
                self._budget.release(entry.size_bytes)
            evicted_count = len(self._dynamic)
            self._dynamic.clear()
        logger.info(
            f"Folder 2 dynamic cache cleared: {evicted_count} files | "
            f"{evicted_size / BYTES_PER_GB:.2f} GB freed"
        )

    def stats(self) -> dict:
        logger.debug("Getting cache statistics")
        with self._lock:
            stats = {
                "folder_1_files": len(self._core),
                "folder_2_cached_files": len(self._dynamic),
                "budget_summary": self._budget.summary(),
            }
        return stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_dynamic(self, file_path: str) -> np.ndarray:
        logger.info(f"Loading dynamic file: {file_path}")
        array = np.load(file_path, allow_pickle=False)
        entry = CachedEntry.from_array(array, is_core=False)
        logger.debug(f"Loaded array size: {entry.size_bytes / (1024**2):.2f} MB")

        if entry.size_bytes > self._budget.folder_2_budget_bytes:
            logger.error(
                f"File '{file_path}' ({entry.size_bytes / BYTES_PER_GB:.3f} GB) "
                f"exceeds entire folder 2 budget ({self._budget.folder_2_budget_bytes / BYTES_PER_GB:.2f} GB)"
            )
            raise MemoryError(
                f"File '{file_path}' ({entry.size_bytes / BYTES_PER_GB:.3f} GB) "
                f"exceeds the entire folder 2 budget "
                f"({self._budget.folder_2_budget_bytes / BYTES_PER_GB:.2f} GB)."
            )

        evicted_count = 0
        while not self._budget.can_fit(entry.size_bytes) and self._dynamic:
            logger.debug(f"Budget full, evicting LRU entry")
            self._evict_lru()
            evicted_count += 1

        if evicted_count > 0:
            logger.info(f"Evicted {evicted_count} files to make space")

        self._dynamic[file_path] = entry
        self._dynamic.move_to_end(file_path)
        self._budget.allocate(entry.size_bytes)
        logger.info(
            f"Dynamic cache: {len(self._dynamic)} files, "
            f"{self._budget.used_dynamic_bytes / BYTES_PER_GB:.2f} / "
            f"{self._budget.folder_2_budget_bytes / BYTES_PER_GB:.2f} GB used"
        )

        return array

    def _evict_lru(self) -> None:
        lru_path, lru_entry = self._dynamic.popitem(last=False)
        self._budget.release(lru_entry.size_bytes)
        logger.debug(f"Evicted LRU file: {lru_path} ({lru_entry.size_bytes / (1024**2):.2f} MB)")