"""
memory_manager.py

Plans which of the selected files can stay resident within a single memory
budget, then loads them as float32 tensors.

Data carrying transition information is mandatory and is reserved first.
Whatever budget remains is divided equally between the selected volunteers
and spent on data without transition information, which is dropped when it
does not fit.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from dataset_registry import (
    PairColumns,
    RegistryColumns,
    build_modality_pairs,
    select_pair_rows,
)

logger = logging.getLogger(__name__)

BYTES_PER_GB = 1024 ** 3

TRAIN = "train"
TEST = "test"

# Optional data follows the transitions split when one volunteer supplies both.
SPLIT_WEIGHTS = {TRAIN: 0.9, TEST: 0.1}


@dataclass
class MemoryBudget:
    """A single limit covering every array held in memory."""

    total_budget_gb: float

    def __post_init__(self) -> None:
        if self.total_budget_gb <= 0:
            raise ValueError("total_budget_gb must be positive.")

    @property
    def total_budget_bytes(self) -> int:
        return int(self.total_budget_gb * BYTES_PER_GB)

    def summary(self) -> str:
        return f"MemoryBudget | Total: {self.total_budget_gb:.2f} GiB"


@dataclass
class ResidencyPlan:
    """The outcome of fitting the selected data into the budget."""

    optional_train: pd.DataFrame
    optional_test: pd.DataFrame
    mandatory_bytes: int
    optional_bytes: int
    budget_bytes: int
    volunteer_bytes: dict[tuple[str, str], int] = field(default_factory=dict)
    dropped_pairs: int = 0
    dropped_examples: int = 0

    @property
    def total_bytes(self) -> int:
        return self.mandatory_bytes + self.optional_bytes

    def summary(self) -> str:
        lines = [
            "[ResidencyPlan]",
            f"  Budget            : {self.budget_bytes / BYTES_PER_GB:.2f} GiB",
            f"  Transitions (kept): {self.mandatory_bytes / BYTES_PER_GB:.2f} GiB",
            f"  Just states (kept): {self.optional_bytes / BYTES_PER_GB:.2f} GiB",
            f"  Total resident    : {self.total_bytes / BYTES_PER_GB:.2f} GiB "
            f"({self.total_bytes / self.budget_bytes * 100:.1f}%)",
            f"  Dropped           : {self.dropped_pairs} pairs / "
            f"{self.dropped_examples} examples",
        ]
        for (volunteer, split), size in sorted(self.volunteer_bytes.items()):
            lines.append(
                f"    {volunteer} {split:<5}: {size / BYTES_PER_GB:.3f} GiB optional"
            )
        return "\n".join(lines)


def plan_resident_set(
    mandatory_train: pd.DataFrame,
    mandatory_test: pd.DataFrame,
    optional_train: pd.DataFrame,
    optional_test: pd.DataFrame,
    budget: MemoryBudget,
    seed: int = 42,
) -> ResidencyPlan:
    """Decide which optional rows fit, before any array is loaded.

    Planning uses the measured ``resident_bytes`` of each file, so the
    result is known without touching the data.
    """
    col = RegistryColumns
    budget_bytes = budget.total_budget_bytes
    mandatory_bytes = int(
        mandatory_train[col.RESIDENT_BYTES].sum()
        + mandatory_test[col.RESIDENT_BYTES].sum()
    )

    if mandatory_bytes > budget_bytes:
        raise MemoryError(
            "Transition data alone needs "
            f"{mandatory_bytes / BYTES_PER_GB:.2f} GiB but the budget is "
            f"{budget.total_budget_gb:.2f} GiB. Raise total_budget_gb: this "
            "data is mandatory and is never dropped."
        )

    remaining = budget_bytes - mandatory_bytes
    sources = {TRAIN: optional_train, TEST: optional_test}
    pairs = {split: build_modality_pairs(df) for split, df in sources.items()}

    buckets: dict[tuple[str, str], pd.DataFrame] = {}
    for split, pair_df in pairs.items():
        if pair_df.empty:
            continue
        for volunteer, group in pair_df.groupby(col.VOLUNTEER_ID):
            buckets[(str(volunteer), split)] = group

    if not buckets:
        return ResidencyPlan(
            optional_train=optional_train.iloc[0:0].copy(),
            optional_test=optional_test.iloc[0:0].copy(),
            mandatory_bytes=mandatory_bytes,
            optional_bytes=0,
            budget_bytes=budget_bytes,
        )

    allocations = _allocate_by_volunteer(buckets, remaining)
    rng = np.random.default_rng(seed)
    kept: dict[str, list[pd.DataFrame]] = {TRAIN: [], TEST: []}
    volunteer_bytes: dict[tuple[str, str], int] = {}
    kept_keys: set = set()

    for bucket_key in sorted(buckets):
        group = buckets[bucket_key].sort_values(PairColumns.PAIR_KEY)
        quota = allocations[bucket_key]
        used = 0
        chosen: list[int] = []
        for position in rng.permutation(len(group)):
            candidate = group.iloc[int(position)]
            size = int(candidate[PairColumns.PAIR_BYTES])
            if used + size > quota:
                continue
            chosen.append(int(candidate.name))
            used += size
        volunteer_bytes[bucket_key] = used
        kept_keys.update(chosen)
        if chosen:
            kept[bucket_key[1]].append(group.loc[sorted(chosen)])

    kept_pairs = {
        split: (
            pd.concat(parts, ignore_index=False)
            if parts
            else pairs[split].iloc[0:0]
        )
        for split, parts in kept.items()
    }
    result = {
        split: select_pair_rows(sources[split], kept_pairs[split])
        for split in (TRAIN, TEST)
    }

    optional_bytes = int(
        sum(df[col.RESIDENT_BYTES].sum() for df in result.values())
    )
    offered = pd.concat([pairs[TRAIN], pairs[TEST]], ignore_index=True)
    kept_examples = int(
        sum(df[col.SAMPLES].sum() for df in result.values()) // 2
    )
    plan = ResidencyPlan(
        optional_train=result[TRAIN],
        optional_test=result[TEST],
        mandatory_bytes=mandatory_bytes,
        optional_bytes=optional_bytes,
        budget_bytes=budget_bytes,
        volunteer_bytes=volunteer_bytes,
        dropped_pairs=len(offered) - sum(len(df) for df in kept_pairs.values()),
        dropped_examples=int(offered[PairColumns.SAMPLES].sum()) - kept_examples,
    )
    logger.info("%s", plan.summary())
    return plan


def _allocate_by_volunteer(
    buckets: dict[tuple[str, str], pd.DataFrame],
    remaining: int,
) -> dict[tuple[str, str], int]:
    """Split the leftover budget equally between volunteers.

    A volunteer supplying both splits divides its share 90/10, matching the
    transitions split. Volunteers wanting less than their share release the
    difference to the others, so no budget is left unused while data is
    still being dropped.
    """
    volunteers = sorted({key[0] for key in buckets})
    weights = {
        key: (
            SPLIT_WEIGHTS[key[1]]
            if sum(1 for other in buckets if other[0] == key[0]) > 1
            else 1.0
        )
        / len(volunteers)
        for key in buckets
    }
    demands = {
        key: int(group[PairColumns.PAIR_BYTES].sum())
        for key, group in buckets.items()
    }

    allocations: dict[tuple[str, str], int] = {}
    pending = set(buckets)
    available = float(remaining)

    while pending:
        weight_total = sum(weights[key] for key in pending)
        if weight_total <= 0:
            break
        satisfied = [
            key
            for key in pending
            if demands[key] <= available * weights[key] / weight_total
        ]
        if not satisfied:
            for key in pending:
                allocations[key] = int(available * weights[key] / weight_total)
            break
        for key in satisfied:
            allocations[key] = demands[key]
            available -= demands[key]
            pending.discard(key)

    return allocations


class TensorStore:
    """Holds every resident array as a float32 tensor, keyed by file path.

    Arrays keep their native shape: IMU stays (examples, windows, time,
    channels) and MMG stays (examples, windows, scales, time, channels).
    """

    def __init__(self, budget: MemoryBudget):
        self._budget = budget
        self._tensors: dict[str, Tensor] = {}
        self._resident_bytes = 0

    @property
    def resident_bytes(self) -> int:
        return self._resident_bytes

    def load(self, file_paths: list[str]) -> None:
        """Load every planned file into memory as float32."""
        unique_paths = sorted(set(file_paths))
        total = len(unique_paths)
        logger.info("Loading %d files into memory as float32", total)

        for index, path in enumerate(unique_paths, start=1):
            if path in self._tensors:
                continue
            # mmap avoids holding the float64 source in the heap.
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
            size = tensor.nelement() * tensor.element_size()

            if self._resident_bytes + size > self._budget.total_budget_bytes:
                raise MemoryError(
                    f"Loading '{path}' would exceed the memory budget "
                    f"({self._budget.total_budget_gb:.2f} GiB). The residency "
                    "plan and the data on disk disagree."
                )

            self._tensors[path] = tensor
            self._resident_bytes += size

            if index % max(1, total // 10) == 0 or index == total:
                logger.info(
                    "Loading progress: %d/%d (%.0f%%) | %.2f GiB resident",
                    index, total, index / total * 100,
                    self._resident_bytes / BYTES_PER_GB,
                )

        logger.info(
            "Resident set ready: %d files | %.2f GiB of %.2f GiB budget",
            len(self._tensors),
            self._resident_bytes / BYTES_PER_GB,
            self._budget.total_budget_gb,
        )

    def get(self, file_path: str) -> Tensor:
        return self._tensors[file_path]
