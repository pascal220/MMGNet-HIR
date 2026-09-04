"""
memory_manager.py

Plans which of the selected samples can stay resident within a single
memory budget, then reads exactly those samples into float32 tensors.

Samples carrying transition information are mandatory and are reserved
first. Whatever budget remains is divided equally between the selected
volunteers and spent on samples without transition information, which are
dropped individually when they do not fit.

Only selected samples are read. A file is opened through a memory map and
just the chosen rows are copied, so a bucket that wants 93 samples from a
3448-sample recording pays for 93.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from dataset_registry import (
    MODALITY_PATH_COLUMN,
    RegistryColumns,
    SampleColumns,
    sample_bytes,
)

logger = logging.getLogger(__name__)

BYTES_PER_GB = 1024 ** 3

TRAIN = "train"
TEST = "test"

# Optional data follows the transitions split when one volunteer supplies both.
SPLIT_WEIGHTS = {TRAIN: 0.9, TEST: 0.1}


@dataclass
class MemoryBudget:
    """A single limit covering every tensor held in memory."""

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
    """The outcome of fitting the selected samples into the budget."""

    optional_train: pd.DataFrame
    optional_test: pd.DataFrame
    mandatory_bytes: int
    optional_bytes: int
    budget_bytes: int
    volunteer_bytes: dict[tuple[str, str], int] = field(default_factory=dict)
    dropped_samples: int = 0

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
            f"  Dropped           : {self.dropped_samples} samples",
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
    """Decide which optional samples fit, before any array is read.

    Every sample carries its own float32 cost, so the result is exact
    without touching the data.
    """
    budget_bytes = budget.total_budget_bytes
    mandatory_bytes = sample_bytes(mandatory_train) + sample_bytes(mandatory_test)

    if mandatory_bytes > budget_bytes:
        raise MemoryError(
            "Transition data alone needs "
            f"{mandatory_bytes / BYTES_PER_GB:.2f} GiB but the budget is "
            f"{budget.total_budget_gb:.2f} GiB. Raise total_budget_gb: this "
            "data is mandatory and is never dropped."
        )

    remaining = budget_bytes - mandatory_bytes
    sources = {TRAIN: optional_train, TEST: optional_test}

    buckets: dict[tuple[str, str], pd.DataFrame] = {}
    for split, df in sources.items():
        if df.empty:
            continue
        for volunteer, group in df.groupby(RegistryColumns.VOLUNTEER_ID):
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
    kept: dict[str, list[np.ndarray]] = {TRAIN: [], TEST: []}
    volunteer_bytes: dict[tuple[str, str], int] = {}

    for bucket_key in sorted(buckets):
        group = buckets[bucket_key].sort_values(
            [SampleColumns.PAIR_KEY, SampleColumns.SAMPLE_INDEX]
        )
        quota = allocations[bucket_key]
        costs = group[SampleColumns.SAMPLE_BYTES].to_numpy(dtype=np.int64)
        order = rng.permutation(len(group))
        affordable = int(np.searchsorted(costs[order].cumsum(), quota, side="right"))
        chosen = group.index.to_numpy()[order[:affordable]]
        volunteer_bytes[bucket_key] = int(costs[order[:affordable]].sum())
        if affordable:
            kept[bucket_key[1]].append(chosen)

    result: dict[str, pd.DataFrame] = {}
    for split, parts in kept.items():
        if parts:
            positions = np.sort(np.concatenate(parts))
            result[split] = sources[split].loc[positions].reset_index(drop=True)
        else:
            result[split] = sources[split].iloc[0:0].copy()

    offered = len(optional_train) + len(optional_test)
    kept_count = len(result[TRAIN]) + len(result[TEST])
    plan = ResidencyPlan(
        optional_train=result[TRAIN],
        optional_test=result[TEST],
        mandatory_bytes=mandatory_bytes,
        optional_bytes=sample_bytes(result[TRAIN]) + sample_bytes(result[TEST]),
        budget_bytes=budget_bytes,
        volunteer_bytes=volunteer_bytes,
        dropped_samples=offered - kept_count,
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
    demands = {key: sample_bytes(group) for key, group in buckets.items()}

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


def load_samples(samples: pd.DataFrame, modality: str) -> Tensor:
    """Read the selected samples of one modality into a single tensor.

    The destination is allocated once and filled in place, since
    concatenating parts would briefly need twice the memory. Rows keep the
    order of ``samples``, so the returned tensor lines up with the sample
    table and with the other modality.
    """
    if modality not in MODALITY_PATH_COLUMN:
        raise ValueError(f"Unknown modality '{modality}'.")
    if samples.empty:
        raise ValueError("Cannot load an empty sample table.")

    path_column = MODALITY_PATH_COLUMN[modality]
    samples = samples.reset_index(drop=True)
    total = len(samples)

    probe = np.load(samples.at[0, path_column], mmap_mode="r", allow_pickle=False)
    item_shape = tuple(int(dim) for dim in probe.shape[1:])
    destination = torch.empty((total, *item_shape), dtype=torch.float32)
    expected = destination.nelement() * destination.element_size()
    logger.info(
        "Reading %d %s samples into a %.2f GiB tensor",
        total, modality, expected / BYTES_PER_GB,
    )

    groups = list(samples.groupby(path_column, sort=True))
    for number, (path, group) in enumerate(groups, start=1):
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if tuple(int(dim) for dim in array.shape[1:]) != item_shape:
            raise ValueError(
                f"'{path}' has item shape {array.shape[1:]}, expected {item_shape}."
            )
        wanted = group[SampleColumns.SAMPLE_INDEX].to_numpy(dtype=np.int64)
        # Read in file order, then scatter to the caller's row order.
        read_order = np.argsort(wanted)
        block = np.asarray(array[wanted[read_order]], dtype=np.float32)
        rows = group.index.to_numpy()[read_order]
        destination[torch.from_numpy(rows)] = torch.from_numpy(block)

        if number % max(1, len(groups) // 10) == 0 or number == len(groups):
            logger.debug(
                "%s progress: %d/%d files", modality, number, len(groups)
            )

    return destination
