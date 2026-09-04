"""Verification harness for sample-level selection, residency and tensors."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

import numpy as np
import pandas as pd
import torch

from dataset_registry import (
    LABEL_TO_CLASS,
    MODALITY_PATH_COLUMN,
    VALID_TRANSITION_VALUES,
    DatasetRegistry,
    RegistryColumns as C,
    SampleColumns as S,
    build_sample_table,
    sample_bytes,
)
from datasets import ModalityTensors, SingleModalityDataset
from main import ExperimentConfig, _build_bundles, _select_experiment_data
from memory_manager import BYTES_PER_GB, MemoryBudget, plan_resident_set

CHECKS: list[str] = []
STRATA = [C.VOLUNTEER_ID, C.CLASS_LABEL, C.TRANSITION_INFO]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    CHECKS.append(message)


def sample_keys(df: pd.DataFrame) -> set:
    return set(zip(df[S.PAIR_KEY], df[S.SAMPLE_INDEX]))


def run_pipeline(config, registry, f1, f2):
    selected = _select_experiment_data(registry, f1, f2, config)
    budget = MemoryBudget(total_budget_gb=config.total_budget_gb)
    plan = plan_resident_set(
        selected.mandatory_train,
        selected.mandatory_test,
        selected.optional_train,
        selected.optional_test,
        budget=budget,
        seed=config.seed,
    )
    train = pd.concat([selected.mandatory_train, plan.optional_train], ignore_index=True)
    test = pd.concat([selected.mandatory_test, plan.optional_test], ignore_index=True)
    return selected, plan, train, test, budget


def verify_selection(label, config, selected, plan, train, test):
    print(f"\n=== {label} ===")
    print(plan.summary())

    check(plan.total_bytes <= plan.budget_bytes, f"{label}: planned bytes within budget")
    check(
        plan.total_bytes == sample_bytes(train) + sample_bytes(test),
        f"{label}: plan totals match the kept sample tables",
    )

    mandatory = sample_keys(selected.mandatory_train) | sample_keys(selected.mandatory_test)
    kept = sample_keys(train) | sample_keys(test)
    check(mandatory <= kept, f"{label}: no transition sample was dropped")

    check(
        not (sample_keys(train) & sample_keys(test)),
        f"{label}: train and test are disjoint at sample level",
    )
    check(
        len(sample_keys(train)) == len(train)
        and len(sample_keys(test)) == len(test),
        f"{label}: no sample is selected twice",
    )

    transitions = pd.concat([train, test])
    transitions = transitions[transitions[C.FOLDER] == "folder_1"]
    check(
        transitions[C.TRANSITION_INFO].isin(VALID_TRANSITION_VALUES).all(),
        f"{label}: every transition sample has a valid marker",
    )

    # The 1.1x cap is counted in samples, per (volunteer, class).
    for split, df in (("train", train), ("test", test)):
        optional = df[df[C.FOLDER] == "folder_2"]
        if optional.empty:
            continue
        demand = df[df[C.FOLDER] == "folder_1"].groupby(
            [C.VOLUNTEER_ID, C.CLASS_LABEL]
        ).size()
        supply = optional.groupby([C.VOLUNTEER_ID, C.CLASS_LABEL]).size()
        for key, got in supply.items():
            cap = int(np.floor(demand.get(key, 0) * config.just_states_ratio))
            check(
                got <= cap,
                f"{label}: {split} {key[0]}/{LABEL_TO_CLASS[int(key[1])]} "
                f"kept {got} <= cap {cap}",
            )


def verify_test_share(label, config, mandatory_train, mandatory_test):
    """Every stratum must give at least test_fraction of its samples to test."""
    everything = pd.concat([mandatory_train, mandatory_test])
    per_stratum = everything.groupby(STRATA).size()
    in_test = mandatory_test.groupby(STRATA).size().reindex(per_stratum.index).fillna(0)

    worst = 1.0
    for key, total in per_stratum.items():
        n_test = int(in_test[key])
        expected = max(1, int(np.ceil(total * config.test_fraction)))
        if n_test != expected:
            raise AssertionError(
                f"{label}: stratum {key} has {n_test} test samples, expected {expected}"
            )
        worst = min(worst, n_test / total)
    check(
        worst >= config.test_fraction,
        f"{label}: every one of {len(per_stratum)} strata gives at least "
        f"{config.test_fraction:.0%} to test (worst {worst:.1%})",
    )

    classes = set(mandatory_test[C.CLASS_LABEL].unique())
    check(
        classes == set(LABEL_TO_CLASS),
        f"{label}: all seven classes are present in test",
    )
    check(
        set(mandatory_test[C.TRANSITION_INFO].unique()) == set(VALID_TRANSITION_VALUES),
        f"{label}: all five transition values are present in test",
    )


def verify_tensors(train, test, budget):
    bundles = _build_bundles(train, test)
    expected_shapes = {"imu": (4, 125, 6), "mmg": (4, 40, 125, 5)}

    resident = sum(b.nbytes for b in bundles.values())
    check(
        resident <= budget.total_budget_bytes,
        f"resident tensors {resident / BYTES_PER_GB:.2f} GiB within budget",
    )

    for name, bundle in bundles.items():
        split, modality = name.split("_")
        samples = train if split == "train" else test

        check(bundle.data.dtype == torch.float32, f"{name} is float32")
        check(
            bundle.item_shape == expected_shapes[modality],
            f"{name} item shape is {expected_shapes[modality]}",
        )
        check(
            len(bundle.data) == len(samples),
            f"{name} holds one row per selected sample ({len(bundle.data)})",
        )
        check(
            len(bundle.labels) == len(bundle.data)
            and len(bundle.metadata) == len(bundle.data),
            f"{name} labels and metadata are row-aligned with the data",
        )
        check(
            bundle.data.is_contiguous(),
            f"{name} is a single contiguous tensor",
        )

        dataset = SingleModalityDataset(bundle)
        check(len(dataset) == len(bundle.data), f"{name} dataset length matches")

        # Rows must equal the source file slice, read independently.
        rng = np.random.default_rng(0)
        for index in rng.choice(len(bundle.data), size=min(5, len(bundle.data)), replace=False):
            index = int(index)
            meta = bundle.metadata.iloc[index]
            source = np.load(meta["source_file"], mmap_mode="r")
            reference = torch.from_numpy(
                np.asarray(source[int(meta["source_sample_index"])], dtype=np.float32)
            )
            check(
                torch.equal(bundle.data[index], reference),
                f"{name} row {index} equals its source slice",
            )
            item, label = dataset[index]
            check(torch.equal(item, reference), f"{name} dataset item {index} matches")
            check(
                int(label) == int(meta[C.CLASS_LABEL])
                and meta["activity_class_name"] == LABEL_TO_CLASS[int(meta[C.CLASS_LABEL])],
                f"{name} label and class name agree at row {index}",
            )
            check(
                meta["source_file"] == samples.iloc[index][MODALITY_PATH_COLUMN[modality.upper()]],
                f"{name} metadata points at the selected file at row {index}",
            )

    # The two modalities must describe the same events, row for row.
    for split in ("train", "test"):
        imu, mmg = bundles[f"{split}_imu"], bundles[f"{split}_mmg"]
        check(
            imu.metadata[C.VOLUNTEER_ID].equals(mmg.metadata[C.VOLUNTEER_ID])
            and imu.metadata["source_sample_index"].equals(
                mmg.metadata["source_sample_index"]
            )
            and torch.equal(imu.labels, mmg.labels),
            f"{split}: IMU and MMG rows describe the same events in the same order",
        )
    return bundles


def main() -> None:
    registry = DatasetRegistry()
    f1, f2 = registry.build_dual_folder("data/transitions", "data/just_states")
    volunteers = sorted(f1[C.VOLUNTEER_ID].unique())
    print(f"Volunteers: {volunteers}")
    print(
        f"Whole dataset as float32: "
        f"{sample_bytes(build_sample_table(pd.concat([f1, f2]))) / BYTES_PER_GB:.2f} GiB"
    )

    # Single volunteer: the 10% draw applies here.
    cfg = ExperimentConfig(
        setup="same_volunteer", same_volunteer_id=volunteers[0], total_budget_gb=24.0
    )
    selected, plan, train, test, budget = run_pipeline(cfg, registry, f1, f2)
    verify_selection(f"same_volunteer {volunteers[0]}", cfg, selected, plan, train, test)
    verify_test_share(
        f"same_volunteer {volunteers[0]}", cfg,
        selected.mandatory_train, selected.mandatory_test,
    )
    verify_tensors(train, test, budget)

    _, _, train2, test2, _ = run_pipeline(cfg, registry, f1, f2)
    check(
        sample_keys(train) == sample_keys(train2)
        and sample_keys(test) == sample_keys(test2),
        "seed 42 reproduces the same sample-level selection",
    )

    # Multi-volunteer: selection only, to keep the run quick.
    cfg2 = ExperimentConfig(
        setup="separate_volunteers",
        train_volunteer_count=len(volunteers) - 2,
        test_volunteer_count=2,
        total_budget_gb=24.0,
    )
    selected2, plan2, train_m, test_m, _ = run_pipeline(cfg2, registry, f1, f2)
    verify_selection("separate_volunteers", cfg2, selected2, plan2, train_m, test_m)
    check(
        not (set(train_m[C.VOLUNTEER_ID]) & set(test_m[C.VOLUNTEER_ID])),
        "no volunteer appears in both train and test",
    )

    # Squeeze the budget so individual samples get dropped.
    mandatory_gib = (
        sample_bytes(selected2.mandatory_train) + sample_bytes(selected2.mandatory_test)
    ) / BYTES_PER_GB
    optional_gib = (
        sample_bytes(selected2.optional_train) + sample_bytes(selected2.optional_test)
    ) / BYTES_PER_GB
    cfg3 = ExperimentConfig(
        setup="separate_volunteers",
        train_volunteer_count=len(volunteers) - 2,
        test_volunteer_count=2,
        total_budget_gb=round(mandatory_gib + optional_gib * 0.3, 3),
    )
    selected3, plan3, train_s, test_s, _ = run_pipeline(cfg3, registry, f1, f2)
    verify_selection(
        f"squeezed {cfg3.total_budget_gb} GiB", cfg3, selected3, plan3, train_s, test_s
    )
    check(plan3.dropped_samples > 0, "the drop path removes individual samples")

    try:
        plan_resident_set(
            selected3.mandatory_train,
            selected3.mandatory_test,
            selected3.optional_train,
            selected3.optional_test,
            budget=MemoryBudget(total_budget_gb=max(0.01, mandatory_gib / 2)),
        )
    except MemoryError:
        check(True, "transition data over budget raises MemoryError")
    else:
        raise AssertionError("expected MemoryError for an impossible budget")

    print(f"\n{len(CHECKS)} checks passed:")
    for item in CHECKS:
        print(f"  - {item}")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
