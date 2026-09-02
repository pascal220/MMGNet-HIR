"""Verification harness for selection, residency planning, and datasets."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

import numpy as np
import pandas as pd
import torch

from dataset_registry import (
    VALID_TRANSITION_VALUES,
    DatasetRegistry,
    PairColumns,
    RegistryColumns as C,
    build_modality_pairs,
)
from datasets import SingleModalityDataset
from main import ExperimentConfig, _select_experiment_data
from memory_manager import BYTES_PER_GB, MemoryBudget, TensorStore, plan_resident_set

GIB = BYTES_PER_GB
CHECKS: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    CHECKS.append(message)


def run_pipeline(config: ExperimentConfig, registry, f1, f2):
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


def verify_common(label, config, selected, plan, train, test):
    print(f"\n=== {label} ===")
    print(plan.summary())

    check(
        plan.total_bytes <= plan.budget_bytes,
        f"{label}: resident bytes within budget",
    )

    mandatory_paths = set(selected.mandatory_train[C.FILE_PATH]) | set(
        selected.mandatory_test[C.FILE_PATH]
    )
    kept_paths = set(train[C.FILE_PATH]) | set(test[C.FILE_PATH])
    check(
        mandatory_paths <= kept_paths,
        f"{label}: no transitions row was dropped",
    )

    transitions = pd.concat([train, test])
    transitions = transitions[transitions[C.FOLDER] == "folder_1"]
    check(
        transitions[C.TRANSITION_INFO].isin(VALID_TRANSITION_VALUES).all(),
        f"{label}: every transitions row has a valid marker",
    )

    check(
        not (set(train[C.FILE_PATH]) & set(test[C.FILE_PATH])),
        f"{label}: no file appears in both train and test",
    )

    for name, df in (("train", train), ("test", test)):
        pairs = build_modality_pairs(df)
        check(
            len(pairs) * 2 == len(df),
            f"{label}: every {name} recording keeps both IMU and MMG",
        )

    for split, df in (("train", train), ("test", test)):
        optional = df[df[C.FOLDER] == "folder_2"]
        mandatory = df[df[C.FOLDER] == "folder_1"]
        for volunteer, group in optional.groupby(C.VOLUNTEER_ID):
            transitions_examples = mandatory[
                (mandatory[C.VOLUNTEER_ID] == volunteer)
                & (mandatory[C.MODALITY] == "IMU")
            ][C.SAMPLES].sum()
            optional_examples = group[group[C.MODALITY] == "IMU"][C.SAMPLES].sum()
            check(
                optional_examples
                <= np.floor(transitions_examples * config.just_states_ratio) * 7,
                f"{label}: {split} {volunteer} respects the example cap",
            )

    sizes = [v for v in plan.volunteer_bytes.values() if v > 0]
    if len(sizes) > 1:
        spread = (max(sizes) - min(sizes)) / max(sizes)
        print(f"  per-volunteer optional spread: {spread:.1%}")


def verify_cap_in_examples(registry, f1, f2, config):
    selected = _select_experiment_data(registry, f1, f2, config)
    for split, trans, opt in (
        ("train", selected.mandatory_train, selected.optional_train),
        ("test", selected.mandatory_test, selected.optional_test),
    ):
        if opt.empty:
            continue
        imu_trans = trans[trans[C.MODALITY] == "IMU"]
        imu_opt = opt[opt[C.MODALITY] == "IMU"]
        demand = imu_trans.groupby([C.VOLUNTEER_ID, C.CLASS_LABEL])[C.SAMPLES].sum()
        supply = imu_opt.groupby([C.VOLUNTEER_ID, C.CLASS_LABEL])[C.SAMPLES].sum()
        for key, got in supply.items():
            cap = int(np.floor(demand.get(key, 0) * config.just_states_ratio))
            check(
                got <= cap,
                f"1.1x cap in examples honoured for {key} ({got} <= {cap})",
            )


def verify_dataset(train, test, budget):
    store = TensorStore(budget=budget)
    store.load(train[C.FILE_PATH].tolist() + test[C.FILE_PATH].tolist())

    measured = 0
    for path in set(train[C.FILE_PATH]) | set(test[C.FILE_PATH]):
        tensor = store.get(path)
        measured += tensor.nelement() * tensor.element_size()
        check(tensor.dtype == torch.float32, "arrays are cached as float32")
    planned = int(
        pd.concat([train, test]).drop_duplicates(C.FILE_PATH)[C.RESIDENT_BYTES].sum()
    )
    check(
        measured == planned,
        f"planned resident bytes match reality ({measured} == {planned})",
    )

    expected_shapes = {"IMU": (4, 125, 6), "MMG": (4, 40, 125, 5)}
    for modality, expected in expected_shapes.items():
        subset = train[train[C.MODALITY] == modality]
        dataset = SingleModalityDataset(subset, store)
        check(
            len(dataset) == subset[C.SAMPLES].sum(),
            f"{modality} dataset length equals its example count "
            f"({len(dataset)})",
        )
        check(
            dataset.item_shape == expected,
            f"{modality} item shape is {expected}, native and uncollapsed",
        )

        index = len(dataset) // 3
        tensor, label = dataset[index]
        check(tuple(tensor.shape) == expected, f"{modality} item tensor shape")

        meta = dataset.get_metadata(index)
        source = np.load(meta[C.FILE_PATH], mmap_mode="r")
        reference = torch.from_numpy(
            np.asarray(source[meta["example_index"]], dtype=np.float32)
        )
        check(
            torch.equal(tensor, reference),
            f"{modality} item equals the source file slice (no averaging)",
        )
        check(
            int(label) == meta[C.CLASS_LABEL],
            f"{modality} label matches its registry row",
        )
        check(
            {C.VOLUNTEER_ID, C.FOLDER, C.TRANSITION_INFO} <= set(meta),
            f"{modality} metadata is preserved for analysis",
        )


def main() -> None:
    registry = DatasetRegistry()
    f1, f2 = registry.build_dual_folder("data/transitions", "data/just_states")
    volunteers = sorted(f1[C.VOLUNTEER_ID].unique())
    print(f"Volunteers: {volunteers}")

    total_gib = (f1[C.RESIDENT_BYTES].sum() + f2[C.RESIDENT_BYTES].sum()) / GIB
    print(f"Whole dataset resident as float32: {total_gib:.2f} GiB")

    # Single volunteer, generous budget.
    cfg = ExperimentConfig(
        setup="same_volunteer",
        same_volunteer_id=volunteers[0],
        total_budget_gb=24.0,
    )
    selected, plan, train, test, budget = run_pipeline(cfg, registry, f1, f2)
    verify_common(f"same_volunteer {volunteers[0]}", cfg, selected, plan, train, test)
    verify_cap_in_examples(registry, f1, f2, cfg)
    verify_dataset(train, test, budget)

    # 10% test draw, stratified per transition value, applied to pairs.
    trans = pd.concat([selected.mandatory_train, selected.mandatory_test])
    trans_pairs = build_modality_pairs(trans)
    test_pairs = build_modality_pairs(selected.mandatory_test)
    for value, group in trans_pairs.groupby(C.TRANSITION_INFO):
        n_test = len(test_pairs[test_pairs[C.TRANSITION_INFO] == value])
        expected = max(1, int(np.ceil(len(group) * cfg.test_fraction)))
        check(
            n_test == expected,
            f"transition '{value}': {n_test} test pairs of {len(group)}",
        )
        check(
            n_test / len(group) >= cfg.test_fraction,
            f"transition '{value}': test share {n_test / len(group):.1%} "
            f"is at least {cfg.test_fraction:.0%}",
        )

    # Reproducibility.
    _, _, train2, test2, _ = run_pipeline(cfg, registry, f1, f2)
    check(
        train[C.FILE_PATH].tolist() == train2[C.FILE_PATH].tolist()
        and test[C.FILE_PATH].tolist() == test2[C.FILE_PATH].tolist(),
        "seed 42 reproduces the same selection and residency plan",
    )

    # Multi-volunteer.
    cfg2 = ExperimentConfig(
        setup="separate_volunteers",
        train_volunteer_count=len(volunteers) - 2,
        test_volunteer_count=2,
        total_budget_gb=24.0,
    )
    selected2, plan2, train_m, test_m, budget2 = run_pipeline(cfg2, registry, f1, f2)
    verify_common("separate_volunteers", cfg2, selected2, plan2, train_m, test_m)
    check(
        not (set(train_m[C.VOLUNTEER_ID]) & set(test_m[C.VOLUNTEER_ID])),
        "no volunteer appears in both train and test",
    )

    # Squeeze the budget so the drop path runs.
    mandatory_gib = (
        selected2.mandatory_train[C.RESIDENT_BYTES].sum()
        + selected2.mandatory_test[C.RESIDENT_BYTES].sum()
    ) / GIB
    optional_gib = (
        selected2.optional_train[C.RESIDENT_BYTES].sum()
        + selected2.optional_test[C.RESIDENT_BYTES].sum()
    ) / GIB
    cfg3 = ExperimentConfig(
        setup="separate_volunteers",
        train_volunteer_count=len(volunteers) - 2,
        test_volunteer_count=2,
        total_budget_gb=round(mandatory_gib + optional_gib * 0.3, 3),
    )
    selected3, plan3, train_s, test_s, _ = run_pipeline(cfg3, registry, f1, f2)
    verify_common(
        f"squeezed budget {cfg3.total_budget_gb} GiB",
        cfg3, selected3, plan3, train_s, test_s,
    )
    check(plan3.dropped_pairs > 0, "the drop path engages under a tight budget")

    # Mandatory data alone over budget must fail loudly.
    try:
        plan_resident_set(
            selected3.mandatory_train,
            selected3.mandatory_test,
            selected3.optional_train,
            selected3.optional_test,
            budget=MemoryBudget(total_budget_gb=max(0.01, mandatory_gib / 2)),
        )
    except MemoryError:
        check(True, "transitions data over budget raises MemoryError")
    else:
        raise AssertionError("expected MemoryError for an impossible budget")

    print(f"\n{len(CHECKS)} checks passed:")
    for item in CHECKS[:200]:
        print(f"  - {item}")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
