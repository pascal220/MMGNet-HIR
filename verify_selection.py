"""Temporary verification of the new selection spec (no array loading)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import numpy as np
import pandas as pd

from dataset_registry import DatasetRegistry, RegistryColumns as C
from main import ExperimentConfig, _select_experiment_data

registry = DatasetRegistry()
f1, f2 = registry.build_dual_folder("data/transitions", "data/just_states", load_shapes=False)

volunteers = sorted(f1[C.VOLUNTEER_ID].unique())
print(f"Volunteers: {volunteers}")


def report(name, train, test, trans):
    print(f"\n=== {name} ===")
    for split_name, df in [("train", train), ("test", test)]:
        g = df.groupby([C.CLASS_LABEL, C.MODALITY, C.FOLDER]).size().unstack(C.FOLDER, fill_value=0)
        g["ratio_js/trans"] = (g.get("folder_2", 0) / g.get("folder_1", 1)).round(3)
        print(f"--- {split_name}: {len(df)} rows ---")
        print(g.to_string())
    overlap = set(train[C.FILE_PATH]) & set(test[C.FILE_PATH])
    assert not overlap, f"train/test overlap: {len(overlap)} files"
    # every transitions row must carry a valid value
    t1 = pd.concat([train, test])
    t1 = t1[t1[C.FOLDER] == "folder_1"]
    assert t1[C.TRANSITION_INFO].isin({"100m", "50m", "0", "50", "100"}).all()
    print(f"selected_transitions: {len(trans)} rows; no overlap; all transition values valid")


# --- same_volunteer mode ---
cfg = ExperimentConfig(setup="same_volunteer", same_volunteer_id=volunteers[0])
train, test, trans = _select_experiment_data(registry, f1, f2, cfg)
report(f"same_volunteer ({volunteers[0]})", train, test, trans)

# check 10% per (modality, transition value) on transitions rows
tt = pd.concat([train, test])
tt = tt[tt[C.FOLDER] == "folder_1"]
te = test[test[C.FOLDER] == "folder_1"]
for (mod, val), grp in tt.groupby([C.MODALITY, C.TRANSITION_INFO]):
    n = len(grp)
    n_test = len(te[(te[C.MODALITY] == mod) & (te[C.TRANSITION_INFO] == val)])
    expected = max(1, int(np.floor(n * 0.10)))
    assert n_test == expected, (mod, val, n, n_test, expected)
print("10% per (modality, transition value) rule verified")

# reproducibility
train2, test2, _ = _select_experiment_data(registry, f1, f2, cfg)
assert train[C.FILE_PATH].tolist() == train2[C.FILE_PATH].tolist()
assert test[C.FILE_PATH].tolist() == test2[C.FILE_PATH].tolist()
print("seed-42 reproducibility verified")

# --- separate_volunteers mode ---
n_vol = len(volunteers)
cfg2 = ExperimentConfig(setup="separate_volunteers",
                        train_volunteer_count=max(1, n_vol - 1),
                        test_volunteer_count=1)
train, test, trans = _select_experiment_data(registry, f1, f2, cfg2)
report(f"separate_volunteers ({n_vol - 1}+1)", train, test, trans)
tv = set(train[C.VOLUNTEER_ID]) & set(test[C.VOLUNTEER_ID])
assert not tv, f"volunteer leakage: {tv}"
print("no volunteer leakage between train and test")
print("\nALL CHECKS PASSED")
