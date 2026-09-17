from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import data_loader
from data_loader import ExperimentConfig, _log_split_summary, _select_experiment_data
from dataset_registry import DatasetRegistry, RegistryColumns as C


def _registry_rows(volunteer: str, folder: str, transition: str | None) -> list[dict]:
    """Create paired IMU/MMG rows with two samples per recording."""
    rows = []
    for modality in ("IMU", "MMG"):
        rows.append(
            {
                C.FILE_PATH: f"{volunteer}_{folder}_{modality}.npy",
                C.VOLUNTEER_ID: volunteer,
                C.MODALITY: modality,
                C.CLASS_LABEL: 0,
                C.TRANSITION_INFO: transition,
                C.FOLDER: folder,
                C.SAMPLES: 2,
                C.RESIDENT_BYTES: 40,
            }
        )
    return rows


class DataLoaderTests(unittest.TestCase):
    def test_mode_is_derived_from_same_volunteer_id(self) -> None:
        self.assertEqual(ExperimentConfig().setup, "separate_volunteers")
        self.assertEqual(
            ExperimentConfig(same_volunteer_id="N004").setup,
            "same_volunteer",
        )

    def test_separate_mode_builds_just_state_pool_for_selected_volunteers_only(self) -> None:
        transitions = pd.DataFrame(
            sum((_registry_rows(f"N00{number}", "folder_1", "0") for number in range(1, 4)), [])
        )
        just_states = pd.DataFrame(
            sum((_registry_rows(f"N00{number}", "folder_2", None) for number in range(1, 4)), [])
        )
        captured_pool_volunteers: list[set[str]] = []
        real_build_sample_table = data_loader.build_sample_table

        def capture_pool(frame: pd.DataFrame) -> pd.DataFrame:
            if frame[C.FOLDER].eq("folder_2").all():
                captured_pool_volunteers.append(set(frame[C.VOLUNTEER_ID]))
            return real_build_sample_table(frame)

        with patch("data_loader.build_sample_table", side_effect=capture_pool):
            selected = _select_experiment_data(
                DatasetRegistry(),
                transitions,
                just_states,
                ExperimentConfig(train_volunteer_count=1, test_volunteer_count=1),
            )

        expected = set(selected.train_volunteer_ids) | set(selected.test_volunteer_ids)
        self.assertEqual(captured_pool_volunteers, [expected])
        selected_samples = pd.concat(
            [selected.mandatory_train, selected.mandatory_test,
             selected.optional_train, selected.optional_test]
        )
        self.assertEqual(set(selected_samples[C.VOLUNTEER_ID]), expected)

    def test_split_summary_reports_requested_and_loaded_human_readable_sizes(self) -> None:
        requested = pd.DataFrame(
            {
                C.FOLDER: ["folder_1", "folder_2"],
                "sample_bytes": [1024, 2048],
                "imu_file_path": ["imu-a.npy", "imu-b.npy"],
                "mmg_file_path": ["mmg-a.npy", "mmg-b.npy"],
            }
        )
        resident = requested.iloc[:1].copy()

        with self.assertLogs("data_loader", level="INFO") as logs:
            _log_split_summary("Train", requested, resident)

        output = "\n".join(logs.output)
        self.assertIn("Train requested: 2 samples | 3.00 KiB", output)
        self.assertIn("Train loaded:    1 samples | 1.00 KiB", output)
        self.assertIn("dropped 1 just_states (2.00 KiB)", output)


if __name__ == "__main__":
    unittest.main()