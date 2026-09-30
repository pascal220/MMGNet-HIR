from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "models"), str(ROOT / "evaluation")]

from data_loader import ExperimentConfig
from evaluation_common import row_normalised_percent, transition_accuracy
from run_selection import (
    ModelNotAvailableError,
    RunVerificationError,
    select_trained_run,
)
from summarise_best_trials import collect_best_trials, summarise
from training_experiment import file_sha256, metadata_fingerprint

TRAIN_METADATA = pd.DataFrame({"volunteer_id": ["N004", "N004"], "source_sample_index": [0, 1]})


def _prepared(volunteer: str | None = "4", seed: int = 42) -> SimpleNamespace:
    config = ExperimentConfig(
        same_volunteer_id=volunteer, train_volunteer_count=5, test_volunteer_count=5,
        total_budget_gb=10.0, seed=seed, test_fraction=0.1, just_states_ratio=1.05,
    )
    return SimpleNamespace(
        experiment=SimpleNamespace(config=config),
        input_mode="windowed",
        train_metadata=TRAIN_METADATA,
    )


def _write_run(
    root: Path,
    run_id: str,
    *,
    model_key: str = "imu_cnn_windowed",
    volunteer: str | None = "N004",
    seed: int = 42,
    completed: str = "2026-09-29T00:00:00+00:00",
    fingerprint: str | None = None,
    best_params: dict | None = None,
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / f"{run_id}.pt"
    checkpoint.write_bytes(run_id.encode())
    manifest = {
        "status": "completed",
        "run_id": run_id,
        "completed_at_utc": completed,
        "model": {"key": model_key, "input_mode": "windowed", "parent_checkpoints": []},
        "data": {
            "experiment_config": {
                "same_volunteer_id": volunteer, "seed": seed, "test_fraction": 0.1,
                "just_states_ratio": 1.05, "total_budget_gb": 10.0,
                "train_volunteer_count": 5, "test_volunteer_count": 5, "batch_size": 32,
            },
            "metadata_fingerprint_sha256": fingerprint or metadata_fingerprint(TRAIN_METADATA),
        },
        "artifacts": {"checkpoint": checkpoint.name, "checkpoint_sha256": file_sha256(checkpoint)},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    best = {
        "objective_value": 0.8, "validation_accuracy": 0.8, "validation_macro_f1": 0.8,
        "best_epoch": 40, "best_trial_number": 10,
        "best_params": best_params or {"optimizer": "Adam", "adam_lr": 0.001},
    }
    (run_dir / "best_trial.json").write_text(json.dumps(best), encoding="utf-8")
    return run_dir


class RunSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_latest_matching_run_is_selected_across_volunteer_spellings(self) -> None:
        _write_run(self.root, "old", volunteer="N004", completed="2026-09-19T00:00:00+00:00")
        _write_run(self.root, "new", volunteer="04", completed="2026-09-29T00:00:00+00:00")
        _write_run(self.root, "other", volunteer="N018", completed="2026-09-30T00:00:00+00:00")
        _write_run(self.root, "mmg", model_key="mmg_cnn_windowed", completed="2026-09-30T00:00:00+00:00")

        run = select_trained_run(_prepared("4"), "imu_cnn_windowed", self.root)

        self.assertEqual(run.run_id, "new")

    def test_missing_model_reports_near_misses_and_train_command(self) -> None:
        _write_run(self.root, "seed7", seed=7)

        with self.assertRaises(ModelNotAvailableError) as raised:
            select_trained_run(_prepared("4"), "imu_cnn_windowed", self.root)

        message = str(raised.exception)
        self.assertIn("seed7", message)
        self.assertIn("seed: trained=7, now=42", message)
        self.assertIn("python main.py --train --same-volunteer-id 4", message)

    def test_multi_volunteer_split_does_not_match_single_volunteer_runs(self) -> None:
        _write_run(self.root, "single")

        with self.assertRaises(ModelNotAvailableError) as raised:
            select_trained_run(_prepared(None), "imu_cnn_windowed", self.root)

        self.assertNotIn("single", str(raised.exception))
        self.assertIn("--train-volunteer-count 5 --test-volunteer-count 5", str(raised.exception))

    def test_fingerprint_mismatch_stops_with_error(self) -> None:
        _write_run(self.root, "leaky", fingerprint="0" * 64)

        with self.assertRaises(RunVerificationError):
            select_trained_run(_prepared("4"), "imu_cnn_windowed", self.root)

    def test_changed_checkpoint_stops_with_error(self) -> None:
        run_dir = _write_run(self.root, "changed")
        (run_dir / "changed.pt").write_bytes(b"overwritten")

        with self.assertRaises(RunVerificationError):
            select_trained_run(_prepared("4"), "imu_cnn_windowed", self.root)


class MetricTests(unittest.TestCase):
    def test_transition_accuracy_ignores_just_states_rows(self) -> None:
        metadata = pd.DataFrame({"transition_info": ["100m", "100m", None, "0", np.nan]})
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 0, 3, 4, 0])

        result = transition_accuracy(y_true, y_pred, metadata)

        self.assertEqual(result["100m"], {"n": 2, "correct": 1, "accuracy": 0.5})
        self.assertEqual(result["0"], {"n": 1, "correct": 1, "accuracy": 1.0})
        self.assertIsNone(result["50"]["accuracy"])

    def test_row_normalised_percent(self) -> None:
        percent = row_normalised_percent(np.array([[3, 1], [0, 0]]))

        np.testing.assert_allclose(percent[0], [75.0, 25.0])
        self.assertTrue(np.isnan(percent[1]).all())


class BestTrialSummaryTests(unittest.TestCase):
    def test_latest_run_per_volunteer_and_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_run(root, "a_old", volunteer="N004", completed="2026-09-19T00:00:00+00:00",
                       best_params={"optimizer": "SGD", "sgd_lr": 0.5})
            _write_run(root, "a_new", volunteer="04", completed="2026-09-29T00:00:00+00:00",
                       best_params={"optimizer": "Adam", "adam_lr": 0.001})
            _write_run(root, "b", volunteer="N018",
                       best_params={"optimizer": "Adam", "adam_lr": 0.003})
            _write_run(root, "multi", volunteer=None)

            runs = collect_best_trials(root)
            summary = summarise(runs)["imu_cnn_windowed"]

        self.assertEqual(sorted(runs["run_id"]), ["a_new", "b"])
        self.assertEqual(summary["volunteers"], ["N004", "N018"])
        lr = summary["numeric"]["param.adam_lr"]
        self.assertAlmostEqual(lr["mean"], 0.002)
        self.assertAlmostEqual(lr["range"], 0.002)
        self.assertAlmostEqual(lr["std"], np.std([0.001, 0.003], ddof=1))
        self.assertNotIn("param.sgd_lr", summary["numeric"])
        self.assertEqual(summary["categorical"]["param.optimizer"]["counts"], {"Adam": 2})


if __name__ == "__main__":
    unittest.main()
