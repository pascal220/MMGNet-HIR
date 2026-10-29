from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("scripts", "models", "train", "evaluation")] + [str(ROOT)]

import fusion_cnn_amputee_model as cnn

# main configures file logging at import time; keep tests free of that side effect.
with patch("logging.basicConfig"), patch("logging.FileHandler"):
    import main
import amputee_binary_evaluation_common as binary_eval
import amputee_train_common
from amputee_binary_evaluation_common import BinarySpec, evaluate_binary_type, overall_metrics
from binary_bundle import bundle_path, load_bundle, save_bundle
from binary_tasks import BINARY_TASKS, task_model_key, task_prepared
from data_loader import (
    ExperimentConfig,
    ExperimentData,
    PreparedData,
    amputee_experiment_config,
)
from dataset_registry import RegistryColumns as C
from fusion_train_amputee import MODELS, train_fusion_amputee_binary_type
from run_selection import ModelNotAvailableError, RunVerificationError
from training_experiment import metadata_fingerprint

MARKERS = ("0", "50", "100", "50m", "100m")
CLASS_ORDER = {"sit": 0, "stand": 1, "walking": 2, "sit_to_stand": 3, "stand_to_sit": 4}


def _metadata(count: int, prefix: str) -> pd.DataFrame:
    return pd.DataFrame({
        C.VOLUNTEER_ID: ["A003"] * count,
        "imu_source_file": [f"{prefix}{i}.npy" for i in range(count)],
        "source_sample_index": list(range(count)),
        C.TRANSITION_INFO: [MARKERS[i % len(MARKERS)] for i in range(count)],
    })


def _prepared(train: int = 50, test: int = 25, seed: int = 0) -> PreparedData:
    torch.manual_seed(seed)
    config = amputee_experiment_config(3, "type1", just_states_ratio=1.05)
    return PreparedData(
        experiment=ExperimentData(config, {}, {}),
        input_mode="single_window",
        model_target="fusion",
        X_imu_train=torch.randn(train, 6, 125),
        X_cwt_train=torch.randn(train, 5, 40, 125),
        y_train=torch.arange(train) % 5,
        X_imu_test=torch.randn(test, 6, 125),
        X_cwt_test=torch.randn(test, 5, 40, 125),
        y_test=torch.arange(test) % 5,
        train_metadata=_metadata(train, "train"),
        test_metadata=_metadata(test, "test"),
    )


class TaskDefinitionTests(unittest.TestCase):
    def test_tasks_match_the_requested_problems(self) -> None:
        self.assertEqual(
            {task.name: task.class_names for task in BINARY_TASKS},
            {
                "sit_vs_sit_to_stand": ("sit", "sit_to_stand"),
                "sit_to_stand_vs_stand": ("sit_to_stand", "stand"),
                "stand_to_sit_vs_sit": ("stand_to_sit", "sit"),
                "stand_vs_walking": ("stand", "walking"),
                "stand_walking_stand_to_sit": ("stand", "walking", "stand_to_sit"),
            },
        )
        self.assertEqual([task.num_classes for task in BINARY_TASKS], [2, 2, 2, 2, 3])

    def test_label_subset_defines_classes_of_the_experiment(self) -> None:
        config = ExperimentConfig(amputee_id="A003", data_type="type1", label_subset=(4, 0))

        self.assertEqual(config.label_to_class, {0: "stand_to_sit", 1: "sit"})
        with self.assertRaises(ValueError):
            ExperimentConfig(same_volunteer_id=4, label_subset=(0, 1)).validate()
        with self.assertRaises(ValueError):
            ExperimentConfig(amputee_id="A003", data_type="type1", label_subset=(0, 9)).validate()


class TaskPreparedTests(unittest.TestCase):
    def test_rows_are_filtered_and_relabelled_in_task_order(self) -> None:
        prepared = _prepared()
        task = BINARY_TASKS[2]  # stand_to_sit (4) -> 0, sit (0) -> 1

        data = task_prepared(prepared, task)

        keep_train = torch.nonzero((prepared.y_train == 4) | (prepared.y_train == 0)).squeeze(1)
        self.assertEqual(len(data.y_train), 20)
        self.assertEqual(len(data.y_test), 10)
        self.assertTrue(torch.equal(data.X_imu_train, prepared.X_imu_train[keep_train]))
        self.assertTrue(torch.equal(
            data.y_train, torch.where(prepared.y_train[keep_train] == 4, 0, 1),
        ))
        self.assertEqual(
            list(data.train_metadata["imu_source_file"]),
            [f"train{i}.npy" for i in keep_train.tolist()],
        )
        self.assertEqual(data.experiment.config.label_to_class, {0: "stand_to_sit", 1: "sit"})
        self.assertEqual(data.experiment.num_classes, 2)
        self.assertEqual(len(prepared.y_train), 50)

    def test_every_class_of_a_three_class_task_is_kept(self) -> None:
        data = task_prepared(_prepared(), BINARY_TASKS[-1])

        self.assertEqual(sorted(set(data.y_train.tolist())), [0, 1, 2])
        self.assertEqual(data.experiment.num_classes, 3)

    def test_training_inputs_can_be_left_out_for_evaluation(self) -> None:
        data = task_prepared(_prepared(), BINARY_TASKS[0], train_inputs=False)

        self.assertEqual(len(data.X_imu_train), 0)
        self.assertEqual(len(data.y_train), len(data.train_metadata))
        self.assertEqual(len(data.X_imu_test), len(data.y_test))

    def test_invalid_inputs_are_rejected(self) -> None:
        prepared = _prepared()
        subset = task_prepared(prepared, BINARY_TASKS[0])
        healthy = _prepared()
        object.__setattr__(healthy.experiment, "config", ExperimentConfig(same_volunteer_id=4))
        missing_class = _prepared()
        missing_class.y_train[missing_class.y_train == 3] = 0

        with self.assertRaises(ValueError):
            task_prepared(subset, BINARY_TASKS[1])
        with self.assertRaises(ValueError):
            task_prepared(healthy, BINARY_TASKS[0])
        with self.assertRaises(ValueError):
            task_prepared(missing_class, BINARY_TASKS[0])


def _fake_run(root: Path, name: str, model: torch.nn.Module, manifest_data: dict) -> dict:
    checkpoint = root / f"{name}.pt"
    torch.save({"model_state_dict": model.state_dict(), "model_config": model.config}, checkpoint)
    manifest = root / f"{name}.json"
    manifest.write_text(json.dumps({
        "data": manifest_data,
        "artifacts": {"checkpoint_sha256": f"sha-{name}"},
    }), encoding="utf-8")
    return {"run_id": name, "checkpoint_path": str(checkpoint), "manifest_path": str(manifest)}


def _write_bundle(root: Path, prepared: PreparedData, **overrides) -> Path:
    """Bundle of tiny untrained models whose manifests describe ``prepared``'s split."""
    config = asdict(prepared.experiment.config)
    config.update(overrides)
    runs = {}
    for task in BINARY_TASKS:
        data = task_prepared(prepared, task, train_inputs=False)
        model = cnn.FusionCNNAmputee(num_classes=task.num_classes)
        runs[task.name] = _fake_run(root, task.name, model, {
            "experiment_config": config,
            "metadata_fingerprint_sha256": metadata_fingerprint(data.train_metadata),
        })
    path = bundle_path(root, cnn.MODEL_KEY, "A003", "type1")
    return save_bundle(
        path, model_key=cnn.MODEL_KEY, input_mode="single_window",
        amputee_id="A003", data_type="type1", runs=runs,
    )


class BundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_one_file_holds_every_task_model(self) -> None:
        path = _write_bundle(self.root, _prepared())

        bundle = load_bundle(path)

        self.assertEqual(path.name, f"best_{cnn.MODEL_KEY}_A003_type1_binary.pt")
        self.assertEqual(list(bundle["tasks"]), [task.name for task in BINARY_TASKS])
        entry = bundle["tasks"]["stand_walking_stand_to_sit"]
        self.assertEqual(entry["class_names"], ["stand", "walking", "stand_to_sit"])
        self.assertEqual(entry["model_config"]["num_classes"], 3)
        model = cnn.FusionCNNAmputee.from_config(entry["model_config"])
        model.load_state_dict(entry["model_state_dict"])

    def test_a_bundle_of_other_tasks_is_rejected(self) -> None:
        path = _write_bundle(self.root, _prepared())

        with self.assertRaises(ValueError):
            load_bundle(path, BINARY_TASKS[:2])
        torch.save({"format": "other"}, self.root / "other.pt")
        with self.assertRaises(ValueError):
            load_bundle(self.root / "other.pt")


class TrainingTests(unittest.TestCase):
    def test_every_task_is_trained_for_each_model_and_bundled(self) -> None:
        prepared = _prepared()
        seen: list[dict] = []

        def fake_run(**kwargs):
            seen.append(kwargs)
            return {"run_id": f"run{len(seen)}"}

        with (
            patch("amputee_train_common.run_training_experiment", side_effect=fake_run),
            patch("amputee_train_common.save_bundle", return_value=Path("bundle.pt")) as save,
        ):
            result = train_fusion_amputee_binary_type(prepared, n_trials=3, checkpoint_dir="ck")

        self.assertEqual(list(result), ["fusion_cnn", "fusion_gru"])
        self.assertEqual(len(seen), 2 * len(BINARY_TASKS))
        expected = [
            (task_model_key(model.model_key, task), task.num_classes)
            for model in MODELS for task in BINARY_TASKS
        ]
        self.assertEqual([(k["model_key"], k["num_classes"]) for k in seen], expected)
        for kwargs, (_, classes) in zip(seen, expected):
            self.assertEqual(kwargs["config"].n_trials, 3)
            self.assertEqual(kwargs["prepared"].experiment.num_classes, classes)
            self.assertEqual(len(kwargs["input_tensors"][0]), len(kwargs["prepared"].y_train))
            self.assertIsNone(kwargs.get("legacy_checkpoint_path"))
            tuner = kwargs["tuner_factory"](None, None, {"device": "cpu"})
            self.assertEqual(tuner.num_classes, classes)
        self.assertEqual(save.call_count, 2)
        self.assertEqual(
            save.call_args_list[0].args[0], bundle_path("ck", MODELS[0].model_key, "A003", "type1"),
        )
        self.assertEqual(list(save.call_args_list[0].kwargs["runs"]), [t.name for t in BINARY_TASKS])
        self.assertEqual(result["fusion_cnn"]["bundle_path"], "bundle.pt")

    def test_no_bundle_is_written_without_a_checkpoint_dir(self) -> None:
        with (
            patch("amputee_train_common.run_training_experiment", return_value={}),
            patch("amputee_train_common.save_bundle") as save,
        ):
            result = train_fusion_amputee_binary_type(_prepared(), checkpoint_dir=None)

        save.assert_not_called()
        self.assertIsNone(result["fusion_cnn"]["bundle_path"])

    def test_healthy_data_is_rejected(self) -> None:
        prepared = _prepared()
        object.__setattr__(prepared.experiment, "config", ExperimentConfig(same_volunteer_id=4))

        with self.assertRaises(ValueError):
            amputee_train_common.train_amputee_type_binary(prepared, MODELS, "single_window")


class EvaluationTests(unittest.TestCase):
    SPECS = (BinarySpec("FusionCNN", cnn.MODEL_KEY, cnn.FusionCNNAmputee.from_config),)

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _evaluate(self, prepared: PreparedData) -> dict:
        return evaluate_binary_type(
            prepared, self.SPECS, "single_window",
            checkpoint_dir=self.root, output_dir=self.root / "out", device="cpu",
        )

    def test_every_model_is_scored_and_summarised_overall(self) -> None:
        prepared = _prepared()
        _write_bundle(self.root, prepared)

        result = self._evaluate(prepared)

        model = result["models"]["FusionCNN"]
        self.assertEqual(list(model["tasks"]), [task.name for task in BINARY_TASKS])
        self.assertEqual(
            [m["test_rows"] for m in model["tasks"].values()], [10, 10, 10, 10, 15],
        )
        self.assertEqual(
            np.asarray(model["tasks"]["stand_walking_stand_to_sit"]["confusion_matrix_counts"]).shape,
            (3, 3),
        )
        overall = model["overall"]
        self.assertEqual(overall["pooled_rows"], 55)
        self.assertAlmostEqual(
            overall["mean_accuracy"], np.mean([m["accuracy"] for m in model["tasks"].values()]),
        )
        self.assertAlmostEqual(
            overall["pooled_accuracy"],
            sum(m["accuracy"] * m["test_rows"] for m in model["tasks"].values()) / 55,
        )
        out = self.root / "out" / "FusionCNN"
        for task in BINARY_TASKS:
            self.assertTrue((out / f"confusion_matrix_{task.name}.png").is_file())
            predictions = pd.read_csv(out / f"predictions_{task.name}.csv")
            self.assertEqual(len(predictions), model["tasks"][task.name]["test_rows"])
            self.assertTrue(set(predictions["class_true"]) <= set(task.class_names))
        self.assertTrue((out / "transition_accuracy.png").is_file())
        saved = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["overall"]["pooled_rows"], 55)

    def test_summary_lists_each_model_and_the_overall_row(self) -> None:
        prepared = _prepared()
        _write_bundle(self.root, prepared)
        results = {"type1": self._evaluate(prepared)}

        summary = binary_eval._summary_rows(results)
        binary_eval._plot_summary(summary, "title", self.root / "summary.png")

        self.assertEqual(
            list(summary["task"]), [task.name for task in BINARY_TASKS] + [binary_eval.OVERALL_LABEL],
        )
        overall = summary.iloc[-1]
        self.assertAlmostEqual(overall["accuracy"], summary["accuracy"].iloc[:-1].mean())
        self.assertEqual(overall["test_rows"], 55)
        self.assertTrue((self.root / "summary.png").is_file())

    def test_models_trained_on_another_split_are_rejected(self) -> None:
        prepared = _prepared()
        _write_bundle(self.root, prepared, seed=7)

        with self.assertRaises(RunVerificationError) as raised:
            self._evaluate(prepared)

        self.assertIn("--binary-classifiers", str(raised.exception))

    def test_models_trained_on_test_samples_are_rejected(self) -> None:
        prepared = _prepared()
        _write_bundle(self.root, prepared)
        leaked = _prepared()
        leaked.test_metadata["imu_source_file"] = leaked.train_metadata["imu_source_file"][:25].to_numpy()
        leaked.test_metadata["source_sample_index"] = np.arange(25)

        with self.assertRaises(RunVerificationError):
            self._evaluate(leaked)

    def test_missing_bundle_stops_before_any_data_is_loaded(self) -> None:
        with patch("amputee_binary_evaluation_common.run_per_amputee_type") as load:
            with self.assertRaises(ModelNotAvailableError) as raised:
                binary_eval.evaluate_amputee_binary(
                    3, "single_window", self.SPECS, checkpoint_dir=self.root,
                    output_root=self.root / "results",
                    seed=42, test_fraction=0.1, just_states_ratio=1.05,
                )

        load.assert_not_called()
        self.assertIn("--binary-classifiers", str(raised.exception))
        self.assertIn("A003", str(raised.exception))

    def test_overall_metrics_pool_rows_and_average_models(self) -> None:
        tasks = {
            "a": {"test_rows": 4, "accuracy": 0.5, "balanced_accuracy": 0.5, "macro_f1": 0.4,
                  "confusion_matrix_counts": [[1, 1], [1, 1]]},
            "b": {"test_rows": 6, "accuracy": 1.0, "balanced_accuracy": 1.0, "macro_f1": 1.0,
                  "confusion_matrix_counts": [[3, 0], [0, 3]]},
        }

        overall = overall_metrics(tasks)

        self.assertAlmostEqual(overall["mean_accuracy"], 0.75)
        self.assertAlmostEqual(overall["pooled_accuracy"], 0.8)
        self.assertEqual(overall["pooled_rows"], 10)


def _named_mock(name: str) -> MagicMock:
    mock = MagicMock(return_value={})
    mock.__name__ = name
    return mock


class CliTests(unittest.TestCase):
    DATA_KWARGS = dict(
        total_budget_gb=10.0, seed=42, test_fraction=0.1, just_states_ratio=1.05, batch_size=32,
    )

    def _main(self, *argv: str) -> int:
        with patch.object(sys, "argv", ["main.py", *argv]), \
                contextlib.redirect_stderr(io.StringIO()):
            return main.main()

    def test_flag_selects_the_binary_entry_points(self) -> None:
        for mode, train_name, eval_name in (
            ("windowed", "train_fusion_windows_amputee_binary", "evaluate_fusion_windows_amputee_binary"),
            ("single_window", "train_fusion_amputee_binary", "evaluate_fusion_single_window_amputee_binary"),
        ):
            train, evaluate = _named_mock("train"), _named_mock("evaluate")
            with self.subTest(mode=mode), \
                    patch(f"main.{train_name}", train), patch(f"main.{eval_name}", evaluate):
                exit_code = self._main(
                    "--amputee-id", "3", "--train", "--test", "--binary-classifiers",
                    "--input-mode", mode, "--n-trials", "5",
                )

            self.assertEqual(exit_code, 0)
            train.assert_called_once_with("A003", n_trials=5, **self.DATA_KWARGS)
            evaluate.assert_called_once_with("A003", **self.DATA_KWARGS)

    def test_flag_off_keeps_the_five_class_entry_points(self) -> None:
        train, binary = _named_mock("train"), _named_mock("binary")
        with patch("main.train_fusion_windows_amputee", train), \
                patch("main.train_fusion_windows_amputee_binary", binary):
            self.assertEqual(self._main("--amputee-id", "3", "--train"), 0)

        train.assert_called_once()
        binary.assert_not_called()

    def test_flag_requires_an_amputee(self) -> None:
        with self.assertRaises(SystemExit):
            self._main("--same-volunteer-id", "4", "--train", "--binary-classifiers")


if __name__ == "__main__":
    unittest.main()
