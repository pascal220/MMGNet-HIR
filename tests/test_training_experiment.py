from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import optuna
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "models")]

from training_experiment import (
    RunArtifacts,
    TrainingRunConfig,
    _make_dataset,
    balanced_class_weights,
    best_epoch_metrics,
    export_study,
    normalize_training_params,
    run_training_experiment,
)
from fusion_cnn_model import FusionCNNTuner
from fusion_cnn_window_model import FusionCNNWindowTuner
from fusion_gru_model import FusionGRUTuner
from fusion_gru_window_model import FusionGRUWindowTuner
from imu_cnn_model import IntentCNNTuner
from imu_cnn_window_model import IntentCNNWindowTuner
from mmg_cnn_model import LocomotionMMGCNNTuner
from mmg_cnn_window_model import LocomotionMMGCNNWindowTuner


class _FakeFigure:
    def write_html(self, path: str | Path) -> None:
        Path(path).write_text("<html>plot</html>", encoding="utf-8")


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(2, 7)
        self.config = {"in_features": 2, "num_classes": 7}

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer(inputs)


class _TinyTrainer:
    def __init__(self, model: _TinyModel, hyperparams: dict) -> None:
        self.model = model
        self.cfg = hyperparams
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

    def fit(self, train_loader, val_loader=None, epochs=1, verbose=True):
        if val_loader is not None:
            raise AssertionError("Final refit must not retain validation data.")
        self.history["train_loss"] = [0.5] * epochs
        self.history["train_acc"] = [0.5] * epochs
        return self.history

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": self.model.config,
                "history": self.history,
                "cfg": self.cfg,
            },
            path,
        )


class _TinyTuner:
    def __init__(self, train_loader, val_loader, search_space: dict) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.search = {"epochs": 2, **search_space}
        self.study = None

    def run(
        self,
        n_trials,
        timeout,
        show_progress,
        storage,
        study_name,
        load_if_exists,
    ):
        self.study = optuna.create_study(
            direction="maximize",
            storage=storage,
            study_name=study_name,
            load_if_exists=load_if_exists,
        )

        def objective(trial: optuna.Trial) -> float:
            trial.suggest_categorical("optimizer", ["Adam"])
            trial.suggest_categorical("batch_size", [8])
            trial.suggest_float("adam_lr", 1e-3, 1e-3)
            trial.suggest_float("adam_wd", 1e-4, 1e-4)
            trial.suggest_float("adam_beta1", 0.9, 0.9)
            trial.suggest_float("adam_beta2", 0.999, 0.999)
            trial.set_user_attr("best_epoch", 2)
            trial.set_user_attr("validation_accuracy", 0.8)
            trial.set_user_attr("validation_macro_f1", 0.7)
            return 0.75

        self.study.optimize(objective, n_trials=n_trials, timeout=timeout)
        return self._build_best_model()

    def _build_best_model(self) -> _TinyModel:
        return _TinyModel()


@dataclass(frozen=True)
class _TinyExperimentConfig:
    setup: str = "separate_volunteers"
    same_volunteer_id: str | None = None
    train_volunteer_count: int = 8
    test_volunteer_count: int = 2
    total_budget_gb: float = 1.0
    seed: int = 42
    test_fraction: float = 0.10
    just_states_ratio: float = 1.10
    batch_size: int = 8


class _TestSealedPrepared:
    def __init__(self) -> None:
        labels = torch.arange(7).repeat_interleave(10)
        self.y_train = labels
        self.X_imu_train = torch.randn(len(labels), 2)
        self.train_metadata = pd.DataFrame(
            {
                "imu_source_file": [f"sample-{index}.npy" for index in range(len(labels))],
                "source_sample_index": list(range(len(labels))),
                "volunteer_id": ["N001"] * len(labels),
            }
        )
        self.input_mode = "single_window"
        self.model_target = "standalone"
        self.experiment = SimpleNamespace(config=_TinyExperimentConfig())

    @property
    def X_imu_test(self):
        raise AssertionError("Training accessed sealed test inputs.")

    @property
    def y_test(self):
        raise AssertionError("Training accessed sealed test labels.")


class TrainingExperimentTests(unittest.TestCase):
    def test_all_tuners_support_deterministic_best_model_rebuild(self) -> None:
        tuner_classes = (
            IntentCNNTuner,
            IntentCNNWindowTuner,
            LocomotionMMGCNNTuner,
            LocomotionMMGCNNWindowTuner,
            FusionCNNTuner,
            FusionGRUTuner,
            FusionCNNWindowTuner,
            FusionGRUWindowTuner,
        )
        for tuner_class in tuner_classes:
            with self.subTest(tuner=tuner_class.__name__):
                self.assertTrue(callable(getattr(tuner_class, "_build_best_model", None)))

    def test_best_epoch_uses_accuracy_and_f1_from_same_epoch(self) -> None:
        history = {
            "val_acc": [0.95, 0.70],
            "val_f1": [0.20, 0.80],
        }

        selected = best_epoch_metrics(history)

        self.assertEqual(selected["best_epoch"], 2)
        self.assertAlmostEqual(selected["validation_accuracy"], 0.70)
        self.assertAlmostEqual(selected["validation_macro_f1"], 0.80)
        self.assertAlmostEqual(selected["objective_value"], 0.75)

    def test_balanced_class_weights(self) -> None:
        labels = torch.tensor([0, 0, 1, 1, 1, 1])

        weights = balanced_class_weights(labels, num_classes=2)

        self.assertEqual(weights, [1.5, 0.75])

    def test_balanced_class_weights_rejects_missing_class(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing training classes"):
            balanced_class_weights(torch.tensor([0, 0, 1]), num_classes=3)

    def test_normalize_adam_params(self) -> None:
        normalized = normalize_training_params(
            {
                "optimizer": "Adam",
                "batch_size": 64,
                "adam_lr": 0.002,
                "adam_wd": 0.0001,
                "adam_beta1": 0.91,
                "adam_beta2": 0.998,
                "lr_factor": 0.4,
                "lr_patience": 7,
            },
            best_epoch=13,
            class_weights=[1.0, 2.0],
        )

        self.assertEqual(normalized["lr"], 0.002)
        self.assertEqual(normalized["weight_decay"], 0.0001)
        self.assertEqual(normalized["batch_size"], 64)
        self.assertEqual(normalized["epochs"], 13)
        self.assertNotIn("adam_lr", normalized)

    def test_nested_pair_dataset_matches_window_fusion_trainers(self) -> None:
        first = torch.arange(6).reshape(3, 2)
        second = torch.arange(9).reshape(3, 3)
        labels = torch.tensor([0, 1, 2])

        dataset = _make_dataset((first, second), labels, nested_inputs=True)
        (actual_first, actual_second), actual_label = dataset[1]

        self.assertTrue(torch.equal(actual_first, first[1]))
        self.assertTrue(torch.equal(actual_second, second[1]))
        self.assertEqual(actual_label.item(), 1)

    def test_export_study_writes_csv_and_each_html_plot(self) -> None:
        study = optuna.create_study(direction="maximize")

        def objective(trial: optuna.Trial) -> float:
            trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            return 0.75

        study.optimize(objective, n_trials=1)

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            plots_dir = run_dir / "plots"
            plots_dir.mkdir()
            artifacts = RunArtifacts(
                run_id="test-run",
                run_dir=run_dir,
                checkpoint=run_dir / "model.pt",
                study_database=run_dir / "study.sqlite3",
                trials_csv=run_dir / "trials.csv",
                history_json=run_dir / "training_history.json",
                manifest_json=run_dir / "manifest.json",
                plots_dir=plots_dir,
            )
            with (
                patch("training_experiment.plot_optimization_history", return_value=_FakeFigure()),
                patch("training_experiment.plot_param_importances", return_value=_FakeFigure()),
                patch("training_experiment.plot_parallel_coordinate", return_value=_FakeFigure()),
                patch("training_experiment.plot_slice", return_value=_FakeFigure()),
            ):
                result = export_study(study, artifacts)

            self.assertTrue(artifacts.trials_csv.is_file())
            self.assertIn("params_learning_rate", artifacts.trials_csv.read_text("utf-8"))
            self.assertEqual(len(result["generated"]), 4)
            self.assertEqual(result["skipped"], {})
            for relative_path in result["generated"]:
                self.assertTrue((run_dir / relative_path).is_file())

    def test_runner_persists_artifacts_without_accessing_test_data(self) -> None:
        prepared = _TestSealedPrepared()
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("training_experiment.plot_optimization_history", return_value=_FakeFigure()),
                patch("training_experiment.plot_param_importances", return_value=_FakeFigure()),
                patch("training_experiment.plot_parallel_coordinate", return_value=_FakeFigure()),
                patch("training_experiment.plot_slice", return_value=_FakeFigure()),
            ):
                result = run_training_experiment(
                    prepared=prepared,
                    model_key="tiny_model",
                    input_tensors=(prepared.X_imu_train,),
                    tuner_factory=lambda train, val, search: _TinyTuner(train, val, search),
                    trainer_factory=_TinyTrainer,
                    config=TrainingRunConfig(
                        n_trials=1,
                        timeout=None,
                        artifact_root=directory,
                        show_progress=False,
                    ),
                )

            run_dir = Path(result["artifact_dir"])
            self.assertTrue(Path(result["checkpoint_path"]).is_file())
            self.assertTrue(Path(result["study_path"]).is_file())
            self.assertTrue(Path(result["trials_csv_path"]).is_file())
            manifest = __import__("json").loads(
                Path(result["manifest_path"]).read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["data"]["test_set_accessed"])
            self.assertEqual(manifest["optimization"]["selection"]["best_epoch"], 2)
            self.assertEqual(manifest["final_refit"]["training_params"]["batch_size"], 8)
            self.assertEqual(len(list((run_dir / "plots").glob("*.html"))), 4)

            with (
                patch("training_experiment.plot_optimization_history", return_value=_FakeFigure()),
                patch("training_experiment.plot_param_importances", return_value=_FakeFigure()),
                patch("training_experiment.plot_parallel_coordinate", return_value=_FakeFigure()),
                patch("training_experiment.plot_slice", return_value=_FakeFigure()),
            ):
                resumed = run_training_experiment(
                    prepared=prepared,
                    model_key="tiny_model",
                    input_tensors=(prepared.X_imu_train,),
                    tuner_factory=lambda train, val, search: _TinyTuner(train, val, search),
                    trainer_factory=_TinyTrainer,
                    config=TrainingRunConfig(
                        n_trials=1,
                        timeout=None,
                        artifact_root=directory,
                        resume_run_id=result["run_id"],
                        show_progress=False,
                    ),
                )

            resumed_manifest = __import__("json").loads(
                Path(resumed["manifest_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(resumed["run_id"], result["run_id"])
            self.assertEqual(resumed_manifest["optimization"]["completed_trials_total"], 2)


if __name__ == "__main__":
    unittest.main()
