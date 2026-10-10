from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("scripts", "models", "train", "evaluation")] + [str(ROOT)]

import fusion_amputee_common
import fusion_cnn_amputee_model as cnn
import fusion_cnn_window_amputee_model as cnn_window
import fusion_gru_amputee_model as gru
import fusion_gru_window_amputee_model as gru_window

# main configures file logging at import time; keep tests free of that side effect.
with patch("logging.basicConfig"), patch("logging.FileHandler"):
    import main
from data_loader import (
    ExperimentConfig,
    _select_experiment_data,
    amputee_experiment_config,
    run_per_amputee_type,
)
from dataset_registry import (
    AMPUTEE_LABEL_TO_CLASS,
    AmputeeDatasetRegistry,
    RegistryColumns as C,
    normalize_amputee_id,
    normalize_data_type,
)
from evaluation_common import compute_metrics
from file_parser import AmputeeFileNameParser
from fusion_amputee_common import INVALID_GEOMETRY_ATTR
from fusion_single_window_eval_amputee import (
    evaluate_fusion_single_window_amputee,
    evaluate_fusion_single_window_amputee_type,
)
from fusion_train_amputee import train_fusion_amputee
from fusion_windows_train_amputee import train_fusion_windows_amputee_type
from run_selection import ModelNotAvailableError, require_trained_runs, select_trained_run
from training_experiment import file_sha256, metadata_fingerprint

AMPUTEE_CLASS_NAMES = ["sit", "stand", "walking", "sit_to_stand", "stand_to_sit"]
TRANSITION_CLASSES = AMPUTEE_CLASS_NAMES + ["standin_to_stand", "walk_to_stand"]
MARKERS = ("0", "50", "100", "50m", "100m")
TRAIN_METADATA = pd.DataFrame({
    "volunteer_id": ["A003"], "imu_source_file": ["a.npy"], "source_sample_index": [0],
})


def _write_sparse(path: Path, shape: tuple[int, ...]) -> None:
    np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=shape).flush()


def _make_amputee_data(root: Path, n_transition: int = 5, n_state: int = 10) -> None:
    """Write zero-filled files named like data/amputee for both recording types."""
    for folder in ("transitions", "just_states"):
        (root / folder).mkdir(parents=True)
    for data_type in ("type1", "type2"):
        for name in TRANSITION_CLASSES:
            for marker in MARKERS:
                stem = f"A003_{data_type}_{{}}_{name}_{marker}.npy"
                _write_sparse(root / "transitions" / f"Last_Series_{stem.format('IMU')}",
                              (n_transition, 4, 125, 6))
                _write_sparse(root / "transitions" / f"Last_Series_Wavelet_{stem.format('MMG')}",
                              (n_transition, 4, 40, 125, 5))
        for name in AMPUTEE_CLASS_NAMES:
            stem = f"A003_{data_type}_{{}}_{name}.npy"
            _write_sparse(root / "just_states" / f"Last_Series_{stem.format('IMU')}",
                          (n_state, 4, 125, 6))
            _write_sparse(root / "just_states" / f"Last_Series_Wavelet_{stem.format('MMG')}",
                          (n_state, 4, 40, 125, 5))


def _sample_keys(frame: pd.DataFrame, file_column: str, index_column: str) -> set[tuple]:
    return set(zip(frame[file_column], frame[index_column]))


class AmputeeParsingTests(unittest.TestCase):
    def test_filename_fields_are_parsed(self) -> None:
        meta = AmputeeFileNameParser().parse("data/Last_Series_A003_type1_IMU_sit_to_stand_50m.npy")

        self.assertEqual(meta.volunteer_id, "A003")
        self.assertEqual(meta.data_type, "type1")
        self.assertEqual(meta.modality, "IMU")
        self.assertEqual(meta.activity_class, "sit_to_stand")
        self.assertEqual(meta.transition_point, "50m")

    def test_arrivals_at_stand_are_trained_as_stand(self) -> None:
        parser = AmputeeFileNameParser()
        for source in ("standin_to_stand", "walk_to_stand"):
            meta = parser.parse(f"Last_Series_Wavelet_A003_type2_MMG_{source}_100.npy")
            self.assertEqual(meta.activity_class, "stand")
            self.assertEqual(meta.source_class, source)
            self.assertEqual(meta.data_type, "type2")

    def test_classes_outside_the_amputee_set_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AmputeeFileNameParser().parse("Last_Series_A003_type1_IMU_stairs_up_0.npy")

    def test_identifiers_are_normalised(self) -> None:
        self.assertEqual(normalize_amputee_id(3), "A003")
        self.assertEqual(normalize_amputee_id("a3"), "A003")
        self.assertEqual(normalize_data_type(2), "type2")
        self.assertEqual(normalize_data_type("Type1"), "type1")
        with self.assertRaises(ValueError):
            normalize_amputee_id("N003")


class AmputeeConfigTests(unittest.TestCase):
    def test_amputee_setup_has_five_classes(self) -> None:
        config = amputee_experiment_config("a3", 2)

        self.assertEqual((config.amputee_id, config.data_type), ("A003", "type2"))
        self.assertEqual(config.setup, "amputee")
        self.assertEqual(config.label_to_class, AMPUTEE_LABEL_TO_CLASS)
        self.assertEqual([config.label_to_class[i] for i in range(5)], AMPUTEE_CLASS_NAMES)

    def test_invalid_combinations_are_rejected(self) -> None:
        for config in (
            ExperimentConfig(amputee_id="A003", data_type="type1", same_volunteer_id=4),
            ExperimentConfig(amputee_id="A003"),
            ExperimentConfig(data_type="type1"),
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()


class AmputeeSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        _make_amputee_data(cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _registries(self, data_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        return AmputeeDatasetRegistry().build_type_registries(
            self.root / "transitions", self.root / "just_states", data_type,
        )

    def test_type_registry_contains_only_the_requested_type(self) -> None:
        transitions, just_states = self._registries("type2")

        for frame in (transitions, just_states):
            self.assertEqual(set(frame[C.DATA_TYPE]), {"type2"})
            self.assertTrue(frame[C.FILE_PATH].str.contains("_type2_").all())
        self.assertEqual(set(transitions[C.SOURCE_CLASS]), set(TRANSITION_CLASSES))

    def test_split_is_disjoint_and_every_transition_stratum_is_tested(self) -> None:
        transitions, just_states = self._registries("type1")
        for seed in (1, 42):
            config = amputee_experiment_config("A003", "type1", seed=seed, test_fraction=0.2)
            selected = _select_experiment_data(
                AmputeeDatasetRegistry(), transitions, just_states, config,
            )
            train = pd.concat([selected.mandatory_train, selected.optional_train])
            test = pd.concat([selected.mandatory_test, selected.optional_test])

            with self.subTest(seed=seed):
                self.assertFalse(
                    _sample_keys(train, "imu_file_path", "sample_index")
                    & _sample_keys(test, "imu_file_path", "sample_index")
                )
                strata = set(zip(selected.mandatory_test[C.SOURCE_CLASS],
                                 selected.mandatory_test[C.TRANSITION_INFO]))
                self.assertEqual(strata, {(n, m) for n in TRANSITION_CLASSES for m in MARKERS})
                both = pd.concat([train, test])
                self.assertEqual(set(both[C.DATA_TYPE]), {"type1"})
                self.assertLessEqual(set(both[C.CLASS_LABEL]), set(range(5)))

    def test_each_type_is_prepared_as_its_own_dataset(self) -> None:
        seen: list[str] = []

        def check(prepared) -> str:
            data_type = prepared.experiment.config.data_type
            seen.append(data_type)
            self.assertEqual(prepared.experiment.num_classes, 5)
            self.assertEqual(prepared.experiment.class_names, AMPUTEE_CLASS_NAMES)
            self.assertEqual(tuple(prepared.X_cwt_train.shape[1:]), (5, 40, 125, 4))
            for metadata in (prepared.train_metadata, prepared.test_metadata):
                self.assertEqual(set(metadata[C.DATA_TYPE]), {data_type})
            self.assertFalse(
                _sample_keys(prepared.train_metadata, "imu_source_file", "source_sample_index")
                & _sample_keys(prepared.test_metadata, "imu_source_file", "source_sample_index")
            )
            self.assertEqual(set(prepared.y_test.tolist()), set(range(5)))
            return data_type

        results = run_per_amputee_type(
            3, "windowed", check,
            data_root=self.root, total_budget_gb=1.0, test_fraction=0.2, just_states_ratio=1.05,
        )

        self.assertEqual(seen, ["type1", "type2"])
        self.assertEqual(results, {"type1": "type1", "type2": "type2"})


def _amputee_prepared(data_type: str, input_mode: str = "single_window") -> SimpleNamespace:
    return SimpleNamespace(
        experiment=SimpleNamespace(
            config=amputee_experiment_config(3, data_type, just_states_ratio=1.05),
            num_classes=5,
        ),
        input_mode=input_mode,
        model_target="fusion",
        train_metadata=TRAIN_METADATA,
        test_metadata=TRAIN_METADATA.iloc[0:0],
        X_imu_train=torch.zeros(1),
        X_cwt_train=torch.zeros(1),
    )


def _write_amputee_run(
    root: Path,
    run_id: str,
    model_key: str = cnn.MODEL_KEY,
    data_type: str | None = "type1",
    *,
    amputee_id: str | None = "A003",
    same_volunteer_id: str | None = None,
    input_mode: str = "single_window",
) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "model.pt"
    checkpoint.write_bytes(run_id.encode())
    manifest = {
        "status": "completed",
        "run_id": run_id,
        "completed_at_utc": "2026-10-01T00:00:00+00:00",
        "model": {"key": model_key, "input_mode": input_mode, "parent_checkpoints": []},
        "data": {
            "experiment_config": {
                "amputee_id": amputee_id, "data_type": data_type,
                "same_volunteer_id": same_volunteer_id, "seed": 42, "test_fraction": 0.1,
                "just_states_ratio": 1.05, "total_budget_gb": 10.0,
                "train_volunteer_count": 8, "test_volunteer_count": 2, "batch_size": 32,
            },
            "metadata_fingerprint_sha256": metadata_fingerprint(TRAIN_METADATA),
        },
        "artifacts": {"checkpoint": checkpoint.name, "checkpoint_sha256": file_sha256(checkpoint)},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class AmputeeRunSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_run_of_the_same_type_is_selected(self) -> None:
        _write_amputee_run(self.root, "type1_run", data_type="type1")
        _write_amputee_run(self.root, "type2_run", data_type="type2")

        run = select_trained_run(_amputee_prepared("type2"), cnn.MODEL_KEY, self.root)

        self.assertEqual(run.run_id, "type2_run")

    def test_other_type_or_healthy_runs_never_match(self) -> None:
        _write_amputee_run(self.root, "type2_run", data_type="type2")
        _write_amputee_run(
            self.root, "healthy_run", data_type=None, amputee_id=None, same_volunteer_id="N003",
        )

        with self.assertRaises(ModelNotAvailableError) as raised:
            select_trained_run(_amputee_prepared("type1"), cnn.MODEL_KEY, self.root)

        message = str(raised.exception)
        self.assertIn("amputee A003 type1", message)
        self.assertIn("python main.py --train --amputee-id A003", message)
        self.assertIn("--model-target fusion", message)

    def test_require_trained_runs_reports_every_missing_model_and_type(self) -> None:
        _write_amputee_run(self.root, "cnn_type1", cnn.MODEL_KEY, "type1")
        splits = [_amputee_prepared("type1"), _amputee_prepared("type2")]

        with self.assertRaises(ModelNotAvailableError) as raised:
            require_trained_runs(splits, (cnn.MODEL_KEY, gru.MODEL_KEY), self.root)

        lines = str(raised.exception).splitlines()
        missing = [line for line in lines if line.startswith("No trained")]
        self.assertEqual(len(missing), 3)
        self.assertFalse(any(f"'{cnn.MODEL_KEY}'" in line and "type1" in line for line in missing))
        self.assertIn("Train it first", lines[-1])

    def test_require_trained_runs_passes_when_every_model_exists(self) -> None:
        for data_type in ("type1", "type2"):
            for key in (cnn.MODEL_KEY, gru.MODEL_KEY):
                _write_amputee_run(self.root, f"{key}_{data_type}", key, data_type)

        require_trained_runs(
            [_amputee_prepared("type1"), _amputee_prepared("type2")],
            (cnn.MODEL_KEY, gru.MODEL_KEY), self.root,
        )


class AmputeeEvaluationTests(unittest.TestCase):
    def test_metrics_use_the_five_amputee_labels(self) -> None:
        metadata = pd.DataFrame({"transition_info": ["0", "50", None, "100", "50m"]})
        y_true = np.array([0, 1, 2, 3, 4])

        metrics = compute_metrics(y_true, y_true, metadata, labels=range(5))

        self.assertEqual(np.asarray(metrics["confusion_matrix_counts"]).shape, (5, 5))
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_missing_models_stop_evaluation_before_any_data_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "evaluation"
            with patch("amputee_evaluation_common.run_per_amputee_type") as load:
                with self.assertRaises(ModelNotAvailableError):
                    evaluate_fusion_single_window_amputee(
                        3, artifact_root=tmp, output_root=str(output_root),
                    )

            load.assert_not_called()
            self.assertFalse(output_root.exists())

    def test_type_evaluator_rejects_healthy_data(self) -> None:
        prepared = SimpleNamespace(
            experiment=SimpleNamespace(config=ExperimentConfig(same_volunteer_id=4)),
            input_mode="single_window",
            model_target="fusion",
        )

        with self.assertRaises(ValueError):
            evaluate_fusion_single_window_amputee_type(prepared)


MODEL_CASES = (
    (cnn.FusionCNNAmputee, (2, 6, 125), (2, 5, 40, 125)),
    (gru.FusionGRUAmputee, (2, 6, 125), (2, 5, 40, 125)),
    (cnn_window.FusionCNNWindowAmputee, (2, 6, 125, 4), (2, 5, 40, 125, 4)),
    (gru_window.FusionGRUWindowAmputee, (2, 6, 125, 4), (2, 5, 40, 125, 4)),
)


class AmputeeModelTests(unittest.TestCase):
    def test_models_are_trainable_end_to_end_and_rebuild_from_config(self) -> None:
        torch.manual_seed(0)
        for model_cls, imu_shape, mmg_shape in MODEL_CASES:
            with self.subTest(model=model_cls.__name__):
                model = model_cls(device="cpu").eval()
                imu, mmg = torch.randn(imu_shape), torch.randn(mmg_shape)

                self.assertTrue(all(p.requires_grad for p in model.parameters()))
                with torch.no_grad():
                    output = model(imu, mmg)
                self.assertEqual(tuple(output.shape), (2, 5))

                clone = model_cls.from_config(dict(model.config), device="cpu").eval()
                clone.load_state_dict(model.state_dict())
                with torch.no_grad():
                    torch.testing.assert_close(clone(imu, mmg), output)

    def test_one_trial_search_builds_a_five_class_model(self) -> None:
        torch.manual_seed(0)
        data = TensorDataset(
            torch.randn(16, 6, 125), torch.randn(16, 5, 40, 125), torch.arange(16) % 5,
        )
        loader = DataLoader(data, batch_size=8)
        tuner = cnn.FusionCNNAmputeeTuner(
            loader, loader,
            search_space={"epochs": 1, "device": "cpu", "batch_size": [8], "seed": 0},
        )

        with contextlib.redirect_stdout(io.StringIO()):
            model = tuner.run(n_trials=1, show_progress=False)

        trained = [t for t in tuner.study.trials if not t.user_attrs.get(INVALID_GEOMETRY_ATTR)]
        self.assertEqual(len(trained), 1)
        with torch.no_grad():
            self.assertEqual(tuple(model.cpu()(data[:2][0], data[:2][1]).shape), (2, 5))


class _ScriptedTuner(cnn.FusionCNNAmputeeTuner):
    """Tuner whose trials follow a script instead of training models."""

    NAME = "ScriptedTuner"

    def __init__(self, outcomes: list[str]):
        super().__init__(None, None, search_space={"device": "cpu", "seed": 0})
        self._outcomes = iter(outcomes)

    def _objective(self, trial: optuna.Trial) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        outcome = next(self._outcomes)
        if outcome == "invalid":
            trial.set_user_attr(INVALID_GEOMETRY_ATTR, True)
            raise optuna.exceptions.TrialPruned()
        if outcome == "oom":
            trial.set_user_attr("out_of_memory", True)
            raise optuna.exceptions.TrialPruned()
        return 0.5

    def _build_best_model(self) -> torch.nn.Module:
        return torch.nn.Identity()

    def search_quietly(self, n_trials: int) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.run(n_trials=n_trials, show_progress=False)


class AmputeeTrialBudgetTests(unittest.TestCase):
    def test_invalid_geometries_are_redrawn_outside_the_budget(self) -> None:
        tuner = _ScriptedTuner(["invalid", "invalid", "ok", "invalid", "ok"])

        tuner.search_quietly(n_trials=2)

        states = [t.state for t in tuner.study.trials]
        self.assertEqual(len(states), 5)
        self.assertEqual(states.count(optuna.trial.TrialState.COMPLETE), 2)

    def test_pruned_trials_that_trained_use_the_budget(self) -> None:
        tuner = _ScriptedTuner(["oom", "ok", "ok"])

        tuner.search_quietly(n_trials=2)

        self.assertEqual(len(tuner.study.trials), 2)

    def test_no_completed_trial_raises_a_clear_error(self) -> None:
        tuner = _ScriptedTuner(["invalid", "oom"])

        with self.assertRaisesRegex(RuntimeError, "Increase the trial budget"):
            tuner.search_quietly(n_trials=1)

    def test_attempts_are_capped(self) -> None:
        tuner = _ScriptedTuner(["invalid"] * 10)

        with patch.object(fusion_amputee_common, "MAX_ATTEMPTS_PER_TRIAL", 3):
            with self.assertRaises(RuntimeError):
                tuner.search_quietly(n_trials=1)

        self.assertEqual(len(tuner.study.trials), 3)


class AmputeeTrainingTests(unittest.TestCase):
    def test_both_models_train_in_order_with_per_type_checkpoints(self) -> None:
        with patch("amputee_train_common.run_training_experiment", return_value="ok") as run:
            result = train_fusion_windows_amputee_type(
                _amputee_prepared("type2", "windowed"), n_trials=7, checkpoint_dir="ck",
            )

        self.assertEqual(result, {"fusion_cnn": "ok", "fusion_gru": "ok"})
        calls = [call.kwargs for call in run.call_args_list]
        self.assertEqual(
            [kwargs["model_key"] for kwargs in calls],
            [cnn_window.MODEL_KEY, gru_window.MODEL_KEY],
        )
        self.assertEqual(
            calls[0]["legacy_checkpoint_path"],
            f"ck/best_{cnn_window.MODEL_KEY}_A003_type2.pt",
        )
        for kwargs, tuner_cls in zip(
            calls,
            (cnn_window.FusionCNNWindowAmputeeTuner, gru_window.FusionGRUWindowAmputeeTuner),
        ):
            self.assertTrue(kwargs["nested_inputs"])
            self.assertEqual(kwargs["num_classes"], 5)
            self.assertEqual(kwargs["config"].n_trials, 7)
            tuner = kwargs["tuner_factory"](None, None, {"device": "cpu"})
            self.assertIsInstance(tuner, tuner_cls)
            self.assertEqual(tuner.num_classes, 5)

    def test_healthy_data_is_rejected(self) -> None:
        prepared = _amputee_prepared("type1")
        prepared.experiment.config = ExperimentConfig(same_volunteer_id=4)

        with self.assertRaises(ValueError):
            train_fusion_windows_amputee_type(prepared)

    def test_one_run_trains_every_type_in_turn(self) -> None:
        trained: list[tuple[str, int]] = []

        def record(prepared, models, input_mode, **kwargs):
            trained.append((prepared.experiment.config.data_type, kwargs["n_trials"]))
            return {}

        def prepare(amputee_id, data_type, **kwargs):
            return SimpleNamespace(config=amputee_experiment_config(amputee_id, data_type, **kwargs))

        with (
            patch("fusion_train_amputee.train_amputee_type", side_effect=record),
            patch("data_loader.prepare_amputee_experiment_data", side_effect=prepare),
            patch("data_loader.prepare_training_data",
                  side_effect=lambda experiment, **_: SimpleNamespace(experiment=experiment)),
        ):
            results = train_fusion_amputee(3, n_trials=4, seed=7)

        self.assertEqual(trained, [("type1", 4), ("type2", 4)])
        self.assertEqual(list(results), ["type1", "type2"])


def _named_mock(name: str) -> MagicMock:
    mock = MagicMock(return_value={})
    mock.__name__ = name
    return mock


class AmputeeCliTests(unittest.TestCase):
    DATA_KWARGS = dict(
        total_budget_gb=10.0, seed=42, test_fraction=0.1, just_states_ratio=1.05, batch_size=32,
    )

    def _main(self, *argv: str) -> int:
        with patch.object(sys, "argv", ["main.py", *argv]), \
                contextlib.redirect_stderr(io.StringIO()):
            return main.main()

    def test_amputee_training_uses_the_trial_budget_for_every_type(self) -> None:
        train = _named_mock("train")
        with patch("main.train_fusion_windows_amputee", train):
            self.assertEqual(self._main("--amputee-id", "3", "--train", "--n-trials", "7"), 0)

        train.assert_called_once_with("A003", n_trials=7, **self.DATA_KWARGS)

    def test_amputee_testing_evaluates_every_type(self) -> None:
        evaluate = _named_mock("evaluate")
        with patch("main.evaluate_fusion_single_window_amputee", evaluate):
            exit_code = self._main("--amputee-id", "A3", "--test", "--input-mode", "single_window")

        self.assertEqual(exit_code, 0)
        evaluate.assert_called_once_with("A003", **self.DATA_KWARGS)

    def test_invalid_amputee_arguments_are_rejected(self) -> None:
        for argv in (
            ("--amputee-id", "3", "--train", "--same-volunteer-id", "4"),
            ("--amputee-id", "3", "--train", "--test-volunteer-count", "2"),
            ("--amputee-id", "3", "--train", "--model-target", "standalone"),
            ("--amputee-id", "N3", "--train"),
            ("--amputee-id", "3", "--train", "--n-trials", "0"),
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                self._main(*argv)

    def test_healthy_training_forwards_the_trial_budget_only_when_given(self) -> None:
        prepared = SimpleNamespace(input_mode="single_window", model_target="standalone")
        for argv, expected in (((), {}), (("--n-trials", "3"), {"n_trials": 3})):
            mmg, imu = _named_mock("mmg"), _named_mock("imu")
            with (
                patch("main.prepare_experiment_data"),
                patch("main.prepare_training_data", return_value=prepared),
                patch("main._log_prepared_summary"),
                patch("main.train_and_evaluate_mmg_cnn", mmg),
                patch("main.train_and_evaluate_imu_cnn", imu),
            ):
                exit_code = self._main(
                    "--same-volunteer-id", "4", "--train", "--input-mode", "single_window", *argv,
                )

            with self.subTest(argv=argv):
                self.assertEqual(exit_code, 0)
                mmg.assert_called_once_with(prepared, **expected)
                imu.assert_called_once_with(prepared, **expected)


if __name__ == "__main__":
    unittest.main()
