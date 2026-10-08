from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "models"), str(ROOT / "tests")]

import fusion_cnn_model
import fusion_cnn_window_model
import fusion_gru_model
import fusion_gru_window_model
from imu_cnn_model import IntentCNN
from imu_cnn_window_model import IntentCNNWindow
from mmg_cnn_model import LocomotionMMGCNN
from mmg_cnn_window_model import LocomotionMMGCNNWindow
from test_fusion_checkpoint import (
    SINGLE_IMU, SINGLE_MMG, WINDOW_IMU, WINDOW_MMG, _save_backbone,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

HIDDEN = [64, 128, 256, 512]
BATCHES = [64, 128, 200]
DROPOUT = (0.0, 0.1)

# (module, tuner class name, trainer class name, windowed, parameters searched)
TUNERS = (
    (fusion_cnn_model, "FusionCNNTuner", "FusionCNNTrainer", False,
     {"fc_hidden": HIDDEN, "dropout": DROPOUT}),
    (fusion_gru_model, "FusionGRUTuner", "FusionGRUTrainer", False,
     {"gru_hidden": HIDDEN, "fc_hidden": HIDDEN, "fc_dropout": DROPOUT}),
    (fusion_cnn_window_model, "FusionCNNWindowTuner", "FusionCNNWindowTrainer", True,
     {"hidden_dim": HIDDEN, "dropout_rate": DROPOUT}),
    (fusion_gru_window_model, "FusionGRUWindowTuner", "FusionGRUWindowTrainer", True,
     {"gru_hidden_dim": HIDDEN, "fc_hidden_dim": HIDDEN, "fc_dropout": DROPOUT}),
)


class FusionSearchSpaceTests(unittest.TestCase):
    def test_search_spaces(self) -> None:
        for module, tuner_name, _, _, searched in TUNERS:
            search = getattr(module, tuner_name)._SEARCH
            with self.subTest(tuner=tuner_name):
                self.assertEqual(search["batch_size"], BATCHES)
                for key, expected in searched.items():
                    self.assertEqual(search[key], expected)
                for removed in (
                    "gru_num_layers", "gru_dropout", "gru_layers",
                    "n_fc_layers", "n_hidden_layers", "fc_hidden_dims", "hidden_dims",
                    "seq_len",
                ):
                    self.assertNotIn(removed, search)


class FusionTunerTrialTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        torch.manual_seed(0)
        single = (
            (IntentCNN(**SINGLE_IMU), SINGLE_IMU),
            (LocomotionMMGCNN(**SINGLE_MMG), SINGLE_MMG),
        )
        windowed = (
            (IntentCNNWindow(**WINDOW_IMU), WINDOW_IMU),
            (LocomotionMMGCNNWindow(**WINDOW_MMG), WINDOW_MMG),
        )
        self.paths = {}
        for name, parents in (("single", single), ("window", windowed)):
            self.paths[name] = (
                _save_backbone(root / f"{name}_imu.pt", *parents[0]),
                _save_backbone(root / f"{name}_mmg.pt", *parents[1]),
            )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    @staticmethod
    def _loader(windowed: bool) -> DataLoader:
        extra = (4,) if windowed else ()
        imu = torch.randn(8, 6, 125, *extra)
        mmg = torch.randn(8, 5, 40, 125, *extra)
        labels = torch.randint(0, 7, (8,))
        if not windowed:
            return DataLoader(TensorDataset(imu, mmg, labels), batch_size=4)

        class Pairs(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return len(labels)

            def __getitem__(self, i):
                return (imu[i], mmg[i]), labels[i]

        return DataLoader(Pairs(), batch_size=4)

    def _tuner(self, module, tuner_name, windowed):
        imu, mmg = self.paths["window" if windowed else "single"]
        return getattr(module, tuner_name)(
            self._loader(windowed), self._loader(windowed), imu, mmg,
            search_space={"device": "cpu", "epochs": 1},
        )

    def test_trials_use_new_parameters_and_best_model_builds(self) -> None:
        for module, tuner_name, _, windowed, searched in TUNERS:
            with self.subTest(tuner=tuner_name):
                tuner = self._tuner(module, tuner_name, windowed)
                study = optuna.create_study(direction="maximize")
                study.optimize(tuner._objective, n_trials=2)
                params = study.best_trial.params
                self.assertIn(params["batch_size"], BATCHES)
                for key, expected in searched.items():
                    if isinstance(expected, tuple):
                        self.assertTrue(expected[0] <= params[key] <= expected[1])
                    else:
                        self.assertIn(params[key], expected)
                tuner._best_params = params
                self.assertIsNotNone(tuner._build_best_model())

    def test_out_of_memory_prunes_trial_instead_of_aborting(self) -> None:
        for module, tuner_name, trainer_name, windowed, _ in TUNERS:
            with self.subTest(tuner=tuner_name):
                tuner = self._tuner(module, tuner_name, windowed)
                study = optuna.create_study(direction="maximize")
                oom = torch.cuda.OutOfMemoryError("simulated")
                with mock.patch.object(
                    getattr(module, trainer_name), "fit", side_effect=oom
                ):
                    study.optimize(tuner._objective, n_trials=1)
                trial = study.trials[0]
                self.assertEqual(trial.state, optuna.trial.TrialState.PRUNED)
                self.assertTrue(trial.user_attrs["out_of_memory"])


if __name__ == "__main__":
    unittest.main()
