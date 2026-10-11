from __future__ import annotations

import sys
import unittest
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "models")]

from optuna_early_stop import PlateauStop

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _run(values: list[float], **stop_kwargs) -> tuple[int, PlateauStop]:
    values = iter(values)
    stop = PlateauStop(**stop_kwargs)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: next(values), n_trials=50, callbacks=[stop])
    return len(study.trials), stop


class PlateauStopTests(unittest.TestCase):
    def test_stops_after_patience_without_improvement(self) -> None:
        count, stop = _run([0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.9] + [0.1] * 43, patience=3)
        self.assertEqual(count, 5)
        self.assertTrue(stop.triggered)

    def test_gain_below_min_delta_does_not_reset(self) -> None:
        count, _ = _run([0.5, 0.5005, 0.501, 0.9] + [0.1] * 46, patience=2, min_delta=1e-2)
        self.assertEqual(count, 3)

    def test_steady_improvement_runs_full_budget(self) -> None:
        count, stop = _run([0.01 * i for i in range(1, 51)], patience=3)
        self.assertEqual(count, 50)
        self.assertFalse(stop.triggered)

    def test_skip_attr_trials_are_not_counted(self) -> None:
        stop = PlateauStop(patience=2, skip_attr="skip")
        study = optuna.create_study(direction="maximize")

        def objective(trial: optuna.Trial) -> float:
            if trial.number > 0:
                trial.set_user_attr("skip", True)
            return 0.5

        study.optimize(objective, n_trials=6, callbacks=[stop])
        self.assertFalse(stop.triggered)

    def test_rejects_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            PlateauStop(patience=0)
        with self.assertRaises(ValueError):
            PlateauStop(patience=1, min_delta=-1)


if __name__ == "__main__":
    unittest.main()
