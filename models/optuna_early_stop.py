"""Optuna callback that ends a study once the best value stops improving."""

from __future__ import annotations

import optuna
from optuna.trial import TrialState


class PlateauStop:
    """Stop after ``patience`` consecutive trials without a ``min_delta`` improvement.

    Complete and pruned trials count; trials tagged with ``skip_attr`` (rejected
    before any training) do not. History is rescanned each call, so resumed
    studies are judged on their earlier trials too.
    """

    def __init__(self, patience: int, min_delta: float = 1e-3, skip_attr: str | None = None):
        if patience < 1:
            raise ValueError("patience must be at least 1.")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative.")
        self.patience = patience
        self.min_delta = min_delta
        self.skip_attr = skip_attr
        self.triggered = False

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        sign = 1.0 if study.direction == optuna.study.StudyDirection.MAXIMIZE else -1.0
        best: float | None = None
        since_best = 0
        for past in study.get_trials(
            deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED)
        ):
            if self.skip_attr and past.user_attrs.get(self.skip_attr):
                continue
            if (
                past.state == TrialState.COMPLETE
                and past.value is not None
                and (best is None or sign * past.value > best + self.min_delta)
            ):
                best = sign * past.value
                since_best = 0
            else:
                since_best += 1
        if since_best >= self.patience:
            self.triggered = True
            study.stop()
