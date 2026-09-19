"""Small reusable early-stopping helper for trainer fit loops."""

from __future__ import annotations

from typing import Any, Mapping

import torch


class EarlyStopping:
    """Track a monotonically decreasing metric and optionally restore best weights.

    The helper is intentionally opt-in: if ``early_stopping_patience`` is absent
    or ``None`` in a trainer config, calls become no-ops. This keeps Optuna trial
    training unchanged unless the caller explicitly enables early stopping.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        enabled: bool,
        monitor: str = "train_loss",
        patience: int = 10,
        min_delta: float = 1e-4,
        restore_best_weights: bool = True,
    ) -> None:
        self.model = model
        self.enabled = enabled
        self.monitor = monitor
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.restore_best_weights = bool(restore_best_weights)
        self.best_value: float | None = None
        self.best_epoch: int | None = None
        self.stopped_epoch: int | None = None
        self.wait_count = 0
        self._best_state: dict[str, torch.Tensor] | None = None
        self._restored = False

    @classmethod
    def from_config(cls, model: torch.nn.Module, cfg: Mapping[str, Any]) -> "EarlyStopping":
        patience = cfg.get("early_stopping_patience")
        enabled = patience is not None
        patience_value = 10 if patience is None else int(patience)
        if enabled and patience_value < 1:
            raise ValueError("early_stopping_patience must be at least 1 or None.")
        return cls(
            model,
            enabled=enabled,
            monitor=str(cfg.get("early_stopping_monitor", "train_loss")),
            patience=patience_value,
            min_delta=float(cfg.get("early_stopping_min_delta", 1e-4)),
            restore_best_weights=bool(cfg.get("restore_best_weights", True)),
        )

    def update(self, epoch: int, metrics: Mapping[str, float | None]) -> bool:
        """Record a new epoch and return True when training should stop."""
        if not self.enabled:
            return False
        value = metrics.get(self.monitor)
        if value is None:
            raise ValueError(
                f"Early stopping monitor '{self.monitor}' is unavailable for this training run."
            )
        value = float(value)
        if self.best_value is None or value < self.best_value - self.min_delta:
            self.best_value = value
            self.best_epoch = int(epoch)
            self.wait_count = 0
            if self.restore_best_weights:
                self._best_state = {
                    key: tensor.detach().cpu().clone()
                    for key, tensor in self.model.state_dict().items()
                }
        else:
            self.wait_count += 1

        if self.wait_count >= self.patience:
            self.stopped_epoch = int(epoch)
            return True
        return False

    def finalize(self) -> None:
        """Restore the best recorded weights, if configured."""
        if (
            self.enabled
            and self.restore_best_weights
            and self._best_state is not None
        ):
            self.model.load_state_dict(self._best_state)
            self._restored = True

    def summary(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "monitor": self.monitor,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_epoch": self.best_epoch,
            "best_value": self.best_value,
            "stopped_epoch": self.stopped_epoch,
            "epochs_without_improvement": self.wait_count,
            "restored_best_weights": self._restored,
        }
