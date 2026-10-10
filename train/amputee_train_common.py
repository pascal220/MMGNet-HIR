"""Shared training sequence for the end-to-end amputee fusion models.

Each amputee data type is an independent dataset. For one prepared type, the
fusion models are tuned and refit one after another with the standard
``run_training_experiment`` protocol (grouped validation holdout, Optuna
search, full-development refit, run manifest). No parent checkpoint is used:
the backbones are trained from scratch inside every fusion model.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from data_loader import PreparedData
from split_utils import validate_prepared_data
from training_experiment import TrainingRunConfig, run_training_experiment

DEFAULT_CHECKPOINT_DIR = "checkpoints"


@dataclass(frozen=True)
class AmputeeModel:
    """One fusion model to train: result name, run key, tuner and trainer."""

    name: str
    model_key: str
    tuner_cls: Any
    trainer_cls: Any


def train_amputee_type(
    prepared: PreparedData,
    models: Sequence[AmputeeModel],
    input_mode: str,
    *,
    n_trials: int = 50,
    timeout: int | None = None,
    artifact_root: str = "results/training",
    run_label: str | None = None,
    checkpoint_dir: str | None = DEFAULT_CHECKPOINT_DIR,
    show_progress: bool = True,
    device: str = "auto",
) -> dict[str, Any]:
    """Tune and refit ``models`` in order on one prepared amputee data type.

    ``n_trials`` is the Optuna trial budget of each model. A copy of every
    final checkpoint is written to ``checkpoint_dir`` with the amputee and
    data type in its name. Test tensors are never read.
    """
    validate_prepared_data(prepared, input_mode, "fusion")
    config = prepared.experiment.config
    if config.setup != "amputee":
        raise ValueError("Amputee training expects data from prepare_amputee_experiment_data.")
    num_classes = prepared.experiment.num_classes

    results: dict[str, Any] = {}
    for model in models:
        alias = (
            None if checkpoint_dir is None
            else f"{checkpoint_dir}/best_{model.model_key}_{config.amputee_id}_{config.data_type}.pt"
        )
        results[model.name] = run_training_experiment(
            prepared=prepared,
            model_key=model.model_key,
            input_tensors=(prepared.X_imu_train, prepared.X_cwt_train),
            tuner_factory=lambda train, val, search, tuner_cls=model.tuner_cls: tuner_cls(
                train, val, num_classes=num_classes, search_space=search,
            ),
            trainer_factory=model.trainer_cls,
            config=TrainingRunConfig(
                n_trials=n_trials,
                timeout=timeout,
                artifact_root=artifact_root,
                run_label=run_label,
                seed=config.seed,
                show_progress=show_progress,
                device=device,
            ),
            legacy_checkpoint_path=alias,
            nested_inputs=input_mode == "windowed",
            num_classes=num_classes,
        )
    return results
