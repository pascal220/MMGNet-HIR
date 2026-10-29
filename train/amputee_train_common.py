"""Shared training sequence for the end-to-end amputee fusion models.

Each amputee data type is an independent dataset. For one prepared type, the
fusion models are tuned and refit one after another with the standard
``run_training_experiment`` protocol (grouped validation holdout, Optuna
search, full-development refit, run manifest). No parent checkpoint is used:
the backbones are trained from scratch inside every fusion model.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from binary_bundle import bundle_path, save_bundle
from binary_tasks import BINARY_TASKS, task_model_key, task_prepared
from data_loader import PreparedData
from device_utils import release_cuda_memory
from split_utils import validate_prepared_data
from training_experiment import TrainingRunConfig, run_training_experiment

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR = "checkpoints"


@dataclass(frozen=True)
class AmputeeModel:
    """One fusion model to train: result name, run key, tuner and trainer."""

    name: str
    model_key: str
    tuner_cls: Any
    trainer_cls: Any


def _run_config(
    seed: int,
    *,
    n_trials: int,
    early_stop_patience: int | None,
    timeout: int | None,
    artifact_root: str,
    run_label: str | None,
    show_progress: bool,
    device: str,
) -> TrainingRunConfig:
    return TrainingRunConfig(
        n_trials=n_trials,
        early_stop_patience=early_stop_patience,
        timeout=timeout,
        artifact_root=artifact_root,
        run_label=run_label,
        seed=seed,
        show_progress=show_progress,
        device=device,
    )


def train_amputee_type(
    prepared: PreparedData,
    models: Sequence[AmputeeModel],
    input_mode: str,
    *,
    n_trials: int = 100,
    early_stop_patience: int | None = 20,
    timeout: int | None = None,
    artifact_root: str = "results/training",
    run_label: str | None = None,
    checkpoint_dir: str | None = DEFAULT_CHECKPOINT_DIR,
    show_progress: bool = True,
    device: str = "auto",
) -> dict[str, Any]:
    """Tune and refit ``models`` in order on one prepared amputee data type.

    ``n_trials`` is the Optuna trial budget of each model; the search ends early
    after ``early_stop_patience`` trials without improvement (None disables it).
    A copy of every
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
            config=_run_config(
                config.seed,
                n_trials=n_trials, early_stop_patience=early_stop_patience, timeout=timeout,
                artifact_root=artifact_root, run_label=run_label,
                show_progress=show_progress, device=device,
            ),
            legacy_checkpoint_path=alias,
            nested_inputs=input_mode == "windowed",
            num_classes=num_classes,
        )
    return results


def train_amputee_type_binary(
    prepared: PreparedData,
    models: Sequence[AmputeeModel],
    input_mode: str,
    *,
    n_trials: int = 100,
    early_stop_patience: int | None = 20,
    timeout: int | None = None,
    artifact_root: str = "results/training",
    run_label: str | None = None,
    checkpoint_dir: str | None = DEFAULT_CHECKPOINT_DIR,
    show_progress: bool = True,
    device: str = "auto",
) -> dict[str, Any]:
    """Tune and refit every binary task for each of ``models`` on one data type.

    Each task of ``BINARY_TASKS`` is a separate experiment on the rows of its
    classes (own validation holdout, Optuna search of ``n_trials`` and refit)
    and keeps its run folder under ``artifact_root``. The final models of one
    architecture are then written to a single bundle file in ``checkpoint_dir``
    (no bundle when it is None). Test tensors are never read.
    """
    validate_prepared_data(prepared, input_mode, "fusion")
    config = prepared.experiment.config
    if config.setup != "amputee":
        raise ValueError("Amputee training expects data from prepare_amputee_experiment_data.")
    run_config = _run_config(
        config.seed,
        n_trials=n_trials, early_stop_patience=early_stop_patience, timeout=timeout,
        artifact_root=artifact_root, run_label=run_label,
        show_progress=show_progress, device=device,
    )

    results: dict[str, Any] = {}
    for model in models:
        runs: dict[str, Any] = {}
        for task in BINARY_TASKS:
            logger.info(
                "Amputee %s, %s: training %s task %s (%s).",
                config.amputee_id, config.data_type, model.name, task.name,
                ", ".join(task.class_names),
            )
            task_data = task_prepared(prepared, task)
            runs[task.name] = run_training_experiment(
                prepared=task_data,
                model_key=task_model_key(model.model_key, task),
                input_tensors=(task_data.X_imu_train, task_data.X_cwt_train),
                tuner_factory=lambda train, val, search, tuner_cls=model.tuner_cls, k=task.num_classes: (
                    tuner_cls(train, val, num_classes=k, search_space=search)
                ),
                trainer_factory=model.trainer_cls,
                config=run_config,
                nested_inputs=input_mode == "windowed",
                num_classes=task.num_classes,
            )
            del task_data
            release_cuda_memory()
        bundle = None
        if checkpoint_dir is not None:
            bundle = str(save_bundle(
                bundle_path(checkpoint_dir, model.model_key, config.amputee_id, config.data_type),
                model_key=model.model_key, input_mode=input_mode,
                amputee_id=config.amputee_id, data_type=config.data_type, runs=runs,
            ))
            logger.info("Binary-classifier bundle written to %s", bundle)
        results[model.name] = {"tasks": runs, "bundle_path": bundle}
    return results
