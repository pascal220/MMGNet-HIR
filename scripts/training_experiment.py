"""Shared, test-sealed Optuna training and artifact management.

The public train entry points supply model-specific tuner/trainer factories. This
module owns the reproducible experiment lifecycle and intentionally never reads
``PreparedData`` test tensors: test evaluation belongs to the evaluation stage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
)
from torch.utils.data import DataLoader, Dataset, TensorDataset

from data_loader import PreparedData
from dataset_registry import LABEL_TO_CLASS
from device_utils import describe_device, device_details, normalize_device_request, resolve_device
from split_utils import GROUP_COLUMNS, split_train_validation

NUM_CLASSES = 7
OBJECTIVE_NAME = "0.5 * validation_accuracy + 0.5 * validation_macro_f1"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingRunConfig:
    """Settings shared by every model-specific training entry point."""

    n_trials: int = 100
    timeout: int | None = None
    artifact_root: str = "results/training"
    run_label: str | None = None
    resume_run_id: str | None = None
    val_fraction: float = 0.10
    seed: int = 42
    show_progress: bool = True
    device: str = "auto"
    final_refit_epochs: int | None = 100
    final_refit_early_stopping_patience: int | None = 10
    final_refit_early_stopping_min_delta: float = 1e-4
    final_refit_restore_best_weights: bool = True
    descriptive_checkpoint_alias: bool = False

    def validate(self) -> None:
        if self.n_trials < 1:
            raise ValueError("n_trials must be at least 1.")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be positive or None.")
        if not 0 < self.val_fraction < 1:
            raise ValueError("val_fraction must be between 0 and 1.")
        if self.final_refit_epochs is not None and self.final_refit_epochs < 1:
            raise ValueError("final_refit_epochs must be at least 1 or None.")
        if (
            self.final_refit_early_stopping_patience is not None
            and self.final_refit_early_stopping_patience < 1
        ):
            raise ValueError(
                "final_refit_early_stopping_patience must be at least 1 or None."
            )
        if self.final_refit_early_stopping_min_delta < 0:
            raise ValueError("final_refit_early_stopping_min_delta must be non-negative.")
        normalize_device_request(self.device)


@dataclass(frozen=True)
class RunArtifacts:
    """Paths belonging to one independently reproducible model run."""

    run_id: str
    run_dir: Path
    checkpoint: Path
    study_journal: Path
    trials_csv: Path
    best_trial_json: Path
    history_json: Path
    manifest_json: Path
    plots_dir: Path


TunerFactory = Callable[[DataLoader, DataLoader, dict[str, Any]], Any]
TrainerFactory = Callable[..., Any]




@runtime_checkable
class PreparedDataLike(Protocol):
    """Structural shape ``run_training_experiment`` relies on from ``PreparedData``."""

    @property
    def experiment(self) -> Any: ...

    @property
    def input_mode(self) -> Any: ...

    @property
    def model_target(self) -> Any: ...

    @property
    def y_train(self) -> torch.Tensor: ...

    @property
    def train_metadata(self) -> pd.DataFrame: ...


class _NestedPairDataset(Dataset):
    """Yield ``((first_input, second_input), label)`` for windowed fusion."""

    def __init__(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        self.first = first
        self.second = second
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (self.first[index], self.second[index]), self.labels[index]


def combined_metric(accuracy: float, macro_f1: float) -> float:
    """Return the scalar objective used consistently throughout selection."""
    return 0.5 * float(accuracy) + 0.5 * float(macro_f1)


def best_epoch_metrics(history: Mapping[str, Sequence[float]]) -> dict[str, float | int]:
    """Select accuracy and F1 from the same best validation epoch."""
    accuracies = list(history.get("val_acc", []))
    f1_scores = list(history.get("val_f1", []))
    if not accuracies or len(accuracies) != len(f1_scores):
        raise ValueError("Validation accuracy and F1 histories must be non-empty and aligned.")
    scores = [combined_metric(acc, f1) for acc, f1 in zip(accuracies, f1_scores)]
    index = int(np.argmax(scores))
    return {
        "best_epoch": index + 1,
        "validation_accuracy": float(accuracies[index]),
        "validation_macro_f1": float(f1_scores[index]),
        "objective_value": float(scores[index]),
    }


def balanced_class_weights(labels: torch.Tensor, num_classes: int = NUM_CLASSES) -> list[float]:
    """Compute balanced inverse-frequency weights from training labels only."""
    values = labels.detach().cpu().to(dtype=torch.long)
    counts = torch.bincount(values, minlength=num_classes).to(dtype=torch.float64)
    if len(counts) != num_classes or torch.any(counts == 0):
        missing = torch.where(counts == 0)[0].tolist()
        raise ValueError(f"Cannot compute class weights; missing training classes: {missing}.")
    weights = len(values) / (num_classes * counts)
    return [float(value) for value in weights.tolist()]


def normalize_training_params(
    raw_params: dict[str, Any],
    *,
    best_epoch: int,
    class_weights: Sequence[float],
    fixed_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate Optuna parameter names into the trainer's canonical schema."""
    params = dict(raw_params)
    optimizer = params["optimizer"]
    normalized: dict[str, Any] = {
        "optimizer": optimizer,
        "batch_size": int(params["batch_size"]),
        "epochs": int(best_epoch),
        "class_weights": [float(value) for value in class_weights],
    }
    if optimizer == "SGD":
        normalized.update(
            lr=float(params["sgd_lr"]),
            momentum=float(params["sgd_momentum"]),
            weight_decay=float(params["sgd_wd"]),
        )
    elif optimizer == "Adam":
        normalized.update(
            lr=float(params["adam_lr"]),
            weight_decay=float(params["adam_wd"]),
            beta1=float(params["adam_beta1"]),
            beta2=float(params["adam_beta2"]),
        )
    else:
        raise ValueError(f"Unsupported optimizer in best trial: {optimizer!r}.")

    fixed = fixed_params or {}
    for key in ("lr_step_size", "lr_decay", "lr_factor", "lr_patience", "lr_min"):
        if key in params:
            normalized[key] = params[key]
        elif key in fixed and not isinstance(fixed[key], (list, tuple, dict)):
            normalized[key] = fixed[key]
    return normalized


def _slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-_")
    return value or "run"


def create_run_artifacts(
    model_key: str,
    prepared: PreparedDataLike,
    config: TrainingRunConfig,
) -> RunArtifacts:
    """Create or reopen the directory for one model run."""
    root = Path(config.artifact_root)
    if config.resume_run_id:
        run_id = _slug(config.resume_run_id)
        run_dir = root / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Cannot resume missing run directory: {run_dir}")
    else:
        experiment = prepared.experiment.config
        if experiment.setup == "same_volunteer":
            volunteer = _slug(str(experiment.same_volunteer_id))
            data_tag = f"same-{volunteer}"
        else:
            data_tag = f"separate-v{experiment.train_volunteer_count}"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        parts = [model_key, prepared.input_mode, data_tag, timestamp]
        if config.run_label:
            parts.append(_slug(config.run_label))
        run_id = "__".join(_slug(part) for part in parts)
        run_dir = root / run_id
        suffix = 1
        while run_dir.exists():
            run_dir = root / f"{run_id}-{suffix}"
            suffix += 1
        run_id = run_dir.name
        run_dir.mkdir(parents=True)

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        checkpoint=run_dir / f"{run_id}.pt",
        study_journal=run_dir / "study.journal",
        trials_csv=run_dir / "trials.csv",
        best_trial_json=run_dir / "best_trial.json",
        history_json=run_dir / "training_history.json",
        manifest_json=run_dir / "manifest.json",
        plots_dir=plots_dir,
    )


def journal_storage(path: Path) -> JournalStorage:
    """Return file-backed Optuna journal storage with no database dependency."""
    return JournalStorage(JournalFileBackend(str(path)))


def export_study(study: optuna.Study, artifacts: RunArtifacts) -> dict[str, Any]:
    """Export all trials to CSV and best-effort interactive HTML plots."""
    study.trials_dataframe().to_csv(artifacts.trials_csv, index=False)
    plotters = {
        "optimization_history.html": plot_optimization_history,
        "parameter_importances.html": plot_param_importances,
        "parallel_coordinate.html": plot_parallel_coordinate,
        "slice.html": plot_slice,
    }
    generated: list[str] = []
    skipped: dict[str, str] = {}
    for filename, plotter in plotters.items():
        try:
            plotter(study).write_html(artifacts.plots_dir / filename)
            generated.append(str(Path("plots") / filename))
        except (ValueError, RuntimeError) as exc:
            skipped[filename] = str(exc)
    return {"generated": generated, "skipped": skipped}


def _close_study_storage(study: optuna.Study) -> None:
    """Release Optuna's storage session after a completed training run."""
    storage = study._storage
    remove_session = getattr(storage, "remove_session", None)
    if callable(remove_session):
        remove_session()
    backend = getattr(storage, "_backend", storage)
    if backend is not storage:
        remove_session = getattr(backend, "remove_session", None)
        if callable(remove_session):
            remove_session()
    engine = getattr(backend, "engine", None)
    if engine is not None:
        engine.dispose()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parent_checkpoint_record(path: Path, artifact_root: Path) -> dict[str, Any]:
    """Describe a fusion parent and link it to its run manifest when available."""
    checkpoint_hash = _sha256(path)
    record: dict[str, Any] = {"path": str(path), "sha256": checkpoint_hash}
    candidates = [path.parent / "manifest.json"]
    if artifact_root.is_dir():
        candidates.extend(artifact_root.glob("*/manifest.json"))
    for manifest_path in candidates:
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        artifact = manifest.get("artifacts", {})
        aliases = {str(alias) for alias in artifact.get("checkpoint_aliases", [])}
        if (
            artifact.get("checkpoint_sha256") == checkpoint_hash
            or str(path) in aliases
        ):
            record.update(
                run_id=manifest.get("run_id"),
                manifest_path=str(manifest_path),
                model_key=manifest.get("model", {}).get("key"),
            )
            break
    return record


def _metadata_fingerprint(metadata: pd.DataFrame) -> str:
    normalized = metadata.reset_index(drop=True).astype(str)
    return hashlib.sha256(normalized.to_csv(index=False).encode("utf-8")).hexdigest()


def _split_fingerprint(metadata: pd.DataFrame, indices: np.ndarray) -> str:
    selected = metadata.iloc[indices][GROUP_COLUMNS].reset_index(drop=True).astype(str)
    return hashlib.sha256(selected.to_csv(index=False).encode("utf-8")).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_info() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
            capture_output=True, text=True, check=True,
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True), encoding="utf-8")


def _class_distribution(labels: torch.Tensor, num_classes: int) -> dict[str, int]:
    counts = torch.bincount(labels.detach().cpu().long(), minlength=num_classes)
    return {str(index): int(count) for index, count in enumerate(counts.tolist())}


def _final_training_accuracy(history: Mapping[str, Sequence[float] | Mapping[str, Any]]) -> float:
    """Return the training accuracy corresponding to the saved final weights."""
    accuracies = list(history.get("train_acc", []))
    if not accuracies:
        raise ValueError("Final training history must include train_acc.")
    early_stopping = history.get("early_stopping")
    if isinstance(early_stopping, Mapping) and early_stopping.get("restored_best_weights"):
        best_epoch = early_stopping.get("best_epoch")
        if isinstance(best_epoch, int) and 1 <= best_epoch <= len(accuracies):
            return float(accuracies[best_epoch - 1])
    return float(accuracies[-1])


def _volunteers(metadata: pd.DataFrame) -> list[str]:
    columns = [column for column in metadata.columns if "volunteer" in str(column).lower()]
    if not columns:
        return []
    return sorted(metadata[columns[0]].dropna().astype(str).unique().tolist())


def _descriptive_checkpoint_alias(
    path: Path,
    experiment: Any,
    saved_at: datetime,
    final_accuracy: float,
) -> Path:
    """Append date, training-volunteer tag, and final accuracy to an alias."""
    if experiment.setup == "same_volunteer":
        volunteer_tag = f"V{experiment.same_volunteer_id}"
    else:
        volunteer_tag = f"V{experiment.train_volunteer_count}"
    accuracy_tag = f"A{final_accuracy * 100:.2f}"
    return path.with_name(
        f"{path.stem}_{saved_at.strftime('%Y-%m-%d')}_{volunteer_tag}_{accuracy_tag}"
        f"{path.suffix}"
    )


def _make_dataset(
    input_tensors: Sequence[torch.Tensor],
    labels: torch.Tensor,
    nested_inputs: bool,
) -> Dataset:
    if nested_inputs:
        if len(input_tensors) != 2:
            raise ValueError("Nested model input requires exactly two tensors.")
        return _NestedPairDataset(input_tensors[0], input_tensors[1], labels)
    return TensorDataset(*input_tensors, labels)


def run_training_experiment(
    *,
    prepared: PreparedDataLike,
    model_key: str,
    input_tensors: Sequence[torch.Tensor],
    tuner_factory: TunerFactory,
    trainer_factory: TrainerFactory,
    config: TrainingRunConfig | None = None,
    legacy_checkpoint_path: str | None = None,
    parent_checkpoints: Sequence[str] = (),
    nested_inputs: bool = False,
    initial_batch_size: int | None = None,
    num_classes: int = NUM_CLASSES,
) -> dict[str, Any]:
    """Tune and refit one model without accessing the reserved test tensors."""
    run_config = config or TrainingRunConfig(seed=prepared.experiment.config.seed)
    run_config.validate()
    device = resolve_device(run_config.device)
    runtime_device = device_details(run_config.device, device)
    logger.info("Compute device: %s", describe_device(runtime_device))
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    if any(len(tensor) != len(prepared.y_train) for tensor in input_tensors):
        raise ValueError("All model inputs must be row-aligned with y_train.")
    parent_artifacts = []
    for raw_path in parent_checkpoints:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Required parent checkpoint does not exist: {path}")
        parent_artifacts.append(
            _parent_checkpoint_record(path, Path(run_config.artifact_root))
        )

    seed_everything(run_config.seed)
    artifacts = create_run_artifacts(model_key, prepared, run_config)
    train_idx, val_idx = split_train_validation(
        prepared.y_train,
        prepared.train_metadata,
        val_fraction=run_config.val_fraction,
        seed=run_config.seed,
    )
    search_weights = balanced_class_weights(prepared.y_train[train_idx], num_classes)

    train_dataset = _make_dataset(
        [tensor[train_idx] for tensor in input_tensors],
        prepared.y_train[train_idx],
        nested_inputs,
    )
    val_dataset = _make_dataset(
        [tensor[val_idx] for tensor in input_tensors],
        prepared.y_train[val_idx],
        nested_inputs,
    )
    loader_batch_size = initial_batch_size or prepared.experiment.config.batch_size
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(run_config.seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=loader_batch_size,
        num_workers=0,
        pin_memory=pin_memory,
    )

    tuner = tuner_factory(
        train_loader,
        val_loader,
        {
            "class_weights": search_weights,
            "seed": run_config.seed,
            "device": str(device),
        },
    )
    storage = journal_storage(artifacts.study_journal)
    tuner.run(
        n_trials=run_config.n_trials,
        timeout=run_config.timeout,
        show_progress=run_config.show_progress,
        storage=storage,
        study_name=artifacts.run_id,
        load_if_exists=bool(run_config.resume_run_id),
    )
    study: optuna.Study = tuner.study
    if study is None:
        raise RuntimeError("The tuner completed without exposing its Optuna study.")

    plots = export_study(study, artifacts)
    best_trial = study.best_trial
    if best_trial.value is None:
        raise RuntimeError("Best trial is missing its single-objective value.")
    selection = {
        "best_trial_number": int(best_trial.number),
        "best_epoch": int(best_trial.user_attrs["best_epoch"]),
        "validation_accuracy": float(best_trial.user_attrs["validation_accuracy"]),
        "validation_macro_f1": float(best_trial.user_attrs["validation_macro_f1"]),
        "objective_value": float(best_trial.value),
    }

    best_trial_summary = {
        "model_key": model_key,
        "study_name": study.study_name,
        "best_trial_number": selection["best_trial_number"],
        "objective": OBJECTIVE_NAME,
        "objective_value": selection["objective_value"],
        "best_epoch": selection["best_epoch"],
        "validation_accuracy": selection["validation_accuracy"],
        "validation_macro_f1": selection["validation_macro_f1"],
        "best_params": dict(best_trial.params),
    }
    _write_json(artifacts.best_trial_json, best_trial_summary)

    final_weights = balanced_class_weights(prepared.y_train, num_classes)
    final_refit_epochs = run_config.final_refit_epochs or 100
    training_params = normalize_training_params(
        best_trial.params,
        best_epoch=final_refit_epochs,
        class_weights=final_weights,
        fixed_params=tuner.search,
    )
    training_params["device"] = str(device)
    if run_config.final_refit_early_stopping_patience is not None:
        training_params["early_stopping_patience"] = int(
            run_config.final_refit_early_stopping_patience
        )
        training_params["early_stopping_monitor"] = "train_loss"
        training_params["early_stopping_min_delta"] = float(
            run_config.final_refit_early_stopping_min_delta
        )
        training_params["restore_best_weights"] = bool(
            run_config.final_refit_restore_best_weights
        )
    final_dataset = _make_dataset(input_tensors, prepared.y_train, nested_inputs)
    final_loader = DataLoader(
        final_dataset,
        batch_size=int(training_params["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(run_config.seed),
    )
    seed_everything(run_config.seed)
    best_model = tuner._build_best_model()
    trainer = trainer_factory(best_model, training_params)
    history = trainer.fit(
        final_loader,
        val_loader=None,
        epochs=int(training_params["epochs"]),
        verbose=True,
    )
    trainer.save(str(artifacts.checkpoint))
    _write_json(artifacts.history_json, history)

    aliases: list[str] = []
    saved_at = datetime.now(timezone.utc)
    final_training_accuracy = _final_training_accuracy(history)
    if legacy_checkpoint_path:
        alias = Path(legacy_checkpoint_path)
        if run_config.descriptive_checkpoint_alias:
            alias = _descriptive_checkpoint_alias(
                alias,
                prepared.experiment.config,
                saved_at,
                final_training_accuracy,
            )
        if alias.resolve() != artifacts.checkpoint.resolve():
            alias.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifacts.checkpoint, alias)
            aliases.append(str(alias))

    experiment_config = asdict(prepared.experiment.config)
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "run_id": artifacts.run_id,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.perf_counter() - started_clock,
        "model": {
            "key": model_key,
            "class": type(best_model).__name__,
            "config": getattr(best_model, "config", {}),
            "input_mode": prepared.input_mode,
            "model_target": prepared.model_target,
            "input_shapes": [list(tensor.shape[1:]) for tensor in input_tensors],
            "num_classes": num_classes,
            "label_mapping": LABEL_TO_CLASS,
            "total_parameters": sum(parameter.numel() for parameter in best_model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in best_model.parameters() if parameter.requires_grad
            ),
            "parent_checkpoints": parent_artifacts,
        },
        "data": {
            "experiment_config": experiment_config,
            "development_rows": len(prepared.y_train),
            "development_volunteers": _volunteers(prepared.train_metadata),
            "class_distribution": _class_distribution(prepared.y_train, num_classes),
            "metadata_fingerprint_sha256": _metadata_fingerprint(prepared.train_metadata),
            "validation_protocol": "single grouped stratified holdout",
            "validation_fraction": run_config.val_fraction,
            "group_columns": GROUP_COLUMNS,
            "search_train_rows": len(train_idx),
            "validation_rows": len(val_idx),
            "search_train_fingerprint_sha256": _split_fingerprint(prepared.train_metadata, train_idx),
            "validation_fingerprint_sha256": _split_fingerprint(prepared.train_metadata, val_idx),
            "test_set_accessed": False,
        },
        "optimization": {
            "study_name": study.study_name,
            "direction": str(study.direction.name).lower(),
            "objective": OBJECTIVE_NAME,
            "requested_additional_trials": run_config.n_trials,
            "completed_trials_total": len(study.trials),
            "timeout_seconds": run_config.timeout,
            "sampler": f"TPESampler(seed={run_config.seed})",
            "pruner": "MedianPruner(n_startup_trials=5,n_warmup_steps=10,interval_steps=1)",
            "search_space": {
                key: value
                for key, value in tuner.search.items()
                if key not in {"class_weights", "device", "seed"}
            },
            "raw_best_params": best_trial.params,
            "selection": selection,
            "search_class_weights": search_weights,
        },
        "final_refit": {
            "policy": "all non-test development data for up to configured final epoch count with train-loss early stopping",
            "epochs": int(training_params["epochs"]),
            "actual_epochs": len(history.get("train_loss", [])),
            "final_training_accuracy": final_training_accuracy,
            "training_params": training_params,
            "early_stopping": history.get("early_stopping", {"enabled": False}),
            "class_weights": final_weights,
            "history_path": artifacts.history_json.name,
        },
        "artifacts": {
            "checkpoint": artifacts.checkpoint.name,
            "checkpoint_sha256": _sha256(artifacts.checkpoint),
            "checkpoint_aliases": aliases,
            "checkpoint_alias_saved_at_utc": saved_at.isoformat(),
            "study_journal": artifacts.study_journal.name,
            "trials_csv": artifacts.trials_csv.name,
            "best_trial_json": artifacts.best_trial_json.name,
            "plots": plots,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "compute_device": runtime_device,
            "optuna": _package_version("optuna"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": _package_version("scikit-learn"),
            "plotly": _package_version("plotly"),
            "git": _git_info(),
            "seed": run_config.seed,
        },
    }
    _write_json(artifacts.manifest_json, manifest)

    _close_study_storage(study)

    return {
        "run_id": artifacts.run_id,
        "artifact_dir": str(artifacts.run_dir),
        "checkpoint_path": str(artifacts.checkpoint),
        "manifest_path": str(artifacts.manifest_json),
        "best_trial_json_path": str(artifacts.best_trial_json),
        "study_path": str(artifacts.study_journal),
        "trials_csv_path": str(artifacts.trials_csv),
        "history": history,
        "selection": selection,
        "best_params": dict(best_trial.params),
        "training_params": training_params,
    }


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable model construction."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
