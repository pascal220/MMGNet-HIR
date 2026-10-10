"""Shared test-set evaluation for the four model families.

Each evaluation entry point supplies the models to compare; this module selects
their latest verified runs, predicts the sealed test split, and writes one
row-normalised confusion matrix per model plus a combined bar chart of accuracy
per transition marker.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from data_loader import PreparedData
from dataset_registry import (
    LABEL_TO_CLASS,
    DatasetRegistry,
    RegistryColumns,
    normalize_amputee_id,
    normalize_data_type,
)
from device_utils import resolve_device
from run_selection import (
    DEFAULT_ARTIFACT_ROOT,
    ModelNotAvailableError,
    TrainedRun,
    select_trained_run,
)
from training_experiment import file_sha256

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT = "results/evaluation"
TRANSITION_ORDER: tuple[str, ...] = ("100m", "50m", "0", "50", "100")
CLASS_NAMES = [LABEL_TO_CLASS[label] for label in sorted(LABEL_TO_CLASS)]

# Receives the checkpoint's model_config and returns an untrained model of that
# architecture; every checkpoint is self-contained, so no other file is needed.
ModelBuilder = Callable[[dict[str, Any]], torch.nn.Module]


@dataclass(frozen=True)
class ModelSpec:
    """One model to evaluate: display label, run key, builder and test inputs."""

    label: str
    model_key: str
    build: ModelBuilder
    inputs: tuple[torch.Tensor, ...]


def _load_model(
    spec: ModelSpec,
    run: TrainedRun,
    prepared: PreparedData,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(run.checkpoint, map_location=device)
    model = spec.build(dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    predictions = []
    for batch in DataLoader(TensorDataset(*inputs), batch_size=batch_size):
        logits = model(*(tensor.to(device, dtype=torch.float32) for tensor in batch))
        predictions.append(logits.argmax(dim=1).cpu())
    return torch.cat(predictions).numpy()


def transition_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Accuracy per transition marker; just_states rows carry no marker and are skipped."""
    markers = metadata[RegistryColumns.TRANSITION_INFO].to_numpy()
    result: dict[str, dict[str, Any]] = {}
    for marker in TRANSITION_ORDER:
        mask = markers == marker
        n = int(mask.sum())
        correct = int((y_true[mask] == y_pred[mask]).sum())
        result[marker] = {"n": n, "correct": correct, "accuracy": correct / n if n else None}
    return result


def row_normalised_percent(counts: np.ndarray) -> np.ndarray:
    """Convert a confusion matrix to percentages of each true class (NaN for empty rows)."""
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(
        counts * 100.0, totals,
        out=np.full(counts.shape, np.nan), where=totals > 0,
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: pd.DataFrame,
    labels: Sequence[int] | None = None,
) -> dict[str, Any]:
    labels = sorted(LABEL_TO_CLASS) if labels is None else list(labels)
    counts = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "test_rows": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix_counts": counts.tolist(),
        "confusion_matrix_row_percent": row_normalised_percent(counts).tolist(),
        "transition_accuracy": transition_accuracy(y_true, y_pred, metadata),
    }


def _plot_confusion_matrix(
    percent: np.ndarray,
    title: str,
    path: Path,
    class_names: Sequence[str] = CLASS_NAMES,
) -> None:
    figure = Figure(figsize=(8, 7), constrained_layout=True)
    axes = figure.subplots()
    image = axes.imshow(percent, cmap="Blues", vmin=0, vmax=100)
    figure.colorbar(image, ax=axes, label="% of true class")
    for row in range(percent.shape[0]):
        for column in range(percent.shape[1]):
            value = percent[row, column]
            if not np.isnan(value):
                axes.text(
                    column, row, f"{value:.1f}", ha="center", va="center",
                    color="white" if value > 50 else "black", fontsize=9,
                )
    ticks = range(len(class_names))
    axes.set_xticks(ticks, class_names, rotation=45, ha="right")
    axes.set_yticks(ticks, class_names)
    axes.set_xlabel("Predicted class")
    axes.set_ylabel("True class")
    axes.set_title(title)
    figure.savefig(path, dpi=150)


def _plot_transition_accuracy(
    results: dict[str, dict[str, dict[str, Any]]],
    title: str,
    path: Path,
) -> None:
    labels = list(results)
    counts = next(iter(results.values()))
    x = np.arange(len(TRANSITION_ORDER))
    width = 0.8 / len(labels)

    figure = Figure(figsize=(9, 5.5), constrained_layout=True)
    axes = figure.subplots()
    for index, label in enumerate(labels):
        values = [
            np.nan if results[label][m]["accuracy"] is None
            else results[label][m]["accuracy"] * 100
            for m in TRANSITION_ORDER
        ]
        bars = axes.bar(x + (index - (len(labels) - 1) / 2) * width, values, width, label=label)
        axes.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    axes.set_xticks(x, [f"{m}\n(n={counts[m]['n']})" for m in TRANSITION_ORDER])
    axes.set_xlabel("Transition marker")
    axes.set_ylabel("Accuracy (%)")
    axes.set_ylim(0, 110)
    axes.set_title(title)
    figure.legend(loc="outside right center")
    axes.grid(axis="y", alpha=0.3)
    figure.savefig(path, dpi=150)


def _output_dir(prepared: PreparedData, output_root: str | Path) -> Path:
    config = prepared.experiment.config
    if getattr(config, "amputee_id", None) is not None:
        data_tag = (
            f"amputee-{normalize_amputee_id(config.amputee_id)}-"
            f"{normalize_data_type(config.data_type)}"
        )
    elif config.same_volunteer_id is not None:
        data_tag = f"same-{DatasetRegistry.normalize_volunteer_id(config.same_volunteer_id)}"
    else:
        data_tag = (
            f"separate-train{config.train_volunteer_count}-test{config.test_volunteer_count}"
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(output_root) / f"{prepared.model_target}__{prepared.input_mode}__{data_tag}__{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _data_title(prepared: PreparedData) -> str:
    config = prepared.experiment.config
    if getattr(config, "amputee_id", None) is not None:
        return (
            f"Amputee {normalize_amputee_id(config.amputee_id)} "
            f"({normalize_data_type(config.data_type)})"
        )
    if config.same_volunteer_id is not None:
        return f"Volunteer {DatasetRegistry.normalize_volunteer_id(config.same_volunteer_id)}"
    volunteers = sorted(prepared.test_metadata[RegistryColumns.VOLUNTEER_ID].unique())
    return f"Unseen volunteers {', '.join(volunteers)}"


def _label_to_class(prepared: PreparedData) -> dict[int, str]:
    return getattr(prepared.experiment.config, "label_to_class", LABEL_TO_CLASS)


def evaluate_models(
    prepared: PreparedData,
    specs: Sequence[ModelSpec],
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    output_dir: str | Path | None = None,
    device: str = "auto",
    batch_size: int = 128,
) -> dict[str, Any]:
    """Evaluate every model in ``specs`` on the test split and save plots and metrics.

    All models must be available; otherwise nothing is evaluated. Outputs go to
    ``output_dir`` when given, else to a new timestamped folder in ``output_root``.
    """
    runs: dict[str, TrainedRun] = {}
    missing: list[str] = []
    for spec in specs:
        try:
            runs[spec.label] = select_trained_run(prepared, spec.model_key, artifact_root)
        except ModelNotAvailableError as exc:
            missing.append(str(exc))
    if missing:
        lines = (line for message in missing for line in message.splitlines())
        raise ModelNotAvailableError("\n".join(dict.fromkeys(lines)))

    torch_device = resolve_device(device)
    y_true = prepared.y_test.cpu().numpy()
    metadata = prepared.test_metadata.reset_index(drop=True)
    if output_dir is None:
        output_dir = _output_dir(prepared, output_root)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    data_title = _data_title(prepared)
    mode = f"{prepared.input_mode}, {prepared.model_target}"
    label_to_class = _label_to_class(prepared)
    labels = sorted(label_to_class)
    class_names = [label_to_class[label] for label in labels]

    predictions = metadata.copy()
    predictions["y_true"] = y_true
    models: dict[str, Any] = {}
    for spec in specs:
        run = runs[spec.label]
        model = _load_model(spec, run, prepared, torch_device)
        y_pred = _predict(model, spec.inputs, torch_device, batch_size)
        predictions[f"pred_{spec.label}"] = y_pred

        metrics = compute_metrics(y_true, y_pred, metadata, labels)
        models[spec.label] = {
            "model_key": spec.model_key,
            "run_id": run.run_id,
            "checkpoint": str(run.checkpoint),
            "checkpoint_sha256": file_sha256(run.checkpoint),
            **metrics,
        }
        _plot_confusion_matrix(
            np.asarray(metrics["confusion_matrix_row_percent"]),
            f"{spec.label} confusion matrix: {data_title}\n({mode})\nrun {run.run_id}",
            output_dir / f"confusion_matrix_{spec.label}.png",
            class_names,
        )
        logger.info(
            "%s (%s): accuracy=%.4f macro_f1=%.4f | per marker: %s",
            spec.label, run.run_id, metrics["accuracy"], metrics["macro_f1"],
            {m: v["accuracy"] for m, v in metrics["transition_accuracy"].items()},
        )

    _plot_transition_accuracy(
        {label: result["transition_accuracy"] for label, result in models.items()},
        f"Accuracy per transition marker: {data_title}\n({mode})",
        output_dir / "transition_accuracy.png",
    )
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    summary = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_mode": prepared.input_mode,
        "model_target": prepared.model_target,
        "experiment_config": asdict(prepared.experiment.config),
        "class_names": class_names,
        "test_volunteers": sorted(metadata[RegistryColumns.VOLUNTEER_ID].unique()),
        "models": models,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Evaluation outputs written to %s", output_dir)
    return {"output_dir": str(output_dir), "models": models}
