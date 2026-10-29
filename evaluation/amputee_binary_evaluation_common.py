"""Evaluation of the amputee binary-classifier sets across every data type.

One call evaluates, for each amputee data type, the bundle of every fusion
architecture. Each task model is tested on the sealed test rows of its own
classes, so the outputs show every model separately and one overall summary:

    <output_root>/fusion__<input_mode>__amputee-<id>__binary__<stamp>/
        type1/<model>/   confusion_matrix_<task>.png, predictions_<task>.csv,
                         transition_accuracy.png, metrics.json
        type2/<model>/   (same)
        summary.json  summary.csv  summary_metrics.png

``overall`` aggregates one bundle: the unweighted mean of the task models'
accuracy, balanced accuracy and macro F1, and the pooled accuracy over every
(task, test row) prediction. A test row of a class shared by several tasks is
counted once per task.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from binary_bundle import bundle_path, load_bundle
from binary_tasks import BINARY_TASKS, task_model_key, task_prepared
from data_loader import (
    AMPUTEE_DATA_TYPES,
    PreparedData,
    amputee_experiment_config,
    run_per_amputee_type,
)
from dataset_registry import RegistryColumns, normalize_amputee_id, normalize_data_type
from device_utils import release_cuda_memory, resolve_device
from evaluation_common import (
    DEFAULT_OUTPUT_ROOT,
    ModelBuilder,
    _data_title,
    _plot_confusion_matrix,
    _plot_transition_accuracy,
    _predict,
    compute_metrics,
)
from run_selection import ModelNotAvailableError, split_differences, verify_recorded_run
from split_utils import validate_prepared_data

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR = "checkpoints"
SUMMARY_METRICS: tuple[str, ...] = ("accuracy", "balanced_accuracy", "macro_f1")
OVERALL_LABEL = "OVERALL"


@dataclass(frozen=True)
class BinarySpec:
    """One architecture whose binary-classifier bundle is evaluated."""

    label: str
    model_key: str
    build: ModelBuilder


def overall_metrics(tasks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-task metrics of one bundle into one overall summary."""
    rows = sum(metrics["test_rows"] for metrics in tasks.values())
    correct = sum(int(np.trace(np.asarray(m["confusion_matrix_counts"]))) for m in tasks.values())
    return {
        "tasks": len(tasks),
        **{
            f"mean_{name}": float(np.mean([metrics[name] for metrics in tasks.values()]))
            for name in SUMMARY_METRICS
        },
        "pooled_rows": rows,
        "pooled_correct": correct,
        "pooled_accuracy": correct / rows if rows else None,
    }


def _evaluate_bundle(
    prepared: PreparedData,
    spec: BinarySpec,
    input_mode: str,
    *,
    checkpoint_dir: str | Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    config = prepared.experiment.config
    path = bundle_path(
        checkpoint_dir, spec.model_key,
        normalize_amputee_id(config.amputee_id), normalize_data_type(config.data_type),
    )
    bundle = load_bundle(path)
    if bundle["model_key"] != spec.model_key or bundle["input_mode"] != input_mode:
        raise ValueError(
            f"{path} holds {bundle['model_key']} ({bundle['input_mode']}) models, "
            f"not {spec.model_key} ({input_mode})."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    data_title = _data_title(prepared)
    mode = f"{input_mode}, fusion"

    tasks: dict[str, dict[str, Any]] = {}
    for task in BINARY_TASKS:
        entry = bundle["tasks"][task.name]
        data = task_prepared(prepared, task, train_inputs=False)
        verify_recorded_run(
            entry["run_id"], task_model_key(spec.model_key, task), entry["manifest"], data,
        )
        model = spec.build(dict(entry["model_config"]))
        model.load_state_dict(entry["model_state_dict"])
        model.to(device).eval()

        y_true = data.y_test.cpu().numpy()
        metadata = data.test_metadata
        y_pred = _predict(model, (data.X_imu_test, data.X_cwt_test), device, batch_size)
        metrics = compute_metrics(y_true, y_pred, metadata, list(range(task.num_classes)))
        tasks[task.name] = {
            "classes": list(task.class_names),
            "run_id": entry["run_id"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            **metrics,
        }
        _plot_confusion_matrix(
            np.asarray(metrics["confusion_matrix_row_percent"]),
            f"{spec.label} {task.name}: {data_title}\n({mode})\nrun {entry['run_id']}",
            output_dir / f"confusion_matrix_{task.name}.png",
            list(task.class_names),
        )
        predictions = metadata.copy()
        predictions["y_true"] = y_true
        predictions[f"pred_{spec.label}"] = y_pred
        predictions["class_true"] = [task.class_names[label] for label in y_true]
        predictions["class_pred"] = [task.class_names[label] for label in y_pred]
        predictions.to_csv(output_dir / f"predictions_{task.name}.csv", index=False)
        logger.info(
            "%s %s: accuracy=%.4f macro_f1=%.4f (%d test rows)",
            spec.label, task.name, metrics["accuracy"], metrics["macro_f1"], metrics["test_rows"],
        )
        del model, data
        release_cuda_memory()

    overall = overall_metrics(tasks)
    _plot_transition_accuracy(
        {name: result["transition_accuracy"] for name, result in tasks.items()},
        f"Accuracy per transition marker: {spec.label}, {data_title}\n({mode})",
        output_dir / "transition_accuracy.png",
        show_counts=False,
    )
    result = {
        "model_key": spec.model_key,
        "bundle": str(path),
        "tasks": tasks,
        "overall": overall,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                "input_mode": input_mode,
                "experiment_config": asdict(config),
                "test_volunteers": sorted(
                    prepared.test_metadata[RegistryColumns.VOLUNTEER_ID].unique()
                ),
                **result,
            },
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    return result


def evaluate_binary_type(
    prepared: PreparedData,
    specs: Sequence[BinarySpec],
    input_mode: str,
    *,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    output_dir: str | Path,
    device: str = "auto",
    batch_size: int = 128,
) -> dict[str, Any]:
    """Evaluate the binary-classifier bundle of every ``specs`` entry on one data type."""
    validate_prepared_data(prepared, input_mode, "fusion")
    if prepared.experiment.config.setup != "amputee":
        raise ValueError("Amputee evaluation expects data from prepare_amputee_experiment_data.")
    torch_device = resolve_device(device)
    output_dir = Path(output_dir)
    models = {
        spec.label: _evaluate_bundle(
            prepared, spec, input_mode,
            checkpoint_dir=checkpoint_dir, output_dir=output_dir / spec.label,
            device=torch_device, batch_size=batch_size,
        )
        for spec in specs
    }
    return {"output_dir": str(output_dir), "models": models}


def _summary_rows(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for data_type, result in results.items():
        for label, model in result["models"].items():
            base = {"data_type": data_type, "model": label}
            for name, metrics in model["tasks"].items():
                rows.append({
                    **base, "task": name, "classes": " / ".join(metrics["classes"]),
                    "test_rows": metrics["test_rows"],
                    **{metric: metrics[metric] for metric in SUMMARY_METRICS},
                })
            overall = model["overall"]
            rows.append({
                **base, "task": OVERALL_LABEL, "classes": f"mean of {overall['tasks']} models",
                "test_rows": overall["pooled_rows"],
                **{metric: overall[f"mean_{metric}"] for metric in SUMMARY_METRICS},
                "pooled_accuracy": overall["pooled_accuracy"],
            })
    return pd.DataFrame(rows)


def _plot_summary(summary: pd.DataFrame, title: str, path: Path) -> None:
    groups = list(summary.groupby(["data_type", "model"], sort=False))
    columns = min(2, len(groups))
    grid_rows = math.ceil(len(groups) / columns)
    figure = Figure(figsize=(8.5 * columns, 5.5 * grid_rows), constrained_layout=True)
    axes_grid = np.atleast_1d(figure.subplots(grid_rows, columns, squeeze=False)).ravel()
    width = 0.8 / len(SUMMARY_METRICS)
    for axes, ((data_type, model), frame) in zip(axes_grid, groups):
        x = np.arange(len(frame))
        for index, name in enumerate(SUMMARY_METRICS):
            bars = axes.bar(
                x + (index - (len(SUMMARY_METRICS) - 1) / 2) * width,
                frame[name].to_numpy() * 100, width, label=name.replace("_", " "),
            )
            axes.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
        axes.set_xticks(
            x, [f"{row.task}\n(n={row.test_rows})" for row in frame.itertuples()],
            rotation=25, ha="right", fontsize=8,
        )
        axes.set_ylabel("Score (%)")
        axes.set_ylim(0, 112)
        axes.set_title(f"{data_type}, {model}")
        axes.grid(axis="y", alpha=0.3)
    for axes in axes_grid[len(groups):]:
        axes.set_visible(False)
    figure.suptitle(title)
    figure.legend(*axes_grid[0].get_legend_handles_labels(), loc="outside right center")
    figure.savefig(path, dpi=150)


def _require_bundles(
    amputee_id: str,
    input_mode: str,
    data_types: Sequence[str],
    specs: Sequence[BinarySpec],
    checkpoint_dir: str | Path,
    config_kwargs: dict[str, Any],
) -> None:
    """Raise unless every bundle exists and was trained with the current split settings."""
    problems: list[str] = []
    train_command = ""
    for data_type in data_types:
        config = amputee_experiment_config(amputee_id, data_type, **config_kwargs)
        train_command = (
            f"python main.py --train --amputee-id {amputee_id} --input-mode {input_mode} "
            f"--model-target fusion --binary-classifiers --seed {config.seed} "
            f"--test-fraction {config.test_fraction} "
            f"--just-states-ratio {config.just_states_ratio} "
            f"--total-budget-gb {config.total_budget_gb}"
        )
        for spec in specs:
            path = bundle_path(checkpoint_dir, spec.model_key, amputee_id, data_type)
            if not path.is_file():
                problems.append(f"No binary-classifier bundle for {spec.label} ({data_type}): {path}")
                continue
            bundle = load_bundle(path)
            for name, entry in bundle["tasks"].items():
                differences = split_differences(
                    entry["manifest"]["data"]["experiment_config"], asdict(config),
                )
                if differences:
                    changed = ", ".join(
                        f"{key}: trained={old!r}, now={new!r}"
                        for key, (old, new) in differences.items()
                    )
                    problems.append(f"{path} (task {name}) uses other split settings ({changed}).")
                    break
    if problems:
        raise ModelNotAvailableError("\n".join([*problems, f"  Train it first with: {train_command}"]))


def evaluate_amputee_binary(
    amputee_id: int | str,
    input_mode: str,
    specs: Sequence[BinarySpec],
    *,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    device: str = "auto",
    data_types: Sequence[str] = AMPUTEE_DATA_TYPES,
    **data_kwargs: Any,
) -> dict[str, Any]:
    """Evaluate the bundles of ``specs`` on every amputee data type and summarise them.

    Every bundle must exist before any data is loaded. ``data_kwargs`` are
    forwarded to ``prepare_amputee_experiment_data`` and must match the training
    settings.
    """
    amputee_id = normalize_amputee_id(amputee_id)
    data_types = [normalize_data_type(data_type) for data_type in data_types]
    config_kwargs = {key: value for key, value in data_kwargs.items() if key != "data_root"}
    _require_bundles(amputee_id, input_mode, data_types, specs, checkpoint_dir, config_kwargs)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(output_root) / f"fusion__{input_mode}__amputee-{amputee_id}__binary__{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_per_amputee_type(
        amputee_id, input_mode,
        lambda prepared: evaluate_binary_type(
            prepared, specs, input_mode,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir / prepared.experiment.config.data_type,
            device=device,
        ),
        data_types=data_types, **data_kwargs,
    )

    summary = _summary_rows(results)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                "amputee_id": amputee_id,
                "input_mode": input_mode,
                "model_target": "fusion",
                "data_settings": data_kwargs,
                "tasks": {
                    task.name: list(task.class_names) for task in BINARY_TASKS
                },
                "types": results,
            },
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    _plot_summary(
        summary,
        f"Amputee {amputee_id}: binary-classifier test scores ({input_mode}, fusion)",
        output_dir / "summary_metrics.png",
    )
    logger.info("Binary-classifier evaluation summary written to %s", output_dir)
    return {"output_dir": str(output_dir), "types": results}
