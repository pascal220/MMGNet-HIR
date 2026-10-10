"""Evaluation of the amputee fusion models across every data type.

One call evaluates each amputee data type in turn. The types are independent
datasets with their own sealed test splits and trained models. Before any data
is loaded, every model of every type must have a matching run (all or nothing).
Outputs go to one folder per call:

    <output_root>/fusion__<input_mode>__amputee-<id>__<stamp>/
        type1/  type2/          per-type confusion matrices, predictions, metrics
        summary.json  summary.csv  summary_metrics.png
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from data_loader import (
    AMPUTEE_DATA_TYPES,
    PreparedData,
    amputee_experiment_config,
    run_per_amputee_type,
)
from dataset_registry import normalize_amputee_id, normalize_data_type
from evaluation_common import DEFAULT_OUTPUT_ROOT
from run_selection import DEFAULT_ARTIFACT_ROOT, require_trained_runs

logger = logging.getLogger(__name__)

SUMMARY_METRICS: tuple[str, ...] = ("accuracy", "balanced_accuracy", "macro_f1")

# Evaluates one prepared data type into the given folder; returns evaluate_models' result.
TypeEvaluator = Callable[..., dict[str, Any]]


def _summary_rows(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "data_type": data_type,
            "model": label,
            "model_key": metrics["model_key"],
            "run_id": metrics["run_id"],
            "test_rows": metrics["test_rows"],
            **{name: metrics[name] for name in SUMMARY_METRICS},
        }
        for data_type, result in results.items()
        for label, metrics in result["models"].items()
    ]
    return pd.DataFrame(rows)


def _plot_summary(summary: pd.DataFrame, title: str, path: Path) -> None:
    x = np.arange(len(summary))
    width = 0.8 / len(SUMMARY_METRICS)
    figure = Figure(figsize=(max(7.0, 2.2 * len(summary)), 5.5), constrained_layout=True)
    axes = figure.subplots()
    for index, name in enumerate(SUMMARY_METRICS):
        bars = axes.bar(
            x + (index - (len(SUMMARY_METRICS) - 1) / 2) * width,
            summary[name].to_numpy() * 100, width, label=name.replace("_", " "),
        )
        axes.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    axes.set_xticks(x, [f"{row.data_type}\n{row.model}" for row in summary.itertuples()])
    axes.set_ylabel("Score (%)")
    axes.set_ylim(0, 110)
    axes.set_title(title)
    figure.legend(loc="outside right center")
    axes.grid(axis="y", alpha=0.3)
    figure.savefig(path, dpi=150)


def evaluate_amputee(
    amputee_id: int | str,
    input_mode: str,
    evaluate_type: TypeEvaluator,
    model_keys: Sequence[str],
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    device: str = "auto",
    data_types: Sequence[str] = AMPUTEE_DATA_TYPES,
    **data_kwargs: Any,
) -> dict[str, Any]:
    """Evaluate ``model_keys`` on every amputee data type and write a combined summary.

    ``evaluate_type(prepared, *, artifact_root, output_dir, device)`` evaluates
    one prepared type. ``data_kwargs`` are forwarded to
    ``prepare_amputee_experiment_data`` and must match the training settings.
    """
    amputee_id = normalize_amputee_id(amputee_id)
    data_types = [normalize_data_type(data_type) for data_type in data_types]
    config_kwargs = {key: value for key, value in data_kwargs.items() if key != "data_root"}
    require_trained_runs(
        [
            SimpleNamespace(
                experiment=SimpleNamespace(
                    config=amputee_experiment_config(amputee_id, data_type, **config_kwargs)
                ),
                input_mode=input_mode,
            )
            for data_type in data_types
        ],
        model_keys,
        artifact_root,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(output_root) / f"fusion__{input_mode}__amputee-{amputee_id}__{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(prepared: PreparedData) -> dict[str, Any]:
        return evaluate_type(
            prepared,
            artifact_root=artifact_root,
            output_dir=output_dir / prepared.experiment.config.data_type,
            device=device,
        )

    results = run_per_amputee_type(
        amputee_id, input_mode, evaluate, data_types=data_types, **data_kwargs,
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
                "types": results,
            },
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    _plot_summary(
        summary,
        f"Amputee {amputee_id}: test scores per data type ({input_mode}, fusion)",
        output_dir / "summary_metrics.png",
    )
    logger.info("Amputee evaluation summary written to %s", output_dir)
    return {"output_dir": str(output_dir), "types": results}
