"""Summarise best_trial.json across single-volunteer training runs.

Uses the latest completed run per (model, volunteer) and writes one CSV row per
run plus a JSON summary per model: n, mean, sample std, min, max and range of
every numeric field, and value counts of categorical hyperparameters.

Run from the repository root:
    python evaluation/summarise_best_trials.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from dataset_registry import DatasetRegistry
from run_selection import DEFAULT_ARTIFACT_ROOT, TrainedRun, load_completed_runs

METRIC_FIELDS = (
    "objective_value",
    "validation_accuracy",
    "validation_macro_f1",
    "best_epoch",
    "best_trial_number",
)
ID_COLUMNS = ["model_key", "volunteer_id", "run_id", "completed_at_utc"]
PARAM_PREFIX = "param."


def collect_best_trials(artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT) -> pd.DataFrame:
    """Return one row per latest single-volunteer run with its best-trial values."""
    latest: dict[tuple[str, str], TrainedRun] = {}
    for run in load_completed_runs(artifact_root):
        volunteer = run.experiment_config.get("same_volunteer_id")
        if volunteer is None:
            continue
        key = (run.model_key, DatasetRegistry.normalize_volunteer_id(volunteer))
        if key not in latest or run.completed_at > latest[key].completed_at:
            latest[key] = run

    rows = []
    for (model_key, volunteer), run in sorted(latest.items()):
        best = json.loads((run.run_dir / "best_trial.json").read_text(encoding="utf-8"))
        row: dict[str, Any] = {
            "model_key": model_key,
            "volunteer_id": volunteer,
            "run_id": run.run_id,
            "completed_at_utc": run.manifest["completed_at_utc"],
        }
        row.update({field: best[field] for field in METRIC_FIELDS})
        row.update({f"{PARAM_PREFIX}{name}": value for name, value in best["best_params"].items()})
        rows.append(row)
    return pd.DataFrame(rows, columns=None if rows else ID_COLUMNS)


def summarise(runs: pd.DataFrame) -> dict[str, Any]:
    """Return numeric statistics and categorical counts per model_key."""
    summary: dict[str, Any] = {}
    for model_key, group in runs.groupby("model_key", sort=True):
        numeric: dict[str, Any] = {}
        categorical: dict[str, Any] = {}
        for column in group.columns.drop(ID_COLUMNS):
            values = group[column].dropna()
            if values.empty:
                continue
            if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
                numeric[column] = {
                    "n": int(len(values)),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else None,
                    "min": values.min().item(),
                    "max": values.max().item(),
                    "range": (values.max() - values.min()).item(),
                }
            else:
                counts = values.astype(str).value_counts().sort_index()
                categorical[column] = {
                    "n": int(len(values)),
                    "counts": {str(value): int(count) for value, count in counts.items()},
                }
        summary[str(model_key)] = {
            "runs": int(len(group)),
            "volunteers": sorted(group["volunteer_id"]),
            "numeric": numeric,
            "categorical": categorical,
        }
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    for model_key, result in summary.items():
        print(f"\n=== {model_key}: {result['runs']} runs ({', '.join(result['volunteers'])}) ===")
        table = pd.DataFrame(result["numeric"]).T
        print(table.to_string(float_format=lambda value: f"{value:.6g}"))
        for column, info in result["categorical"].items():
            print(f"{column}: {info['counts']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarise best_trial.json across single-volunteer training runs."
    )
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", default="results/evaluation/best_trials")
    args = parser.parse_args()

    runs = collect_best_trials(args.artifact_root)
    if runs.empty:
        print(f"No completed single-volunteer runs found in {args.artifact_root}.")
        return 1

    summary = summarise(runs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_dir / "best_trials_runs.csv", index=False)
    (output_dir / "best_trials_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _print_summary(summary)
    print(f"\nSaved {output_dir / 'best_trials_runs.csv'} and {output_dir / 'best_trials_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
