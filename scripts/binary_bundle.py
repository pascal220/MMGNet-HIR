"""One-file storage of the binary-classifier set of an amputee data type.

A bundle holds every task model of one architecture for one amputee data type:
weights, architecture config, class order, and a copy of each training run's
manifest. The manifests let evaluation confirm the models were trained on the
current split without needing the individual run folders, which remain only as
the audit trail of each search.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from binary_tasks import BINARY_TASKS, BinaryTask

BUNDLE_FORMAT = "binary_classifier_bundle"
BUNDLE_VERSION = 1


def bundle_path(
    checkpoint_dir: str | Path,
    model_key: str,
    amputee_id: str,
    data_type: str,
) -> Path:
    """Default location of the bundle of one architecture and amputee data type."""
    return Path(checkpoint_dir) / f"best_{model_key}_{amputee_id}_{data_type}_binary.pt"


def save_bundle(
    path: str | Path,
    *,
    model_key: str,
    input_mode: str,
    amputee_id: str,
    data_type: str,
    runs: Mapping[str, Mapping[str, Any]],
    tasks: tuple[BinaryTask, ...] = BINARY_TASKS,
) -> Path:
    """Write the models of ``runs`` (task name -> ``run_training_experiment`` result)."""
    entries: dict[str, Any] = {}
    for task in tasks:
        run = runs[task.name]
        checkpoint = torch.load(run["checkpoint_path"], map_location="cpu")
        manifest = json.loads(Path(run["manifest_path"]).read_text(encoding="utf-8"))
        entries[task.name] = {
            "labels": list(task.labels),
            "class_names": list(task.class_names),
            "run_id": run["run_id"],
            "checkpoint_sha256": manifest["artifacts"]["checkpoint_sha256"],
            "model_config": checkpoint["model_config"],
            "model_state_dict": checkpoint["model_state_dict"],
            "manifest": manifest,
        }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "format": BUNDLE_FORMAT,
            "version": BUNDLE_VERSION,
            "model_key": model_key,
            "input_mode": input_mode,
            "amputee_id": amputee_id,
            "data_type": data_type,
            "tasks": entries,
        },
        temporary,
    )
    os.replace(temporary, path)
    return path


def load_bundle(
    path: str | Path,
    tasks: tuple[BinaryTask, ...] = BINARY_TASKS,
) -> dict[str, Any]:
    """Load a bundle and check it holds exactly the expected tasks."""
    bundle = torch.load(path, map_location="cpu")
    if not isinstance(bundle, dict) or bundle.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"{path} is not a binary-classifier bundle.")
    entries = bundle["tasks"]
    expected = {task.name: list(task.labels) for task in tasks}
    found = {name: list(entry["labels"]) for name, entry in entries.items()}
    if found != expected:
        raise ValueError(
            f"{path} holds tasks {found}, but this version of the code defines {expected}. "
            "Retrain the binary classifiers."
        )
    return bundle
