"""Select trained runs whose recorded data split matches the prepared data.

A run may only be reused (for evaluation, or as a frozen fusion backbone) when
it was trained on exactly the split held in memory; otherwise its "unseen" test
samples may have been part of its training data.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from data_loader import reconstruct_train_metadata
from dataset_registry import DatasetRegistry, normalize_amputee_id, normalize_data_type
from split_utils import GROUP_COLUMNS
from training_experiment import PreparedDataLike, file_sha256, metadata_fingerprint

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT_ROOT = "results/training"

# Experiment settings that change which samples land in train and test. The memory
# budget is deliberately excluded: it only drops optional samples within each side,
# and verification rebuilds a run's training set under the budget it recorded.
SPLIT_SETTINGS: dict[str, tuple[str, ...]] = {
    "same_volunteer": (
        "same_volunteer_id", "seed", "test_fraction", "just_states_ratio",
    ),
    "separate_volunteers": (
        "train_volunteer_count", "test_volunteer_count", "seed", "just_states_ratio",
    ),
    "amputee": (
        "amputee_id", "data_type", "seed", "test_fraction", "just_states_ratio",
    ),
}
_DATA_IDENTITY = {
    "setup", "same_volunteer_id", "train_volunteer_count", "test_volunteer_count",
    "amputee_id", "data_type",
}


class ModelNotAvailableError(LookupError):
    """No completed training run matches the requested model and data split."""


class RunVerificationError(RuntimeError):
    """A selected run's artifacts no longer match the recorded training state."""


@dataclass(frozen=True)
class TrainedRun:
    """One completed run directory under the training artifact root."""

    run_id: str
    run_dir: Path
    manifest: dict[str, Any]

    @property
    def model_key(self) -> str:
        return self.manifest["model"]["key"]

    @property
    def input_mode(self) -> str:
        return self.manifest["model"]["input_mode"]

    @property
    def experiment_config(self) -> dict[str, Any]:
        return self.manifest["data"]["experiment_config"]

    @property
    def checkpoint(self) -> Path:
        return self.run_dir / self.manifest["artifacts"]["checkpoint"]

    @property
    def completed_at(self) -> datetime:
        return datetime.fromisoformat(self.manifest["completed_at_utc"])

    @property
    def parent_checkpoints(self) -> list[dict[str, Any]]:
        return list(self.manifest["model"].get("parent_checkpoints", []))


def load_completed_runs(artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT) -> list[TrainedRun]:
    """Return every run under ``artifact_root`` whose manifest reports completion."""
    runs: list[TrainedRun] = []
    for manifest_path in sorted(Path(artifact_root).glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable manifest %s: %s", manifest_path, exc)
            continue
        if manifest.get("status") == "completed":
            runs.append(TrainedRun(manifest_path.parent.name, manifest_path.parent, manifest))
    return runs


def _setup(settings: Mapping[str, Any]) -> str:
    # Manifests written before amputee support have no amputee_id key.
    if settings.get("amputee_id") is not None:
        return "amputee"
    return "separate_volunteers" if settings.get("same_volunteer_id") is None else "same_volunteer"


def _normalise(name: str, value: Any) -> Any:
    if value is None:
        return value
    if name == "same_volunteer_id":
        return DatasetRegistry.normalize_volunteer_id(value)
    if name == "amputee_id":
        return normalize_amputee_id(value)
    if name == "data_type":
        return normalize_data_type(value)
    return value


def split_differences(
    recorded: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, tuple[Any, Any]]:
    """Return ``{setting: (recorded, current)}`` for split settings that differ."""
    recorded_setup, current_setup = _setup(recorded), _setup(current)
    if recorded_setup != current_setup:
        return {"setup": (recorded_setup, current_setup)}

    differences: dict[str, tuple[Any, Any]] = {}
    for name in SPLIT_SETTINGS[current_setup]:
        old = _normalise(name, recorded.get(name))
        new = _normalise(name, current.get(name))
        if isinstance(old, float) or isinstance(new, float):
            same = old is not None and new is not None and math.isclose(old, new)
        else:
            same = old == new
        if not same:
            differences[name] = (old, new)
    return differences


def describe_split(settings: Mapping[str, Any]) -> str:
    """Return a short human-readable description of the data split."""
    if _setup(settings) == "amputee":
        return (
            f"amputee {_normalise('amputee_id', settings['amputee_id'])} "
            f"{_normalise('data_type', settings['data_type'])}"
        )
    if _setup(settings) == "same_volunteer":
        return f"volunteer {_normalise('same_volunteer_id', settings['same_volunteer_id'])}"
    return (
        f"{settings['train_volunteer_count']} train / "
        f"{settings['test_volunteer_count']} unseen test volunteers"
    )


def _current_settings(prepared: PreparedDataLike) -> dict[str, Any]:
    config = prepared.experiment.config
    return {
        name: getattr(config, name, None)
        for names in SPLIT_SETTINGS.values()
        for name in names
    }


def _train_command(model_key: str, prepared: PreparedDataLike) -> str:
    config = prepared.experiment.config
    target = "fusion" if model_key.startswith("fusion") else "standalone"
    amputee_id = getattr(config, "amputee_id", None)
    if amputee_id is not None:
        # One amputee training run always trains every data type.
        data = f"--amputee-id {normalize_amputee_id(amputee_id)}"
    elif config.same_volunteer_id is not None:
        data = f"--same-volunteer-id {config.same_volunteer_id}"
    else:
        data = (
            f"--train-volunteer-count {config.train_volunteer_count} "
            f"--test-volunteer-count {config.test_volunteer_count}"
        )
    return (
        f"python main.py --train {data} --input-mode {prepared.input_mode} "
        f"--model-target {target} --seed {config.seed} "
        f"--test-fraction {config.test_fraction} "
        f"--just-states-ratio {config.just_states_ratio} "
        f"--total-budget-gb {config.total_budget_gb}"
    )


def _missing_run_message(
    model_key: str,
    prepared: PreparedDataLike,
    candidates: list[TrainedRun],
    artifact_root: str | Path,
) -> str:
    settings = _current_settings(prepared)
    lines = [
        f"No trained '{model_key}' model ({prepared.input_mode}) in {artifact_root} "
        f"matches the current split ({describe_split(settings)}, seed={settings['seed']})."
    ]
    near_misses = [
        (run, differences)
        for run in candidates
        if (differences := split_differences(run.experiment_config, settings))
        and not _DATA_IDENTITY & differences.keys()
    ]
    for run, differences in near_misses:
        changed = ", ".join(
            f"{name}: trained={old!r}, now={new!r}" for name, (old, new) in differences.items()
        )
        lines.append(f"  Run {run.run_id} uses other split settings ({changed}).")
    lines.append(f"  Train it first with: {_train_command(model_key, prepared)}")
    return "\n".join(lines)


def _sample_keys(metadata: Any) -> set[tuple[str, str]]:
    return {
        (str(path).replace("\\", "/"), str(index))
        for path, index in zip(metadata[GROUP_COLUMNS[0]], metadata[GROUP_COLUMNS[1]])
    }


def _verify_trained_on_split(
    run_id: str,
    model_key: str,
    manifest: Mapping[str, Any],
    prepared: PreparedDataLike,
) -> None:
    """Check a run's training samples came from this split and exclude its test samples.

    A run trained under another memory budget used another subset of the same
    split, so its training metadata is rebuilt under that budget and compared.
    """
    recorded = manifest["data"]["metadata_fingerprint_sha256"]
    train_metadata = prepared.train_metadata
    matches = metadata_fingerprint(train_metadata) == recorded
    budget = manifest["data"]["experiment_config"].get("total_budget_gb")
    if not matches and budget is not None and not math.isclose(
        budget, prepared.experiment.config.total_budget_gb
    ):
        train_metadata = reconstruct_train_metadata(
            prepared.experiment, prepared.input_mode, budget
        )
        matches = metadata_fingerprint(train_metadata) == recorded
    if not matches:
        raise RunVerificationError(
            f"Run {run_id} was trained on a different split than the one prepared now "
            "(training-metadata fingerprint mismatch), so its test samples cannot be "
            "guaranteed unseen. This happens when the data files or the selection code "
            "differ from training. Retrain with: "
            f"{_train_command(model_key, prepared)}"
        )
    if _sample_keys(train_metadata) & _sample_keys(prepared.test_metadata):
        raise RunVerificationError(
            f"Run {run_id} was trained on samples that are in the current test set."
        )


def verify_run(run: TrainedRun, prepared: PreparedDataLike) -> None:
    """Check the checkpoint is unchanged and the run was trained on this split."""
    if not run.checkpoint.is_file():
        raise RunVerificationError(f"Run {run.run_id} is missing its checkpoint {run.checkpoint}.")
    if file_sha256(run.checkpoint) != run.manifest["artifacts"]["checkpoint_sha256"]:
        raise RunVerificationError(
            f"Checkpoint {run.checkpoint} has changed since run {run.run_id} was trained."
        )
    _verify_trained_on_split(run.run_id, run.model_key, run.manifest, prepared)


def _has_split_marker(run: TrainedRun, prepared: PreparedDataLike) -> bool:
    """Windowed multi-volunteer runs must carry the ``separate-v<N>`` folder marker."""
    config = prepared.experiment.config
    if (
        prepared.input_mode != "windowed"
        or config.same_volunteer_id is not None
        or getattr(config, "amputee_id", None) is not None
    ):
        return True
    return f"separate-v{config.train_volunteer_count}" in run.run_id


def _candidate_runs(
    prepared: PreparedDataLike,
    model_key: str,
    artifact_root: str | Path,
) -> tuple[list[TrainedRun], list[TrainedRun]]:
    """Return ``(candidates, matching)`` runs of ``model_key`` for the prepared split."""
    settings = _current_settings(prepared)
    candidates = [
        run for run in load_completed_runs(artifact_root)
        if run.model_key == model_key and run.input_mode == prepared.input_mode
    ]
    matching = [
        run for run in candidates
        if not split_differences(run.experiment_config, settings)
        and _has_split_marker(run, prepared)
    ]
    return candidates, matching


def select_trained_run(
    prepared: PreparedDataLike,
    model_key: str,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> TrainedRun:
    """Return the latest verified run of ``model_key`` trained on the prepared split."""
    candidates, matching = _candidate_runs(prepared, model_key, artifact_root)
    if not matching:
        raise ModelNotAvailableError(
            _missing_run_message(model_key, prepared, candidates, artifact_root)
        )
    run = max(matching, key=lambda candidate: candidate.completed_at)
    verify_run(run, prepared)
    logger.info("Selected %s run %s (completed %s).", model_key, run.run_id, run.completed_at)
    return run


def require_trained_runs(
    splits: Sequence[PreparedDataLike],
    model_keys: Sequence[str],
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> None:
    """Raise ``ModelNotAvailableError`` unless every model has a run for every split.

    Only manifests are read, so ``splits`` may be lightweight objects exposing
    ``experiment.config`` and ``input_mode`` and no data needs to be loaded.
    Checkpoint and split fingerprints are still verified by ``select_trained_run``.
    """
    missing: list[str] = []
    for prepared in splits:
        for model_key in model_keys:
            candidates, matching = _candidate_runs(prepared, model_key, artifact_root)
            if not matching:
                missing.append(
                    _missing_run_message(model_key, prepared, candidates, artifact_root)
                )
    if missing:
        lines = list(dict.fromkeys(line for message in missing for line in message.splitlines()))
        commands = [line for line in lines if line.lstrip().startswith("Train it first")]
        raise ModelNotAvailableError(
            "\n".join([line for line in lines if line not in commands] + commands)
        )


def verify_parent_checkpoints(run: TrainedRun, prepared: PreparedDataLike) -> list[Path]:
    """Return a fusion run's frozen parent checkpoints after checking they are unchanged."""
    paths: list[Path] = []
    for record in run.parent_checkpoints:
        path = Path(record["path"])
        if not path.is_file():
            raise RunVerificationError(f"Run {run.run_id} parent checkpoint {path} is missing.")
        if file_sha256(path) != record["sha256"]:
            raise RunVerificationError(
                f"Run {run.run_id} parent checkpoint {path} has changed since training."
            )
        manifest_path = record.get("manifest_path")
        if manifest_path is None:
            raise RunVerificationError(
                f"Run {run.run_id} parent checkpoint {path} is not linked to a training run."
            )
        parent = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        _verify_trained_on_split(
            f"{run.run_id} parent {parent['run_id']}",
            parent["model"]["key"],
            parent,
            prepared,
        )
        paths.append(path)
    return paths
