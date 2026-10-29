"""Binary and small multi-class sub-problems of the five-class amputee task.

Each task keeps the rows of a few amputee classes from the ordinary train and
test split and relabels them 0..k-1 in the order listed in ``BinaryTask.labels``.
Because the rows come from the existing split, every task is tested on a subset
of the same sealed test samples the five-class model is tested on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import pandas as pd
import torch

from data_loader import PreparedData
from dataset_registry import AMPUTEE_CLASS_TO_LABEL, AMPUTEE_LABEL_TO_CLASS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BinaryTask:
    """One sub-problem: a name and the amputee labels it separates."""

    name: str
    labels: tuple[int, ...]

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(AMPUTEE_LABEL_TO_CLASS[label] for label in self.labels)


def _task(name: str, *classes: str) -> BinaryTask:
    return BinaryTask(name, tuple(AMPUTEE_CLASS_TO_LABEL[class_name] for class_name in classes))


BINARY_TASKS: tuple[BinaryTask, ...] = (
    _task("sit_vs_sit_to_stand", "sit", "sit_to_stand"),
    _task("sit_to_stand_vs_stand", "sit_to_stand", "stand"),
    _task("stand_to_sit_vs_sit", "stand_to_sit", "sit"),
    _task("stand_vs_walking", "stand", "walking"),
    _task("stand_walking_stand_to_sit", "stand", "walking", "stand_to_sit"),
)


def task_model_key(model_key: str, task: BinaryTask) -> str:
    """Run key of ``task`` for the architecture ``model_key``."""
    return f"{model_key}-{task.name}"


def _relabel(y: torch.Tensor, task: BinaryTask) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the positions of the task's rows and their new labels."""
    new_labels = torch.full_like(y, -1)
    for index, label in enumerate(task.labels):
        new_labels[y == label] = index
    keep = torch.nonzero(new_labels >= 0).squeeze(1)
    return keep, new_labels[keep]


def _rows(metadata: pd.DataFrame, keep: torch.Tensor) -> pd.DataFrame:
    return metadata.iloc[keep.numpy()].reset_index(drop=True)


def task_prepared(
    prepared: PreparedData,
    task: BinaryTask,
    *,
    train_inputs: bool = True,
) -> PreparedData:
    """Return ``prepared`` restricted to the rows of ``task`` and relabelled.

    The result carries a config whose ``label_subset`` names the task's classes,
    so ``num_classes``, class names and run manifests follow the task.
    ``train_inputs=False`` leaves the training input tensors empty (labels and
    metadata are still filtered); evaluation needs only those.
    """
    config = prepared.experiment.config
    if config.setup != "amputee":
        raise ValueError("Binary classifiers are only defined for amputee data.")
    if config.label_subset is not None:
        raise ValueError("The data is already restricted to a binary task.")
    train_keep, y_train = _relabel(prepared.y_train, task)
    test_keep, y_test = _relabel(prepared.y_test, task)
    for split, labels in (("training", y_train), ("test", y_test)):
        missing = [
            name for index, name in enumerate(task.class_names)
            if not bool((labels == index).any())
        ]
        if missing and split == "training":
            raise ValueError(f"Task {task.name} has no {split} rows of {missing}.")
        if missing:
            logger.warning("Task %s has no %s rows of %s.", task.name, split, missing)

    def inputs(tensor: torch.Tensor, keep: torch.Tensor, wanted: bool = True) -> torch.Tensor:
        return tensor[keep] if wanted else tensor[:0]

    return replace(
        prepared,
        experiment=replace(
            prepared.experiment, config=replace(config, label_subset=task.labels),
        ),
        X_imu_train=inputs(prepared.X_imu_train, train_keep, train_inputs),
        X_cwt_train=inputs(prepared.X_cwt_train, train_keep, train_inputs),
        y_train=y_train,
        X_imu_test=inputs(prepared.X_imu_test, test_keep),
        X_cwt_test=inputs(prepared.X_cwt_test, test_keep),
        y_test=y_test,
        train_metadata=_rows(prepared.train_metadata, train_keep),
        test_metadata=_rows(prepared.test_metadata, test_keep),
    )
