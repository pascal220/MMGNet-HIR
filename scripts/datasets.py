"""
datasets.py

One PyTorch item per recorded example. IMU and MMG stay separate, since
they train separate models; any fusion happens inside the model's input
layers rather than here.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from dataset_registry import LABEL_TO_CLASS, RegistryColumns
from memory_manager import TensorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModalityShape:
    """The input geometry a model must accept, with named axes.

    Windows are four sequential views of the same stream, each shifted by
    50 ms, so the first and last are 200 ms apart. ``scales`` is present
    for MMG only, where it counts wavelet scales.
    """

    windows: int
    time_steps: int
    channels: int
    scales: Optional[int] = None

    @property
    def item_shape(self) -> tuple[int, ...]:
        """Shape of a single example; batches arrive as (batch, *item_shape)."""
        if self.scales is None:
            return (self.windows, self.time_steps, self.channels)
        return (self.windows, self.scales, self.time_steps, self.channels)

    @classmethod
    def from_array_shape(cls, array_shape: tuple[int, ...]) -> "ModalityShape":
        """Build a descriptor from a full array shape including the example axis."""
        item = tuple(int(dim) for dim in array_shape[1:])
        if len(item) == 3:
            windows, time_steps, channels = item
            return cls(windows=windows, time_steps=time_steps, channels=channels)
        if len(item) == 4:
            windows, scales, time_steps, channels = item
            return cls(
                windows=windows,
                scales=scales,
                time_steps=time_steps,
                channels=channels,
            )
        raise ValueError(
            f"Unsupported array shape {array_shape}: expected 4 axes (IMU) "
            "or 5 axes (MMG)."
        )

    def describe(self) -> str:
        if self.scales is None:
            return (
                f"windows={self.windows}, time_steps={self.time_steps}, "
                f"channels={self.channels}"
            )
        return (
            f"windows={self.windows}, scales={self.scales}, "
            f"time_steps={self.time_steps}, channels={self.channels}"
        )


class SingleModalityDataset(Dataset):
    """Exposes every example of one modality as an individual item.

    A file holds many examples along its leading axis, so the dataset index
    addresses (file, example) rather than whole files.

    Parameters
    ----------
    registry_df : pd.DataFrame
        Registry rows for a single modality.
    store : TensorStore
        Holds the float32 tensors for those rows.
    transform : Callable, optional
        Applied to the example tensor.
    """

    def __init__(
        self,
        registry_df: pd.DataFrame,
        store: TensorStore,
        transform: Optional[Callable] = None,
    ):
        if registry_df.empty:
            raise ValueError("Cannot build a dataset from an empty registry.")

        self._df = registry_df.reset_index(drop=True)
        self._store = store
        self._transform = transform

        samples = self._df[RegistryColumns.SAMPLES].to_numpy(dtype=np.int64)
        self._paths = self._df[RegistryColumns.FILE_PATH].to_numpy()
        self._row_position = np.repeat(np.arange(len(self._df)), samples)
        self._example_position = np.concatenate(
            [np.arange(count, dtype=np.int64) for count in samples]
        )
        self._labels = torch.from_numpy(
            self._df[RegistryColumns.CLASS_LABEL].to_numpy(dtype=np.int64)
        )
        self._shape = self._resolve_shape()

        logger.info(
            "%s created: %d examples from %d files | item shape %s (%s)",
            self.__class__.__name__,
            len(self),
            len(self._df),
            self._shape.item_shape,
            self._shape.describe(),
        )

    def __len__(self) -> int:
        return int(self._row_position.size)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        row_position = int(self._row_position[index])
        tensor = self._store.get(self._paths[row_position])[
            int(self._example_position[index])
        ]
        if self._transform is not None:
            tensor = self._transform(tensor)
        return tensor, self._labels[row_position]

    @property
    def shape_spec(self) -> ModalityShape:
        return self._shape

    @property
    def item_shape(self) -> tuple[int, ...]:
        return self._shape.item_shape

    def class_counts(self) -> pd.Series:
        """Examples per class label, not files per class label."""
        return (
            self._df.groupby(RegistryColumns.CLASS_LABEL)[RegistryColumns.SAMPLES]
            .sum()
            .sort_index()
        )

    def get_class_name(self, label: int) -> str:
        return LABEL_TO_CLASS[label]

    def get_metadata(self, index: int) -> dict:
        """Full provenance of one example, for later analysis."""
        row = self._df.iloc[int(self._row_position[index])]
        metadata = row.to_dict()
        metadata["example_index"] = int(self._example_position[index])
        metadata["activity_class_name"] = LABEL_TO_CLASS[
            int(row[RegistryColumns.CLASS_LABEL])
        ]
        return metadata

    def _resolve_shape(self) -> ModalityShape:
        shapes = {
            tuple(shape) for shape in self._df[RegistryColumns.ARRAY_SHAPE]
        }
        item_shapes = {shape[1:] for shape in shapes}
        if len(item_shapes) != 1:
            raise ValueError(
                f"Registry mixes incompatible array geometries: {sorted(item_shapes)}"
            )
        return ModalityShape.from_array_shape(next(iter(shapes)))
