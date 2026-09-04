"""
datasets.py

One PyTorch item per recorded sample. IMU and MMG stay separate, since
they train separate models; any fusion happens inside the model's input
layers rather than here.

A split is held as a single resident tensor of shape (samples, *item)
together with a row-aligned label tensor and metadata frame. A tensor
cannot carry strings such as volunteer IDs or class names, so metadata
travels beside the data at the same row position rather than inside it.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from dataset_registry import (
    LABEL_TO_CLASS,
    MODALITY_PATH_COLUMN,
    RegistryColumns,
    SampleColumns,
)
from memory_manager import load_samples

logger = logging.getLogger(__name__)

SOURCE_FILE = "source_file"
SOURCE_SAMPLE_INDEX = "source_sample_index"
ACTIVITY_CLASS_NAME = "activity_class_name"

METADATA_COLUMNS = [
    RegistryColumns.VOLUNTEER_ID,
    ACTIVITY_CLASS_NAME,
    RegistryColumns.CLASS_LABEL,
    RegistryColumns.TRANSITION_INFO,
    RegistryColumns.FOLDER,
    SOURCE_FILE,
    SOURCE_SAMPLE_INDEX,
]


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
        """Shape of a single sample; batches arrive as (batch, *item_shape)."""
        if self.scales is None:
            return (self.windows, self.time_steps, self.channels)
        return (self.windows, self.scales, self.time_steps, self.channels)

    @classmethod
    def from_item_shape(cls, item: tuple[int, ...]) -> "ModalityShape":
        """Build a descriptor from the shape of one sample."""
        item = tuple(int(dim) for dim in item)
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
            f"Unsupported item shape {item}: expected 3 axes (IMU) or 4 axes (MMG)."
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


@dataclass(frozen=True)
class ModalityTensors:
    """One resident split of one modality.

    ``data``, ``labels`` and ``metadata`` share a row order: row ``i`` of
    each describes the same sample. Row ``i`` also matches row ``i`` of the
    other modality's bundle for the same split, because both are built from
    the same sample table.
    """

    modality: str
    data: Tensor
    labels: Tensor
    metadata: pd.DataFrame

    def __post_init__(self) -> None:
        if not (len(self.data) == len(self.labels) == len(self.metadata)):
            raise ValueError(
                "data, labels and metadata must have the same number of rows: "
                f"{len(self.data)}, {len(self.labels)}, {len(self.metadata)}"
            )

    @classmethod
    def from_samples(cls, samples: pd.DataFrame, modality: str) -> "ModalityTensors":
        """Read the selected samples and attach their metadata."""
        modality = modality.upper()
        if modality not in MODALITY_PATH_COLUMN:
            raise ValueError(f"Unknown modality '{modality}'.")
        if samples.empty:
            raise ValueError(f"No samples selected for modality {modality}.")

        samples = samples.reset_index(drop=True)
        data = load_samples(samples, modality)
        labels = torch.from_numpy(
            samples[RegistryColumns.CLASS_LABEL].to_numpy(dtype="int64")
        )

        metadata = pd.DataFrame(
            {
                RegistryColumns.VOLUNTEER_ID: samples[RegistryColumns.VOLUNTEER_ID],
                ACTIVITY_CLASS_NAME: samples[RegistryColumns.CLASS_LABEL].map(
                    LABEL_TO_CLASS
                ),
                RegistryColumns.CLASS_LABEL: samples[RegistryColumns.CLASS_LABEL],
                RegistryColumns.TRANSITION_INFO: samples[
                    RegistryColumns.TRANSITION_INFO
                ],
                RegistryColumns.FOLDER: samples[RegistryColumns.FOLDER],
                SOURCE_FILE: samples[MODALITY_PATH_COLUMN[modality]],
                SOURCE_SAMPLE_INDEX: samples[SampleColumns.SAMPLE_INDEX],
            }
        )[METADATA_COLUMNS]

        bundle = cls(
            modality=modality, data=data, labels=labels, metadata=metadata
        )
        logger.info(
            "%s tensor ready: %s | %.2f GiB | %d source files",
            modality,
            tuple(data.shape),
            bundle.nbytes / 1024 ** 3,
            samples[MODALITY_PATH_COLUMN[modality]].nunique(),
        )
        return bundle

    @property
    def nbytes(self) -> int:
        return self.data.nelement() * self.data.element_size()

    @property
    def shape_spec(self) -> ModalityShape:
        return ModalityShape.from_item_shape(tuple(self.data.shape[1:]))

    @property
    def item_shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.data.shape[1:])

    def class_counts(self) -> pd.Series:
        """Samples per class label."""
        return (
            self.metadata.groupby(RegistryColumns.CLASS_LABEL)
            .size()
            .sort_index()
        )


class SingleModalityDataset(Dataset):
    """Exposes every resident sample of one modality as an individual item.

    Parameters
    ----------
    tensors : ModalityTensors
        A resident split of one modality.
    transform : Callable, optional
        Applied to the sample tensor.
    """

    def __init__(
        self,
        tensors: ModalityTensors,
        transform: Optional[Callable] = None,
    ):
        self._tensors = tensors
        self._transform = transform

        logger.info(
            "%s created: %d %s samples | item shape %s (%s)",
            self.__class__.__name__,
            len(self),
            tensors.modality,
            tensors.item_shape,
            tensors.shape_spec.describe(),
        )

    def __len__(self) -> int:
        return int(self._tensors.data.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        item = self._tensors.data[index]
        if self._transform is not None:
            item = self._transform(item)
        return item, self._tensors.labels[index]

    @property
    def tensors(self) -> ModalityTensors:
        return self._tensors

    @property
    def shape_spec(self) -> ModalityShape:
        return self._tensors.shape_spec

    @property
    def item_shape(self) -> tuple[int, ...]:
        return self._tensors.item_shape

    def class_counts(self) -> pd.Series:
        return self._tensors.class_counts()

    def get_class_name(self, label: int) -> str:
        return LABEL_TO_CLASS[label]

    def get_metadata(self, index: int) -> dict:
        """Full provenance of one sample, for later analysis."""
        return self._tensors.metadata.iloc[int(index)].to_dict()
