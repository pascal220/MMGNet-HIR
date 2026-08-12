"""
dataset_registry.py 

Now supports building separate registries per folder and exposes
a unified dual-folder build method.
"""

import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from file_parser import FileMetadata, FileNameParser


# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------

class RegistryColumns:
    FILE_PATH = "file_path"
    VOLUNTEER_ID = "volunteer_id"
    MODALITY = "modality"
    ACTIVITY_CLASS = "activity_class"
    CLASS_LABEL = "class_label"
    IS_TRANSITION_CLASS = "is_transition_class"
    TRANSITION_POINT = "transition_point"
    HAS_TRANSITION_INFO = "has_transition_info"
    WIDTH = "width"
    HEIGHT = "height"
    CHANNELS = "channels"
    SAMPLES = "samples"
    IS_1D = "is_1d"
    FOLDER = "folder"                  # ← NEW: tracks which folder a file came from


CLASS_TO_LABEL: dict[str, int] = {
    "sit": 0,
    "stand": 1,
    "walk": 2,
    "sit-to-stand": 3,
    "stand-to-sit": 4,
    "stair-ascent": 5,
    "stair-descent": 6,
}

LABEL_TO_CLASS: dict[int, str] = {v: k for k, v in CLASS_TO_LABEL.items()}


# ---------------------------------------------------------------------------
# Registry Builder
# ---------------------------------------------------------------------------

class DatasetRegistry:
    """
    Scans data directories and builds Pandas DataFrame registries.

    Supports a dual-folder workflow where folder 1 is always fully
    loaded and folder 2 is loaded up to a configurable memory limit.

    Parameters
    ----------
    parser : FileNameParser, optional
        Custom parser instance.
    """

    def __init__(self, parser: Optional[FileNameParser] = None):
        self._parser = parser or FileNameParser()
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_from_folder(
        self,
        directory: Union[str, Path],
        folder_tag: str,
        load_shapes: bool = True,
    ) -> pd.DataFrame:
        """
        Scan a single directory and return its registry DataFrame.

        Parameters
        ----------
        directory : str | Path
            Root directory to scan recursively.
        folder_tag : str
            Label stored in the FOLDER column (e.g. 'folder_1', 'folder_2').
        load_shapes : bool
            If True, reads array shapes from .npy file headers.

        Returns
        -------
        pd.DataFrame
            Registry for this folder.
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        records = []

        for file_path in sorted(directory.rglob("*.npy")):
            record = self._process_file(file_path, load_shapes, folder_tag)
            if record is not None:
                records.append(record)

        df = pd.DataFrame(records)
        df = self._cast_dtypes(df)

        print(
            f"[DatasetRegistry] '{folder_tag}' | {len(df)} files | "
            f"{df[RegistryColumns.VOLUNTEER_ID].nunique()} volunteers"
        )

        return df

    def build_dual_folder(
        self,
        folder_1: Union[str, Path],
        folder_2: Union[str, Path],
        load_shapes: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build separate registries for two folders.

        Folder 1 is intended to be fully loaded into memory.
        Folder 2 is loaded up to a configurable memory budget.

        Parameters
        ----------
        folder_1 : str | Path
            Primary data directory (always fully loaded).
        folder_2 : str | Path
            Secondary data directory (loaded up to memory limit).
        load_shapes : bool
            If True, reads array shapes from .npy file headers.

        Returns
        -------
        df_folder_1 : pd.DataFrame
        df_folder_2 : pd.DataFrame
        """
        df_1 = self.build_from_folder(folder_1, folder_tag="folder_1",
                                       load_shapes=load_shapes)
        df_2 = self.build_from_folder(folder_2, folder_tag="folder_2",
                                       load_shapes=load_shapes)

        self._df = pd.concat([df_1, df_2], ignore_index=True)

        return df_1, df_2

    def filter_by_modality(self, df: pd.DataFrame, modality: str) -> pd.DataFrame:
        return df[
            df[RegistryColumns.MODALITY] == modality.upper()
        ].reset_index(drop=True)

    def filter_by_volunteer(self, df: pd.DataFrame, volunteer_id: str) -> pd.DataFrame:
        return df[
            df[RegistryColumns.VOLUNTEER_ID] == volunteer_id.upper()
        ].reset_index(drop=True)

    def get_transition_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[
            df[RegistryColumns.HAS_TRANSITION_INFO]
        ].reset_index(drop=True)

    def summary(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [
            RegistryColumns.FOLDER,
            RegistryColumns.MODALITY,
            RegistryColumns.ACTIVITY_CLASS,
        ]
        return (
            df.groupby(cols)
            .size()
            .reset_index(name="count")
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_file(
        self,
        file_path: Path,
        load_shapes: bool,
        folder_tag: str,
    ) -> Optional[dict]:
        try:
            metadata: FileMetadata = self._parser.parse(str(file_path))
            record = self._metadata_to_record(metadata, folder_tag)

            if load_shapes:
                shape = self._load_shape(file_path)
                if shape is not None:
                    record.update(self._shape_to_record(shape))

            return record

        except ValueError as exc:
            print(f"[DatasetRegistry] Skipping '{file_path.name}': {exc}")
            return None

    @staticmethod
    def _metadata_to_record(metadata: FileMetadata, folder_tag: str) -> dict:
        return {
            RegistryColumns.FILE_PATH: metadata.file_path,
            RegistryColumns.VOLUNTEER_ID: metadata.volunteer_id,
            RegistryColumns.MODALITY: metadata.modality,
            RegistryColumns.ACTIVITY_CLASS: metadata.activity_class,
            RegistryColumns.CLASS_LABEL: CLASS_TO_LABEL[metadata.activity_class],
            RegistryColumns.IS_TRANSITION_CLASS: metadata.is_transition_class,
            RegistryColumns.TRANSITION_POINT: metadata.transition_point,
            RegistryColumns.HAS_TRANSITION_INFO: metadata.has_transition_info,
            RegistryColumns.FOLDER: folder_tag,
        }

    @staticmethod
    def _load_shape(file_path: Path) -> Optional[tuple]:
        try:
            import numpy as np
            with open(file_path, "rb") as f:
                version = np.lib.format.read_magic(f)
                shape, _, _ = np.lib.format._read_array_header(f, version)
            return shape
        except Exception as exc:
            print(f"[DatasetRegistry] Could not read shape of '{file_path}': {exc}")
            return None

    @staticmethod
    def _shape_to_record(shape: tuple) -> dict:
        if len(shape) != 4:
            return {}
        width, height, channels, samples = shape
        return {
            RegistryColumns.WIDTH: width,
            RegistryColumns.HEIGHT: height,
            RegistryColumns.CHANNELS: channels,
            RegistryColumns.SAMPLES: samples,
            RegistryColumns.IS_1D: height == 1,
        }

    @staticmethod
    def _cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        col = RegistryColumns
        dtype_map = {
            col.CLASS_LABEL: "int8",
            col.IS_TRANSITION_CLASS: "bool",
            col.HAS_TRANSITION_INFO: "bool",
        }
        for column, dtype in dtype_map.items():
            if column in df.columns:
                df[column] = df[column].astype(dtype)
        if col.IS_1D in df.columns:
            df[col.IS_1D] = df[col.IS_1D].astype("bool")
        return df