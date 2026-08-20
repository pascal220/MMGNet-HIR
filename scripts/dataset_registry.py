"""
dataset_registry.py 

Now supports building separate registries per folder and exposes
a unified dual-folder build method.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from file_parser import FileMetadata, FileNameParser

logger = logging.getLogger(__name__)


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
    TRANSITION_INFO = "transition_info"
    SAMPLES = "samples"
    NO_WINDOWS = "no_windows"
    HEIGHT = "height"
    WIDTH = "width"
    CHANNELS = "channels"
    FOLDER = "folder"                  


CLASS_TO_LABEL: dict[str, int] = {
    "sit": 0,
    "stand": 1,
    "walking": 2,
    "sit_to_stand": 3,
    "stand_to_sit": 4,
    "stairs_up": 5,
    "stairs_down": 6,
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
        logger.info(f"Building registry for {folder_tag} from {directory}")
        directory = Path(directory)

        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            raise FileNotFoundError(f"Directory not found: {directory}")

        logger.debug("Scanning directory recursively for .npy files")
        records = []
        file_count = 0

        for file_path in sorted(directory.rglob("*.npy")):
            file_count += 1
            record = self._process_file(file_path, load_shapes, folder_tag)
            if record is not None:
                records.append(record)
            if file_count % 100 == 0:
                logger.debug(f"Processed {file_count} files from {folder_tag}")

        logger.info(f"Found {file_count} .npy files in {folder_tag}")
        df = pd.DataFrame(records)
        df = self._cast_dtypes(df)
        df = self._add_file_size_column(df)

        volunteer_count = df[RegistryColumns.VOLUNTEER_ID].nunique()
        logger.info(
            f"Registry '{folder_tag}' complete: {len(df)} files | "
            f"{volunteer_count} volunteers"
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
        logger.info("Building dual-folder registries")
        logger.debug(f"Folder 1: {folder_1}")
        logger.debug(f"Folder 2: {folder_2}")
        
        df_1 = self.build_from_folder(folder_1, folder_tag="folder_1",
                                       load_shapes=load_shapes)
        df_2 = self.build_from_folder(folder_2, folder_tag="folder_2",
                                       load_shapes=load_shapes)

        self._df = pd.concat([df_1, df_2], ignore_index=True)
        logger.info(f"Dual-folder registries complete: {len(df_1)} + {len(df_2)} files")

        return df_1, df_2

    def filter_by_modality(self, df: pd.DataFrame, modality: str) -> pd.DataFrame:
        logger.debug(f"Filtering {len(df)} samples by modality '{modality}'")
        filtered = df[
            df[RegistryColumns.MODALITY] == modality.upper()
        ].reset_index(drop=True)
        logger.debug(f"Filtered result: {len(filtered)} samples")
        return filtered

    def filter_by_volunteer(self, df: pd.DataFrame, volunteer_id: str) -> pd.DataFrame:
        volunteer_id = self.normalize_volunteer_id(volunteer_id)
        logger.debug(f"Filtering {len(df)} samples by volunteer '{volunteer_id}'")
        filtered = df[
            df[RegistryColumns.VOLUNTEER_ID] == volunteer_id
        ].reset_index(drop=True)
        logger.debug(f"Filtered result: {len(filtered)} samples")
        return filtered

    @staticmethod
    def normalize_volunteer_id(volunteer_id: Union[int, str]) -> str:
        """Return a canonical volunteer identifier such as ``N004``."""
        if isinstance(volunteer_id, (int, np.integer)):
            if volunteer_id < 0:
                raise ValueError("Volunteer number must be non-negative.")
            return f"N{int(volunteer_id):03d}"

        if not isinstance(volunteer_id, str):
            raise TypeError("volunteer_id must be an integer or string.")

        value = volunteer_id.strip().upper()
        if value.isdigit():
            return f"N{int(value):03d}"
        if value.startswith("N") and value[1:].isdigit():
            return f"N{int(value[1:]):03d}"
        raise ValueError(
            f"Invalid volunteer ID '{volunteer_id}'. Use an integer or an ID such as N004."
        )

    @staticmethod
    def _validate_registry(df: pd.DataFrame) -> None:
        """Validate the columns required by volunteer-selection methods."""
        required = {
            RegistryColumns.VOLUNTEER_ID,
            RegistryColumns.CLASS_LABEL,
            RegistryColumns.FILE_PATH,
        }
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Registry is missing required columns: {sorted(missing)}")
        if df.empty:
            raise ValueError("Cannot select volunteers from an empty registry.")

    def select_volunteers_split(
        self,
        df: pd.DataFrame,
        train_count: int,
        test_count: int,
        seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Randomly split distinct volunteers into train and test registries."""
        self._validate_registry(df)
        if not isinstance(train_count, (int, np.integer)) or train_count < 1:
            raise ValueError("train_count must be a positive integer.")
        if not isinstance(test_count, (int, np.integer)) or test_count < 1:
            raise ValueError("test_count must be a positive integer.")

        volunteers = np.array(
            sorted(df[RegistryColumns.VOLUNTEER_ID].dropna().unique())
        )
        required_count = int(train_count) + int(test_count)
        if required_count > len(volunteers):
            raise ValueError(
                f"Requested {required_count} volunteers, but only "
                f"{len(volunteers)} are available: {volunteers.tolist()}"
            )

        selected = np.random.default_rng(seed).permutation(volunteers)
        train_ids = selected[:train_count]
        test_ids = selected[train_count:required_count]
        logger.info("Volunteer split (seed=%d): train=%s, test=%s", seed,
                    train_ids.tolist(), test_ids.tolist())

        train_df = df[df[RegistryColumns.VOLUNTEER_ID].isin(train_ids)].copy()
        test_df = df[df[RegistryColumns.VOLUNTEER_ID].isin(test_ids)].copy()
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def select_test_coverage(
        self,
        df: pd.DataFrame,
        volunteer_ids: list[str],
        minimum_transition_infos: int = 5,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Select mandatory test rows for every class and volunteer.

        One row is selected for each distinct ``transition_info`` value.  The
        selection is restricted to the transitions folder, whose rows are the
        mandatory core records in this project.
        """
        self._validate_registry(df)
        if minimum_transition_infos < 1:
            raise ValueError("minimum_transition_infos must be positive.")
        if RegistryColumns.TRANSITION_INFO not in df.columns:
            raise ValueError("Registry does not contain transition_info metadata.")

        volunteers = [self.normalize_volunteer_id(value) for value in volunteer_ids]
        candidates = df[
            (df[RegistryColumns.VOLUNTEER_ID].isin(volunteers))
            & (df[RegistryColumns.FOLDER] == "folder_1")
            & (df[RegistryColumns.TRANSITION_INFO].notna())
        ]
        rng = np.random.default_rng(seed)
        selected_parts: list[pd.DataFrame] = []

        for volunteer_id in volunteers:
            volunteer_df = candidates[
                candidates[RegistryColumns.VOLUNTEER_ID] == volunteer_id
            ]
            for class_label in sorted(CLASS_TO_LABEL.values()):
                class_df = volunteer_df[
                    volunteer_df[RegistryColumns.CLASS_LABEL] == class_label
                ]
                transition_values = sorted(
                    class_df[RegistryColumns.TRANSITION_INFO].unique()
                )
                if len(transition_values) < minimum_transition_infos:
                    raise ValueError(
                        f"Volunteer {volunteer_id}, class {class_label} has "
                        f"only {len(transition_values)} distinct transition_info "
                        f"values; {minimum_transition_infos} are required."
                    )

                chosen_values = rng.choice(
                    transition_values,
                    size=minimum_transition_infos,
                    replace=False,
                )
                for transition_value in chosen_values:
                    rows = class_df[
                        class_df[RegistryColumns.TRANSITION_INFO] == transition_value
                    ].sort_values(RegistryColumns.FILE_PATH)
                    selected_parts.append(rows.iloc[[0]])

        if not selected_parts:
            raise ValueError("No transition test-coverage rows were found.")
        result = pd.concat(selected_parts, ignore_index=True)
        logger.info(
            "Selected %d mandatory test rows (%d per volunteer minimum)",
            len(result),
            minimum_transition_infos * df[RegistryColumns.CLASS_LABEL].nunique(),
        )
        return result

    def get_transition_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug(f"Extracting transition samples from {len(df)} samples")
        filtered = df[
            df[RegistryColumns.TRANSITION_INFO].notna()
        ].reset_index(drop=True)
        logger.debug(f"Found {len(filtered)} transition samples")
        return filtered

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
            logger.debug(f"Parsed {file_path.name}: {metadata.volunteer_id} {metadata.modality} {metadata.activity_class}")
            record = self._metadata_to_record(metadata, folder_tag)

            if load_shapes:
                shape = self._load_shape(file_path)
                
                if shape is not None:
                    record.update(self._shape_to_record(shape))
                    logger.debug(f"Shape loaded for {file_path.name}: {shape}")

            return record

        except ValueError as exc:
            logger.warning(f"Skipping '{file_path.name}': {exc}")
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
            RegistryColumns.TRANSITION_INFO: metadata.transition_point,
            RegistryColumns.FOLDER: folder_tag,
        }

    @staticmethod
    def _load_shape(file_path: Path) -> Optional[tuple]:
        try:
            import numpy as np
            array = np.load(file_path, mmap_mode="r", allow_pickle=False)
            return tuple(array.shape)
        except Exception as exc:
            print(f"[DatasetRegistry] Could not read shape of '{file_path}': {exc}")
            return None

    @staticmethod
    def _shape_to_record(shape: tuple) -> dict:
        """
        Shapes are (samples, ..., height, width); any axes between the
        leading sample axis and the trailing two spatial axes are folded
        into a single channel count (see datasets.py:_to_tensor).
        """
        if len(shape) < 3:
            return {}
        samples = shape[0]
        no_windows = shape[1]
        width, channels = shape[-2], shape[-1]
        if len(shape) > 4:
            height = shape[2]
        else:
            height = 1
            
        return {
            RegistryColumns.WIDTH: width,
            RegistryColumns.HEIGHT: height,
            RegistryColumns.CHANNELS: channels,
            RegistryColumns.SAMPLES: samples,
            RegistryColumns.NO_WINDOWS: no_windows,
        }

    @staticmethod
    def _cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        col = RegistryColumns
        dtype_map = {
            col.CLASS_LABEL: "int8",
            col.IS_TRANSITION_CLASS: "bool",
        }
        for column, dtype in dtype_map.items():
            if column in df.columns:
                df[column] = df[column].astype(dtype)
        return df

    @staticmethod
    def _add_file_size_column(df: pd.DataFrame) -> pd.DataFrame:
        """Add the actual on-disk size of each registered file in bytes.

        Shape-based estimates are unsuitable here: they can omit dimensions,
        assume the wrong dtype, and do not include the NumPy file header.
        """
        if df.empty:
            df["file_size_bytes"] = pd.Series(dtype="int64")
            return df

        df["file_size_bytes"] = df[RegistryColumns.FILE_PATH].apply(os.path.getsize)
        return df