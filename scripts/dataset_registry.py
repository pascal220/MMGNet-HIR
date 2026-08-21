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

# The only transition markers allowed in the transitions folder.
VALID_TRANSITION_VALUES: frozenset[str] = frozenset({"100m", "50m", "0", "50", "100"})


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

    def get_valid_transitions(
        self,
        df: pd.DataFrame,
        volunteer_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Return transitions-folder rows with a valid transition value.

        Rows in folder_1 whose ``transition_info`` is not one of the five
        expected values are dropped with a warning.
        """
        self._validate_registry(df)
        if RegistryColumns.TRANSITION_INFO not in df.columns:
            raise ValueError("Registry does not contain transition_info metadata.")

        in_folder = df[RegistryColumns.FOLDER] == "folder_1"
        valid = df[RegistryColumns.TRANSITION_INFO].isin(VALID_TRANSITION_VALUES)
        invalid = df[in_folder & ~valid]
        if not invalid.empty:
            logger.warning(
                "Dropping %d transitions rows with unexpected transition "
                "values: %s",
                len(invalid),
                sorted(
                    invalid[RegistryColumns.TRANSITION_INFO].dropna().unique()
                ),
            )

        mask = in_folder & valid
        if volunteer_ids is not None:
            normalized = [self.normalize_volunteer_id(v) for v in volunteer_ids]
            mask &= df[RegistryColumns.VOLUNTEER_ID].isin(normalized)
        return df[mask].copy()

    def split_transitions_by_fraction(
        self,
        df: pd.DataFrame,
        volunteer_id: Union[int, str],
        test_fraction: float = 0.10,
        seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split one volunteer's transitions rows into train and test sets.

        The draw is stratified per (modality, transition value): each group
        contributes ``max(1, floor(test_fraction * n))`` rows to the test
        set and the remainder to the training set. Rows keep every registry
        column, so downstream bucketing uses class labels only.
        """
        if not 0 < test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1.")
        volunteer_id = self.normalize_volunteer_id(volunteer_id)
        candidates = self.get_valid_transitions(df, [volunteer_id])
        if candidates.empty:
            raise ValueError(
                f"No valid transitions rows found for volunteer {volunteer_id}."
            )

        rng = np.random.default_rng(seed)
        train_parts: list[pd.DataFrame] = []
        test_parts: list[pd.DataFrame] = []
        group_keys = [RegistryColumns.MODALITY, RegistryColumns.TRANSITION_INFO]

        for (modality, value), group in candidates.groupby(group_keys, sort=True):
            group = group.sort_values(RegistryColumns.FILE_PATH)
            if len(group) < 2:
                raise ValueError(
                    f"Volunteer {volunteer_id}, {modality} transition "
                    f"'{value}' has only {len(group)} row(s); at least 2 "
                    "are required to form a train/test split."
                )
            n_test = max(1, int(np.floor(len(group) * test_fraction)))
            test_idx = rng.choice(
                group.index.to_numpy(), size=n_test, replace=False
            )
            test_parts.append(group.loc[np.sort(test_idx)])
            train_parts.append(group.drop(index=test_idx))

        train_df = pd.concat(train_parts, ignore_index=True)
        test_df = pd.concat(test_parts, ignore_index=True)
        logger.info(
            "Transitions split for %s (seed=%d): %d train rows, %d test rows",
            volunteer_id, seed, len(train_df), len(test_df),
        )
        return train_df, test_df

    def match_just_states(
        self,
        transitions_df: pd.DataFrame,
        just_states_df: pd.DataFrame,
        ratio: float = 1.10,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Sample just_states rows to match transitions counts per bucket.

        For each (volunteer, class label, modality) bucket present in
        ``transitions_df``, ``floor(ratio * transitions count)`` rows are
        drawn at random from ``just_states_df``. If fewer rows are
        available, all of them are used and a warning is logged.
        """
        if ratio <= 0:
            raise ValueError("ratio must be positive.")
        pool = just_states_df[
            just_states_df[RegistryColumns.FOLDER] == "folder_2"
        ]
        if transitions_df.empty or pool.empty:
            return pool.iloc[0:0].copy()

        rng = np.random.default_rng(seed)
        bucket_keys = [
            RegistryColumns.VOLUNTEER_ID,
            RegistryColumns.CLASS_LABEL,
            RegistryColumns.MODALITY,
        ]
        parts: list[pd.DataFrame] = []

        for (volunteer, label, modality), count in (
            transitions_df.groupby(bucket_keys).size().items()
        ):
            target = int(np.floor(count * ratio))
            if target == 0:
                continue
            available = pool[
                (pool[RegistryColumns.VOLUNTEER_ID] == volunteer)
                & (pool[RegistryColumns.CLASS_LABEL] == label)
                & (pool[RegistryColumns.MODALITY] == modality)
            ].sort_values(RegistryColumns.FILE_PATH)
            if len(available) <= target:
                if len(available) < target:
                    logger.warning(
                        "just_states shortfall for %s class %d %s: wanted "
                        "%d rows, only %d available",
                        volunteer, label, modality, target, len(available),
                    )
                parts.append(available)
                continue
            chosen = rng.choice(
                available.index.to_numpy(), size=target, replace=False
            )
            parts.append(available.loc[np.sort(chosen)])

        if not parts:
            return pool.iloc[0:0].copy()
        result = pd.concat(parts, ignore_index=True)
        logger.info(
            "Matched %d just_states rows to %d transitions rows (ratio=%.2f)",
            len(result), len(transitions_df), ratio,
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