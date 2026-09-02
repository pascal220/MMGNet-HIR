"""
dataset_registry.py 

Now supports building separate registries per folder and exposes
a unified dual-folder build method.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union, cast

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
    ARRAY_SHAPE = "array_shape"
    FILE_SIZE_BYTES = "file_size_bytes"
    ARRAY_NBYTES = "array_nbytes"
    RESIDENT_BYTES = "resident_bytes"
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

# Arrays are cached as float32, so residency is costed at 4 bytes per element.
FLOAT32_ITEMSIZE = 4

# Stands in for NaN transition_info so pandas can join on the column.
NO_TRANSITION_KEY = "__none__"

MODALITIES: tuple[str, str] = ("IMU", "MMG")


# ---------------------------------------------------------------------------
# IMU/MMG pairing
# ---------------------------------------------------------------------------

class PairColumns:
    PAIR_KEY = "pair_key"
    SAMPLES = "samples"
    IMU_PATH = "imu_file_path"
    MMG_PATH = "mmg_file_path"
    PAIR_BYTES = "pair_bytes"


def build_modality_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Group registry rows into one row per IMU/MMG recording pair.

    Both modalities describe the same recording, so selection and dropping
    operate on pairs. That keeps the IMU and MMG models trained and tested
    on an identical set of recordings, which is what makes their scores
    comparable.
    """
    col = RegistryColumns
    if df.empty:
        return pd.DataFrame(
            columns=[
                PairColumns.PAIR_KEY, col.VOLUNTEER_ID, col.CLASS_LABEL,
                col.TRANSITION_INFO, col.FOLDER, PairColumns.SAMPLES,
                PairColumns.IMU_PATH, PairColumns.MMG_PATH,
                PairColumns.PAIR_BYTES,
            ]
        )

    work = df.copy()
    work[PairColumns.PAIR_KEY] = (
        work[col.VOLUNTEER_ID].astype(str)
        + "|" + work[col.CLASS_LABEL].astype(str)
        + "|" + work[col.TRANSITION_INFO].fillna(NO_TRANSITION_KEY).astype(str)
        + "|" + work[col.FOLDER].astype(str)
    )

    duplicated = work.duplicated([PairColumns.PAIR_KEY, col.MODALITY])
    if duplicated.any():
        raise ValueError(
            "Registry contains multiple files for the same pair key and "
            f"modality: {sorted(work.loc[duplicated, PairColumns.PAIR_KEY].unique())}"
        )

    sides = {
        modality: work[work[col.MODALITY] == modality].set_index(
            PairColumns.PAIR_KEY
        )
        for modality in MODALITIES
    }
    unmatched = sides["IMU"].index.symmetric_difference(sides["MMG"].index)
    if len(unmatched):
        raise ValueError(
            f"{len(unmatched)} recordings lack an IMU/MMG counterpart: "
            f"{sorted(unmatched)[:5]}"
        )

    imu, mmg = sides["IMU"], sides["MMG"].reindex(sides["IMU"].index)
    mismatched = imu[col.SAMPLES] != mmg[col.SAMPLES]
    if mismatched.any():
        raise ValueError(
            "IMU and MMG example counts differ for: "
            f"{sorted(imu.index[mismatched])[:5]}"
        )

    pairs = pd.DataFrame(
        {
            col.VOLUNTEER_ID: imu[col.VOLUNTEER_ID],
            col.CLASS_LABEL: imu[col.CLASS_LABEL],
            col.TRANSITION_INFO: imu[col.TRANSITION_INFO],
            col.FOLDER: imu[col.FOLDER],
            PairColumns.SAMPLES: imu[col.SAMPLES],
            PairColumns.IMU_PATH: imu[col.FILE_PATH],
            PairColumns.MMG_PATH: mmg[col.FILE_PATH],
            PairColumns.PAIR_BYTES: imu[col.RESIDENT_BYTES] + mmg[col.RESIDENT_BYTES],
        }
    )
    return pairs.reset_index().sort_values(PairColumns.PAIR_KEY).reset_index(drop=True)


def select_pair_rows(df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Return the registry rows belonging to the given pairs."""
    paths = set(pairs[PairColumns.IMU_PATH]) | set(pairs[PairColumns.MMG_PATH])
    return df[df[RegistryColumns.FILE_PATH].isin(paths)].reset_index(drop=True)


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
    ) -> pd.DataFrame:
        """
        Scan a single directory and return its registry DataFrame.

        Parameters
        ----------
        directory : str | Path
            Root directory to scan recursively.
        folder_tag : str
            Label stored in the FOLDER column (e.g. 'folder_1', 'folder_2').

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
            record = self._process_file(file_path, folder_tag)
            if record is not None:
                records.append(record)
            if file_count % 100 == 0:
                logger.debug(f"Processed {file_count} files from {folder_tag}")

        logger.info(f"Found {file_count} .npy files in {folder_tag}")
        df = pd.DataFrame(records)
        df = self._cast_dtypes(df)

        volunteer_count = df[RegistryColumns.VOLUNTEER_ID].nunique()
        logger.info(
            f"Registry '{folder_tag}' complete: {len(df)} files | "
            f"{volunteer_count} volunteers | "
            f"{df[RegistryColumns.SAMPLES].sum()} examples | "
            f"{df[RegistryColumns.RESIDENT_BYTES].sum() / (1024 ** 3):.2f} GiB resident"
        )

        return df

    def build_dual_folder(
        self,
        folder_1: Union[str, Path],
        folder_2: Union[str, Path],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build separate registries for the transitions and just_states folders.

        Returns
        -------
        df_folder_1 : pd.DataFrame
        df_folder_2 : pd.DataFrame
        """
        logger.info("Building dual-folder registries")
        logger.debug(f"Folder 1: {folder_1}")
        logger.debug(f"Folder 2: {folder_2}")

        df_1 = self.build_from_folder(folder_1, folder_tag="folder_1")
        df_2 = self.build_from_folder(folder_2, folder_tag="folder_2")

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

        The draw is stratified per transition value and its unit is an
        IMU/MMG pair: each group contributes
        ``max(1, ceil(test_fraction * n))`` pairs to the test set and the
        remainder to the training set. Rounding up guarantees every
        transition value reaches at least ``test_fraction`` of its pairs
        rather than falling short of it. Rows keep every registry column,
        so downstream bucketing uses class labels only.
        """
        if not 0 < test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1.")
        volunteer_id = self.normalize_volunteer_id(volunteer_id)
        candidates = self.get_valid_transitions(df, [volunteer_id])
        if candidates.empty:
            raise ValueError(
                f"No valid transitions rows found for volunteer {volunteer_id}."
            )

        pairs = build_modality_pairs(candidates)
        rng = np.random.default_rng(seed)
        train_keys: list[int] = []
        test_keys: list[int] = []

        for value, group in pairs.groupby(RegistryColumns.TRANSITION_INFO, sort=True):
            group = group.sort_values(PairColumns.PAIR_KEY)
            if len(group) < 2:
                raise ValueError(
                    f"Volunteer {volunteer_id}, transition '{value}' has only "
                    f"{len(group)} pair(s); at least 2 are required to form a "
                    "train/test split."
                )
            n_test = max(1, int(np.ceil(len(group) * test_fraction)))
            positions = rng.permutation(len(group))
            test_keys.extend(group.index[positions[:n_test]])
            train_keys.extend(group.index[positions[n_test:]])

        train_df = select_pair_rows(candidates, pairs.loc[sorted(train_keys)])
        test_df = select_pair_rows(candidates, pairs.loc[sorted(test_keys)])
        logger.info(
            "Transitions split for %s (seed=%d): %d train rows / %d examples, "
            "%d test rows / %d examples",
            volunteer_id, seed,
            len(train_df), train_df[RegistryColumns.SAMPLES].sum(),
            len(test_df), test_df[RegistryColumns.SAMPLES].sum(),
        )
        return train_df, test_df

    def match_just_states(
        self,
        transitions_df: pd.DataFrame,
        just_states_df: pd.DataFrame,
        ratio: float = 1.10,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Sample just_states rows to cap examples against transitions.

        For each (volunteer, class label) bucket, whole IMU/MMG pairs are
        drawn at random until adding another would exceed
        ``floor(ratio * transitions examples)``. The cap counts individual
        examples, not files: a just_states file holds far more examples than
        a transitions file, so a file-count cap would not bound imbalance.
        """
        if ratio <= 0:
            raise ValueError("ratio must be positive.")
        pool = just_states_df[
            just_states_df[RegistryColumns.FOLDER] == "folder_2"
        ]
        if transitions_df.empty or pool.empty:
            return pool.iloc[0:0].copy()

        transition_pairs = build_modality_pairs(transitions_df)
        pool_pairs = build_modality_pairs(pool)
        rng = np.random.default_rng(seed)
        bucket_keys = [RegistryColumns.VOLUNTEER_ID, RegistryColumns.CLASS_LABEL]
        chosen_keys: list[int] = []

        demand = transition_pairs.groupby(bucket_keys)[PairColumns.SAMPLES].sum()
        for key, examples in demand.items():
            volunteer, label = cast(tuple, key)
            budget = int(np.floor(examples * ratio))
            if budget <= 0:
                continue
            available = pool_pairs[
                (pool_pairs[RegistryColumns.VOLUNTEER_ID] == volunteer)
                & (pool_pairs[RegistryColumns.CLASS_LABEL] == label)
            ].sort_values(PairColumns.PAIR_KEY)
            if available.empty:
                logger.warning(
                    "No just_states data for %s class %d", volunteer, label
                )
                continue

            used = 0
            for position in rng.permutation(len(available)):
                candidate = available.iloc[int(position)]
                size = int(candidate[PairColumns.SAMPLES])
                if used + size > budget:
                    continue
                chosen_keys.append(cast(int, candidate.name))
                used += size
            if used < budget:
                logger.debug(
                    "just_states cap not filled for %s class %d: %d of %d "
                    "examples", volunteer, label, used, budget,
                )

        if not chosen_keys:
            return pool.iloc[0:0].copy()
        result = select_pair_rows(pool, pool_pairs.loc[sorted(chosen_keys)])
        logger.info(
            "Matched %d just_states examples to %d transitions examples "
            "(cap ratio=%.2f)",
            result[RegistryColumns.SAMPLES].sum(),
            transitions_df[RegistryColumns.SAMPLES].sum(),
            ratio,
        )
        return result

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
        folder_tag: str,
    ) -> Optional[dict]:
        try:
            metadata: FileMetadata = self._parser.parse(str(file_path))
        except ValueError as exc:
            logger.warning(f"Skipping '{file_path.name}': {exc}")
            return None

        array_info = self._read_array_info(file_path)
        if array_info is None:
            return None

        record = self._metadata_to_record(metadata, folder_tag)
        record.update(array_info)
        return record

    @staticmethod
    def _metadata_to_record(metadata: FileMetadata, folder_tag: str) -> dict:
        return {
            RegistryColumns.FILE_PATH: metadata.file_path,
            RegistryColumns.VOLUNTEER_ID: metadata.volunteer_id,
            RegistryColumns.MODALITY: metadata.modality,
            RegistryColumns.ACTIVITY_CLASS: metadata.activity_class,
            RegistryColumns.CLASS_LABEL: CLASS_TO_LABEL[metadata.activity_class],
            RegistryColumns.TRANSITION_INFO: metadata.transition_point,
            RegistryColumns.FOLDER: folder_tag,
        }

    @staticmethod
    def _read_array_info(file_path: Path) -> Optional[dict]:
        """Measure a file without materialising it.

        ``file_size_bytes`` comes from the filesystem and ``array_nbytes``
        from the .npy header, so neither figure is estimated from shapes.
        ``resident_bytes`` re-costs the same element count as float32,
        which is how arrays are held in memory.
        """
        try:
            array = np.load(file_path, mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            logger.warning(f"Could not read array header of '{file_path}': {exc}")
            return None

        shape = tuple(int(dim) for dim in array.shape)
        if len(shape) < 2:
            logger.warning(
                f"Skipping '{file_path.name}': expected a leading example "
                f"axis plus at least one feature axis, got shape {shape}"
            )
            return None

        nbytes = int(array.nbytes)
        elements = nbytes // array.dtype.itemsize
        return {
            RegistryColumns.SAMPLES: shape[0],
            RegistryColumns.ARRAY_SHAPE: shape,
            RegistryColumns.FILE_SIZE_BYTES: os.path.getsize(file_path),
            RegistryColumns.ARRAY_NBYTES: nbytes,
            RegistryColumns.RESIDENT_BYTES: elements * FLOAT32_ITEMSIZE,
        }

    @staticmethod
    def _cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        col = RegistryColumns
        dtype_map: dict[str, np.dtype] = {
            col.CLASS_LABEL: np.dtype("int8"),
            col.SAMPLES: np.dtype("int64"),
            col.FILE_SIZE_BYTES: np.dtype("int64"),
            col.ARRAY_NBYTES: np.dtype("int64"),
            col.RESIDENT_BYTES: np.dtype("int64"),
        }
        for column, dtype in dtype_map.items():
            if column in df.columns:
                df[column] = df[column].astype(dtype)
        return df