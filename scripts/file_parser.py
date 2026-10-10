"""
file_parser.py

Handles parsing of .npy filenames into structured metadata.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_MODALITIES = {"MMG", "IMU"}

VALID_CLASSES = {
    "sit", "stand", "walking",
    "sit_to_stand", "stand_to_sit",
    "stairs_up", "stairs_down",
}

TRANSITION_CLASSES = {
    "sit_to_stand", "stand_to_sit",
    "stairs_up", "stairs_down",
}

STEADY_STATE_CLASSES = {"sit", "stand", "walking"}

# Transition point suffixes found in the 'transitions' folder are numeric
# markers (percent/position through the transition), e.g. "0", "50", "100",
# "50m", "100m" - validated with a regex rather than a fixed set.
VALID_TRANSITION_POINT_PATTERN = re.compile(r"^\d+m?$")

# Amputee recordings cover five classes. Two extra filename classes record
# different ways of arriving at a stand and are trained as "stand".
AMPUTEE_CLASSES = {
    "sit", "stand", "walking",
    "sit_to_stand", "stand_to_sit",
}

AMPUTEE_CLASS_ALIASES: dict[str, str] = {
    "standin_to_stand": "stand",
    "walk_to_stand": "stand",
}

# ---------------------------------------------------------------------------
# Dataclass — structured metadata container
# ---------------------------------------------------------------------------


@dataclass
class FileMetadata:
    """Structured container for all metadata extracted from a .npy filename."""

    file_path: str
    volunteer_id: str                          # e.g. "N001"
    modality: str                              # "MMG" or "IMU"
    activity_class: str                        # e.g. "sit", "walk"
    is_transition_class: bool                  # True if sit-to-stand, etc.
    transition_point: Optional[str] = None     # e.g. "pre_transition"
    has_transition_info: bool = field(init=False)
    # Amputee-only: recording type ("type1"/"type2") and the class named in
    # the filename before aliasing (e.g. "walk_to_stand" for class "stand").
    data_type: Optional[str] = None
    source_class: Optional[str] = None

    def __post_init__(self):
        self.has_transition_info = self.transition_point is not None


# ---------------------------------------------------------------------------
# Parser Class
# ---------------------------------------------------------------------------

class FileNameParser:
    """
    Parses .npy filenames into structured FileMetadata objects.

    Expected filename formats:
        <prefix>_<volunteer_id>_<modality>_<class>.npy
        <prefix>_<volunteer_id>_<modality>_<class>_<transition_point>.npy

    Examples:
        trial01_N001_MMG_sit.npy
        trial01_N001_IMU_sit-to-stand_pre_transition.npy
    """

    # Regex: captures volunteer ID (N0XX), modality, class, and optional
    # transition point from the filename stem
    _PATTERN = re.compile(
        r".*?(N0\d+)"                          # volunteer ID
        r"_(MMG|IMU)"                          # modality
        r"_([\w-]+?)"                          # activity class
        r"(?:_(\d+m?))?"                       # optional transition point
        r"$",
        re.IGNORECASE
    )

    def parse(self, file_path: str) -> FileMetadata:
        """
        Parse a single file path into a FileMetadata object.

        Parameters
        ----------
        file_path : str
            Full or relative path to the .npy file.

        Returns
        -------
        FileMetadata
            Populated metadata object.

        Raises
        ------
        ValueError
            If the filename does not match the expected pattern or contains
            unrecognised modality / class values.
        """
        logger.debug(f"Parsing file: {file_path}")
        stem = os.path.splitext(os.path.basename(file_path))[0]
        logger.debug(f"Filename stem: {stem}")
        match = self._PATTERN.match(stem)

        if not match:
            logger.error(f"Filename does not match expected pattern: {stem}")
            raise ValueError(
                f"Filename '{stem}' does not match the expected naming convention."
            )

        volunteer_id = match.group(1).upper()
        modality = match.group(2).upper()
        activity_class = match.group(3).lower()
        transition_point_raw = match.group(4)

        logger.debug(f"Extracted: volunteer={volunteer_id}, modality={modality}, class={activity_class}, transition={transition_point_raw}")

        self._validate_modality(modality, stem)
        self._validate_class(activity_class, stem)

        transition_point = self._resolve_transition_point(
            transition_point_raw, activity_class, stem
        )
        logger.debug(f"Resolved transition point: {transition_point}")

        metadata = FileMetadata(
            file_path=file_path,
            volunteer_id=volunteer_id,
            modality=modality,
            activity_class=activity_class,
            is_transition_class=activity_class in TRANSITION_CLASSES,
            transition_point=transition_point,
        )
        logger.debug("FileMetadata created successfully")
        return metadata

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_modality(modality: str, stem: str) -> None:
        logger.debug(f"Validating modality: {modality}")
        if modality not in VALID_MODALITIES:
            logger.error(f"Invalid modality '{modality}' in file '{stem}'")
            raise ValueError(
                f"Unrecognised modality '{modality}' in file '{stem}'. "
                f"Expected one of {VALID_MODALITIES}."
            )
        logger.debug(f"Modality '{modality}' is valid")

    @staticmethod
    def _validate_class(activity_class: str, stem: str) -> None:
        logger.debug(f"Validating activity class: {activity_class}")
        if activity_class not in VALID_CLASSES:
            logger.error(f"Invalid activity class '{activity_class}' in file '{stem}'")
            raise ValueError(
                f"Unrecognised activity class '{activity_class}' in file "
                f"'{stem}'. Expected one of {VALID_CLASSES}."
            )
        logger.debug(f"Activity class '{activity_class}' is valid")

    @staticmethod
    def _resolve_transition_point(
        raw: Optional[str],
        activity_class: str,
        stem: str
    ) -> Optional[str]:
        """
        Validate and return the transition point string if present.
        Only transition-class files should carry transition point info.
        """
        if raw is None:
            logger.debug("No transition point provided")
            return None

        normalised = raw.lower()
        logger.debug(f"Validating transition point: {normalised}")

        if not VALID_TRANSITION_POINT_PATTERN.match(normalised):
            logger.error(f"Invalid transition point '{raw}' in file '{stem}'")
            raise ValueError(
                f"Unrecognised transition point '{raw}' in file '{stem}'. "
                f"Expected a numeric marker matching {VALID_TRANSITION_POINT_PATTERN.pattern}."
            )

        logger.debug(f"Transition point '{normalised}' is valid")
        return normalised

class AmputeeFileNameParser(FileNameParser):
    """
    Parses amputee .npy filenames into FileMetadata objects.

    Expected filename formats:
        <prefix>_<amputee_id>_<data_type>_<modality>_<class>.npy
        <prefix>_<amputee_id>_<data_type>_<modality>_<class>_<transition_point>.npy

    Examples:
        Last_Series_A003_type1_IMU_sit_50m.npy
        Last_Series_Wavelet_A003_type2_MMG_walk_to_stand_100.npy

    Aliased classes (see ``AMPUTEE_CLASS_ALIASES``) are stored under their
    target class; the filename class is kept in ``source_class``.
    """

    _PATTERN = re.compile(
        r"(?:^|.*_)(A\d+)"                     # amputee ID
        r"_(type\d+)"                          # recording type
        r"_(MMG|IMU)"                          # modality
        r"_([a-z_]+?)"                         # activity class
        r"(?:_(\d+m?))?"                       # optional transition point
        r"$",
        re.IGNORECASE
    )

    def parse(self, file_path: str) -> FileMetadata:
        stem = os.path.splitext(os.path.basename(file_path))[0]
        match = self._PATTERN.match(stem)
        if not match:
            raise ValueError(
                f"Filename '{stem}' does not match the amputee naming convention."
            )

        amputee_id = match.group(1).upper()
        data_type = match.group(2).lower()
        modality = match.group(3).upper()
        source_class = match.group(4).lower()
        activity_class = AMPUTEE_CLASS_ALIASES.get(source_class, source_class)

        self._validate_modality(modality, stem)
        if activity_class not in AMPUTEE_CLASSES:
            raise ValueError(
                f"Unrecognised amputee activity class '{source_class}' in file "
                f"'{stem}'. Expected one of "
                f"{sorted(AMPUTEE_CLASSES | set(AMPUTEE_CLASS_ALIASES))}."
            )

        return FileMetadata(
            file_path=file_path,
            volunteer_id=amputee_id,
            modality=modality,
            activity_class=activity_class,
            is_transition_class=activity_class in TRANSITION_CLASSES,
            transition_point=self._resolve_transition_point(
                match.group(5), activity_class, stem
            ),
            data_type=data_type,
            source_class=source_class,
        )
