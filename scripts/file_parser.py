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

# Note: "stair_up"/"stair_down" (IMU files) and "stairs_up"/"stairs_down"
# (MMG files) both occur in the dataset - the source data itself is
# inconsistent about the singular/plural spelling.
VALID_CLASSES = {
    "sit", "stand", "walking",
    "sit_to_stand", "stand_to_sit",
    "stair_up", "stair_down",
    "stairs_up", "stairs_down",
}

TRANSITION_CLASSES = {
    "sit_to_stand", "stand_to_sit",
    "stair_up", "stair_down",
    "stairs_up", "stairs_down",
}

STEADY_STATE_CLASSES = {"sit", "stand", "walking"}

# Transition point suffixes found in the 'transitions' folder are numeric
# markers (percent/position through the transition), e.g. "0", "50", "100",
# "50m", "100m" - validated with a regex rather than a fixed set.
VALID_TRANSITION_POINT_PATTERN = re.compile(r"^\d+m?$")

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
        logger.debug(f"FileMetadata created successfully")
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