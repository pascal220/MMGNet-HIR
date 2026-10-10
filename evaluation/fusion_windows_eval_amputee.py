"""Test-set evaluation of the windowed amputee fusion models.

``evaluate_fusion_windows_amputee`` evaluates FusionCNN and FusionGRU on every
amputee data type (four models for two types) in one call.
"""

import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fusion_cnn_window_amputee_model as cnn
import fusion_gru_window_amputee_model as gru
from amputee_evaluation_common import evaluate_amputee
from data_loader import AMPUTEE_DATA_TYPES, PreparedData
from evaluation_common import DEFAULT_OUTPUT_ROOT, ModelSpec, evaluate_models
from run_selection import DEFAULT_ARTIFACT_ROOT
from split_utils import validate_prepared_data

INPUT_MODE = "windowed"
MODEL_KEYS = (cnn.MODEL_KEY, gru.MODEL_KEY)


def evaluate_fusion_windows_amputee_type(
    prepared: PreparedData,
    *,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    output_dir: str | Path | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    """Evaluate the latest windowed FusionCNN and FusionGRU of one amputee data type."""
    validate_prepared_data(prepared, INPUT_MODE, "fusion")
    if prepared.experiment.config.setup != "amputee":
        raise ValueError("Amputee evaluation expects data from prepare_amputee_experiment_data.")
    inputs = (prepared.X_imu_test, prepared.X_cwt_test)
    specs = (
        ModelSpec("FusionCNN", cnn.MODEL_KEY, cnn.FusionCNNWindowAmputee.from_config, inputs),
        ModelSpec("FusionGRU", gru.MODEL_KEY, gru.FusionGRUWindowAmputee.from_config, inputs),
    )
    return evaluate_models(
        prepared, specs,
        artifact_root=artifact_root, output_root=output_root,
        output_dir=output_dir, device=device,
    )


def evaluate_fusion_windows_amputee(
    amputee_id: int | str,
    *,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    device: str = "auto",
    data_types: Sequence[str] = AMPUTEE_DATA_TYPES,
    **data_kwargs: Any,
) -> dict[str, Any]:
    """Evaluate both windowed fusion models on every amputee data type.

    ``data_kwargs`` must match the settings the models were trained with.
    """
    return evaluate_amputee(
        amputee_id, INPUT_MODE, evaluate_fusion_windows_amputee_type, MODEL_KEYS,
        artifact_root=artifact_root, output_root=output_root, device=device,
        data_types=data_types, **data_kwargs,
    )
