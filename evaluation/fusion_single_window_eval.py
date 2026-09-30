import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from data_loader import PreparedData
from evaluation_common import DEFAULT_OUTPUT_ROOT, ModelSpec, evaluate_models
from fusion_cnn_model import FusionCNN
from fusion_gru_model import FusionGRU
from run_selection import DEFAULT_ARTIFACT_ROOT
from split_utils import validate_prepared_data


def _fusion_cnn(config, parents):
    intent_cnn_path, gesture_cnn_path = parents
    return FusionCNN(
        intent_cnn_path=intent_cnn_path,
        gesture_cnn_path=gesture_cnn_path,
        num_classes=config["num_classes"],
        fc_hidden=config["fc_hidden"],
    )


def _fusion_gru(config, parents):
    intent_cnn_path, gesture_cnn_path = parents
    return FusionGRU(
        intent_cnn_path=intent_cnn_path,
        gesture_cnn_path=gesture_cnn_path,
        num_classes=config["num_classes"],
        gru_hidden=config["gru_hidden"],
        gru_layers=config["gru_layers"],
        gru_dropout=config["gru_dropout"],
        fc_hidden=config["fc_hidden"],
    )


def evaluate_fusion_single_window(
    prepared: PreparedData,
    *,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    device: str = "auto",
):
    """Evaluate the latest single-window FusionCNN and FusionGRU, scoring every window."""
    validate_prepared_data(prepared, "single_window", "fusion")
    inputs = (prepared.X_imu_test, prepared.X_cwt_test)
    specs = (
        ModelSpec("FusionCNN", "fusion_cnn", _fusion_cnn, inputs),
        ModelSpec("FusionGRU", "fusion_gru", _fusion_gru, inputs),
    )
    return evaluate_models(
        prepared, specs,
        artifact_root=artifact_root, output_root=output_root, device=device,
    )
