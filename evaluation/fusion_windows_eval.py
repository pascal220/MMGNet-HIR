import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from data_loader import PreparedData
from evaluation_common import DEFAULT_OUTPUT_ROOT, ModelSpec, evaluate_models
from fusion_cnn_window_model import FusionCNNWindow
from fusion_gru_window_model import FusionGRUWindow
from run_selection import DEFAULT_ARTIFACT_ROOT
from split_utils import validate_prepared_data


def _fusion_cnn(config, parents):
    imu_checkpoint, mmg_checkpoint = parents
    return FusionCNNWindow(
        **{**config, "imu_checkpoint": imu_checkpoint, "mmg_checkpoint": mmg_checkpoint}
    )


def _fusion_gru(config, parents):
    imu_checkpoint, mmg_checkpoint = parents
    return FusionGRUWindow(
        **{**config, "imu_checkpoint": imu_checkpoint, "mmg_checkpoint": mmg_checkpoint}
    )


def evaluate_fusion_windows(
    prepared: PreparedData,
    *,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    device: str = "auto",
):
    """Evaluate the latest windowed FusionCNN and FusionGRU on the sealed test split."""
    validate_prepared_data(prepared, "windowed", "fusion")
    inputs = (prepared.X_imu_test, prepared.X_cwt_test)
    specs = (
        ModelSpec("FusionCNN", "fusion_cnn_windowed", _fusion_cnn, inputs),
        ModelSpec("FusionGRU", "fusion_gru_windowed", _fusion_gru, inputs),
    )
    return evaluate_models(
        prepared, specs,
        artifact_root=artifact_root, output_root=output_root, device=device,
    )
