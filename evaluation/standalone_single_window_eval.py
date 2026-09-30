import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from data_loader import PreparedData
from evaluation_common import DEFAULT_OUTPUT_ROOT, ModelSpec, evaluate_models
from imu_cnn_model import IntentCNN
from mmg_cnn_model import LocomotionMMGCNN
from run_selection import DEFAULT_ARTIFACT_ROOT
from split_utils import validate_prepared_data


def evaluate_standalone_single_window(
    prepared: PreparedData,
    *,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    device: str = "auto",
):
    """Evaluate the latest single-window IMU and MMG CNNs, scoring every window."""
    validate_prepared_data(prepared, "single_window", "standalone")
    specs = (
        ModelSpec(
            "IMU", "imu_cnn",
            lambda config, _: IntentCNN(**config),
            (prepared.X_imu_test,),
        ),
        ModelSpec(
            "MMG", "mmg_cnn",
            lambda config, _: LocomotionMMGCNN(**config),
            (prepared.X_cwt_test,),
        ),
    )
    return evaluate_models(
        prepared, specs,
        artifact_root=artifact_root, output_root=output_root, device=device,
    )
