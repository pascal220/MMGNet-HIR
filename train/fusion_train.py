import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from data_loader import PreparedData
from fusion_cnn_model import FusionCNNTrainer, FusionCNNTuner
from fusion_gru_model import FusionGRUTrainer, FusionGRUTuner
from split_utils import validate_prepared_data
from training_experiment import TrainingRunConfig, run_training_experiment

NUM_CLASSES = 7
INTENT_CNN_PATH = "checkpoints/best_intent_cnn.pt"
GESTURE_CNN_PATH = "checkpoints/best_mmg_cnn.pt"
FUSION_CNN_PATH = "checkpoints/best_fusion_cnn.pt"
FUSION_GRU_PATH = "checkpoints/best_fusion_gru.pt"


def train_and_evaluate_fusion(
    prepared: PreparedData,
    batch_size: int | None = None,
    intent_cnn_path: str = INTENT_CNN_PATH,
    gesture_cnn_path: str = GESTURE_CNN_PATH,
    fusion_cnn_checkpoint_path: str | None = FUSION_CNN_PATH,
    fusion_gru_checkpoint_path: str | None = FUSION_GRU_PATH,
    *,
    n_trials: int = 50,
    timeout: int | None = 3600,
    artifact_root: str = "results/training",
    run_label: str | None = None,
    cnn_resume_run_id: str | None = None,
    gru_resume_run_id: str | None = None,
    show_progress: bool = True,
    device: str = "auto",
):
    """Tune and refit single-window FusionCNN and FusionGRU models.

    The standalone backbones remain frozen. Test tensors are deliberately not
    read; this function returns validation-selection metadata and artifact paths.
    """
    validate_prepared_data(prepared, "single_window", "fusion")
    common = {
        "prepared": prepared,
        "input_tensors": (prepared.X_imu_train, prepared.X_cwt_train),
        "parent_checkpoints": (intent_cnn_path, gesture_cnn_path),
        "initial_batch_size": batch_size,
    }

    cnn_config = TrainingRunConfig(
        n_trials=n_trials,
        timeout=timeout,
        artifact_root=artifact_root,
        run_label=run_label,
        resume_run_id=cnn_resume_run_id,
        seed=prepared.experiment.config.seed,
        show_progress=show_progress,
        device=device,
    )
    cnn_result = run_training_experiment(
        **common,
        model_key="fusion_cnn",
        tuner_factory=lambda train, val, search: FusionCNNTuner(
            train_loader=train,
            val_loader=val,
            intent_cnn_path=intent_cnn_path,
            gesture_cnn_path=gesture_cnn_path,
            num_classes=NUM_CLASSES,
            search_space=search,
        ),
        trainer_factory=FusionCNNTrainer,
        config=cnn_config,
        legacy_checkpoint_path=fusion_cnn_checkpoint_path,
    )

    gru_config = TrainingRunConfig(
        n_trials=n_trials,
        timeout=timeout,
        artifact_root=artifact_root,
        run_label=run_label,
        resume_run_id=gru_resume_run_id,
        seed=prepared.experiment.config.seed,
        show_progress=show_progress,
        device=device,
    )
    gru_result = run_training_experiment(
        **common,
        model_key="fusion_gru",
        tuner_factory=lambda train, val, search: FusionGRUTuner(
            train_loader=train,
            val_loader=val,
            intent_cnn_path=intent_cnn_path,
            gesture_cnn_path=gesture_cnn_path,
            num_classes=NUM_CLASSES,
            search_space=search,
        ),
        trainer_factory=FusionGRUTrainer,
        config=gru_config,
        legacy_checkpoint_path=fusion_gru_checkpoint_path,
    )
    return {"fusion_cnn": cnn_result, "fusion_gru": gru_result}
