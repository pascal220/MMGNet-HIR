import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from data_loader import PreparedData
from mmg_cnn_window_model import (
    LocomotionMMGCNNWindowTrainer,
    LocomotionMMGCNNWindowTuner,
)
from split_utils import validate_prepared_data
from training_experiment import TrainingRunConfig, run_training_experiment


def train_and_evaluate_mmg_cnn_windows(
    prepared: PreparedData,
    batch_size: int | None = None,
    checkpoint_path: str | None = "checkpoints/best_window_mmg_cnn.pt",
    *,
    n_trials: int = 50,
    timeout: int | None = 3600,
    artifact_root: str = "results/training",
    run_label: str | None = None,
    resume_run_id: str | None = None,
    show_progress: bool = True,
):
    """Tune and refit a windowed MMG CNN without accessing test data."""
    validate_prepared_data(prepared, "windowed", "standalone")
    config = TrainingRunConfig(
        n_trials=n_trials,
        timeout=timeout,
        artifact_root=artifact_root,
        run_label=run_label,
        resume_run_id=resume_run_id,
        seed=prepared.experiment.config.seed,
        show_progress=show_progress,
    )
    return run_training_experiment(
        prepared=prepared,
        model_key="mmg_cnn_windowed",
        input_tensors=(prepared.X_cwt_train,),
        tuner_factory=lambda train, val, search: LocomotionMMGCNNWindowTuner(
            train, val, in_channels=5, num_classes=7, search_space=search
        ),
        trainer_factory=LocomotionMMGCNNWindowTrainer,
        config=config,
        legacy_checkpoint_path=checkpoint_path,
        initial_batch_size=batch_size,
    )
