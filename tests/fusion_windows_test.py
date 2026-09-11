import sys
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from fusion_cnn_window_model import (
    FusionCNNWindow,
    FusionCNNWindowTrainer,
    FusionCNNWindowTuner,
)
from fusion_gru_window_model import (
    FusionGRUWindow,
    FusionGRUWindowTrainer,
    FusionGRUWindowTuner,
)
from split_utils import split_train_validation


# Windowed input shapes:
#   IMU: (batch, 6, 125, 4)
#   MMG: (batch, 5, 40, 125, 4)
NUM_CLASSES      = 7
N_TRIALS         = 50
TIMEOUT          = 3600
TRAIN_EPOCHS     = 100

# Paths to pretrained windowed standalone model checkpoints
INTENT_CNN_PATH  = "checkpoints/best_window_intent_cnn.pt"
GESTURE_CNN_PATH = "checkpoints/best_window_mmg_cnn.pt"

# Output checkpoint paths
FUSION_CNN_PATH  = "checkpoints/best_window_fusion_cnn.pt"
FUSION_GRU_PATH  = "checkpoints/best_window_fusion_gru.pt"


def train_and_evaluate(
    X_imu_train,
    X_cwt_train,
    y_train,
    X_imu_test,
    X_cwt_test,
    y_test,
    train_metadata,
    batch_size=32,
    intent_cnn_path=INTENT_CNN_PATH,
    gesture_cnn_path=GESTURE_CNN_PATH,
    fusion_cnn_checkpoint_path=FUSION_CNN_PATH,
    fusion_gru_checkpoint_path=FUSION_GRU_PATH,
):
    """Train and evaluate windowed FusionCNNWindow and FusionGRUWindow models.

    Args:
        X_imu_train: IMU tensor shaped (N, 6, 125, 4).
        X_cwt_train: MMG/CWT tensor shaped (N, 5, 40, 125, 4).
        y_train: Training labels shaped (N,).
        X_imu_test: Test IMU tensor shaped (K, 6, 125, 4).
        X_cwt_test: Test MMG/CWT tensor shaped (K, 5, 40, 125, 4).
        y_test: Test labels shaped (K,).
        train_metadata: Row-aligned metadata for the training tensors, used to
            carve out a stratified, group-aware 10% validation split.
        batch_size: Batch size for DataLoaders.
        intent_cnn_path: Path to the windowed IntentCNN checkpoint.
        gesture_cnn_path: Path to the windowed MMGCNN checkpoint.
        fusion_cnn_checkpoint_path: FusionCNNWindow output checkpoint.
        fusion_gru_checkpoint_path: FusionGRUWindow output checkpoint.

    Returns:
        dict: Training histories and validation/test metrics for both models.
    """
    train_idx, val_idx = split_train_validation(y_train, train_metadata)
    X_imu_val, X_cwt_val, y_val = X_imu_train[val_idx], X_cwt_train[val_idx], y_train[val_idx]
    X_imu_train, X_cwt_train, y_train = (
        X_imu_train[train_idx],
        X_cwt_train[train_idx],
        y_train[train_idx],
    )

    train_loader = DataLoader(
        TensorDataset(X_imu_train, X_cwt_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_imu_val, X_cwt_val, y_val),
        batch_size=batch_size,
    )
    test_loader = DataLoader(
        TensorDataset(X_imu_test, X_cwt_test, y_test),
        batch_size=batch_size,
    )

    print("\n" + "=" * 60)
    print("STUDY 1: FusionCNNWindow — Optuna hyperparameter search")
    print("=" * 60)
    cnn_tuner = FusionCNNWindowTuner(
        train_loader,
        val_loader,
        imu_checkpoint=intent_cnn_path,
        mmg_checkpoint=gesture_cnn_path,
        num_classes=NUM_CLASSES,
    )
    best_fusion_cnn = cnn_tuner.run(n_trials=N_TRIALS, timeout=TIMEOUT)
    cnn_best_params = cnn_tuner.get_best_params()
    cnn_tuner.plot_results(save_dir="optuna_plots_window_fusion_cnn")

    cnn_trainer = FusionCNNWindowTrainer(best_fusion_cnn, cnn_best_params)
    cnn_history = cnn_trainer.fit(
        train_loader,
        val_loader,
        epochs=TRAIN_EPOCHS,
        verbose=True,
    )
    cnn_val_results = cnn_trainer.evaluate(val_loader)
    cnn_test_results = cnn_trainer.evaluate(test_loader)

    # ── Inference ───────────────────────────────────────────────────────────
    # TODO: Implement inference logic here, e.g., using trainer.predict().

    cnn_trainer.save(fusion_cnn_checkpoint_path)

    print("\n" + "=" * 60)
    print("STUDY 2: FusionGRUWindow — Optuna hyperparameter search")
    print("=" * 60)
    gru_tuner = FusionGRUWindowTuner(
        train_loader,
        val_loader,
        imu_checkpoint=intent_cnn_path,
        mmg_checkpoint=gesture_cnn_path,
        num_classes=NUM_CLASSES,
    )
    best_fusion_gru = gru_tuner.run(n_trials=N_TRIALS, timeout=TIMEOUT)
    gru_best_params = gru_tuner.get_best_params()
    gru_tuner.plot_results(save_dir="optuna_plots_window_fusion_gru")

    gru_trainer = FusionGRUWindowTrainer(best_fusion_gru, gru_best_params)
    gru_history = gru_trainer.fit(
        train_loader,
        val_loader,
        epochs=TRAIN_EPOCHS,
        verbose=True,
    )
    gru_val_results = gru_trainer.evaluate(val_loader)
    gru_test_results = gru_trainer.evaluate(test_loader)

    # ── Inference ───────────────────────────────────────────────────────────
    # TODO: Implement inference logic here, e.g., using trainer.predict().

    gru_trainer.save(fusion_gru_checkpoint_path)

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'Val Accuracy':>14} {'Val Macro F1':>14} {'Test Accuracy':>14} {'Test Macro F1':>14}")
    print("-" * 85)
    print(
        f"{'FusionCNNWindow':<20}"
        f"{cnn_val_results['accuracy']:>14.4f}"
        f"{cnn_val_results['f1']:>14.4f}"
        f"{cnn_test_results['accuracy']:>14.4f}"
        f"{cnn_test_results['f1']:>14.4f}"
    )
    print(
        f"{'FusionGRUWindow':<20}"
        f"{gru_val_results['accuracy']:>14.4f}"
        f"{gru_val_results['f1']:>14.4f}"
        f"{gru_test_results['accuracy']:>14.4f}"
        f"{gru_test_results['f1']:>14.4f}"
    )

    return {
        "fusion_cnn": {
            "history": cnn_history,
            "val_results": cnn_val_results,
            "test_results": cnn_test_results,
        },
        "fusion_gru": {
            "history": gru_history,
            "val_results": gru_val_results,
            "test_results": gru_test_results,
        },
    }
