import sys
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from fusion_cnn_model import FusionCNN, FusionCNNTrainer, FusionCNNTuner
from fusion_gru_model import FusionGRU, FusionGRUTrainer, FusionGRUTuner
from split_utils import split_train_validation


# ── Configuration ────────────────────────────────────────────────────────────
NUM_CLASSES      = 7
N_TRIALS         = 50
TIMEOUT          = 3600    # 1 hour per study
TRAIN_EPOCHS     = 100     # final training epochs

# Paths to pre-trained sub-model checkpoints
INTENT_CNN_PATH  = "checkpoints/best_intent_cnn.pt"
GESTURE_CNN_PATH = "checkpoints/best_locomotion_cnn.pt"

# Output checkpoint paths
FUSION_CNN_PATH  = "checkpoints/best_fusion_cnn.pt"
FUSION_GRU_PATH  = "checkpoints/best_fusion_gru.pt"


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
    """Train and evaluate single-window FusionCNN and FusionGRU models.

    Args:
        X_imu_train: Training IMU tensor of shape (N, 6, 125) from single_window input mode
        X_cwt_train: Training CWT tensor of shape (N, 5, 40, 125) from single_window input mode
        y_train: Training labels tensor of shape (N,)
        X_imu_test: Test IMU tensor of shape (M, 6, 125) from single_window input mode
        X_cwt_test: Test CWT tensor of shape (M, 5, 40, 125) from single_window input mode
        y_test: Test labels tensor of shape (M,)
        train_metadata: Row-aligned metadata for the training tensors, used to
            carve out a stratified, group-aware 10% validation split.
        batch_size: Batch size for DataLoaders (default: 32)
        intent_cnn_path: Path to the trained IntentCNN checkpoint
        gesture_cnn_path: Path to the trained LocomotionMMGCNN checkpoint
        fusion_cnn_checkpoint_path: Path to save/load the FusionCNN checkpoint
        fusion_gru_checkpoint_path: Path to save/load the FusionGRU checkpoint

    Returns:
        dict: Training histories and final validation/test results for both fusion models
    """
    # ── Split off a validation set ──────────────────────────────────────────────
    train_idx, val_idx = split_train_validation(y_train, train_metadata)
    X_imu_val, X_cwt_val, y_val = X_imu_train[val_idx], X_cwt_train[val_idx], y_train[val_idx]
    X_imu_train, X_cwt_train, y_train = (
        X_imu_train[train_idx],
        X_cwt_train[train_idx],
        y_train[train_idx],
    )

    # ── Create DataLoaders ─────────────────────────────────────────────────────
    train_loader = DataLoader(
        TensorDataset(X_imu_train, X_cwt_train, y_train),
        batch_size = batch_size,
        shuffle    = True,
    )
    val_loader = DataLoader(
        TensorDataset(X_imu_val, X_cwt_val, y_val),
        batch_size = batch_size,
    )
    test_loader = DataLoader(
        TensorDataset(X_imu_test, X_cwt_test, y_test),
        batch_size = batch_size,
    )

    # ============================================================================
    # Study 1: FusionCNN
    # ============================================================================
    print("\n" + "=" * 60)
    print("STUDY 1: FusionCNN — Optuna hyperparameter search")
    print("=" * 60)

    cnn_tuner  = FusionCNNTuner(
        train_loader     = train_loader,
        val_loader       = val_loader,
        intent_cnn_path  = intent_cnn_path,
        gesture_cnn_path = gesture_cnn_path,
        num_classes      = NUM_CLASSES,
    )
    best_fusion_cnn = cnn_tuner.run(n_trials=N_TRIALS, timeout=TIMEOUT)

    # ── Inspect results ─────────────────────────────────────────────────────────
    cnn_best_params = cnn_tuner.get_best_params()
    cnn_tuner.plot_results(save_dir="optuna_plots_fusion_cnn")

    # ── Train final model with best params ─────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"FusionCNN — Final training for {TRAIN_EPOCHS} epochs")
    print("=" * 60)

    cnn_trainer = FusionCNNTrainer(best_fusion_cnn, cnn_best_params)
    cnn_history = cnn_trainer.fit(
        train_loader,
        val_loader,
        epochs  = TRAIN_EPOCHS,
        verbose = True,
    )

    # ── Evaluate ────────────────────────────────────────────────────────────────
    print("\nFusionCNN — Final validation evaluation:")
    cnn_val_results = cnn_trainer.evaluate(val_loader)
    print("\nFusionCNN — Final test evaluation:")
    cnn_test_results = cnn_trainer.evaluate(test_loader)

    # ── Inference ───────────────────────────────────────────────────────────────
    #TODO: Implement inference logic here, e.g., using trainer.predict() on new data

    # ── Save / Load ─────────────────────────────────────────────────────────────
    cnn_trainer.save(fusion_cnn_checkpoint_path)

    # ============================================================================
    # Study 2: FusionGRU
    # ============================================================================
    print("\n" + "=" * 60)
    print("STUDY 2: FusionGRU — Optuna hyperparameter search")
    print("=" * 60)

    gru_tuner  = FusionGRUTuner(
        train_loader     = train_loader,
        val_loader       = val_loader,
        intent_cnn_path  = intent_cnn_path,
        gesture_cnn_path = gesture_cnn_path,
        num_classes      = NUM_CLASSES,
    )
    best_fusion_gru = gru_tuner.run(n_trials=N_TRIALS, timeout=TIMEOUT)

    # ── Inspect results ─────────────────────────────────────────────────────────
    gru_best_params = gru_tuner.get_best_params()
    gru_tuner.plot_results(save_dir="optuna_plots_fusion_gru")

    # ── Train final model with best params ─────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"FusionGRU — Final training for {TRAIN_EPOCHS} epochs")
    print("=" * 60)

    gru_trainer = FusionGRUTrainer(best_fusion_gru, gru_best_params)
    gru_history = gru_trainer.fit(
        train_loader,
        val_loader,
        epochs  = TRAIN_EPOCHS,
        verbose = True,
    )

    # ── Evaluate ────────────────────────────────────────────────────────────────
    print("\nFusionGRU — Final validation evaluation:")
    gru_val_results = gru_trainer.evaluate(val_loader)
    print("\nFusionGRU — Final test evaluation:")
    gru_test_results = gru_trainer.evaluate(test_loader)

    # ── Inference ───────────────────────────────────────────────────────────────
    #TODO: Implement inference logic here, e.g., using trainer.predict() on new data

    # ── Save / Load ─────────────────────────────────────────────────────────────
    gru_trainer.save(fusion_gru_checkpoint_path)

    # ============================================================================
    # Side-by-side comparison summary (user decides which to use)
    # ============================================================================
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<15} {'Val Accuracy':>14} {'Val Macro F1':>14} {'Test Accuracy':>14} {'Test Macro F1':>14}")
    print("-" * 75)
    print(
        f"{'FusionCNN':<15}"
        f"{cnn_val_results['accuracy']:>14.4f}"
        f"{cnn_val_results['f1']:>14.4f}"
        f"{cnn_test_results['accuracy']:>14.4f}"
        f"{cnn_test_results['f1']:>14.4f}"
    )
    print(
        f"{'FusionGRU':<15}"
        f"{gru_val_results['accuracy']:>14.4f}"
        f"{gru_val_results['f1']:>14.4f}"
        f"{gru_test_results['accuracy']:>14.4f}"
        f"{gru_test_results['f1']:>14.4f}"
    )
    print("=" * 60)
    print(f"\nBoth models saved:")
    print(f"  FusionCNN -> '{fusion_cnn_checkpoint_path}'")
    print(f"  FusionGRU -> '{fusion_gru_checkpoint_path}'")
    print("\nPlease review the results above and select your preferred model.")

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