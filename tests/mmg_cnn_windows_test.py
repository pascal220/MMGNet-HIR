import sys
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from mmg_cnn_window_model import (
    LocomotionMMGCNNWindow,
    LocomotionMMGCNNWindowTrainer,
    LocomotionMMGCNNWindowTuner,
)
from split_utils import split_train_validation


# Windowed input shape: (batch, 5, 40, 125, 4)
IN_CHANNELS  = 5
NUM_CLASSES  = 7
N_TRIALS     = 50
TIMEOUT      = 3600
TRAIN_EPOCHS = 100


def train_and_evaluate(
    X_train,
    y_train,
    X_test,
    y_test,
    train_metadata,
    batch_size=32,
    checkpoint_path="checkpoints/best_window_mmg_cnn.pt",
):
    """Train and evaluate the windowed LocomotionMMGCNNWindow model.

    Args:
        X_train: Tensor shaped (N, 5, 40, 125, 4).
        y_train: Labels shaped (N,).
        X_test: Test tensor shaped (K, 5, 40, 125, 4).
        y_test: Test labels shaped (K,).
        train_metadata: Row-aligned metadata for X_train/y_train, used to carve
            out a stratified, group-aware 10% validation split.
        batch_size: Batch size for DataLoaders.
        checkpoint_path: Output checkpoint path.

    Returns:
        dict: Training history and validation/test metrics.
    """
    train_idx, val_idx = split_train_validation(y_train, train_metadata)
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_train, y_train = X_train[train_idx], y_train[train_idx]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=batch_size
    )

    tuner = LocomotionMMGCNNWindowTuner(
        train_loader,
        val_loader,
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
    )
    best_model = tuner.run(n_trials=N_TRIALS, timeout=TIMEOUT)
    best_params = tuner.get_best_params()
    tuner.plot_results(save_dir="optuna_plots_window_mmg_cnn")

    trainer = LocomotionMMGCNNWindowTrainer(best_model, best_params)
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=TRAIN_EPOCHS,
        verbose=True,
    )

    val_results = trainer.evaluate(val_loader)
    test_results = trainer.evaluate(test_loader)

    # ── Inference ───────────────────────────────────────────────────────────
    # TODO: Implement inference logic here, e.g., using trainer.predict().

    trainer.save(checkpoint_path)
    return {
        "history": history,
        "val_results": val_results,
        "test_results": test_results,
    }
