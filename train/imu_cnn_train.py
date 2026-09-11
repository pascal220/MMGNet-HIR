import sys
from pathlib import Path

from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from imu_cnn_model import IntentCNN, IntentCNNTrainer, IntentCNNTuner
from data_loader import PreparedData
from split_utils import split_train_validation, validate_prepared_data


def train_and_evaluate_imu_cnn(
    prepared: PreparedData,
    batch_size: int | None = None,
    checkpoint_path="checkpoints/intent_cnn.pt",
):
    """Train and evaluate the single-window Intent CNN model.
    
    Args:
        prepared: Single-window standalone tensors and row-aligned metadata.
        batch_size: Optional DataLoader batch-size override.
        checkpoint_path: Path to save/load model checkpoint (default: "checkpoints/intent_cnn.pt")
    
    Returns:
        dict: Training history from the trainer
    """
    validate_prepared_data(prepared, "single_window", "standalone")
    if batch_size is None:
        batch_size = prepared.experiment.config.batch_size
    X_train = prepared.X_imu_train
    y_train = prepared.y_train
    X_test = prepared.X_imu_test
    y_test = prepared.y_test

    # ── Build model ────────────────────────────────────────────────────────────
    model   = IntentCNN(in_channels=6, num_classes=7)
    trainer = IntentCNNTrainer(model)

    # ── Split off a validation set ──────────────────────────────────────────────
    train_idx, val_idx = split_train_validation(y_train, prepared.train_metadata)
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_train, y_train = X_train[train_idx], y_train[train_idx]

    # ── Create DataLoaders ─────────────────────────────────────────────────────
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=batch_size)

    # ── Run Optuna search ───────────────────────────────────────────────────────
    tuner      = IntentCNNTuner(train_loader, val_loader)
    best_model = tuner.run(n_trials=50, timeout=3600)

    # ── Inspect results ─────────────────────────────────────────────────────────
    best_params = tuner.get_best_params()
    tuner.plot_results(save_dir="optuna_plots")

    # ── Train final model with best params ─────────────────────────────────────
    trainer = IntentCNNTrainer(best_model, best_params)
    history = trainer.fit(train_loader, val_loader)

    # ── Evaluate ────────────────────────────────────────────────────────────────
    trainer.evaluate(val_loader)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=batch_size)
    trainer.evaluate(test_loader)

    # ── Inference ───────────────────────────────────────────────────────────────
    #TODO: Implement inference logic here, e.g., using trainer.predict() on new data

    # ── Save / Load ─────────────────────────────────────────────────────────────
    trainer.save(checkpoint_path)
    
    return history