import torch
from torch.utils.data import TensorDataset, DataLoader
from imu_cnn_model import IntentCNN, IntentCNNTrainer, IntentCNNTuner


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32, checkpoint_path="checkpoints/intent_cnn.pt"):
    """Train and evaluate the Intent CNN model.
    
    Args:
        X_train: Training input tensor of shape (N, 6, 125)
        y_train: Training labels tensor of shape (N,)
        X_val: Validation input tensor of shape (M, 6, 125)
        y_val: Validation labels tensor of shape (M,)
        batch_size: Batch size for DataLoaders (default: 32)
        checkpoint_path: Path to save/load model checkpoint (default: "checkpoints/intent_cnn.pt")
    
    Returns:
        dict: Training history from the trainer
    """
    # ── Build model ────────────────────────────────────────────────────────────
    model   = IntentCNN(in_channels=6, num_classes=7)
    trainer = IntentCNNTrainer(model)

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