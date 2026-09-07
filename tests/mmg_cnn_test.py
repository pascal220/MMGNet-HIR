import torch
from torch.utils.data import TensorDataset, DataLoader
from mmg_cnn_model import LocomotionMMGCNN, LocomotionMMGCNNTrainer, LocomotionMMGCNNTuner


# ── Configuration ────────────────────────────────────────────────────────────
IN_CHANNELS  = 5
NUM_CLASSES  = 7
N_TRIALS     = 50
TIMEOUT      = 3600    # 1 hour
TRAIN_EPOCHS = 100     # final training after Optuna


def train_and_evaluate(X_train, y_train, X_test, y_test, batch_size=32, checkpoint_path="checkpoints/best_mmg_cnn.pt"):
    """Train and evaluate the single-window Locomotion MMG CNN model.

    Args:
        X_train: Training input tensor of shape (N, 5, 40, 125) from single_window input mode
        y_train: Training labels tensor of shape (N,)
        X_test: Test input tensor of shape (M, 5, 40, 125) from single_window input mode
        y_test: Test labels tensor of shape (M,)
        batch_size: Batch size for DataLoaders (default: 32)
        checkpoint_path: Path to save/load model checkpoint (default: "checkpoints/best_mmg_cnn.pt")

    Returns:
        dict: Training history and final validation/test results from the trainer

    Note:
        The body still expects validation tensors and will be refactored later
        to split validation data from the training set inside this function.
    """
    # ── Build model ────────────────────────────────────────────────────────────
    model   = LocomotionMMGCNN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    trainer = LocomotionMMGCNNTrainer(model)

    # ── Create DataLoaders ─────────────────────────────────────────────────────
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=batch_size)

    # ── Run Optuna search ───────────────────────────────────────────────────────
    tuner      = LocomotionMMGCNNTuner(train_loader, val_loader,
                                  in_channels=IN_CHANNELS,
                                  num_classes=NUM_CLASSES)
    best_model = tuner.run(n_trials=N_TRIALS, timeout=TIMEOUT)

    # ── Inspect results ─────────────────────────────────────────────────────────
    best_params = tuner.get_best_params()
    tuner.plot_results(save_dir="optuna_plots_dacnn")

    # ── Train final model with best params ─────────────────────────────────────
    trainer = LocomotionMMGCNNTrainer(best_model, best_params)
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs  = TRAIN_EPOCHS,
        verbose = True,
    )

    # ── Evaluate ────────────────────────────────────────────────────────────────
    val_results = trainer.evaluate(val_loader)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
    test_results = trainer.evaluate(test_loader)

    # ── Inference ───────────────────────────────────────────────────────────────
    #TODO: Implement inference logic here, e.g., using trainer.predict() on new data

    # ── Save / Load ─────────────────────────────────────────────────────────────
    trainer.save(checkpoint_path)

    return {
        "history": history,
        "val_results": val_results,
        "test_results": test_results,
    }