# mmg_cnn_test.py
# Test script for GestureCNN / GestureCNNTuner from dacnn_model.py
#
# Input shape : (batch, 5, 40, 125)
#               5 channels (MMG sensors)
#               40 frequency scales (CWT)
#               125 time steps
# Output      : 7 gesture classes

import torch
from torch.utils.data import TensorDataset, DataLoader
from mmg_cnn_model import LocomotionMMGCNN, LocomotionMMGCNNTrainer, LocomotionMMGCNNTuner

# ── Configuration ────────────────────────────────────────────────────────────
IN_CHANNELS  = 5
FREQ_SCALES  = 40
TIME_STEPS   = 125
NUM_CLASSES  = 7
N_TRAIN      = 500
N_VAL        = 100
BATCH_SIZE   = 32
N_TRIALS     = 50
TIMEOUT      = 3600    # 1 hour
TRAIN_EPOCHS = 100     # final training after Optuna

# ── Dummy data (replace with your real CWT scalogram DataLoaders) ────────────
# X shape: (samples, channels, freq_scales, time_steps)
# y shape: (samples,)  -- integer class labels in [0, NUM_CLASSES)
X_train = torch.randn(N_TRAIN, IN_CHANNELS, FREQ_SCALES, TIME_STEPS)
y_train = torch.randint(0, NUM_CLASSES, (N_TRAIN,))
X_val   = torch.randn(N_VAL,   IN_CHANNELS, FREQ_SCALES, TIME_STEPS)
y_val   = torch.randint(0, NUM_CLASSES, (N_VAL,))

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size = BATCH_SIZE,
    shuffle    = True,
)
val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size = BATCH_SIZE,
)

# ── 1. Sanity check: paper default architecture ───────────────────────────────
print("=" * 60)
print("Sanity check: paper default architecture")
print("=" * 60)

default_model   = LocomotionMMGCNN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
default_trainer = LocomotionMMGCNNTrainer(default_model)

# Quick 3-epoch smoke test
default_trainer.fit(train_loader, val_loader, epochs=3, verbose=True)
default_trainer.evaluate(val_loader)

# Single sample inference
sample      = torch.randn(IN_CHANNELS, FREQ_SCALES, TIME_STEPS)
pred        = default_trainer.predict(sample)
probs       = default_trainer.predict_proba(sample)
print(f"\nSingle sample prediction : class {pred.item()}")
print(f"Class probabilities      : {probs.squeeze().numpy().round(3)}")

# ── 2. Run Optuna hyperparameter search ──────────────────────────────────────
print("\n" + "=" * 60)
print("Running Optuna hyperparameter search")
print("=" * 60)

tuner      = LocomotionMMGCNNTuner(train_loader, val_loader,
                              in_channels=IN_CHANNELS,
                              num_classes=NUM_CLASSES)
best_model = tuner.run(n_trials=N_TRIALS, timeout=TIMEOUT)

# Inspect and save plots
best_params = tuner.get_best_params()
tuner.plot_results(save_dir="optuna_plots_dacnn")

# ── 3. Final training with best architecture ─────────────────────────────────
print("\n" + "=" * 60)
print(f"Final training: best architecture for {TRAIN_EPOCHS} epochs")
print("=" * 60)

final_trainer = LocomotionMMGCNNTrainer(best_model, best_params)
history       = final_trainer.fit(
    train_loader,
    val_loader,
    epochs  = TRAIN_EPOCHS,
    verbose = True,
)

# ── 4. Final evaluation ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Final evaluation on validation set")
print("=" * 60)
final_trainer.evaluate(val_loader)

# ── 5. Save the final model ───────────────────────────────────────────────────
final_trainer.save("checkpoints/best_mmg_cnn.pt")

# ── 6. Reload and verify ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Reload and verify")
print("=" * 60)

reloaded_model   = LocomotionMMGCNN(
    in_channels   = best_model.config["in_channels"],
    num_classes   = best_model.config["num_classes"],
    block_filters = best_model.config["block_filters"],
    kernel_sizes  = best_model.config["kernel_sizes"],
    strides       = best_model.config["strides"],
    dropout_rates = best_model.config["dropout_rates"],
    fc_hidden     = best_model.config["fc_hidden"],
)
reloaded_trainer = LocomotionMMGCNNTrainer(reloaded_model, best_params)
reloaded_trainer.load("checkpoints/best_mmg_cnn.pt")
reloaded_trainer.evaluate(val_loader)