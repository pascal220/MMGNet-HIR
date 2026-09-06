# test_fusion.py
# Runs two separate Optuna studies (FusionCNN and FusionGRU),
# trains the best model from each study for the full 100 epochs,
# evaluates both, and saves both.
# The user compares results and selects the preferred model.
#
# DataLoader format: (X_imu, X_cwt, y)
#   X_imu : (batch, 6, 125)       -- raw IMU time series
#   X_cwt : (batch, 5, 40, 125)   -- CWT scalogram images
#   y     : (batch,)              -- integer class labels [0, 7)

import torch
from torch.utils.data import TensorDataset, DataLoader
from fusion_cnn_model import FusionCNN, FusionCNNTrainer, FusionCNNTuner
from fusion_gru_model import FusionGRU, FusionGRUTrainer, FusionGRUTuner

# ── Configuration ────────────────────────────────────────────────────────────
NUM_CLASSES      = 7
N_TRAIN          = 500
N_VAL            = 100
BATCH_SIZE       = 32
N_TRIALS         = 50
TIMEOUT          = 3600    # 1 hour per study
TRAIN_EPOCHS     = 100     # final training epochs

# Paths to pre-trained sub-model checkpoints
INTENT_CNN_PATH  = "checkpoints/best_intent_cnn.pt"
GESTURE_CNN_PATH = "checkpoints/best_locomotion_cnn.pt"

# Output checkpoint paths
FUSION_CNN_PATH  = "checkpoints/best_fusion_cnn.pt"
FUSION_GRU_PATH  = "checkpoints/best_fusion_gru.pt"

# ── Dummy paired data (replace with your real DataLoaders) ───────────────────
# Both X_imu and X_cwt must correspond to the same samples (paired)
X_imu_train = torch.randn(N_TRAIN, 6,  125)
X_cwt_train = torch.randn(N_TRAIN, 5, 40, 125)
y_train     = torch.randint(0, NUM_CLASSES, (N_TRAIN,))

X_imu_val   = torch.randn(N_VAL, 6,  125)
X_cwt_val   = torch.randn(N_VAL, 5, 40, 125)
y_val       = torch.randint(0, NUM_CLASSES, (N_VAL,))

# TensorDataset with 3 tensors -> DataLoader yields (X_imu, X_cwt, y) tuples
train_loader = DataLoader(
    TensorDataset(X_imu_train, X_cwt_train, y_train),
    batch_size = BATCH_SIZE,
    shuffle    = True,
)
val_loader = DataLoader(
    TensorDataset(X_imu_val, X_cwt_val, y_val),
    batch_size = BATCH_SIZE,
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
    intent_cnn_path  = INTENT_CNN_PATH,
    gesture_cnn_path = GESTURE_CNN_PATH,
    num_classes      = NUM_CLASSES,
)
best_fusion_cnn = cnn_tuner.run(n_trials=N_TRIALS, timeout=TIMEOUT)
cnn_best_params = cnn_tuner.get_best_params()
cnn_tuner.plot_results(save_dir="optuna_plots_fusion_cnn")

# ── Final training: FusionCNN ────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"FusionCNN — Final training for {TRAIN_EPOCHS} epochs")
print("=" * 60)

cnn_trainer = FusionCNNTrainer(best_fusion_cnn, cnn_best_params)
cnn_history = cnn_trainer.fit(
    train_loader, val_loader,
    epochs  = TRAIN_EPOCHS,
    verbose = True,
)

print("\nFusionCNN — Final evaluation:")
cnn_results = cnn_trainer.evaluate(val_loader)
cnn_trainer.save(FUSION_CNN_PATH)

# ============================================================================
# Study 2: FusionGRU
# ============================================================================
print("\n" + "=" * 60)
print("STUDY 2: FusionGRU — Optuna hyperparameter search")
print("=" * 60)

gru_tuner  = FusionGRUTuner(
    train_loader     = train_loader,
    val_loader       = val_loader,
    intent_cnn_path  = INTENT_CNN_PATH,
    gesture_cnn_path = GESTURE_CNN_PATH,
    num_classes      = NUM_CLASSES,
)
best_fusion_gru = gru_tuner.run(n_trials=N_TRIALS, timeout=TIMEOUT)
gru_best_params = gru_tuner.get_best_params()
gru_tuner.plot_results(save_dir="optuna_plots_fusion_gru")

# ── Final training: FusionGRU ────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"FusionGRU — Final training for {TRAIN_EPOCHS} epochs")
print("=" * 60)

gru_trainer = FusionGRUTrainer(best_fusion_gru, gru_best_params)
gru_history = gru_trainer.fit(
    train_loader, val_loader,
    epochs  = TRAIN_EPOCHS,
    verbose = True,
)

print("\nFusionGRU — Final evaluation:")
gru_results = gru_trainer.evaluate(val_loader)
gru_trainer.save(FUSION_GRU_PATH)

# ============================================================================
# Side-by-side comparison summary (user decides which to use)
# ============================================================================
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Model':<15} {'Val Accuracy':>14} {'Val Macro F1':>14}")
print("-" * 45)
print(
    f"{'FusionCNN':<15}"
    f"{cnn_results['accuracy']:>14.4f}"
    f"{cnn_results['f1']:>14.4f}"
)
print(
    f"{'FusionGRU':<15}"
    f"{gru_results['accuracy']:>14.4f}"
    f"{gru_results['f1']:>14.4f}"
)
print("=" * 60)
print(f"\nBoth models saved:")
print(f"  FusionCNN -> '{FUSION_CNN_PATH}'")
print(f"  FusionGRU -> '{FUSION_GRU_PATH}'")
print("\nPlease review the results above and select your preferred model.")