# mmg_cnn_window_model.py
# Window-based CNN backbone for MMG locomotion recognition
# Based on Wattanasiri et al., "Gesture Recognition Through Mechanomyogram Signals:
# An Adaptive Framework for Arm Posture Variability"
# IEEE JBHI, Vol. 29, No. 4, April 2025
#
# Amendments vs. original mmg_cnn_model.py:
#   - Input shape  : (batch, 5, 40, 125, 4) with 4 overlapping windows
#   - First layer  : Conv3D on (40, 125, 4) volume dimensions
#   - Window collapse: GlobalMaxPool3D after first conv
#   - Conv Block 1 : Replaced by new Conv3D layer
#   - Rest of architecture: Conv2D blocks 2, 3, 4 unchanged
#   - Optuna HPO   : Added first_conv_filters, first_conv_kernel_freq, first_conv_kernel_time

import os
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.pruners  import MedianPruner
from optuna.samplers import TPESampler
from typing import TypedDict

import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from device_utils import resolve_device

# Optuna visualisation (requires optuna[visualization] and plotly)
try:
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
        plot_slice,
    )
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ============================================================================
# Building Blocks
# ============================================================================

class _ConvBlock2D(nn.Module):
    """
    One convolutional block from Wattanasiri et al. (Fig. 6):
        Conv2d -> BatchNorm2d -> ReLU -> Dropout2d

    Args:
        in_channels  : input feature maps
        out_channels : number of convolutional filters
        kernel_size  : square kernel size (e.g. 7 -> 7x7)
        stride       : convolutional stride (controls spatial downsampling)
        dropout_rate : dropout probability applied after ReLU
        padding      : 'same' padding to preserve spatial dims when stride=1,
                       or 0 when stride > 1 (matches paper's downsampling)
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        stride:       int   = 1,
        dropout_rate: float = 0.25,
        padding:      int   = 0,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size = kernel_size,
                stride      = stride,
                padding     = padding,
                bias        = False,     # BN subsumes bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ============================================================================
# Main Model
# ============================================================================

class LocomotionMMGCNNWindowConfig(TypedDict):
    in_channels: int
    num_classes: int
    first_conv_filters: int
    first_conv_kernel_freq: int
    first_conv_kernel_time: int
    block_filters: list[int]
    kernel_sizes: list[int]
    strides: list[int]
    dropout_rates: list[float]
    fc_hidden: int | None


class LocomotionMMGCNNWindow(nn.Module):
    """
    Window-based 2D CNN for locomotion recognition (Wattanasiri et al., 2025).

    Processes CWT scalogram windows of shape (batch, channels, freq, time, windows).

    The first layer is a Conv3D that processes the 3D volume (freq, time, windows),
    followed by GlobalMaxPool3D to collapse the window dimension. The output then
    feeds into the remaining Conv2D blocks from the original architecture.

    Args:
        in_channels             : CWT input channels / sensors  (default 5)
        num_classes             : output locomotion classes     (default 7)
        first_conv_filters      : number of filters in first Conv3D layer
        first_conv_kernel_freq  : kernel size along frequency dimension
        first_conv_kernel_time  : kernel size along time dimension
        block_filters           : list of filter counts for Conv2D blocks (blocks 2, 3, 4)
        kernel_sizes            : list of square kernel sizes for Conv2D blocks
        strides                 : list of strides for Conv2D blocks
        dropout_rates           : list of dropout rates for Conv2D blocks
        fc_hidden               : hidden units in optional FC layer before softmax
                                  (None = global pool directly to classifier)
    """

    def __init__(
        self,
        in_channels:             int,
        num_classes:             int,
        first_conv_filters:      int,
        first_conv_kernel_freq:  int,
        first_conv_kernel_time:  int,
        block_filters:           list[int],
        kernel_sizes:            list[int],
        strides:                 list[int],
        dropout_rates:           list[float],
        fc_hidden:               int | None = None,
    ):
        super().__init__()

        n = len(block_filters)
        assert len(kernel_sizes)  == n, "kernel_sizes must match block_filters length"
        assert len(strides)       == n, "strides must match block_filters length"
        assert len(dropout_rates) == n, "dropout_rates must match block_filters length"

        # ── New first layer: Conv3D on (40, 125, 4) volume ──────────────────
        self.first_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=first_conv_filters,
            kernel_size=(first_conv_kernel_freq, first_conv_kernel_time, 4),
            padding='same',
            bias=False,
        )
        self.bn_first = nn.BatchNorm3d(first_conv_filters)
        self.relu = nn.ReLU(inplace=True)

        # ── Collapse window dimension via GlobalMaxPool ──────────────────────
        self.window_pool = nn.AdaptiveMaxPool3d((None, None, 1))

        # ── Build Conv2D blocks dynamically (blocks 2, 3, 4 from original) ──
        blocks     = []
        current_ch = first_conv_filters  # Input from first Conv3D layer

        for filters, kernel, stride, dropout in zip(
            block_filters, kernel_sizes, strides, dropout_rates
        ):
            # Use padding=0 when stride > 1 (spatial downsampling, paper style)
            # Use same-style padding when stride = 1 to preserve spatial dims
            padding = kernel // 2 if stride == 1 else 0

            blocks.append(
                _ConvBlock2D(
                    in_channels  = current_ch,
                    out_channels = filters,
                    kernel_size  = kernel,
                    stride       = stride,
                    dropout_rate = dropout,
                    padding      = padding,
                )
            )
            current_ch = filters

        self.conv_blocks = nn.Sequential(*blocks)

        # ── Global average pooling ───────────────────────────────────────────
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # ── Classification head ──────────────────────────────────────────────
        if fc_hidden is not None:
            self.classifier = nn.Sequential(
                nn.Linear(current_ch, fc_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(fc_hidden, num_classes),
            )
        else:
            self.classifier = nn.Linear(current_ch, num_classes)

        # ── Store config for serialisation ───────────────────────────────────
        self.config: LocomotionMMGCNNWindowConfig = {
            "in_channels":             in_channels,
            "num_classes":             num_classes,
            "first_conv_filters":      first_conv_filters,
            "first_conv_kernel_freq":  first_conv_kernel_freq,
            "first_conv_kernel_time":  first_conv_kernel_time,
            "block_filters":           block_filters,
            "kernel_sizes":            kernel_sizes,
            "strides":                 strides,
            "dropout_rates":           dropout_rates,
            "fc_hidden":               fc_hidden,
        }

        self._initialise_weights()

    # ── Weight initialisation ────────────────────────────────────────────────
    def _initialise_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    # ── Forward pass ─────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, 5, 40, 125, 4)  - 5 MMG sensors, 40 freq bins, 125 time steps, 4 windows
        Returns:
            logits : (batch, num_classes)
        """
        # First Conv3D layer
        x = self.first_conv(x)          # (batch, first_conv_filters, 40, 125, 4)
        x = self.bn_first(x)
        x = self.relu(x)

        # Collapse window dimension
        x = self.window_pool(x)         # (batch, first_conv_filters, 40, 125, 1)
        x = x.squeeze(-1)               # (batch, first_conv_filters, 40, 125)

        # Conv2D blocks (unchanged from original)
        x = self.conv_blocks(x)         # (batch, C, H', W')
        x = self.global_pool(x)         # (batch, C,  1,  1)
        x = x.flatten(start_dim=1)      # (batch, C)
        x = self.classifier(x)          # (batch, num_classes)
        return x


# ============================================================================
# Trainer
# ============================================================================

class LocomotionMMGCNNWindowTrainer:
    """
    Manages training, evaluation, inference, and persistence of LocomotionMMGCNNWindow.

    Key differences from LocomotionMMGCNNTrainer:
        - Handles 5D input tensors (batch, channels, freq, time, windows)
        - Uses ReduceLROnPlateau scheduler (matches paper)
        - Supports optional FC hidden layer in classifier head

    Supported optimisers:
        'SGD'  : lr, momentum, weight_decay
        'Adam' : lr, weight_decay, beta1, beta2
    """

    _SGD_DEFAULTS = dict(
        optimizer    = "SGD",
        lr           = 0.005,
        momentum     = 0.9,
        weight_decay = 0.0005,
        # ReduceLROnPlateau params
        lr_factor    = 0.5,
        lr_patience  = 10,
        lr_min       = 1e-6,
        batch_size   = 32,
        epochs       = 100,
    )

    _ADAM_DEFAULTS = dict(
        optimizer    = "Adam",
        lr           = 0.005,
        weight_decay = 1e-4,
        beta1        = 0.9,
        beta2        = 0.999,
        # ReduceLROnPlateau params
        lr_factor    = 0.5,
        lr_patience  = 10,
        lr_min       = 1e-6,
        batch_size   = 32,
        epochs       = 100,
    )

    def __init__(
        self,
        model:       LocomotionMMGCNNWindow,
        hyperparams: dict | None = None,
    ):
        self.cfg = {**self._SGD_DEFAULTS, **(hyperparams or {})}

        # ── Device ───────────────────────────────────────────────────────────
        self.device = resolve_device(self.cfg.get("device", "auto"))

        # ── Model ────────────────────────────────────────────────────────────
        self.model = model.to(self.device)
        class_weights = self.cfg.get("class_weights")
        loss_weights = (
            None
            if class_weights is None
            else torch.as_tensor(class_weights, dtype=torch.float32, device=self.device)
        )
        self.criterion = nn.CrossEntropyLoss(weight=loss_weights)

        # ── Optimiser ────────────────────────────────────────────────────────
        opt_name = self.cfg.get("optimizer", "SGD")

        if opt_name == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr           = self.cfg["lr"],
                momentum     = self.cfg.get("momentum", 0.9),
                weight_decay = self.cfg.get("weight_decay", 0.0005),
            )
        elif opt_name == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr           = self.cfg["lr"],
                weight_decay = self.cfg.get("weight_decay", 1e-4),
                betas        = (
                    self.cfg.get("beta1", 0.9),
                    self.cfg.get("beta2", 0.999),
                ),
            )
        else:
            raise ValueError(f"Unknown optimizer: '{opt_name}'")

        # ── LR Scheduler: ReduceLROnPlateau (matches paper) ──────────────────
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode     = "min",
            factor   = self.cfg.get("lr_factor",   0.5),
            patience = self.cfg.get("lr_patience",  10),
            min_lr   = self.cfg.get("lr_min",      1e-6),
        )

        # ── History ──────────────────────────────────────────────────────────
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [], "val_f1": [],
        }

    # ── Internal epoch runner ────────────────────────────────────────────────
    def _run_epoch(
        self,
        loader:   DataLoader,
        training: bool,
    ) -> tuple[float, float, float]:
        """
        Run one epoch.

        Returns:
            (mean_loss, accuracy, macro_f1)
        """
        self.model.train(training)
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.set_grad_enabled(training):
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device, dtype=torch.float32)
                y_batch = y_batch.to(self.device, dtype=torch.long)

                logits = self.model(X_batch)
                loss   = self.criterion(logits, y_batch)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * X_batch.size(0)
                preds       = logits.argmax(dim=1).cpu().numpy()
                labels      = y_batch.cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels)

        all_preds  = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        n          = len(all_labels)

        mean_loss = total_loss / n
        accuracy  = float((all_preds == all_labels).mean())
        macro_f1  = float(
            f1_score(all_labels, all_preds, average="macro", zero_division=0)
        )
        return mean_loss, accuracy, macro_f1

    # ── Public API ───────────────────────────────────────────────────────────
    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader | None   = None,
        epochs:       int | None          = None,
        verbose:      bool                = True,
        trial:        optuna.Trial | None = None,
    ) -> dict:
        """
        Train the model.

        Args:
            train_loader : DataLoader yielding (X, y)
                           X : (batch, 5, 40, 125, 4)   y : (batch,)
            val_loader   : optional validation DataLoader
            epochs       : overrides cfg['epochs'] if provided
            verbose      : print per-epoch metrics
            trial        : Optuna trial object (enables pruning when provided)

        Returns:
            history dict
        """
        n_epochs = epochs or self.cfg["epochs"]

        for epoch in range(1, n_epochs + 1):
            tr_loss, tr_acc, _ = self._run_epoch(train_loader, training=True)

            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)

            val_str = ""
            val_acc = 0.0
            v_loss  = tr_loss   # fallback for scheduler if no val_loader

            if val_loader is not None:
                v_loss, v_acc, v_f1 = self._run_epoch(val_loader, training=False)
                self.history["val_loss"].append(v_loss)
                self.history["val_acc"].append(v_acc)
                self.history["val_f1"].append(v_f1)
                val_acc = v_acc
                val_str = (
                    f"  |  val_loss: {v_loss:.4f}"
                    f"  val_acc: {v_acc:.4f}"
                    f"  val_f1: {v_f1:.4f}"
                )

                # ── Optuna pruning ──────────────────────────────────────────
                if trial is not None:
                    combined = self._combined_metric(v_acc, v_f1)
                    trial.report(combined, step=epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            # ── ReduceLROnPlateau steps on val_loss ─────────────────────────
            self.scheduler.step(v_loss)

            if verbose:
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch [{epoch:>3}/{n_epochs}]"
                    f"  train_loss: {tr_loss:.4f}"
                    f"  train_acc: {tr_acc:.4f}"
                    f"{val_str}"
                    f"  lr: {current_lr:.6f}"
                )

        return self.history

    # ── Evaluation ───────────────────────────────────────────────────────────
    def evaluate(self, loader: DataLoader) -> dict:
        """
        Evaluate on a DataLoader.

        Returns:
            dict with keys 'loss', 'accuracy', 'f1'
        """
        loss, acc, f1 = self._run_epoch(loader, training=False)
        print(
            f"[Evaluate]"
            f"  loss: {loss:.4f}"
            f"  accuracy: {acc:.4f}"
            f"  macro_f1: {f1:.4f}"
        )
        return {"loss": loss, "accuracy": acc, "f1": f1}

    # ── Inference ────────────────────────────────────────────────────────────
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict class indices.

        Args:
            X : (batch, 5, 40, 125, 4) or (5, 40, 125, 4) for a single sample
        Returns:
            predicted class indices, shape (batch,)
        """
        self.model.eval()
        if X.dim() == 4:
            X = X.unsqueeze(0)
        X = X.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            preds = self.model(X).argmax(dim=1)
        return preds.cpu()

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.

        Args:
            X : (batch, 5, 40, 125, 4) or (5, 40, 125, 4)
        Returns:
            probabilities, shape (batch, num_classes)
        """
        self.model.eval()
        if X.dim() == 4:
            X = X.unsqueeze(0)
        X = X.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            probs = torch.softmax(self.model(X), dim=1)
        return probs.cpu()

    # ── Persistence ──────────────────────────────────────────────────────────
    def save(self, path: str):
        """Save model weights, optimiser state, and config."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "model_state_dict":     self.model.state_dict(),
                "model_config":         self.model.config,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "history":              self.history,
                "cfg":                  self.cfg,
            },
            path,
        )
        print(f"[LocomotionMMGCNNWindowTrainer] Saved to '{path}'")

    def load(self, path: str):
        """Load model weights, optimiser state, and config."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.history = ckpt.get("history", self.history)
        self.cfg     = ckpt.get("cfg",     self.cfg)
        print(f"[LocomotionMMGCNNWindowTrainer] Loaded from '{path}'")

    # ── Helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _combined_metric(acc: float, f1: float) -> float:
        """
        Scalar metric reported to Optuna.
        Equal-weight average of validation accuracy and macro F1.
        """
        return 0.5 * acc + 0.5 * f1


# ============================================================================
# Optuna Tuner
# ============================================================================

class LocomotionMMGCNNWindowTuner:
    """
    Wraps an Optuna study to find the best LocomotionMMGCNNWindow hyperparameters.

    Searches over:
        First layer  : first_conv_filters, first_conv_kernel_freq, first_conv_kernel_time
        Architecture : number of Conv2D blocks (1-3), filters per block,
                       kernel size, stride, dropout per block
        Optimiser    : SGD or Adam (with respective hyperparameters)
        Training     : batch size, lr schedule
        Classifier   : optional FC hidden layer

    Optimisation target:
        Maximise  0.5 * val_accuracy + 0.5 * val_macro_F1

    Usage:
        tuner      = LocomotionMMGCNNWindowTuner(train_loader, val_loader)
        best_model = tuner.run(n_trials=50, timeout=3600)
        tuner.plot_results(save_dir="optuna_plots")
    """

    # Search space bounds
    _SEARCH = dict(
        # NEW: First layer
        first_conv_filters      = [16, 32, 64, 128],
        first_conv_kernel_freq  = [3, 5, 7],
        first_conv_kernel_time  = [3, 5, 7],
        # Architecture (Conv2D blocks)
        n_blocks      = (1, 3),
        filters       = [16, 32, 64, 128],
        kernel_sizes  = [3, 5, 7],
        strides       = [1, 2, 3],
        dropout_rates = (0.1, 0.5),
        # Classifier
        fc_hidden     = [64, 128, 256],
        # Training
        batch_size    = [16, 32, 64],
        epochs        = 50,                         # fixed per trial
        # SGD
        sgd_lr        = (1e-4, 1e-1),
        sgd_momentum  = (0.70, 0.99),
        sgd_wd        = (1e-6, 1e-2),
        # Adam
        adam_lr       = (1e-4, 1e-2),
        adam_wd       = (1e-6, 1e-2),
        adam_beta1    = (0.85, 0.99),
        adam_beta2    = (0.90, 0.9999),
        # ReduceLROnPlateau
        lr_factor     = (0.1, 0.7),
        lr_patience   = (5, 20),
        lr_min        = 1e-7,
    )

    def __init__(
        self,
        train_loader:  DataLoader,
        val_loader:    DataLoader,
        in_channels:   int = 5,
        num_classes:   int = 7,
        search_space:  dict | None = None,
    ):
        """
        Args:
            train_loader  : training DataLoader
            val_loader    : validation DataLoader
            in_channels   : MMG channels (default 5)
            num_classes   : output classes (default 7)
            search_space  : optional dict to override _SEARCH bounds
        """
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.in_channels  = in_channels
        self.num_classes  = num_classes
        self.search       = {**self._SEARCH, **(search_space or {})}
        self.device       = resolve_device(self.search.get("device", "auto"))

        self.study        = None
        self._best_model  = None
        self._best_params = None

    # ── Objective function (called once per trial) ──────────────────────────
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Build, train, and evaluate one candidate configuration.
        Returns the combined metric (higher = better).
        """

        # ── 1. Sample first layer parameters ────────────────────────────────
        first_conv_filters = trial.suggest_categorical(
            "first_conv_filters", self.search["first_conv_filters"]
        )
        first_conv_kernel_freq = trial.suggest_categorical(
            "first_conv_kernel_freq", self.search["first_conv_kernel_freq"]
        )
        first_conv_kernel_time = trial.suggest_categorical(
            "first_conv_kernel_time", self.search["first_conv_kernel_time"]
        )

        # ── 2. Sample architecture (Conv2D blocks) ──────────────────────────
        n_blocks = trial.suggest_int(
            "n_blocks",
            self.search["n_blocks"][0],
            self.search["n_blocks"][1],
        )

        block_filters = []
        kernel_sizes  = []
        strides       = []
        dropout_rates = []

        for i in range(n_blocks):
            filters = trial.suggest_categorical(
                f"block_{i}_filters", self.search["filters"]
            )
            kernel = trial.suggest_categorical(
                f"block_{i}_kernel", self.search["kernel_sizes"]
            )
            stride = trial.suggest_categorical(
                f"block_{i}_stride", self.search["strides"]
            )
            dropout = trial.suggest_float(
                f"block_{i}_dropout",
                self.search["dropout_rates"][0],
                self.search["dropout_rates"][1],
            )

            block_filters.append(filters)
            kernel_sizes.append(kernel)
            strides.append(stride)
            dropout_rates.append(dropout)

        # ── 3. Sample classifier FC hidden layer ────────────────────────────
        fc_hidden_str = trial.suggest_categorical(
            "fc_hidden", [str(x) for x in self.search["fc_hidden"]]
        )
        fc_hidden = None if fc_hidden_str == "None" else int(fc_hidden_str)

        # ── 4. Sample optimiser + hyperparameters ───────────────────────────
        opt_name = trial.suggest_categorical("optimizer", ["SGD", "Adam"])

        if opt_name == "SGD":
            hyperparams = dict(
                optimizer    = "SGD",
                lr           = trial.suggest_float(
                    "sgd_lr", *self.search["sgd_lr"], log=True
                ),
                momentum     = trial.suggest_float(
                    "sgd_momentum", *self.search["sgd_momentum"]
                ),
                weight_decay = trial.suggest_float(
                    "sgd_wd", *self.search["sgd_wd"], log=True
                ),
            )
        else:   # Adam
            hyperparams = dict(
                optimizer    = "Adam",
                lr           = trial.suggest_float(
                    "adam_lr", *self.search["adam_lr"], log=True
                ),
                weight_decay = trial.suggest_float(
                    "adam_wd", *self.search["adam_wd"], log=True
                ),
                beta1        = trial.suggest_float(
                    "adam_beta1", *self.search["adam_beta1"]
                ),
                beta2        = trial.suggest_float(
                    "adam_beta2", *self.search["adam_beta2"]
                ),
            )

        # ── 5. Sample ReduceLROnPlateau parameters ──────────────────────────
        hyperparams["lr_factor"]   = trial.suggest_float(
            "lr_factor", *self.search["lr_factor"]
        )
        hyperparams["lr_patience"] = trial.suggest_int(
            "lr_patience", *self.search["lr_patience"]
        )
        hyperparams["lr_min"]      = self.search["lr_min"]

        # ── 6. Sample batch size ────────────────────────────────────────────
        batch_size = trial.suggest_categorical(
            "batch_size", self.search["batch_size"]
        )
        hyperparams["batch_size"] = batch_size
        hyperparams["epochs"]     = self.search["epochs"]
        hyperparams["class_weights"] = self.search.get("class_weights")
        hyperparams["device"] = str(self.device)

        # Rebuild DataLoaders with trial batch size
        train_ds = self.train_loader.dataset
        val_ds   = self.val_loader.dataset
        generator = torch.Generator().manual_seed(
            int(self.search.get("seed", 42)) + trial.number
        )
        t_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, generator=generator,
            pin_memory=self.device.type == "cuda",
        )
        v_loader = DataLoader(
            val_ds, batch_size=batch_size, pin_memory=self.device.type == "cuda"
        )

        # ── 7. Build model + trainer ────────────────────────────────────────
        model = LocomotionMMGCNNWindow(
            in_channels             = self.in_channels,
            num_classes             = self.num_classes,
            first_conv_filters      = first_conv_filters,
            first_conv_kernel_freq  = first_conv_kernel_freq,
            first_conv_kernel_time  = first_conv_kernel_time,
            block_filters           = block_filters,
            kernel_sizes            = kernel_sizes,
            strides                 = strides,
            dropout_rates           = dropout_rates,
            fc_hidden               = fc_hidden,
        )
        trainer = LocomotionMMGCNNWindowTrainer(model, hyperparams)

        # ── 8. Train (pruning enabled via trial argument) ───────────────────
        try:
            trainer.fit(
                t_loader,
                v_loader,
                epochs  = self.search["epochs"],
                verbose = False,
                trial   = trial,
            )
        except optuna.exceptions.TrialPruned:
            raise

        # ── 9. Return combined metric ───────────────────────────────────────
        scores = [
            LocomotionMMGCNNWindowTrainer._combined_metric(acc, f1)
            for acc, f1 in zip(
                trainer.history["val_acc"], trainer.history["val_f1"]
            )
        ]
        best_index = int(np.argmax(scores))
        trial.set_user_attr("best_epoch", best_index + 1)
        trial.set_user_attr(
            "validation_accuracy", trainer.history["val_acc"][best_index]
        )
        trial.set_user_attr(
            "validation_macro_f1", trainer.history["val_f1"][best_index]
        )
        return float(scores[best_index])

    # ── Run the study ────────────────────────────────────────────────────────
    def run(
        self,
        n_trials:      int  = 50,
        timeout:       int | None = 3600,
        show_progress: bool = True,
        storage: str | None = None,
        study_name: str | None = None,
        load_if_exists: bool = False,
    ) -> LocomotionMMGCNNWindow:
        """
        Run the Optuna hyperparameter search.

        Args:
            n_trials     : maximum number of trials
            timeout      : hard stop in seconds (default 3600 = 1 hour)
            show_progress: display Optuna progress bar

        Returns:
            LocomotionMMGCNNWindow model built with the best found hyperparameters
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.study = optuna.create_study(
            direction = "maximize",
            sampler   = TPESampler(seed=int(self.search.get("seed", 42))),
            pruner    = MedianPruner(
                n_startup_trials  = 5,
                n_warmup_steps    = 10,
                interval_steps    = 1,
            ),
            storage=storage,
            study_name=study_name,
            load_if_exists=load_if_exists,
        )

        print(
            f"\n[LocomotionMMGCNNWindowTuner] Starting Optuna search"
            f"  |  n_trials={n_trials}  timeout={timeout}s"
            f"  |  device={self.device}\n"
        )

        self.study.optimize(
            self._objective,
            n_trials         = n_trials,
            timeout          = timeout,
            show_progress_bar= show_progress,
            gc_after_trial   = True,
        )

        # ── Report results ──────────────────────────────────────────────────
        best          = self.study.best_trial
        self._best_params = best.params

        print(f"\n{'='*60}")
        print(f"[LocomotionMMGCNNWindowTuner] Search complete")
        print(f"  Best trial     : #{best.number}")
        print(f"  Best metric    : {best.value:.4f}  "
              f"(0.5*val_acc + 0.5*val_F1)")
        print(f"  Best params    :")
        for k, v in self._best_params.items():
            print(f"    {k:<30}: {v}")
        print(f"{'='*60}\n")

        # ── Rebuild best model ──────────────────────────────────────────────
        self._best_model = self._build_best_model()
        return self._best_model

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _build_best_model(self) -> LocomotionMMGCNNWindow:
        """Reconstruct LocomotionMMGCNNWindow from best trial params."""
        if self._best_params is None:
            raise RuntimeError("Call run() before _build_best_model()")
        p         = self._best_params
        n_blocks  = p["n_blocks"]

        first_conv_filters      = p["first_conv_filters"]
        first_conv_kernel_freq  = p["first_conv_kernel_freq"]
        first_conv_kernel_time  = p["first_conv_kernel_time"]

        block_filters = [p[f"block_{i}_filters"] for i in range(n_blocks)]
        kernel_sizes  = [p[f"block_{i}_kernel"]  for i in range(n_blocks)]
        strides       = [p[f"block_{i}_stride"]  for i in range(n_blocks)]
        dropout_rates = [p[f"block_{i}_dropout"] for i in range(n_blocks)]

        fc_hidden_str = p["fc_hidden"]
        fc_hidden = None if fc_hidden_str == "None" else int(fc_hidden_str)

        return LocomotionMMGCNNWindow(
            in_channels             = self.in_channels,
            num_classes             = self.num_classes,
            first_conv_filters      = first_conv_filters,
            first_conv_kernel_freq  = first_conv_kernel_freq,
            first_conv_kernel_time  = first_conv_kernel_time,
            block_filters           = block_filters,
            kernel_sizes            = kernel_sizes,
            strides                 = strides,
            dropout_rates           = dropout_rates,
            fc_hidden               = fc_hidden,
        )

    def get_best_params(self) -> dict:
        """Return the best hyperparameter dict found by Optuna."""
        if self._best_params is None:
            raise RuntimeError("Call run() before get_best_params()")
        return self._best_params

    def get_best_model(self) -> LocomotionMMGCNNWindow:
        """Return the LocomotionMMGCNNWindow built with the best hyperparameters."""
        if self._best_model is None:
            raise RuntimeError("Call run() before get_best_model()")
        return self._best_model

    # ── Visualisation ────────────────────────────────────────────────────────
    def plot_results(self, save_dir: str = "optuna_plots"):
        """
        Generate and save Optuna visualisation plots to disk.

        Plots saved:
            1. optimisation_history.html
            2. param_importances.html
            3. parallel_coordinate.html
            4. slice.html

        Args:
            save_dir : directory to save HTML plots
        """
        if self.study is None:
            raise RuntimeError("Call run() before plot_results()")

        if not PLOTLY_AVAILABLE:
            print(
                "[LocomotionMMGCNNWindowTuner] Plotly not installed. "
                "Run: pip install optuna[visualization] plotly"
            )
            return

        os.makedirs(save_dir, exist_ok=True)

        plots = {
            "optimisation_history.html" : plot_optimization_history(self.study),
            "param_importances.html"    : plot_param_importances(self.study),
            "parallel_coordinate.html"  : plot_parallel_coordinate(self.study),
            "slice.html"                : plot_slice(self.study),
        }

        for filename, fig in plots.items():
            path = os.path.join(save_dir, filename)
            fig.write_html(path)
            print(f"[LocomotionMMGCNNWindowTuner] Saved plot: '{path}'")