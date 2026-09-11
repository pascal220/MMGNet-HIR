# imu_cnn_window_model.py
# Window-based CNN backbone for IMU intent recognition
# Based on Su et al., "A CNN-Based Method for Intent Recognition Using Inertial
# Measurement Units and Intelligent Lower Limb Prosthesis"
# IEEE TNSRE, Vol. 27, No. 5, May 2019
#
# Amendments vs. original imu_cnn_model.py:
#   - Input shape  : (batch, 6, 125, 4) with 4 overlapping windows
#   - First layer  : Conv2D on (125, 4) spatial dimensions
#   - Window collapse: GlobalAveragePool2D after first conv
#   - Rest of architecture: Unchanged (Inception blocks, GAP1D, FC)
#   - Optuna HPO   : Added first_conv_filters and first_conv_kernel_width

import ast
import os
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.pruners  import MedianPruner
from optuna.samplers import TPESampler

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

class _ConvBnRelu(nn.Module):
    """Conv1d -> BatchNorm1d -> ReLU."""

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        stride:       int = 1,
        padding:      int = 0,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,          # BN subsumes bias
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _InceptionBlock(nn.Module):
    """
    Inception-style block (Fig. 3, Su et al. 2019).

    Structure:
        Input
          |
        ConvStride  (kernel_a, stride=2, padding=0)   -- halves length
          |
        +------------------+
        |                  |
      Conv_A (kernel_a)  Conv_B (kernel_b)   -- 'same' padding
        |                  |
        +--------+---------+
                 |
           Depth Concat  =>  out_channels * 2 feature maps

    Args:
        in_channels  : input feature maps
        out_channels : filters per branch (total output = out_channels * 2)
        kernel_a     : kernel size for branch A and strided conv (e.g. 3)
        kernel_b     : kernel size for branch B                  (e.g. 5)
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_a:     int = 3,
        kernel_b:     int = 5,
    ):
        super().__init__()

        # Strided conv -- halves temporal dimension
        self.conv_stride = _ConvBnRelu(
            in_channels, out_channels,
            kernel_size=kernel_a,
            stride=2,
            padding=0,
        )

        # Branch A: 'same' padding => padding = kernel_a // 2
        self.conv_a = _ConvBnRelu(
            out_channels, out_channels,
            kernel_size=kernel_a,
            stride=1,
            padding=kernel_a // 2,
        )

        # Branch B: 'same' padding => padding = kernel_b // 2
        self.conv_b = _ConvBnRelu(
            out_channels, out_channels,
            kernel_size=kernel_b,
            stride=1,
            padding=kernel_b // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x        = self.conv_stride(x)
        branch_a = self.conv_a(x)
        branch_b = self.conv_b(x)
        return torch.cat([branch_a, branch_b], dim=1)   # depth concat


# ============================================================================
# Main Model
# ============================================================================

class IntentCNNWindow(nn.Module):
    """
    Window-based CNN for motion intent recognition.

    Handles overlapping window inputs via a Conv2D first layer followed by
    GlobalAveragePool to collapse the window dimension, then feeds into
    the original Inception block architecture.

    Args:
        in_channels        : IMU signal channels (default 6)
        num_classes        : output classes (default 7)
        first_conv_filters : number of filters in first Conv2D layer
        first_conv_kernel_width : kernel width for first Conv2D (depth fixed at 4)
        block_filters      : list of per-branch filter counts for Inception blocks
        kernel_pairs       : list of (kernel_a, kernel_b) tuples for Inception blocks
    """

    def __init__(
        self,
        in_channels:             int,
        num_classes:             int,
        first_conv_filters:      int,
        first_conv_kernel_width: int,
        block_filters:           list[int],
        kernel_pairs:            list[tuple],
    ):
        super().__init__()

        assert len(block_filters) == len(kernel_pairs), (
            "block_filters and kernel_pairs must have the same length"
        )

        # ── New first layer: Conv2D on (125, 4) spatial dimensions ──────────
        self.first_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_conv_filters,
            kernel_size=(first_conv_kernel_width, 4),  # (width, depth=4)
            padding='same',
            bias=False,
        )
        self.bn_first = nn.BatchNorm2d(first_conv_filters)
        self.relu = nn.ReLU(inplace=True)

        # ── Collapse window dimension via GlobalAveragePool ─────────────────
        self.window_pool = nn.AdaptiveAvgPool2d((None, 1))  # (N, C, 125, 1)

        # ── Build Inception blocks dynamically ──────────────────────────────
        blocks       = []
        current_ch   = first_conv_filters

        for filters, (ka, kb) in zip(block_filters, kernel_pairs):
            blocks.append(
                _InceptionBlock(
                    in_channels=current_ch,
                    out_channels=filters,
                    kernel_a=ka,
                    kernel_b=kb,
                )
            )
            current_ch = filters * 2    # depth concat doubles channels

        self.blocks      = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc          = nn.Linear(current_ch, num_classes)

        # Store config for inspection / serialisation
        self.config = dict(
            in_channels             = in_channels,
            num_classes             = num_classes,
            first_conv_filters      = first_conv_filters,
            first_conv_kernel_width = first_conv_kernel_width,
            block_filters           = block_filters,
            kernel_pairs            = kernel_pairs,
        )

        self._initialise_weights()

    # ── Weight initialisation ───────────────────────────────────────────────
    def _initialise_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    # ── Forward pass ────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, 6, 125, 4)  - 6 IMU channels, 125 time steps, 4 windows
        Returns:
            logits : (batch, num_classes)
        """
        # First Conv2D layer
        x = self.first_conv(x)          # (batch, first_conv_filters, 125, 4)
        x = self.bn_first(x)
        x = self.relu(x)

        # Collapse window dimension
        x = self.window_pool(x)         # (batch, first_conv_filters, 125, 1)
        x = x.squeeze(-1)               # (batch, first_conv_filters, 125)

        # Inception blocks (unchanged from original)
        x = self.blocks(x)              # (batch, C, length)
        x = self.global_pool(x)         # (batch, C, 1)
        x = x.squeeze(-1)               # (batch, C)
        x = self.fc(x)                  # (batch, num_classes)
        return x


# ============================================================================
# Trainer
# ============================================================================

class IntentCNNWindowTrainer:
    """
    Manages training, evaluation, inference, and persistence of IntentCNNWindow.

    Accepts a hyperparams dict (typically supplied by Optuna) that controls
    both the optimiser and the learning rate schedule.

    Supported optimisers:
        'SGD'  : lr, momentum, weight_decay, lr_step_size, lr_decay
        'Adam' : lr, weight_decay, beta1, beta2
    """

    _SGD_DEFAULTS = dict(
        optimizer    = "SGD",
        lr           = 0.01,
        momentum     = 0.9,
        weight_decay = 0.0005,
        lr_decay     = 0.1,
        lr_step_size = 25,
        batch_size   = 200,
        epochs       = 50,
    )

    _ADAM_DEFAULTS = dict(
        optimizer    = "Adam",
        lr           = 1e-3,
        weight_decay = 1e-4,
        beta1        = 0.9,
        beta2        = 0.999,
        lr_decay     = 0.1,
        lr_step_size = 25,
        batch_size   = 200,
        epochs       = 50,
    )

    def __init__(
        self,
        model:       IntentCNNWindow,
        hyperparams: dict | None = None,
    ):
        self.cfg = {**self._SGD_DEFAULTS, **(hyperparams or {})}

        # ── Device ──────────────────────────────────────────────────────────
        self.device = resolve_device(self.cfg.get("device", "auto"))

        # ── Model ───────────────────────────────────────────────────────────
        self.model = model.to(self.device)
        class_weights = self.cfg.get("class_weights")
        loss_weights = (
            None
            if class_weights is None
            else torch.as_tensor(class_weights, dtype=torch.float32, device=self.device)
        )
        self.criterion = nn.CrossEntropyLoss(weight=loss_weights)

        # ── Optimiser ───────────────────────────────────────────────────────
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

        # ── LR Scheduler ────────────────────────────────────────────────────
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size = self.cfg.get("lr_step_size", 25),
            gamma     = self.cfg.get("lr_decay",     0.1),
        )

        # ── History ─────────────────────────────────────────────────────────
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [], "val_f1": [],
        }

    # ── Internal epoch runner ───────────────────────────────────────────────
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

    # ── Public API ──────────────────────────────────────────────────────────
    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader | None = None,
        epochs:       int | None        = None,
        verbose:      bool              = True,
        trial:        optuna.Trial | None = None,   # for Optuna pruning
    ) -> dict:
        """
        Train the model.

        Args:
            train_loader : DataLoader yielding (X, y)
                           X : (batch, 6, 125, 4)   y : (batch,)
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
            self.scheduler.step()

            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)

            val_str = ""
            val_acc = 0.0

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
                    # Report combined metric to Optuna
                    combined = self._combined_metric(v_acc, v_f1)
                    trial.report(combined, step=epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            if verbose:
                lr_now = self.scheduler.get_last_lr()[0]
                print(
                    f"Epoch [{epoch:>3}/{n_epochs}]"
                    f"  train_loss: {tr_loss:.4f}"
                    f"  train_acc: {tr_acc:.4f}"
                    f"{val_str}"
                    f"  lr: {lr_now:.6f}"
                )

        return self.history

    # ── Evaluation ──────────────────────────────────────────────────────────
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

    # ── Inference ───────────────────────────────────────────────────────────
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict class indices.

        Args:
            X : (batch, 6, 125, 4) or (6, 125, 4) for a single sample
        Returns:
            predicted class indices, shape (batch,)
        """
        self.model.eval()
        if X.dim() == 3:
            X = X.unsqueeze(0)
        X = X.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            preds = self.model(X).argmax(dim=1)
        return preds.cpu()

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.

        Args:
            X : (batch, 6, 125, 4) or (6, 125, 4)
        Returns:
            probabilities, shape (batch, num_classes)
        """
        self.model.eval()
        if X.dim() == 3:
            X = X.unsqueeze(0)
        X = X.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            probs = torch.softmax(self.model(X), dim=1)
        return probs.cpu()

    # ── Persistence ─────────────────────────────────────────────────────────
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
        print(f"[IntentCNNWindowTrainer] Saved to '{path}'")

    def load(self, path: str):
        """Load model weights, optimiser state, and config."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.history = ckpt.get("history", self.history)
        self.cfg     = ckpt.get("cfg",     self.cfg)
        print(f"[IntentCNNWindowTrainer] Loaded from '{path}'")

    # ── Helpers ─────────────────────────────────────────────────────────────
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

class IntentCNNWindowTuner:
    """
    Wraps an Optuna study to find the best IntentCNNWindow hyperparameters.

    Searches over:
        First layer  : first_conv_filters, first_conv_kernel_width
        Architecture : number of blocks (1-3), filters per block,
                       kernel pair per block
        Optimiser    : SGD or Adam (with respective hyperparameters)
        Training     : batch size, lr schedule

    Optimisation target:
        Maximise  0.5 * val_accuracy + 0.5 * val_macro_F1

    Usage:
        tuner      = IntentCNNWindowTuner(train_loader, val_loader)
        best_model = tuner.run(n_trials=50, timeout=3600)
        tuner.plot_results(save_dir="optuna_plots")
    """

    # Search space bounds
    _SEARCH = dict(
        # NEW: First layer
        first_conv_filters      = [16, 32, 64, 128],
        first_conv_kernel_width = [3, 5, 7],
        # Architecture (Inception blocks)
        n_blocks      = (1, 3),
        filters       = [8, 16, 32, 64],
        kernel_pairs  = [(3, 5), (3, 7), (5, 7)],
        # Training
        batch_size    = [64, 128, 200, 256],
        epochs        = 50,                         # fixed per trial
        # SGD
        sgd_lr        = (1e-4, 1e-1),
        sgd_momentum  = (0.70, 0.99),
        sgd_wd        = (1e-6, 1e-2),
        sgd_decay     = (0.05, 0.50),
        sgd_step      = (10,   40),
        # Adam
        adam_lr       = (1e-4, 1e-2),
        adam_wd       = (1e-6, 1e-2),
        adam_beta1    = (0.85, 0.99),
        adam_beta2    = (0.90, 0.9999),
        # Shared LR schedule
        lr_step_size  = (10, 40),
        lr_decay      = (0.05, 0.50),
    )

    def __init__(
        self,
        train_loader:  DataLoader,
        val_loader:    DataLoader,
        in_channels:   int = 6,
        num_classes:   int = 7,
        search_space:  dict | None = None,
    ):
        """
        Args:
            train_loader  : training DataLoader
            val_loader    : validation DataLoader
            in_channels   : IMU channels (default 6)
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
        first_conv_kernel_width = trial.suggest_categorical(
            "first_conv_kernel_width", self.search["first_conv_kernel_width"]
        )

        # ── 2. Sample architecture (Inception blocks) ───────────────────────
        n_blocks = trial.suggest_int(
            "n_blocks",
            self.search["n_blocks"][0],
            self.search["n_blocks"][1],
        )

        block_filters = []
        kernel_pairs  = []

        for i in range(n_blocks):
            filters = trial.suggest_categorical(
                f"block_{i}_filters", self.search["filters"]
            )
            kpair = trial.suggest_categorical(
                f"block_{i}_kernel_pair",
                [str(k) for k in self.search["kernel_pairs"]]
            )
            block_filters.append(filters)
            kernel_pairs.append(ast.literal_eval(kpair))

        # ── 3. Sample optimiser + hyperparameters ───────────────────────────
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
                lr_step_size = trial.suggest_int(
                    "lr_step_size", *self.search["lr_step_size"]
                ),
                lr_decay     = trial.suggest_float(
                    "lr_decay", *self.search["lr_decay"]
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
                lr_step_size = trial.suggest_int(
                    "lr_step_size", *self.search["lr_step_size"]
                ),
                lr_decay     = trial.suggest_float(
                    "lr_decay", *self.search["lr_decay"]
                ),
            )

        # ── 4. Sample batch size ────────────────────────────────────────────
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

        # ── 5. Build model + trainer ────────────────────────────────────────
        model = IntentCNNWindow(
            in_channels             = self.in_channels,
            num_classes             = self.num_classes,
            first_conv_filters      = first_conv_filters,
            first_conv_kernel_width = first_conv_kernel_width,
            block_filters           = block_filters,
            kernel_pairs            = kernel_pairs,
        )
        trainer = IntentCNNWindowTrainer(model, hyperparams)

        # ── 6. Train (pruning enabled via trial argument) ───────────────────
        try:
            trainer.fit(
                t_loader,
                v_loader,
                epochs  = self.search["epochs"],
                verbose = False,
                trial   = trial,
            )
        except optuna.exceptions.TrialPruned:
            raise   # re-raise so Optuna records the pruning

        # ── 7. Return combined metric ───────────────────────────────────────
        scores = [
            IntentCNNWindowTrainer._combined_metric(acc, f1)
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
        n_trials:    int  = 50,
        timeout:     int | None = 3600,   # seconds (1 hour)
        show_progress: bool = True,
        storage: str | None = None,
        study_name: str | None = None,
        load_if_exists: bool = False,
    ) -> IntentCNNWindow:
        """
        Run the Optuna hyperparameter search.

        Args:
            n_trials     : maximum number of trials
            timeout      : hard stop in seconds (default 3600 = 1 hour)
            show_progress: display Optuna progress bar

        Returns:
            IntentCNNWindow model built with the best found hyperparameters
        """
        # Suppress Optuna's per-trial INFO logs for cleanliness
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.study = optuna.create_study(
            direction = "maximize",
            sampler   = TPESampler(seed=int(self.search.get("seed", 42))),
            pruner    = MedianPruner(
                n_startup_trials  = 5,    # don't prune first 5 trials
                n_warmup_steps    = 10,   # don't prune first 10 epochs
                interval_steps    = 1,
            ),
            storage=storage,
            study_name=study_name,
            load_if_exists=load_if_exists,
        )

        print(
            f"\n[IntentCNNWindowTuner] Starting Optuna search"
            f"  |  n_trials={n_trials}  timeout={timeout}s"
            f"  |  device={self.device}\n"
        )

        self.study.optimize(
            self._objective,
            n_trials         = n_trials,
            timeout          = timeout,
            show_progress_bar= show_progress,
            gc_after_trial   = True,      # free GPU memory between trials
        )

        # ── Report results ──────────────────────────────────────────────────
        best          = self.study.best_trial
        self._best_params = best.params

        print(f"\n{'='*60}")
        print(f"[IntentCNNWindowTuner] Search complete")
        print(f"  Best trial     : #{best.number}")
        print(f"  Best metric    : {best.value:.4f}  "
              f"(0.5*val_acc + 0.5*val_F1)")
        print(f"  Best params    :")
        for k, v in self._best_params.items():
            print(f"    {k:<22}: {v}")
        print(f"{'='*60}\n")

        # ── Rebuild best model ──────────────────────────────────────────────
        self._best_model = self._build_best_model()
        return self._best_model

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _build_best_model(self) -> IntentCNNWindow:
        """Reconstruct IntentCNNWindow from best trial params."""
        if self._best_params is None:
            raise RuntimeError("Call run() before _build_best_model()")
        p         = self._best_params
        n_blocks  = p["n_blocks"]

        first_conv_filters      = p["first_conv_filters"]
        first_conv_kernel_width = p["first_conv_kernel_width"]
        block_filters = [p[f"block_{i}_filters"]    for i in range(n_blocks)]
        kernel_pairs  = [
            ast.literal_eval(p[f"block_{i}_kernel_pair"]) for i in range(n_blocks)
        ]

        return IntentCNNWindow(
            in_channels             = self.in_channels,
            num_classes             = self.num_classes,
            first_conv_filters      = first_conv_filters,
            first_conv_kernel_width = first_conv_kernel_width,
            block_filters           = block_filters,
            kernel_pairs            = kernel_pairs,
        )

    def get_best_params(self) -> dict:
        """Return the best hyperparameter dict found by Optuna."""
        if self._best_params is None:
            raise RuntimeError("Call run() before get_best_params()")
        return self._best_params

    def get_best_model(self) -> IntentCNNWindow:
        """Return the IntentCNNWindow built with the best hyperparameters."""
        if self._best_model is None:
            raise RuntimeError("Call run() before get_best_model()")
        return self._best_model

    # ── Visualisation ────────────────────────────────────────────────────────
    def plot_results(self, save_dir: str = "optuna_plots"):
        """
        Generate and save Optuna visualisation plots to disk.

        Plots saved:
            1. optimisation_history.html  -- metric vs trial number
            2. param_importances.html     -- which params matter most
            3. parallel_coordinate.html   -- hyperparameter relationships
            4. slice.html                 -- metric vs each parameter

        Args:
            save_dir : directory to save HTML plots
        """
        if self.study is None:
            raise RuntimeError("Call run() before plot_results()")

        if not PLOTLY_AVAILABLE:
            print(
                "[IntentCNNWindowTuner] Plotly not installed. "
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
            print(f"[IntentCNNWindowTuner] Saved plot: '{path}'")