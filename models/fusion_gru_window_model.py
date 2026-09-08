# fusion_gru_window_model.py
# Window-based Fusion GRU for multimodal IMU + MMG recognition
#
# Architecture:
#   - Frozen IMU backbone (IntentCNNWindow)
#   - Frozen MMG backbone (LocomotionMMGCNNWindow)
#   - Trainable GRU fusion layer
#   - Trainable FC classifier
#
# Amendments vs. original fusion_gru_model.py:
#   - Input shapes: IMU (batch, 6, 125, 4), MMG (batch, 5, 40, 125, 4)
#   - Backbones: Load window-based models with Conv2D/Conv3D first layers
#   - Optuna HPO: Added first layer parameters for both backbones

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

from imu_cnn_window_model import IntentCNNWindow
from mmg_cnn_window_model import LocomotionMMGCNNWindow

# Optuna visualisation
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
# Frozen Backbones (same as fusion_cnn_window_model.py)
# ============================================================================

class _FrozenBackbones(nn.Module):
    """
    Loads frozen IMU and MMG backbones from pretrained checkpoints.

    Extracts features from both modalities and concatenates them.

    Args:
        imu_checkpoint : path to IntentCNNWindow checkpoint
        mmg_checkpoint : path to LocomotionMMGCNNWindow checkpoint
        device         : torch device
    """

    def __init__(
        self,
        imu_checkpoint: str,
        mmg_checkpoint: str,
        device:         torch.device,
    ):
        super().__init__()

        # ── Load IMU backbone ───────────────────────────────────────────────
        imu_ckpt = torch.load(imu_checkpoint, map_location=device)
        imu_cfg  = imu_ckpt["model_config"]

        self.imu_backbone = IntentCNNWindow(
            in_channels             = imu_cfg["in_channels"],
            num_classes             = imu_cfg["num_classes"],
            first_conv_filters      = imu_cfg["first_conv_filters"],
            first_conv_kernel_width = imu_cfg["first_conv_kernel_width"],
            block_filters           = imu_cfg["block_filters"],
            kernel_pairs            = imu_cfg["kernel_pairs"],
        )
        self.imu_backbone.load_state_dict(imu_ckpt["model_state_dict"])
        self.imu_backbone.eval()

        # Freeze IMU backbone
        for param in self.imu_backbone.parameters():
            param.requires_grad = False

        # ── Load MMG backbone ───────────────────────────────────────────────
        mmg_ckpt = torch.load(mmg_checkpoint, map_location=device)
        mmg_cfg  = mmg_ckpt["model_config"]

        self.mmg_backbone = LocomotionMMGCNNWindow(
            in_channels             = mmg_cfg["in_channels"],
            num_classes             = mmg_cfg["num_classes"],
            first_conv_filters      = mmg_cfg["first_conv_filters"],
            first_conv_kernel_freq  = mmg_cfg["first_conv_kernel_freq"],
            first_conv_kernel_time  = mmg_cfg["first_conv_kernel_time"],
            block_filters           = mmg_cfg["block_filters"],
            kernel_sizes            = mmg_cfg["kernel_sizes"],
            strides                 = mmg_cfg["strides"],
            dropout_rates           = mmg_cfg["dropout_rates"],
            fc_hidden               = mmg_cfg.get("fc_hidden"),
        )
        self.mmg_backbone.load_state_dict(mmg_ckpt["model_state_dict"])
        self.mmg_backbone.eval()

        # Freeze MMG backbone
        for param in self.mmg_backbone.parameters():
            param.requires_grad = False

        # ── Compute feature dimensions ──────────────────────────────────────
        # IMU: features after global pool (before final FC)
        # MMG: features after global pool (before classifier)

        # For IMU: last block output channels * 2 (depth concat in inception)
        imu_last_filters = imu_cfg["block_filters"][-1]
        self.imu_feature_dim = imu_last_filters * 2

        # For MMG: last conv block output channels
        self.mmg_feature_dim = mmg_cfg["block_filters"][-1]

        self.feature_dim = self.imu_feature_dim + self.mmg_feature_dim

        print(
            f"[_FrozenBackbones] Loaded frozen backbones"
            f"  |  IMU features: {self.imu_feature_dim}"
            f"  |  MMG features: {self.mmg_feature_dim}"
            f"  |  Total: {self.feature_dim}"
        )

    def forward(
        self,
        x_imu: torch.Tensor,
        x_mmg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract and concatenate features from both modalities.

        Args:
            x_imu : (batch, 6, 125, 4)
            x_mmg : (batch, 5, 40, 125, 4)

        Returns:
            fused_features : (batch, imu_feature_dim + mmg_feature_dim)
        """
        with torch.no_grad():
            # ── IMU features ────────────────────────────────────────────────
            # Forward through: first_conv → window_pool → blocks → global_pool
            imu_x = x_imu
            imu_x = self.imu_backbone.first_conv(imu_x)
            imu_x = self.imu_backbone.bn_first(imu_x)
            imu_x = self.imu_backbone.relu(imu_x)
            imu_x = self.imu_backbone.window_pool(imu_x)
            imu_x = imu_x.squeeze(-1)
            imu_x = self.imu_backbone.blocks(imu_x)
            imu_x = self.imu_backbone.global_pool(imu_x)
            imu_features = imu_x.squeeze(-1)  # (batch, imu_feature_dim)

            # ── MMG features ────────────────────────────────────────────────
            # Forward through: first_conv → window_pool → conv_blocks → global_pool
            mmg_x = x_mmg
            mmg_x = self.mmg_backbone.first_conv(mmg_x)
            mmg_x = self.mmg_backbone.bn_first(mmg_x)
            mmg_x = self.mmg_backbone.relu(mmg_x)
            mmg_x = self.mmg_backbone.window_pool(mmg_x)
            mmg_x = mmg_x.squeeze(-1)
            mmg_x = self.mmg_backbone.conv_blocks(mmg_x)
            mmg_x = self.mmg_backbone.global_pool(mmg_x)
            mmg_features = mmg_x.flatten(start_dim=1)  # (batch, mmg_feature_dim)

        # ── Concatenate ─────────────────────────────────────────────────────
        fused = torch.cat([imu_features, mmg_features], dim=1)
        return fused


# ============================================================================
# Fusion Model with GRU
# ============================================================================

class FusionGRUWindow(nn.Module):
    """
    Multimodal fusion model combining frozen IMU and MMG backbones with GRU.

    Architecture:
        Frozen IMU backbone → features
        Frozen MMG backbone → features
        Concatenate
        Reshape to sequence: (batch, seq_len, feature_dim)
        Bidirectional GRU
        Trainable FC head → classifier

    The GRU processes the concatenated features as a temporal sequence,
    allowing the model to learn temporal dependencies in the fused representation.

    Args:
        imu_checkpoint : path to pretrained IntentCNNWindow checkpoint
        mmg_checkpoint : path to pretrained LocomotionMMGCNNWindow checkpoint
        num_classes    : output classes
        gru_hidden_dim : hidden units in GRU layer
        gru_num_layers : number of stacked GRU layers
        gru_dropout    : dropout between GRU layers (if num_layers > 1)
        fc_hidden_dims : list of hidden layer sizes for FC head after GRU
        fc_dropout     : dropout probability in FC head
        seq_len        : sequence length for GRU input (default 1)
    """

    def __init__(
        self,
        imu_checkpoint: str,
        mmg_checkpoint: str,
        num_classes:    int,
        gru_hidden_dim: int,
        gru_num_layers: int = 1,
        gru_dropout:    float = 0.0,
        fc_hidden_dims: list[int] | None = None,
        fc_dropout:     float = 0.5,
        seq_len:        int = 1,
    ):
        super().__init__()

        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # ── Frozen backbones ────────────────────────────────────────────────
        self.backbones = _FrozenBackbones(
            imu_checkpoint, mmg_checkpoint, device
        )

        self.seq_len = seq_len

        # ── Bidirectional GRU ───────────────────────────────────────────────
        self.gru = nn.GRU(
            input_size    = self.backbones.feature_dim,
            hidden_size   = gru_hidden_dim,
            num_layers    = gru_num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = gru_dropout if gru_num_layers > 1 else 0.0,
        )

        # GRU output dimension (bidirectional doubles the hidden size)
        gru_output_dim = gru_hidden_dim * 2

        # ── Trainable FC head ───────────────────────────────────────────────
        fc_hidden_dims = fc_hidden_dims or []
        layers         = []
        current_dim    = gru_output_dim

        for hidden_dim in fc_hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=fc_dropout),
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_classes))
        self.fc_head = nn.Sequential(*layers)

        # ── Store config ────────────────────────────────────────────────────
        self.config = dict(
            imu_checkpoint = imu_checkpoint,
            mmg_checkpoint = mmg_checkpoint,
            num_classes    = num_classes,
            gru_hidden_dim = gru_hidden_dim,
            gru_num_layers = gru_num_layers,
            gru_dropout    = gru_dropout,
            fc_hidden_dims = fc_hidden_dims,
            fc_dropout     = fc_dropout,
            seq_len        = seq_len,
        )

        self._initialise_weights()

    def _initialise_weights(self):
        # GRU weights
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

        # FC head weights
        for m in self.fc_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x_imu: torch.Tensor,
        x_mmg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_imu : (batch, 6, 125, 4)
            x_mmg : (batch, 5, 40, 125, 4)

        Returns:
            logits : (batch, num_classes)
        """
        # Extract fused features
        fused = self.backbones(x_imu, x_mmg)  # (batch, feature_dim)

        # Reshape to sequence: (batch, seq_len, feature_dim)
        batch_size = fused.size(0)
        fused_seq  = fused.unsqueeze(1).expand(-1, self.seq_len, -1)

        # GRU forward
        gru_out, _ = self.gru(fused_seq)  # (batch, seq_len, gru_hidden_dim*2)

        # Take last timestep output
        gru_last = gru_out[:, -1, :]      # (batch, gru_hidden_dim*2)

        # FC head
        logits = self.fc_head(gru_last)   # (batch, num_classes)
        return logits


# ============================================================================
# Trainer
# ============================================================================

class FusionGRUWindowTrainer:
    """
    Manages training, evaluation, inference, and persistence of FusionGRUWindow.

    Only the GRU and FC head are trainable; backbones remain frozen.

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
        batch_size   = 64,
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
        batch_size   = 64,
        epochs       = 50,
    )

    def __init__(
        self,
        model:       FusionGRUWindow,
        hyperparams: dict | None = None,
    ):
        self.cfg = {**self._SGD_DEFAULTS, **(hyperparams or {})}

        # ── Device ──────────────────────────────────────────────────────────
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # ── Model ───────────────────────────────────────────────────────────
        self.model     = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # ── Optimiser (only GRU + FC head parameters) ───────────────────────
        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        print(
            f"[FusionGRUWindowTrainer] Trainable parameters: "
            f"{sum(p.numel() for p in trainable_params):,}"
        )

        opt_name = self.cfg.get("optimizer", "SGD")

        if opt_name == "SGD":
            self.optimizer = optim.SGD(
                trainable_params,
                lr           = self.cfg["lr"],
                momentum     = self.cfg.get("momentum", 0.9),
                weight_decay = self.cfg.get("weight_decay", 0.0005),
            )
        elif opt_name == "Adam":
            self.optimizer = optim.Adam(
                trainable_params,
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
            for (X_imu, X_mmg), y_batch in loader:
                X_imu   = X_imu.to(self.device, dtype=torch.float32)
                X_mmg   = X_mmg.to(self.device, dtype=torch.float32)
                y_batch = y_batch.to(self.device, dtype=torch.long)

                logits = self.model(X_imu, X_mmg)
                loss   = self.criterion(logits, y_batch)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping for RNN stability
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=5.0
                    )
                    self.optimizer.step()

                total_loss += loss.item() * y_batch.size(0)
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
        trial:        optuna.Trial | None = None,
    ) -> dict:
        """
        Train the GRU fusion model.

        Args:
            train_loader : DataLoader yielding ((X_imu, X_mmg), y)
                           X_imu : (batch, 6, 125, 4)
                           X_mmg : (batch, 5, 40, 125, 4)
                           y     : (batch,)
            val_loader   : optional validation DataLoader
            epochs       : overrides cfg['epochs'] if provided
            verbose      : print per-epoch metrics
            trial        : Optuna trial object (enables pruning)

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
    def predict(
        self,
        X_imu: torch.Tensor,
        X_mmg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict class indices.

        Args:
            X_imu : (batch, 6, 125, 4) or (6, 125, 4)
            X_mmg : (batch, 5, 40, 125, 4) or (5, 40, 125, 4)

        Returns:
            predicted class indices, shape (batch,)
        """
        self.model.eval()
        if X_imu.dim() == 3:
            X_imu = X_imu.unsqueeze(0)
        if X_mmg.dim() == 4:
            X_mmg = X_mmg.unsqueeze(0)

        X_imu = X_imu.to(self.device, dtype=torch.float32)
        X_mmg = X_mmg.to(self.device, dtype=torch.float32)

        with torch.no_grad():
            preds = self.model(X_imu, X_mmg).argmax(dim=1)
        return preds.cpu()

    def predict_proba(
        self,
        X_imu: torch.Tensor,
        X_mmg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict class probabilities.

        Args:
            X_imu : (batch, 6, 125, 4) or (6, 125, 4)
            X_mmg : (batch, 5, 40, 125, 4) or (5, 40, 125, 4)

        Returns:
            probabilities, shape (batch, num_classes)
        """
        self.model.eval()
        if X_imu.dim() == 3:
            X_imu = X_imu.unsqueeze(0)
        if X_mmg.dim() == 4:
            X_mmg = X_mmg.unsqueeze(0)

        X_imu = X_imu.to(self.device, dtype=torch.float32)
        X_mmg = X_mmg.to(self.device, dtype=torch.float32)

        with torch.no_grad():
            probs = torch.softmax(self.model(X_imu, X_mmg), dim=1)
        return probs.cpu()

    # ── Persistence ─────────────────────────────────────────────────────────
    def save(self, path: str):
        """Save GRU + FC head weights, optimiser state, and config."""
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
        print(f"[FusionGRUWindowTrainer] Saved to '{path}'")

    def load(self, path: str):
        """Load GRU + FC head weights, optimiser state, and config."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.history = ckpt.get("history", self.history)
        self.cfg     = ckpt.get("cfg",     self.cfg)
        print(f"[FusionGRUWindowTrainer] Loaded from '{path}'")

    # ── Helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _combined_metric(acc: float, f1: float) -> float:
        """Scalar metric reported to Optuna."""
        return 0.5 * acc + 0.5 * f1


# ============================================================================
# Optuna Tuner
# ============================================================================

class FusionGRUWindowTuner:
    """
    Wraps an Optuna study to find the best FusionGRUWindow hyperparameters.

    Searches over:
        GRU         : hidden dimension, number of layers, dropout
        FC head     : hidden layer sizes, dropout rate
        Optimiser   : SGD or Adam (with respective hyperparameters)
        Training    : batch size, lr schedule

    Note: Backbone architectures are fixed (loaded from checkpoints).

    Optimisation target:
        Maximise  0.5 * val_accuracy + 0.5 * val_macro_F1

    Usage:
        tuner      = FusionGRUWindowTuner(
            train_loader, val_loader,
            imu_checkpoint="imu_best.pth",
            mmg_checkpoint="mmg_best.pth"
        )
        best_model = tuner.run(n_trials=50, timeout=3600)
        tuner.plot_results(save_dir="optuna_plots")
    """

    _SEARCH = dict(
        # GRU architecture
        gru_hidden_dim = [64, 128, 256, 512],
        gru_num_layers = (1, 3),
        gru_dropout    = (0.0, 0.5),
        seq_len        = [1, 2, 4],
        # FC head architecture
        n_fc_layers    = (0, 2),
        fc_hidden_dims = [64, 128, 256],
        fc_dropout     = (0.2, 0.7),
        # Training
        batch_size     = [32, 64, 128],
        epochs         = 50,
        # SGD
        sgd_lr         = (1e-4, 1e-1),
        sgd_momentum   = (0.70, 0.99),
        sgd_wd         = (1e-6, 1e-2),
        sgd_decay      = (0.05, 0.50),
        sgd_step       = (10,   40),
        # Adam
        adam_lr        = (1e-4, 1e-2),
        adam_wd        = (1e-6, 1e-2),
        adam_beta1     = (0.85, 0.99),
        adam_beta2     = (0.90, 0.9999),
        # Shared LR schedule
        lr_step_size   = (10, 40),
        lr_decay       = (0.05, 0.50),
    )

    def __init__(
        self,
        train_loader:   DataLoader,
        val_loader:     DataLoader,
        imu_checkpoint: str,
        mmg_checkpoint: str,
        num_classes:    int = 7,
        search_space:   dict | None = None,
    ):
        """
        Args:
            train_loader   : training DataLoader
            val_loader     : validation DataLoader
            imu_checkpoint : path to pretrained IntentCNNWindow checkpoint
            mmg_checkpoint : path to pretrained LocomotionMMGCNNWindow checkpoint
            num_classes    : output classes (default 7)
            search_space   : optional dict to override _SEARCH bounds
        """
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.imu_checkpoint = imu_checkpoint
        self.mmg_checkpoint = mmg_checkpoint
        self.num_classes    = num_classes
        self.search         = {**self._SEARCH, **(search_space or {})}

        self.study        = None
        self._best_model  = None
        self._best_params = None

    # ── Objective function ──────────────────────────────────────────────────
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Build, train, and evaluate one candidate configuration.
        Returns the combined metric (higher = better).
        """

        # ── 1. Sample GRU architecture ──────────────────────────────────────
        gru_hidden_dim = trial.suggest_categorical(
            "gru_hidden_dim", self.search["gru_hidden_dim"]
        )
        gru_num_layers = trial.suggest_int(
            "gru_num_layers",
            self.search["gru_num_layers"][0],
            self.search["gru_num_layers"][1],
        )
        gru_dropout = trial.suggest_float(
            "gru_dropout",
            self.search["gru_dropout"][0],
            self.search["gru_dropout"][1],
        ) if gru_num_layers > 1 else 0.0

        seq_len = trial.suggest_categorical(
            "seq_len", self.search["seq_len"]
        )

        # ── 2. Sample FC head architecture ──────────────────────────────────
        n_fc_layers = trial.suggest_int(
            "n_fc_layers",
            self.search["n_fc_layers"][0],
            self.search["n_fc_layers"][1],
        )

        fc_hidden_dims = []
        for i in range(n_fc_layers):
            dim = trial.suggest_categorical(
                f"fc_hidden_dim_{i}", self.search["fc_hidden_dims"]
            )
            fc_hidden_dims.append(dim)

        fc_dropout = trial.suggest_float(
            "fc_dropout",
            self.search["fc_dropout"][0],
            self.search["fc_dropout"][1],
        )

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

        # Rebuild DataLoaders with trial batch size
        train_ds = self.train_loader.dataset
        val_ds   = self.val_loader.dataset
        t_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        v_loader = DataLoader(val_ds,   batch_size=batch_size)

        # ── 5. Build model + trainer ────────────────────────────────────────
        model = FusionGRUWindow(
            imu_checkpoint = self.imu_checkpoint,
            mmg_checkpoint = self.mmg_checkpoint,
            num_classes    = self.num_classes,
            gru_hidden_dim = gru_hidden_dim,
            gru_num_layers = gru_num_layers,
            gru_dropout    = gru_dropout,
            fc_hidden_dims = fc_hidden_dims if fc_hidden_dims else None,
            fc_dropout     = fc_dropout,
            seq_len        = seq_len,
        )
        trainer = FusionGRUWindowTrainer(model, hyperparams)

        # ── 6. Train (pruning enabled) ──────────────────────────────────────
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

        # ── 7. Return combined metric ───────────────────────────────────────
        val_acc = max(trainer.history["val_acc"], default=0.0)
        val_f1  = max(trainer.history["val_f1"],  default=0.0)
        return FusionGRUWindowTrainer._combined_metric(val_acc, val_f1)

    # ── Run the study ────────────────────────────────────────────────────────
    def run(
        self,
        n_trials:      int  = 50,
        timeout:       int  = 3600,
        show_progress: bool = True,
    ) -> FusionGRUWindow:
        """
        Run the Optuna hyperparameter search.

        Args:
            n_trials     : maximum number of trials
            timeout      : hard stop in seconds (default 3600 = 1 hour)
            show_progress: display Optuna progress bar

        Returns:
            FusionGRUWindow model built with the best found hyperparameters
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.study = optuna.create_study(
            direction = "maximize",
            sampler   = TPESampler(seed=42),
            pruner    = MedianPruner(
                n_startup_trials  = 5,
                n_warmup_steps    = 10,
                interval_steps    = 1,
            ),
        )

        print(
            f"\n[FusionGRUWindowTuner] Starting Optuna search"
            f"  |  n_trials={n_trials}  timeout={timeout}s"
            f"  |  device={'cuda' if torch.cuda.is_available() else 'cpu'}\n"
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
        print(f"[FusionGRUWindowTuner] Search complete")
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

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _build_best_model(self) -> FusionGRUWindow:
        """Reconstruct FusionGRUWindow from best trial params."""
        if self._best_params is None:
            raise RuntimeError("Call run() before _build_best_model()")
        p         = self._best_params
        n_fc      = p["n_fc_layers"]

        gru_hidden_dim = p["gru_hidden_dim"]
        gru_num_layers = p["gru_num_layers"]
        gru_dropout    = p.get("gru_dropout", 0.0)
        seq_len        = p["seq_len"]

        fc_hidden_dims = [p[f"fc_hidden_dim_{i}"] for i in range(n_fc)] if n_fc > 0 else None
        fc_dropout     = p["fc_dropout"]

        return FusionGRUWindow(
            imu_checkpoint = self.imu_checkpoint,
            mmg_checkpoint = self.mmg_checkpoint,
            num_classes    = self.num_classes,
            gru_hidden_dim = gru_hidden_dim,
            gru_num_layers = gru_num_layers,
            gru_dropout    = gru_dropout,
            fc_hidden_dims = fc_hidden_dims,
            fc_dropout     = fc_dropout,
            seq_len        = seq_len,
        )

    def get_best_params(self) -> dict:
        """Return the best hyperparameter dict found by Optuna."""
        if self._best_params is None:
            raise RuntimeError("Call run() before get_best_params()")
        return self._best_params

    def get_best_model(self) -> FusionGRUWindow:
        """Return the FusionGRUWindow built with the best hyperparameters."""
        if self._best_model is None:
            raise RuntimeError("Call run() before get_best_model()")
        return self._best_model

    # ── Visualisation ────────────────────────────────────────────────────────
    def plot_results(self, save_dir: str = "optuna_plots"):
        """
        Generate and save Optuna visualisation plots to disk.

        Args:
            save_dir : directory to save HTML plots
        """
        if self.study is None:
            raise RuntimeError("Call run() before plot_results()")

        if not PLOTLY_AVAILABLE:
            print(
                "[FusionGRUWindowTuner] Plotly not installed. "
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
            print(f"[FusionGRUWindowTuner] Saved plot: '{path}'")