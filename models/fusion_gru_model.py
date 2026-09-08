# fusion_gru_model.py
# Fusion architecture 2: GRU fusion of IntentCNN and GestureCNN
#
# Architecture:
#   GAP(IntentCNN backbone) + GAP(GestureCNN backbone)
#       -> Concatenate      (batch, C1+C2)
#       -> Reshape          (batch, 1, C1+C2)   [single time step]
#       -> GRU              (batch, 1, gru_hidden)
#       -> Squeeze seq dim  (batch, gru_hidden)
#       -> FC Layer         (Optuna: hidden units)
#       -> ReLU + Dropout
#       -> Linear           (batch, num_classes)
#       -> Softmax
#
# Both sub-model backbones are frozen.
# Optuna optimises: GRU hidden size, GRU layers, GRU dropout,
#                   FC hidden size, optimiser hyperparameters.
#
# References:
#   IntentCNN  : Su et al., IEEE TNSRE 2019         (cnn_model.py)
#   GestureCNN : Wattanasiri et al., IEEE JBHI 2025  (dacnn_model.py)

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

from imu_cnn_model   import IntentCNN
from mmg_cnn_model import LocomotionMMGCNN

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
# Frozen Backbone Feature Extractor
# ============================================================================

class _FrozenBackbones(nn.Module):
    """
    Identical to the one in fusion_cnn_model.py.
    Loads, freezes, and extracts GAP features from both pre-trained
    sub-models, returning a concatenated (batch, C1+C2) feature vector.

    IntentCNN path:
        x_imu (batch, 6, 125)
        -> IntentCNN.blocks         (batch, C1, L)
        -> AdaptiveAvgPool1d(1)     (batch, C1, 1)
        -> squeeze                  (batch, C1)

    GestureCNN path:
        x_cwt (batch, 5, 40, 125)
        -> GestureCNN.conv_blocks   (batch, C2, H, W)
        -> AdaptiveAvgPool2d((1,1)) (batch, C2, 1, 1)
        -> flatten                  (batch, C2)

    Concatenated output:
        (batch, C1 + C2)
    """

    def __init__(
        self,
        intent_cnn_path:  str,
        gesture_cnn_path: str,
        device:           torch.device,
    ):
        super().__init__()

        # ── Load IntentCNN ───────────────────────────────────────────────────
        intent_ckpt  = torch.load(intent_cnn_path,  map_location=device)
        intent_cfg   = intent_ckpt["model_config"]
        intent_model = IntentCNN(
            in_channels   = intent_cfg["in_channels"],
            num_classes   = intent_cfg["num_classes"],
            block_filters = intent_cfg["block_filters"],
            kernel_pairs  = intent_cfg["kernel_pairs"],
        )
        intent_model.load_state_dict(intent_ckpt["model_state_dict"])

        # ── Load GestureCNN ──────────────────────────────────────────────────
        gesture_ckpt  = torch.load(gesture_cnn_path, map_location=device)
        gesture_cfg   = gesture_ckpt["model_config"]
        gesture_model = LocomotionMMGCNN(
            in_channels   = gesture_cfg["in_channels"],
            num_classes   = gesture_cfg["num_classes"],
            block_filters = gesture_cfg["block_filters"],
            kernel_sizes  = gesture_cfg["kernel_sizes"],
            strides       = gesture_cfg["strides"],
            dropout_rates = gesture_cfg["dropout_rates"],
            fc_hidden     = gesture_cfg["fc_hidden"],
        )
        gesture_model.load_state_dict(gesture_ckpt["model_state_dict"])

        # ── Store backbone feature extractors only ───────────────────────────
        self.intent_backbone  = intent_model.blocks
        self.gesture_backbone = gesture_model.conv_blocks

        # ── Infer output channel sizes ───────────────────────────────────────
        self.c1 = intent_cfg["block_filters"][-1] * 2
        self.c2 = gesture_cfg["block_filters"][-1]

        # ── GAP layers ───────────────────────────────────────────────────────
        self.gap_1d = nn.AdaptiveAvgPool1d(output_size=1)
        self.gap_2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self._freeze()

    def _freeze(self):
        for param in self.intent_backbone.parameters():
            param.requires_grad = False
        for param in self.gesture_backbone.parameters():
            param.requires_grad = False

    @property
    def feature_dim(self) -> int:
        return self.c1 + self.c2

    def forward(
        self,
        x_imu: torch.Tensor,
        x_cwt: torch.Tensor,
    ) -> torch.Tensor:
        f_imu = self.intent_backbone(x_imu)
        f_imu = self.gap_1d(f_imu).squeeze(-1)
        f_cwt = self.gesture_backbone(x_cwt)
        f_cwt = self.gap_2d(f_cwt).flatten(start_dim=1)
        return torch.cat([f_imu, f_cwt], dim=1)


# ============================================================================
# FusionGRU Model
# ============================================================================

class FusionGRU(nn.Module):
    """
    Fusion model — GRU path:

        GAP(IntentCNN) + GAP(GestureCNN)
            -> Concatenate      (batch, C1+C2)
            -> Reshape          (batch, 1, C1+C2)
            -> GRU              (batch, 1, gru_hidden)
            -> Squeeze seq dim  (batch, gru_hidden)
            -> FC               (batch, fc_hidden)
            -> ReLU + Dropout
            -> Linear           (batch, num_classes)

    Only the GRU and FC head are trained. Both backbones are frozen.

    Args:
        intent_cnn_path  : path to saved IntentCNN  .pt checkpoint
        gesture_cnn_path : path to saved GestureCNN .pt checkpoint
        num_classes      : output classes (default 7)
        gru_hidden       : GRU hidden state size
        gru_layers       : number of stacked GRU layers
        gru_dropout      : dropout between GRU layers (only if gru_layers > 1)
        fc_hidden        : hidden units in FC layer after GRU
        device           : torch device
    """

    def __init__(
        self,
        intent_cnn_path:  str,
        gesture_cnn_path: str,
        num_classes:      int                 = 7,
        gru_hidden:       int                 = 64,
        gru_layers:       int                 = 1,
        gru_dropout:      float               = 0.0,
        fc_hidden:        int                 = 128,
        device:           torch.device | None = None,
    ):
        super().__init__()

        device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # ── Frozen backbones ─────────────────────────────────────────────────
        self.backbones = _FrozenBackbones(
            intent_cnn_path, gesture_cnn_path, device
        )
        feat_dim = self.backbones.feature_dim

        # ── GRU layer ────────────────────────────────────────────────────────
        # Input:  (batch, seq_len=1, input_size=C1+C2)
        # Output: (batch, seq_len=1, gru_hidden)
        self.gru = nn.GRU(
            input_size  = feat_dim,
            hidden_size = gru_hidden,
            num_layers  = gru_layers,
            dropout     = gru_dropout if gru_layers > 1 else 0.0,
            batch_first = True,
        )

        # ── Trainable FC head ────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(fc_hidden, num_classes),
        )

        # ── Config for serialisation ─────────────────────────────────────────
        self.config = dict(
            model_type       = "FusionGRU",
            intent_cnn_path  = intent_cnn_path,
            gesture_cnn_path = gesture_cnn_path,
            num_classes      = num_classes,
            gru_hidden       = gru_hidden,
            gru_layers       = gru_layers,
            gru_dropout      = gru_dropout,
            fc_hidden        = fc_hidden,
            feature_dim      = feat_dim,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def forward(
        self,
        x_imu: torch.Tensor,
        x_cwt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_imu : (batch, 6, 125)
            x_cwt : (batch, 5, 40, 125)
        Returns:
            logits : (batch, num_classes)
        """
        features   = self.backbones(x_imu, x_cwt)  # (batch, C1+C2)
        seq        = features.unsqueeze(1)          # (batch, 1, C1+C2)
        gru_out, _ = self.gru(seq)                  # (batch, 1, gru_hidden)
        gru_out    = gru_out.squeeze(1)             # (batch, gru_hidden)
        return self.head(gru_out)                   # (batch, num_classes)


# ============================================================================
# Trainer
# ============================================================================

class FusionGRUTrainer:
    """
    Trainer for FusionGRU.

    DataLoader must yield (X_imu, X_cwt, y) 3-tuples.
    Uses ReduceLROnPlateau. Supports SGD and Adam.
    Only GRU and FC head parameters are updated (backbones are frozen).
    """

    _DEFAULTS = dict(
        optimizer   = "Adam",
        lr          = 1e-3,
        weight_decay= 1e-4,
        beta1       = 0.9,
        beta2       = 0.999,
        lr_factor   = 0.5,
        lr_patience = 10,
        lr_min      = 1e-6,
        batch_size  = 32,
        epochs      = 100,
    )

    def __init__(
        self,
        model:       FusionGRU,
        hyperparams: dict | None = None,
    ):
        self.cfg = {**self._DEFAULTS, **(hyperparams or {})}

        self.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        print(f"[FusionGRUTrainer] Using device: {self.device}")

        self.model     = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        print(
            f"[FusionGRUTrainer] Trainable parameters: "
            f"{sum(p.numel() for p in trainable):,}"
        )

        opt_name = self.cfg.get("optimizer", "Adam")
        if opt_name == "SGD":
            self.optimizer = optim.SGD(
                trainable,
                lr           = self.cfg["lr"],
                momentum     = self.cfg.get("momentum", 0.9),
                weight_decay = self.cfg.get("weight_decay", 1e-4),
            )
        elif opt_name == "Adam":
            self.optimizer = optim.Adam(
                trainable,
                lr           = self.cfg["lr"],
                weight_decay = self.cfg.get("weight_decay", 1e-4),
                betas        = (
                    self.cfg.get("beta1", 0.9),
                    self.cfg.get("beta2", 0.999),
                ),
            )
        else:
            raise ValueError(f"Unknown optimizer: '{opt_name}'")

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode     = "min",
            factor   = self.cfg.get("lr_factor",   0.5),
            patience = self.cfg.get("lr_patience",  10),
            min_lr   = self.cfg.get("lr_min",      1e-6),
        )

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
        self.model.train(training)
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.set_grad_enabled(training):
            for X_imu, X_cwt, y_batch in loader:
                X_imu   = X_imu.to(self.device,  dtype=torch.float32)
                X_cwt   = X_cwt.to(self.device,  dtype=torch.float32)
                y_batch = y_batch.to(self.device, dtype=torch.long)

                logits = self.model(X_imu, X_cwt)
                loss   = self.criterion(logits, y_batch)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * X_imu.size(0)
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())

        all_preds  = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        n          = len(all_labels)

        return (
            total_loss / n,
            float((all_preds == all_labels).mean()),
            float(f1_score(all_labels, all_preds,
                           average="macro", zero_division=0)),
        )

    # ── Public API ───────────────────────────────────────────────────────────
    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader | None   = None,
        epochs:       int | None          = None,
        verbose:      bool                = True,
        trial:        optuna.Trial | None = None,
    ) -> dict:
        n_epochs = epochs or self.cfg["epochs"]

        for epoch in range(1, n_epochs + 1):
            tr_loss, tr_acc, _ = self._run_epoch(train_loader, training=True)
            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)

            val_str = ""
            v_loss  = tr_loss

            if val_loader is not None:
                v_loss, v_acc, v_f1 = self._run_epoch(
                    val_loader, training=False
                )
                self.history["val_loss"].append(v_loss)
                self.history["val_acc"].append(v_acc)
                self.history["val_f1"].append(v_f1)
                val_str = (
                    f"  |  val_loss: {v_loss:.4f}"
                    f"  val_acc: {v_acc:.4f}"
                    f"  val_f1: {v_f1:.4f}"
                )

                if trial is not None:
                    combined = self._combined_metric(v_acc, v_f1)
                    trial.report(combined, step=epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            self.scheduler.step(v_loss)

            if verbose:
                lr_now = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch [{epoch:>3}/{n_epochs}]"
                    f"  train_loss: {tr_loss:.4f}"
                    f"  train_acc: {tr_acc:.4f}"
                    f"{val_str}"
                    f"  lr: {lr_now:.6f}"
                )

        return self.history

    def evaluate(self, loader: DataLoader) -> dict:
        loss, acc, f1 = self._run_epoch(loader, training=False)
        print(
            f"[Evaluate]  loss: {loss:.4f}"
            f"  accuracy: {acc:.4f}  macro_f1: {f1:.4f}"
        )
        return {"loss": loss, "accuracy": acc, "f1": f1}

    def predict(
        self,
        x_imu: torch.Tensor,
        x_cwt: torch.Tensor,
    ) -> torch.Tensor:
        self.model.eval()
        if x_imu.dim() == 2: x_imu = x_imu.unsqueeze(0)
        if x_cwt.dim() == 3: x_cwt = x_cwt.unsqueeze(0)
        x_imu = x_imu.to(self.device, dtype=torch.float32)
        x_cwt = x_cwt.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            preds = self.model(x_imu, x_cwt).argmax(dim=1)
        return preds.cpu()

    def predict_proba(
        self,
        x_imu: torch.Tensor,
        x_cwt: torch.Tensor,
    ) -> torch.Tensor:
        self.model.eval()
        if x_imu.dim() == 2: x_imu = x_imu.unsqueeze(0)
        if x_cwt.dim() == 3: x_cwt = x_cwt.unsqueeze(0)
        x_imu = x_imu.to(self.device, dtype=torch.float32)
        x_cwt = x_cwt.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            probs = torch.softmax(self.model(x_imu, x_cwt), dim=1)
        return probs.cpu()

    def save(self, path: str):
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
        print(f"[FusionGRUTrainer] Saved to '{path}'")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.history = ckpt.get("history", self.history)
        self.cfg     = ckpt.get("cfg",     self.cfg)
        print(f"[FusionGRUTrainer] Loaded from '{path}'")

    @staticmethod
    def _combined_metric(acc: float, f1: float) -> float:
        return 0.5 * acc + 0.5 * f1


# ============================================================================
# Optuna Tuner
# ============================================================================

class FusionGRUTuner:
    """
    Optuna tuner for FusionGRU.

    Searches over:
        GRU hidden size, GRU layers, GRU dropout,
        FC hidden size, optimiser (SGD or Adam),
        batch size, ReduceLROnPlateau factor and patience.

    Usage:
        tuner      = FusionGRUTuner(train_loader, val_loader,
                                    intent_cnn_path, gesture_cnn_path)
        best_model = tuner.run(n_trials=50, timeout=3600)
        tuner.plot_results(save_dir='optuna_plots_fusion_gru')
    """

    _SEARCH = dict(
        fc_hidden    = [64, 128, 256],
        gru_hidden   = [32, 64, 128, 256],
        gru_layers   = (1, 3),
        gru_dropout  = (0.0, 0.5),
        batch_size   = [16, 32, 64, 128],
        epochs       = 50,
        sgd_lr       = (1e-4, 1e-1),
        sgd_momentum = (0.70, 0.99),
        sgd_wd       = (1e-6, 1e-2),
        adam_lr      = (1e-4, 1e-2),
        adam_wd      = (1e-6, 1e-2),
        adam_beta1   = (0.85, 0.99),
        adam_beta2   = (0.90, 0.9999),
        lr_factor    = (0.1, 0.9),
        lr_patience  = (5,   20),
    )

    def __init__(
        self,
        train_loader:     DataLoader,
        val_loader:       DataLoader,
        intent_cnn_path:  str,
        gesture_cnn_path: str,
        num_classes:      int        = 7,
        search_space:     dict | None = None,
    ):
        self.train_loader     = train_loader
        self.val_loader       = val_loader
        self.intent_cnn_path  = intent_cnn_path
        self.gesture_cnn_path = gesture_cnn_path
        self.num_classes      = num_classes
        self.search           = {**self._SEARCH, **(search_space or {})}

        self.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.study        = None
        self._best_model  = None
        self._best_params = None

    def _objective(self, trial: optuna.Trial) -> float:

        gru_hidden  = trial.suggest_categorical(
            "gru_hidden", self.search["gru_hidden"]
        )
        gru_layers  = trial.suggest_int(
            "gru_layers",
            self.search["gru_layers"][0],
            self.search["gru_layers"][1],
        )
        gru_dropout = trial.suggest_float(
            "gru_dropout",
            self.search["gru_dropout"][0],
            self.search["gru_dropout"][1],
        )
        fc_hidden   = trial.suggest_categorical(
            "fc_hidden", self.search["fc_hidden"]
        )
        opt_name    = trial.suggest_categorical("optimizer", ["SGD", "Adam"])

        if opt_name == "SGD":
            hyperparams = dict(
                optimizer    = "SGD",
                lr           = trial.suggest_float(
                    "sgd_lr", *self.search["sgd_lr"], log=True),
                momentum     = trial.suggest_float(
                    "sgd_momentum", *self.search["sgd_momentum"]),
                weight_decay = trial.suggest_float(
                    "sgd_wd", *self.search["sgd_wd"], log=True),
            )
        else:
            hyperparams = dict(
                optimizer    = "Adam",
                lr           = trial.suggest_float(
                    "adam_lr", *self.search["adam_lr"], log=True),
                weight_decay = trial.suggest_float(
                    "adam_wd", *self.search["adam_wd"], log=True),
                beta1        = trial.suggest_float(
                    "adam_beta1", *self.search["adam_beta1"]),
                beta2        = trial.suggest_float(
                    "adam_beta2", *self.search["adam_beta2"]),
            )

        hyperparams["lr_factor"]   = trial.suggest_float(
            "lr_factor", *self.search["lr_factor"])
        hyperparams["lr_patience"] = trial.suggest_int(
            "lr_patience", *self.search["lr_patience"])

        batch_size = trial.suggest_categorical(
            "batch_size", self.search["batch_size"]
        )
        hyperparams["batch_size"] = batch_size
        hyperparams["epochs"]     = self.search["epochs"]

        t_loader = DataLoader(
            self.train_loader.dataset, batch_size=batch_size, shuffle=True
        )
        v_loader = DataLoader(
            self.val_loader.dataset, batch_size=batch_size
        )

        model = FusionGRU(
            intent_cnn_path  = self.intent_cnn_path,
            gesture_cnn_path = self.gesture_cnn_path,
            num_classes      = self.num_classes,
            gru_hidden       = gru_hidden,
            gru_layers       = gru_layers,
            gru_dropout      = gru_dropout,
            fc_hidden        = fc_hidden,
            device           = self.device,
        )
        trainer = FusionGRUTrainer(model, hyperparams)

        try:
            trainer.fit(
                t_loader, v_loader,
                epochs=self.search["epochs"], verbose=False, trial=trial,
            )
        except optuna.exceptions.TrialPruned:
            raise

        val_acc = max(trainer.history["val_acc"], default=0.0)
        val_f1  = max(trainer.history["val_f1"],  default=0.0)
        return FusionGRUTrainer._combined_metric(val_acc, val_f1)

    def run(
        self,
        n_trials:      int  = 50,
        timeout:       int  = 3600,
        show_progress: bool = True,
    ) -> FusionGRU:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.study = optuna.create_study(
            direction = "maximize",
            sampler   = TPESampler(seed=42),
            pruner    = MedianPruner(
                n_startup_trials = 5,
                n_warmup_steps   = 10,
                interval_steps   = 1,
            ),
        )

        print(
            f"\n[FusionGRUTuner] Starting Optuna search"
            f"  |  n_trials={n_trials}  timeout={timeout}s"
            f"  |  device={self.device}\n"
        )

        self.study.optimize(
            self._objective,
            n_trials          = n_trials,
            timeout           = timeout,
            show_progress_bar = show_progress,
            gc_after_trial    = True,
        )

        best              = self.study.best_trial
        self._best_params = best.params

        print(f"\n{'='*60}")
        print(f"[FusionGRUTuner] Search complete")
        print(f"  Best trial  : #{best.number}")
        print(f"  Best metric : {best.value:.4f}")
        print(f"  Best params :")
        for k, v in self._best_params.items():
            print(f"    {k:<25}: {v}")
        print(f"{'='*60}\n")

        p = self._best_params
        self._best_model = FusionGRU(
            intent_cnn_path  = self.intent_cnn_path,
            gesture_cnn_path = self.gesture_cnn_path,
            num_classes      = self.num_classes,
            gru_hidden       = p["gru_hidden"],
            gru_layers       = p["gru_layers"],
            gru_dropout      = p["gru_dropout"],
            fc_hidden        = p["fc_hidden"],
            device           = self.device,
        )
        return self._best_model

    def get_best_params(self) -> dict:
        if self._best_params is None:
            raise RuntimeError("Call run() before get_best_params()")
        return self._best_params

    def get_best_model(self) -> FusionGRU:
        if self._best_model is None:
            raise RuntimeError("Call run() before get_best_model()")
        return self._best_model

    def plot_results(self, save_dir: str = "optuna_plots_fusion_gru"):
        if self.study is None:
            raise RuntimeError("Call run() before plot_results()")
        if not PLOTLY_AVAILABLE:
            print("[FusionGRUTuner] Install optuna[visualization] and plotly")
            return

        os.makedirs(save_dir, exist_ok=True)
        plots = {
            "optimisation_history.html": plot_optimization_history(self.study),
            "param_importances.html"   : plot_param_importances(self.study),
            "parallel_coordinate.html" : plot_parallel_coordinate(self.study),
            "slice.html"               : plot_slice(self.study),
        }
        for filename, fig in plots.items():
            path = os.path.join(save_dir, filename)
            fig.write_html(path)
            print(f"[FusionGRUTuner] Saved: '{path}'")