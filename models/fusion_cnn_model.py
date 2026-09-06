# fusion_cnn_model.py
# Fusion architecture 1: Direct GAP fusion of IntentCNN and GestureCNN
#
# Architecture:
#   GAP(IntentCNN backbone) + GAP(GestureCNN backbone)
#       -> Concatenate  (batch, C1+C2)
#       -> FC Layer     (Optuna: hidden units)
#       -> ReLU + Dropout
#       -> Linear       (batch, num_classes)
#       -> Softmax
#
# Both sub-model backbones are frozen.
# Optuna optimises: FC hidden size + optimiser hyperparameters.
#
# References:
#   IntentCNN  : Su et al., IEEE TNSRE 2019        (cnn_model.py)
#   GestureCNN : Wattanasiri et al., IEEE JBHI 2025 (dacnn_model.py)

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
    Loads, freezes, and extracts GAP features from both pre-trained
    sub-models.

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

        # ── Store backbone feature extractors only (no classifier heads) ─────
        self.intent_backbone  = intent_model.blocks
        self.gesture_backbone = gesture_model.conv_blocks

        # ── Infer output channel sizes ───────────────────────────────────────
        # C1: depth concat doubles channels in each Inception block
        self.c1 = intent_cfg["block_filters"][-1] * 2
        # C2: last filter count in GestureCNN
        self.c2 = gesture_cfg["block_filters"][-1]

        # ── GAP layers ───────────────────────────────────────────────────────
        self.gap_1d = nn.AdaptiveAvgPool1d(output_size=1)
        self.gap_2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # ── Freeze all backbone parameters ───────────────────────────────────
        self._freeze()

    def _freeze(self):
        """Freeze all backbone parameters — no gradients, no updates."""
        for param in self.intent_backbone.parameters():
            param.requires_grad = False
        for param in self.gesture_backbone.parameters():
            param.requires_grad = False

    @property
    def feature_dim(self) -> int:
        """Total concatenated feature dimension C1 + C2."""
        return self.c1 + self.c2

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
            features : (batch, C1 + C2)
        """
        # IntentCNN path
        f_imu = self.intent_backbone(x_imu)    # (batch, C1, L)
        f_imu = self.gap_1d(f_imu)             # (batch, C1, 1)
        f_imu = f_imu.squeeze(-1)              # (batch, C1)

        # GestureCNN path
        f_cwt = self.gesture_backbone(x_cwt)   # (batch, C2, H, W)
        f_cwt = self.gap_2d(f_cwt)             # (batch, C2, 1, 1)
        f_cwt = f_cwt.flatten(start_dim=1)     # (batch, C2)

        return torch.cat([f_imu, f_cwt], dim=1)    # (batch, C1+C2)


# ============================================================================
# FusionCNN Model
# ============================================================================

class FusionCNN(nn.Module):
    """
    Fusion model — direct GAP path:

        GAP(IntentCNN) + GAP(GestureCNN)
            -> Concatenate  (batch, C1+C2)
            -> FC           (batch, fc_hidden)
            -> ReLU + Dropout
            -> Linear       (batch, num_classes)

    Only the FC head is trained. Both backbones are frozen.

    Args:
        intent_cnn_path  : path to saved IntentCNN  .pt checkpoint
        gesture_cnn_path : path to saved GestureCNN .pt checkpoint
        num_classes      : output classes (default 7)
        fc_hidden        : hidden units in FC layer
        device           : torch device
    """

    def __init__(
        self,
        intent_cnn_path:  str,
        gesture_cnn_path: str,
        num_classes:      int                 = 7,
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

        # ── Trainable fusion head ────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(feat_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(fc_hidden, num_classes),
        )

        # ── Config for serialisation ─────────────────────────────────────────
        self.config = dict(
            model_type       = "FusionCNN",
            intent_cnn_path  = intent_cnn_path,
            gesture_cnn_path = gesture_cnn_path,
            num_classes      = num_classes,
            fc_hidden        = fc_hidden,
            feature_dim      = feat_dim,
        )

        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

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
        features = self.backbones(x_imu, x_cwt)    # (batch, C1+C2)
        return self.head(features)                  # (batch, num_classes)


# ============================================================================
# Trainer
# ============================================================================

class FusionCNNTrainer:
    """
    Trainer for FusionCNN.

    DataLoader must yield (X_imu, X_cwt, y) 3-tuples.
    Uses ReduceLROnPlateau. Supports SGD and Adam.
    Only the FC head parameters are updated (backbones are frozen).
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
        model:       FusionCNN,
        hyperparams: dict | None = None,
    ):
        self.cfg = {**self._DEFAULTS, **(hyperparams or {})}

        self.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        print(f"[FusionCNNTrainer] Using device: {self.device}")

        self.model     = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # Only train non-frozen parameters (the FC head)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        print(
            f"[FusionCNNTrainer] Trainable parameters: "
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
        print(f"[FusionCNNTrainer] Saved to '{path}'")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.history = ckpt.get("history", self.history)
        self.cfg     = ckpt.get("cfg",     self.cfg)
        print(f"[FusionCNNTrainer] Loaded from '{path}'")

    @staticmethod
    def _combined_metric(acc: float, f1: float) -> float:
        return 0.5 * acc + 0.5 * f1


# ============================================================================
# Optuna Tuner
# ============================================================================

class FusionCNNTuner:
    """
    Optuna tuner for FusionCNN.

    Searches over:
        FC hidden size + optimiser (SGD or Adam) + batch size
        + ReduceLROnPlateau factor and patience

    Usage:
        tuner      = FusionCNNTuner(train_loader, val_loader,
                                    intent_cnn_path, gesture_cnn_path)
        best_model = tuner.run(n_trials=50, timeout=3600)
        tuner.plot_results(save_dir='optuna_plots_fusion_cnn')
    """

    _SEARCH = dict(
        fc_hidden    = [64, 128, 256, 512],
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

        fc_hidden  = trial.suggest_categorical(
            "fc_hidden", self.search["fc_hidden"]
        )
        opt_name   = trial.suggest_categorical("optimizer", ["SGD", "Adam"])

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

        model = FusionCNN(
            intent_cnn_path  = self.intent_cnn_path,
            gesture_cnn_path = self.gesture_cnn_path,
            num_classes      = self.num_classes,
            fc_hidden        = fc_hidden,
            device           = self.device,
        )
        trainer = FusionCNNTrainer(model, hyperparams)

        try:
            trainer.fit(
                t_loader, v_loader,
                epochs=self.search["epochs"], verbose=False, trial=trial,
            )
        except optuna.exceptions.TrialPruned:
            raise

        val_acc = max(trainer.history["val_acc"], default=0.0)
        val_f1  = max(trainer.history["val_f1"],  default=0.0)
        return FusionCNNTrainer._combined_metric(val_acc, val_f1)

    def run(
        self,
        n_trials:      int  = 50,
        timeout:       int  = 3600,
        show_progress: bool = True,
    ) -> FusionCNN:
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
            f"\n[FusionCNNTuner] Starting Optuna search"
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
        print(f"[FusionCNNTuner] Search complete")
        print(f"  Best trial  : #{best.number}")
        print(f"  Best metric : {best.value:.4f}")
        print(f"  Best params :")
        for k, v in self._best_params.items():
            print(f"    {k:<25}: {v}")
        print(f"{'='*60}\n")

        self._best_model = FusionCNN(
            intent_cnn_path  = self.intent_cnn_path,
            gesture_cnn_path = self.gesture_cnn_path,
            num_classes      = self.num_classes,
            fc_hidden        = self._best_params["fc_hidden"],
            device           = self.device,
        )
        return self._best_model

    def get_best_params(self) -> dict:
        if self._best_params is None:
            raise RuntimeError("Call run() before get_best_params()")
        return self._best_params

    def get_best_model(self) -> FusionCNN:
        if self._best_model is None:
            raise RuntimeError("Call run() before get_best_model()")
        return self._best_model

    def plot_results(self, save_dir: str = "optuna_plots_fusion_cnn"):
        if self.study is None:
            raise RuntimeError("Call run() before plot_results()")
        if not PLOTLY_AVAILABLE:
            print("[FusionCNNTuner] Install optuna[visualization] and plotly")
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
            print(f"[FusionCNNTuner] Saved: '{path}'")