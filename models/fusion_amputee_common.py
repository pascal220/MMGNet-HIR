# fusion_amputee_common.py
# Shared building blocks for the end-to-end amputee fusion models.
#
# Unlike the healthy-volunteer fusion models, which freeze pre-trained IMU and
# MMG backbones, the amputee fusion models build both backbones inside the
# fusion model and train every parameter from scratch together with the head.
# Optuna therefore jointly searches:
#   - the IMU backbone   (``imu_*`` parameters, the standalone IMU search space)
#   - the MMG backbone   (``mmg_*`` parameters, the standalone MMG search space)
#   - the fusion head    (model-specific)
#   - the optimiser, LR schedule, and batch size

from __future__ import annotations

import ast
import os
from typing import Any, ClassVar

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from device_utils import release_cuda_memory, resolve_device
from early_stopping import EarlyStopping
from imu_cnn_model import IntentCNN, IntentCNNTuner
from imu_cnn_window_model import IntentCNNWindow, IntentCNNWindowTuner
from mmg_cnn_model import LocomotionMMGCNN, LocomotionMMGCNNTuner
from mmg_cnn_window_model import LocomotionMMGCNNWindow, LocomotionMMGCNNWindowTuner

AMPUTEE_NUM_CLASSES = 5
IMU_CHANNELS = 6
MMG_CHANNELS = 5
IMU_LENGTH = 125
MMG_FREQ_DIM = 40
MMG_TIME_DIM = 125

# Trials whose sampled backbone cannot fit the input are tagged with this user
# attribute, rejected before any training, and redrawn outside the trial budget.
INVALID_GEOMETRY_ATTR = "invalid_geometry"
# Safety cap on all sampled trials, as a multiple of the trial budget.
MAX_ATTEMPTS_PER_TRIAL = 20


# ============================================================================
# Search spaces
# ============================================================================

def _prefixed(search: dict[str, Any], prefix: str, keys: tuple[str, ...]) -> dict[str, Any]:
    return {f"{prefix}_{key}": search[key] for key in keys}


# Backbone spaces are taken verbatim from the standalone tuners so the two
# never drift apart.
SINGLE_WINDOW_BACKBONE_SEARCH: dict[str, Any] = {
    **_prefixed(IntentCNNTuner._SEARCH, "imu", ("n_blocks", "filters", "kernel_pairs")),
    **_prefixed(
        LocomotionMMGCNNTuner._SEARCH, "mmg",
        ("n_blocks", "filters", "kernel_sizes", "strides", "dropout_rates"),
    ),
}

WINDOWED_BACKBONE_SEARCH: dict[str, Any] = {
    **_prefixed(
        IntentCNNWindowTuner._SEARCH, "imu",
        ("first_conv_filters", "first_conv_kernel_width", "n_blocks", "filters", "kernel_pairs"),
    ),
    **_prefixed(
        LocomotionMMGCNNWindowTuner._SEARCH, "mmg",
        (
            "first_conv_filters", "first_conv_kernel_size", "n_blocks",
            "filters", "kernel_sizes", "strides", "dropout_rates",
        ),
    ),
}

OPTIMISER_SEARCH: dict[str, Any] = dict(
    sgd_lr       = (1e-4, 1e-1),
    sgd_momentum = (0.70, 0.99),
    sgd_wd       = (1e-6, 1e-2),
    adam_lr      = (1e-4, 1e-2),
    adam_wd      = (1e-6, 1e-2),
    adam_beta1   = (0.85, 0.99),
    adam_beta2   = (0.90, 0.9999),
)

PLATEAU_SCHEDULE_SEARCH: dict[str, Any] = dict(
    lr_factor   = (0.1, 0.9),
    lr_patience = (5, 20),
    lr_min      = 1e-6,
)

STEP_SCHEDULE_SEARCH: dict[str, Any] = dict(
    lr_step_size = (10, 40),
    lr_decay     = (0.05, 0.50),
)


def suggest_imu_backbone(
    trial: optuna.trial.BaseTrial,
    search: dict[str, Any],
    *,
    windowed: bool,
) -> dict[str, Any]:
    """Sample an IntentCNN (or IntentCNNWindow) configuration."""
    config: dict[str, Any] = {"in_channels": IMU_CHANNELS}
    if windowed:
        config["first_conv_filters"] = trial.suggest_categorical(
            "imu_first_conv_filters", search["imu_first_conv_filters"]
        )
        config["first_conv_kernel_width"] = trial.suggest_categorical(
            "imu_first_conv_kernel_width", search["imu_first_conv_kernel_width"]
        )

    n_blocks = trial.suggest_int("imu_n_blocks", *search["imu_n_blocks"])
    pair_choices = [str(tuple(pair)) for pair in search["imu_kernel_pairs"]]
    block_filters: list[int] = []
    kernel_pairs: list[tuple[int, int]] = []
    for i in range(n_blocks):
        block_filters.append(
            trial.suggest_categorical(f"imu_block_{i}_filters", search["imu_filters"])
        )
        pair = trial.suggest_categorical(f"imu_block_{i}_kernel_pair", pair_choices)
        kernel_pairs.append(tuple(ast.literal_eval(pair)))

    # Window pooling and the 'same'-padded first conv keep the length at 125.
    if not IntentCNNTuner._blocks_fit_input(kernel_pairs, IMU_LENGTH):
        trial.set_user_attr(INVALID_GEOMETRY_ATTR, True)
        raise optuna.exceptions.TrialPruned(
            "Sampled IMU backbone shrinks the sequence below the kernel size."
        )
    config.update(block_filters=block_filters, kernel_pairs=kernel_pairs)
    return config


def suggest_mmg_backbone(
    trial: optuna.trial.BaseTrial,
    search: dict[str, Any],
    *,
    windowed: bool,
) -> dict[str, Any]:
    """Sample a LocomotionMMGCNN (or LocomotionMMGCNNWindow) configuration."""
    config: dict[str, Any] = {"in_channels": MMG_CHANNELS}
    if windowed:
        filters = trial.suggest_categorical(
            "mmg_first_conv_filters", search["mmg_first_conv_filters"]
        )
        kernel = trial.suggest_categorical(
            "mmg_first_conv_kernel_size", search["mmg_first_conv_kernel_size"]
        )
        config.update(
            first_conv_filters=filters,
            first_conv_kernel_freq=kernel,
            first_conv_kernel_time=kernel,
        )

    n_blocks = trial.suggest_int("mmg_n_blocks", *search["mmg_n_blocks"])
    block_filters: list[int] = []
    kernel_sizes: list[int] = []
    strides: list[int] = []
    dropout_rates: list[float] = []
    for i in range(n_blocks):
        block_filters.append(
            trial.suggest_categorical(f"mmg_block_{i}_filters", search["mmg_filters"][i])
        )
        kernel_sizes.append(
            trial.suggest_categorical(f"mmg_block_{i}_kernel", search["mmg_kernel_sizes"][i])
        )
        strides.append(
            trial.suggest_categorical(f"mmg_block_{i}_stride", search["mmg_strides"][i])
        )
        dropout_rates.append(
            trial.suggest_float(f"mmg_block_{i}_dropout", *search["mmg_dropout_rates"])
        )

    if not LocomotionMMGCNNTuner._blocks_fit_input(
        kernel_sizes, strides, MMG_FREQ_DIM, MMG_TIME_DIM
    ):
        trial.set_user_attr(INVALID_GEOMETRY_ATTR, True)
        raise optuna.exceptions.TrialPruned(
            "Sampled MMG backbone shrinks the feature map below the kernel size."
        )
    config.update(
        block_filters=block_filters,
        kernel_sizes=kernel_sizes,
        strides=strides,
        dropout_rates=dropout_rates,
    )
    return config


def suggest_optimiser(trial: optuna.trial.BaseTrial, search: dict[str, Any]) -> dict[str, Any]:
    """Sample SGD or Adam with the parameter names ``normalize_training_params`` expects."""
    if trial.suggest_categorical("optimizer", ["SGD", "Adam"]) == "SGD":
        return dict(
            optimizer    = "SGD",
            lr           = trial.suggest_float("sgd_lr", *search["sgd_lr"], log=True),
            momentum     = trial.suggest_float("sgd_momentum", *search["sgd_momentum"]),
            weight_decay = trial.suggest_float("sgd_wd", *search["sgd_wd"], log=True),
        )
    return dict(
        optimizer    = "Adam",
        lr           = trial.suggest_float("adam_lr", *search["adam_lr"], log=True),
        weight_decay = trial.suggest_float("adam_wd", *search["adam_wd"], log=True),
        beta1        = trial.suggest_float("adam_beta1", *search["adam_beta1"]),
        beta2        = trial.suggest_float("adam_beta2", *search["adam_beta2"]),
    )


def default_backbone_configs(windowed: bool) -> dict[str, dict[str, Any]]:
    """Valid backbone configurations for constructing a model without a search."""
    if windowed:
        return {
            "imu": dict(
                in_channels=IMU_CHANNELS, first_conv_filters=32, first_conv_kernel_width=5,
                block_filters=[16, 32], kernel_pairs=[(3, 5), (3, 5)],
            ),
            "mmg": dict(
                in_channels=MMG_CHANNELS, first_conv_filters=128,
                first_conv_kernel_freq=3, first_conv_kernel_time=3,
                block_filters=[16, 32, 64], kernel_sizes=[7, 5, 3], strides=[3, 2, 1],
                dropout_rates=[0.05, 0.05, 0.05],
            ),
        }
    return {
        "imu": dict(
            in_channels=IMU_CHANNELS, block_filters=[16, 32], kernel_pairs=[(3, 5), (3, 5)],
        ),
        "mmg": dict(
            in_channels=MMG_CHANNELS, block_filters=[16, 32, 64], kernel_sizes=[7, 5, 3],
            strides=[3, 2, 1], dropout_rates=[0.05, 0.05, 0.05],
        ),
    }


# ============================================================================
# Trainable backbones
# ============================================================================

class AmputeeFusionBackbones(nn.Module):
    """Trainable IMU and MMG feature extractors, concatenated into one vector.

    Each backbone is the standalone model with its classifier replaced by an
    identity, so it returns the globally pooled features. Nothing is frozen.
    """

    def __init__(self, backbone_configs: dict[str, dict[str, Any]], windowed: bool):
        super().__init__()
        imu_config = dict(backbone_configs["imu"])
        mmg_config = dict(backbone_configs["mmg"])
        imu_cls = IntentCNNWindow if windowed else IntentCNN
        mmg_cls = LocomotionMMGCNNWindow if windowed else LocomotionMMGCNN

        # num_classes only sizes the discarded classifier.
        self.imu = imu_cls(num_classes=1, **imu_config)
        self.imu.fc = nn.Identity()
        self.mmg = mmg_cls(num_classes=1, fc_hidden=None, **mmg_config)
        self.mmg.classifier = nn.Identity()

        self.imu_dim = int(imu_config["block_filters"][-1]) * 2  # Inception concat
        self.mmg_dim = int(mmg_config["block_filters"][-1])
        self.feature_dim = self.imu_dim + self.mmg_dim
        self.backbone_configs = {"imu": imu_config, "mmg": mmg_config}

    def forward(self, x_imu: torch.Tensor, x_cwt: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.imu(x_imu), self.mmg(x_cwt)], dim=1)


def init_linear_layers(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


def load_from_checkpoint(cls: Any, path: str, device: torch.device | str | None = None):
    """Load a complete end-to-end model from one checkpoint file."""
    device = resolve_device(device)
    ckpt = torch.load(path, map_location=device)
    model = cls.from_config(ckpt["model_config"], device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


# ============================================================================
# Trainer
# ============================================================================

def combined_metric(accuracy: float, macro_f1: float) -> float:
    return 0.5 * accuracy + 0.5 * macro_f1


def unpack_batch(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accept both ``(imu, mmg, y)`` and windowed ``((imu, mmg), y)`` batches."""
    if len(batch) == 2:
        (x_imu, x_cwt), y = batch
    else:
        x_imu, x_cwt, y = batch
    return x_imu, x_cwt, y


class AmputeeFusionTrainer:
    """Train every parameter of an end-to-end amputee fusion model.

    Subclasses set ``SCHEDULER``: ``"plateau"`` (ReduceLROnPlateau, as the
    single-window fusion models) or ``"step"`` (StepLR, as the windowed ones).
    """

    NAME: ClassVar[str] = "AmputeeFusionTrainer"
    SCHEDULER: ClassVar[str] = "plateau"

    _DEFAULTS: ClassVar[dict[str, Any]] = dict(
        optimizer    = "Adam",
        lr           = 1e-3,
        weight_decay = 1e-4,
        momentum     = 0.9,
        beta1        = 0.9,
        beta2        = 0.999,
        lr_factor    = 0.5,
        lr_patience  = 10,
        lr_min       = 1e-6,
        lr_step_size = 20,
        lr_decay     = 0.5,
        batch_size   = 32,
        epochs       = 100,
    )

    def __init__(self, model: nn.Module, hyperparams: dict | None = None):
        self.cfg = {**self._DEFAULTS, **(hyperparams or {})}
        self.device = resolve_device(self.cfg.get("device", "auto"))
        self.model = model.to(self.device)

        class_weights = self.cfg.get("class_weights")
        loss_weights = (
            None
            if class_weights is None
            else torch.as_tensor(class_weights, dtype=torch.float32, device=self.device)
        )
        self.criterion = nn.CrossEntropyLoss(weight=loss_weights)

        parameters = list(self.model.parameters())
        opt_name = self.cfg["optimizer"]
        if opt_name == "SGD":
            self.optimizer = optim.SGD(
                parameters,
                lr=self.cfg["lr"],
                momentum=self.cfg["momentum"],
                weight_decay=self.cfg["weight_decay"],
            )
        elif opt_name == "Adam":
            self.optimizer = optim.Adam(
                parameters,
                lr=self.cfg["lr"],
                weight_decay=self.cfg["weight_decay"],
                betas=(self.cfg["beta1"], self.cfg["beta2"]),
            )
        else:
            raise ValueError(f"Unknown optimizer: '{opt_name}'")

        if self.SCHEDULER == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.cfg["lr_factor"],
                patience=int(self.cfg["lr_patience"]),
                min_lr=self.cfg["lr_min"],
            )
        elif self.SCHEDULER == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(self.cfg["lr_step_size"]),
                gamma=self.cfg["lr_decay"],
            )
        else:
            raise ValueError(f"Unknown scheduler: '{self.SCHEDULER}'")

        self.history: dict[str, Any] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "val_f1": [],
        }

    def _run_epoch(self, loader: DataLoader, training: bool) -> tuple[float, float, float]:
        self.model.train(training)
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.set_grad_enabled(training):
            for batch in loader:
                x_imu, x_cwt, y_batch = unpack_batch(batch)
                x_imu = x_imu.to(self.device, dtype=torch.float32)
                x_cwt = x_cwt.to(self.device, dtype=torch.float32)
                y_batch = y_batch.to(self.device, dtype=torch.long)

                logits = self.model(x_imu, x_cwt)
                loss = self.criterion(logits, y_batch)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * x_imu.size(0)
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        return (
            total_loss / len(labels),
            float((preds == labels).mean()),
            float(f1_score(labels, preds, average="macro", zero_division=0)),
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int | None = None,
        verbose: bool = True,
        trial: optuna.Trial | None = None,
    ) -> dict:
        n_epochs = epochs or self.cfg["epochs"]
        early_stopping = EarlyStopping.from_config(self.model, self.cfg)
        if verbose:
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"[{self.NAME}] device: {self.device}  trainable parameters: {trainable:,}")

        for epoch in range(1, n_epochs + 1):
            tr_loss, tr_acc, _ = self._run_epoch(train_loader, training=True)
            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)

            val_str = ""
            v_loss = tr_loss
            if val_loader is not None:
                v_loss, v_acc, v_f1 = self._run_epoch(val_loader, training=False)
                self.history["val_loss"].append(v_loss)
                self.history["val_acc"].append(v_acc)
                self.history["val_f1"].append(v_f1)
                val_str = f"  |  val_loss: {v_loss:.4f}  val_acc: {v_acc:.4f}  val_f1: {v_f1:.4f}"

                if trial is not None:
                    trial.report(combined_metric(v_acc, v_f1), step=epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            if self.SCHEDULER == "plateau":
                self.scheduler.step(v_loss)
            else:
                self.scheduler.step()
            should_stop = early_stopping.update(epoch, {"train_loss": tr_loss})

            if verbose:
                lr_now = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch [{epoch:>3}/{n_epochs}]  train_loss: {tr_loss:.4f}"
                    f"  train_acc: {tr_acc:.4f}{val_str}  lr: {lr_now:.6f}"
                )

            if should_stop:
                if verbose:
                    print(
                        f"[EarlyStopping] Stopping at epoch {epoch}; best "
                        f"{early_stopping.monitor} was {early_stopping.best_value:.6f} "
                        f"at epoch {early_stopping.best_epoch}."
                    )
                break

        early_stopping.finalize()
        self.history["early_stopping"] = early_stopping.summary()
        return self.history

    def evaluate(self, loader: DataLoader) -> dict:
        loss, acc, f1 = self._run_epoch(loader, training=False)
        return {"loss": loss, "accuracy": acc, "f1": f1}

    @torch.no_grad()
    def predict_proba(self, x_imu: torch.Tensor, x_cwt: torch.Tensor) -> torch.Tensor:
        """Class probabilities for a batch of samples."""
        self.model.eval()
        logits = self.model(
            x_imu.to(self.device, dtype=torch.float32),
            x_cwt.to(self.device, dtype=torch.float32),
        )
        return torch.softmax(logits, dim=1).cpu()

    def predict(self, x_imu: torch.Tensor, x_cwt: torch.Tensor) -> torch.Tensor:
        """Class indices for a batch of samples."""
        return self.predict_proba(x_imu, x_cwt).argmax(dim=1)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": self.model.config,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "history": self.history,
                "cfg": self.cfg,
            },
            path,
        )
        print(f"[{self.NAME}] Saved to '{path}'")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.history = ckpt.get("history", self.history)
        self.cfg = ckpt.get("cfg", self.cfg)


# ============================================================================
# Optuna tuner
# ============================================================================

class AmputeeFusionTuner:
    """Jointly search both backbones, the fusion head, and the optimiser.

    Subclasses define the model, trainer, whether inputs are windowed, the
    search dictionary, and ``_suggest_head``. The best model is rebuilt by
    replaying the best trial's parameters through the same suggestion code.
    """

    NAME: ClassVar[str] = "AmputeeFusionTuner"
    MODEL_CLS: ClassVar[Any]
    TRAINER_CLS: ClassVar[type[AmputeeFusionTrainer]]
    WINDOWED: ClassVar[bool]
    _SEARCH: ClassVar[dict[str, Any]]

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int = AMPUTEE_NUM_CLASSES,
        search_space: dict | None = None,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.search = {**self._SEARCH, **(search_space or {})}
        self.device = resolve_device(self.search.get("device", "auto"))

        self.study: optuna.Study | None = None
        self._best_model: nn.Module | None = None
        self._best_params: dict[str, Any] | None = None

    def _suggest_head(self, trial: optuna.trial.BaseTrial) -> dict[str, Any]:
        raise NotImplementedError

    def _suggest_model_kwargs(self, trial: optuna.trial.BaseTrial) -> dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "backbone_configs": {
                "imu": suggest_imu_backbone(trial, self.search, windowed=self.WINDOWED),
                "mmg": suggest_mmg_backbone(trial, self.search, windowed=self.WINDOWED),
            },
            **self._suggest_head(trial),
        }

    def _suggest_training(self, trial: optuna.trial.BaseTrial) -> dict[str, Any]:
        hyperparams = suggest_optimiser(trial, self.search)
        if self.TRAINER_CLS.SCHEDULER == "plateau":
            hyperparams["lr_factor"] = trial.suggest_float("lr_factor", *self.search["lr_factor"])
            hyperparams["lr_patience"] = trial.suggest_int(
                "lr_patience", *self.search["lr_patience"]
            )
            hyperparams["lr_min"] = self.search["lr_min"]
        else:
            hyperparams["lr_step_size"] = trial.suggest_int(
                "lr_step_size", *self.search["lr_step_size"]
            )
            hyperparams["lr_decay"] = trial.suggest_float("lr_decay", *self.search["lr_decay"])
        hyperparams["batch_size"] = trial.suggest_categorical(
            "batch_size", self.search["batch_size"]
        )
        hyperparams["epochs"] = self.search["epochs"]
        hyperparams["class_weights"] = self.search.get("class_weights")
        hyperparams["device"] = str(self.device)
        return hyperparams

    def _objective(self, trial: optuna.Trial) -> float:
        model_kwargs = self._suggest_model_kwargs(trial)
        hyperparams = self._suggest_training(trial)
        batch_size = hyperparams["batch_size"]

        pin_memory = self.device.type == "cuda"
        generator = torch.Generator().manual_seed(int(self.search.get("seed", 42)) + trial.number)
        train_loader = DataLoader(
            self.train_loader.dataset, batch_size=batch_size, shuffle=True,
            generator=generator, pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            self.val_loader.dataset, batch_size=batch_size, pin_memory=pin_memory,
        )

        model = trainer = None
        out_of_memory = False
        try:
            model = self.MODEL_CLS(**model_kwargs)
            trainer = self.TRAINER_CLS(model, hyperparams)
            trainer.fit(
                train_loader, val_loader,
                epochs=self.search["epochs"], verbose=False, trial=trial,
            )
        except torch.cuda.OutOfMemoryError:
            out_of_memory = True

        if out_of_memory:
            # Released outside the except block so the traceback no longer
            # pins the failed batch's tensors.
            del trainer, model, train_loader, val_loader
            release_cuda_memory()
            trial.set_user_attr("out_of_memory", True)
            raise optuna.exceptions.TrialPruned(
                f"CUDA out of memory (batch_size={batch_size})."
            )

        history = trainer.history
        del trainer, model
        release_cuda_memory()

        scores = [combined_metric(a, f) for a, f in zip(history["val_acc"], history["val_f1"])]
        best_index = int(np.argmax(scores))
        trial.set_user_attr("best_epoch", best_index + 1)
        trial.set_user_attr("validation_accuracy", history["val_acc"][best_index])
        trial.set_user_attr("validation_macro_f1", history["val_f1"][best_index])
        return float(scores[best_index])

    def run(
        self,
        n_trials: int = 50,
        timeout: int | None = None,
        show_progress: bool = True,
        storage: Any = None,
        study_name: str | None = None,
        load_if_exists: bool = False,
    ) -> nn.Module:
        """Search until ``n_trials`` candidates have been trained, or ``timeout``.

        Sampled backbones that cannot fit the input are rejected before any
        training and redrawn without using the budget. Pruned trials (median
        pruner or CUDA out of memory) did train, so they count.
        """
        if n_trials < 1:
            raise ValueError("n_trials must be at least 1.")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=int(self.search.get("seed", 42))),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1),
            storage=storage,
            study_name=study_name,
            load_if_exists=load_if_exists,
        )
        first_trial = len(self.study.trials)
        max_attempts = n_trials * MAX_ATTEMPTS_PER_TRIAL

        timeout_desc = f"{timeout}s" if timeout is not None else "None"
        print(
            f"\n[{self.NAME}] Starting Optuna search  |  n_trials={n_trials} (trained)"
            f"  timeout={timeout_desc}  |  device={self.device}\n"
        )
        counts = {"trained": 0, "redrawn": 0}
        progress = tqdm(total=n_trials, desc=self.NAME, unit="trial", disable=not show_progress)

        def stop_at_budget(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            if trial.user_attrs.get(INVALID_GEOMETRY_ATTR):
                counts["redrawn"] += 1
                progress.set_postfix(redrawn=counts["redrawn"])
            else:
                counts["trained"] += 1
                progress.update(1)
            if (
                counts["trained"] >= n_trials
                or counts["trained"] + counts["redrawn"] >= max_attempts
            ):
                study.stop()

        try:
            self.study.optimize(
                self._objective,
                n_trials=None,
                timeout=timeout,
                callbacks=[stop_at_budget],
                show_progress_bar=False,
                gc_after_trial=True,
            )
        finally:
            progress.close()

        if counts["trained"] < n_trials and counts["trained"] + counts["redrawn"] >= max_attempts:
            print(
                f"[{self.NAME}] Stopped after {max_attempts} sampled trials with only "
                f"{counts['trained']} trained: most sampled backbones did not fit the input."
            )
        trained = [
            trial for trial in self.study.get_trials(deepcopy=False)[first_trial:]
            if not trial.user_attrs.get(INVALID_GEOMETRY_ATTR)
        ]
        if not self.study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
            out_of_memory = sum(1 for trial in trained if trial.user_attrs.get("out_of_memory"))
            raise RuntimeError(
                f"[{self.NAME}] No trial completed: {len(trained)} trained trial(s) were "
                f"pruned ({out_of_memory} out of CUDA memory) and {counts['redrawn']} "
                "invalid backbone geometries were redrawn. Increase the trial budget "
                "(--n-trials) or the timeout."
            )

        best = self.study.best_trial
        self._best_params = dict(best.params)
        print(f"\n{'=' * 60}\n[{self.NAME}] Search complete")
        print(
            f"  Trials      : {counts['trained']} trained, "
            f"{counts['redrawn']} invalid geometries redrawn"
        )
        print(f"  Best trial  : #{best.number}\n  Best metric : {best.value:.4f}\n  Best params :")
        for key, value in self._best_params.items():
            print(f"    {key:<30}: {value}")
        print(f"{'=' * 60}\n")

        self._best_model = self._build_best_model()
        return self._best_model

    def _build_best_model(self) -> nn.Module:
        """Rebuild the best architecture (untrained) from the best trial."""
        if self._best_params is None:
            raise RuntimeError("Call run() before _build_best_model()")
        fixed = optuna.trial.FixedTrial(self._best_params)
        return self.MODEL_CLS(**self._suggest_model_kwargs(fixed))

    def get_best_params(self) -> dict[str, Any]:
        if self._best_params is None:
            raise RuntimeError("Call run() before get_best_params()")
        return self._best_params

    def get_best_model(self) -> nn.Module:
        if self._best_model is None:
            raise RuntimeError("Call run() before get_best_model()")
        return self._best_model
