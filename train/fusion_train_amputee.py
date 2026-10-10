"""End-to-end single-window fusion training for amputee data.

One run trains every amputee data type in turn: FusionCNN then FusionGRU on
type1, then FusionCNN then FusionGRU on type2. Each type is an independent
dataset with its own split, Optuna studies and run folders. The IMU and MMG
backbones are trained from scratch inside each fusion model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fusion_cnn_amputee_model as cnn
import fusion_gru_amputee_model as gru
from amputee_train_common import DEFAULT_CHECKPOINT_DIR, AmputeeModel, train_amputee_type
from data_loader import AMPUTEE_DATA_TYPES, PreparedData, run_per_amputee_type

INPUT_MODE = "single_window"
MODELS = (
    AmputeeModel("fusion_cnn", cnn.MODEL_KEY, cnn.FusionCNNAmputeeTuner, cnn.FusionCNNAmputeeTrainer),
    AmputeeModel("fusion_gru", gru.MODEL_KEY, gru.FusionGRUAmputeeTuner, gru.FusionGRUAmputeeTrainer),
)


def train_fusion_amputee_type(
    prepared: PreparedData,
    *,
    n_trials: int = 50,
    timeout: int | None = None,
    artifact_root: str = "results/training",
    run_label: str | None = None,
    checkpoint_dir: str | None = DEFAULT_CHECKPOINT_DIR,
    show_progress: bool = True,
    device: str = "auto",
) -> dict[str, Any]:
    """Tune and refit FusionCNN then FusionGRU on one prepared amputee data type."""
    return train_amputee_type(
        prepared, MODELS, INPUT_MODE,
        n_trials=n_trials, timeout=timeout, artifact_root=artifact_root,
        run_label=run_label, checkpoint_dir=checkpoint_dir,
        show_progress=show_progress, device=device,
    )


def train_fusion_amputee(
    amputee_id: int | str,
    *,
    n_trials: int = 50,
    timeout: int | None = None,
    artifact_root: str = "results/training",
    run_label: str | None = None,
    checkpoint_dir: str | None = DEFAULT_CHECKPOINT_DIR,
    show_progress: bool = True,
    device: str = "auto",
    data_types: Sequence[str] = AMPUTEE_DATA_TYPES,
    **data_kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Train both fusion models on every data type, one type after another.

    ``data_kwargs`` (total_budget_gb, seed, test_fraction, just_states_ratio,
    batch_size, data_root) are forwarded to ``prepare_amputee_experiment_data``.
    A failure stops the run before the remaining models and types.
    """
    return run_per_amputee_type(
        amputee_id,
        INPUT_MODE,
        lambda prepared: train_fusion_amputee_type(
            prepared,
            n_trials=n_trials, timeout=timeout, artifact_root=artifact_root,
            run_label=run_label, checkpoint_dir=checkpoint_dir,
            show_progress=show_progress, device=device,
        ),
        data_types=data_types,
        **data_kwargs,
    )
