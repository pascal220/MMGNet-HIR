import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from data_loader import PreparedData, prepare_experiment_data, prepare_training_data
from dataset_registry import normalize_amputee_id
from run_selection import ModelNotAvailableError, RunVerificationError

sys.path.insert(0, str(Path(__file__).resolve().parent / "train"))

from fusion_train import train_and_evaluate_fusion
from fusion_train_amputee import train_fusion_amputee
from fusion_windows_train import train_and_evaluate_fusion_windows
from fusion_windows_train_amputee import train_fusion_windows_amputee
from imu_cnn_train import train_and_evaluate_imu_cnn
from imu_cnn_windows_train import train_and_evaluate_imu_cnn_windows
from mmg_cnn_train import train_and_evaluate_mmg_cnn
from mmg_cnn_windows_train import train_and_evaluate_mmg_cnn_windows

sys.path.insert(0, str(Path(__file__).resolve().parent / "evaluation"))

from fusion_single_window_eval import evaluate_fusion_single_window
from fusion_single_window_eval_amputee import evaluate_fusion_single_window_amputee
from fusion_windows_eval import evaluate_fusion_windows
from fusion_windows_eval_amputee import evaluate_fusion_windows_amputee
from standalone_single_window_eval import evaluate_standalone_single_window
from standalone_windows_eval import evaluate_standalone_windows


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("session_registry.log"),
    ],
)
logger = logging.getLogger(__name__)


def _shape(tensor) -> tuple[int, ...]:
    """Return a plain tuple shape for concise logging."""
    return tuple(int(dim) for dim in tensor.shape)


def _log_prepared_summary(prepared: PreparedData) -> None:
    """Log the prepared tensor shapes and how they can be used later."""
    logger.info("Prepared input mode : %s", prepared.input_mode)
    logger.info("Model target        : %s", prepared.model_target)
    logger.info("IMU train tensor    : %s", _shape(prepared.X_imu_train))
    logger.info("IMU test tensor     : %s", _shape(prepared.X_imu_test))
    logger.info("MMG/CWT train tensor: %s", _shape(prepared.X_cwt_train))
    logger.info("MMG/CWT test tensor : %s", _shape(prepared.X_cwt_test))
    logger.info("Train labels        : %s", _shape(prepared.y_train))
    logger.info("Test labels         : %s", _shape(prepared.y_test))
    logger.info("Train metadata rows : %d", len(prepared.train_metadata))
    logger.info("Test metadata rows  : %d", len(prepared.test_metadata))

    logger.info("PreparedData is ready for the requested operation.")


def _select_train_and_evaluate(prepared: PreparedData):
    """Pick the training entry point matching input_mode and model_target."""
    if prepared.input_mode == "windowed":
        if prepared.model_target == "fusion":
            return train_and_evaluate_fusion_windows
        return train_and_evaluate_mmg_cnn_windows, train_and_evaluate_imu_cnn_windows

    if prepared.model_target == "fusion":
        return train_and_evaluate_fusion
    return train_and_evaluate_mmg_cnn, train_and_evaluate_imu_cnn


def _run_selected_training(prepared: PreparedData, train_kwargs: dict[str, Any]) -> None:
    """Run the selected training entry point(s) in their required order."""
    entry_points = _select_train_and_evaluate(prepared)
    if not isinstance(entry_points, tuple):
        entry_points = (entry_points,)

    for train_and_evaluate in entry_points:
        logger.info("Starting training with %s.", train_and_evaluate.__name__)
        train_and_evaluate(prepared, **train_kwargs)
        logger.info("Completed training with %s.", train_and_evaluate.__name__)


def _select_evaluate(prepared: PreparedData):
    """Pick the evaluation entry point matching input_mode and model_target."""
    if prepared.input_mode == "windowed":
        if prepared.model_target == "fusion":
            return evaluate_fusion_windows
        return evaluate_standalone_windows

    if prepared.model_target == "fusion":
        return evaluate_fusion_single_window
    return evaluate_standalone_single_window


def _run_stage(stage: str, action: Callable[[], object]) -> bool:
    """Run one stage, logging instead of raising; return whether it succeeded."""
    try:
        action()
    except (ModelNotAvailableError, RunVerificationError) as exc:
        logger.error("%s stopped: %s", stage, exc)
        return False
    except Exception:
        logger.exception("%s failed; stopping execution.", stage)
        return False
    return True


def _run_amputee(args: argparse.Namespace, train_kwargs: dict[str, Any]) -> int:
    """Train and/or evaluate the amputee fusion models on every data type in turn."""
    if args.input_mode == "windowed":
        train, evaluate = train_fusion_windows_amputee, evaluate_fusion_windows_amputee
    else:
        train, evaluate = train_fusion_amputee, evaluate_fusion_single_window_amputee
    data_kwargs = {
        "total_budget_gb": args.total_budget_gb,
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "just_states_ratio": args.just_states_ratio,
        "batch_size": args.batch_size,
    }

    if args.train:
        logger.info("Starting amputee training with %s.", train.__name__)
        if not _run_stage(
            "Training", lambda: train(args.amputee_id, **train_kwargs, **data_kwargs)
        ):
            return 1
        logger.info("Completed amputee training with %s.", train.__name__)

    if args.test:
        logger.info("Starting amputee evaluation with %s.", evaluate.__name__)
        if not _run_stage("Evaluation", lambda: evaluate(args.amputee_id, **data_kwargs)):
            return 1
        logger.info("Completed amputee evaluation with %s.", evaluate.__name__)

    return 0


def main() -> int:
    """Prepare data and run the operation selected by the command-line flags."""
    parser = argparse.ArgumentParser(
        description="Prepare volunteer-based train/test tensors."
    )
    parser.add_argument(
        "--same-volunteer-id",
        default=None,
        help="Use one volunteer for both splits (e.g. 4 or N004).",
    )
    parser.add_argument("--train-volunteer-count", type=int, default=None)
    parser.add_argument("--test-volunteer-count", type=int, default=None)
    parser.add_argument(
        "--amputee-id",
        default=None,
        help=(
            "Use one amputee's data from data/amputee (e.g. 3 or A003). Every data "
            "type is a separate dataset with its own same-subject split; one run "
            "trains/evaluates the fusion models on each type in turn."
        ),
    )
    parser.add_argument("--total-budget-gb", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--just-states-ratio", type=float, default=1.05)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help=(
            "Optuna trial budget for each trained model. Defaults to each entry "
            "point's own budget (100 for standalone models, 50 for fusion models)."
        ),
    )
    parser.add_argument(
        "--input-mode",
        choices=["single_window", "windowed"],
        default="windowed",
        help=(
            "Use 'single_window' to expand each of the 4 windows into separate "
            "samples, or 'windowed' to keep all 4 windows inside each sample."
        ),
    )
    parser.add_argument(
        "--model-target",
        choices=["standalone", "fusion"],
        default=None,
        help=(
            "Prepare tensors for standalone IMU/MMG models or paired fusion models "
            "(default: standalone; amputee runs are always fusion)."
        ),
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run the selected training entry point(s).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Evaluate the latest trained models matching this split on the "
            "test set and save plots and metrics under results/evaluation."
        ),
    )
    args = parser.parse_args()

    if not args.train and not args.test:
        parser.error("at least one of --train or --test is required")
    if args.same_volunteer_id is not None and (
        args.train_volunteer_count is not None or args.test_volunteer_count is not None
    ):
        parser.error(
            "--same-volunteer-id cannot be combined with --train-volunteer-count "
            "or --test-volunteer-count"
        )
    if args.n_trials is not None and args.n_trials < 1:
        parser.error("--n-trials must be a positive integer")
    train_kwargs = {} if args.n_trials is None else {"n_trials": args.n_trials}

    if args.amputee_id is not None:
        if (
            args.same_volunteer_id is not None
            or args.train_volunteer_count is not None
            or args.test_volunteer_count is not None
        ):
            parser.error(
                "--amputee-id cannot be combined with --same-volunteer-id, "
                "--train-volunteer-count or --test-volunteer-count"
            )
        if args.model_target == "standalone":
            parser.error("--amputee-id supports only --model-target fusion")
        try:
            args.amputee_id = normalize_amputee_id(args.amputee_id)
        except ValueError as exc:
            parser.error(str(exc))
        return _run_amputee(args, train_kwargs)

    model_target = args.model_target or "standalone"
    train_volunteer_count = 5 if args.train_volunteer_count is None else args.train_volunteer_count
    test_volunteer_count = 5 if args.test_volunteer_count is None else args.test_volunteer_count

    experiment = prepare_experiment_data(
        same_volunteer_id=args.same_volunteer_id,
        train_volunteer_count=train_volunteer_count,
        test_volunteer_count=test_volunteer_count,
        total_budget_gb=args.total_budget_gb,
        seed=args.seed,
        test_fraction=args.test_fraction,
        just_states_ratio=args.just_states_ratio,
        batch_size=args.batch_size,
    )

    prepared = prepare_training_data(
        experiment,
        input_mode=args.input_mode,
        model_target=model_target,
    )

    _log_prepared_summary(prepared)

    if args.train and not _run_stage(
        "Training", lambda: _run_selected_training(prepared, train_kwargs)
    ):
        return 1

    if args.test:
        evaluate = _select_evaluate(prepared)
        logger.info("Starting evaluation with %s.", evaluate.__name__)
        if not _run_stage("Evaluation", lambda: evaluate(prepared)):
            return 1
        logger.info("Completed evaluation with %s.", evaluate.__name__)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
