import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from data_loader import PreparedData, prepare_experiment_data, prepare_training_data

sys.path.insert(0, str(Path(__file__).resolve().parent / "train"))

from fusion_train import train_and_evaluate_fusion
from fusion_windows_train import train_and_evaluate_fusion_windows
from imu_cnn_train import train_and_evaluate_imu_cnn
from imu_cnn_windows_train import train_and_evaluate_imu_cnn_windows
from mmg_cnn_train import train_and_evaluate_mmg_cnn
from mmg_cnn_windows_train import train_and_evaluate_mmg_cnn_windows


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


def _run_selected_training(prepared: PreparedData) -> None:
    """Run the selected training entry point(s) in their required order."""
    entry_points = _select_train_and_evaluate(prepared)
    if not isinstance(entry_points, tuple):
        entry_points = (entry_points,)

    for train_and_evaluate in entry_points:
        logger.info("Starting training with %s.", train_and_evaluate.__name__)
        train_and_evaluate(prepared)
        logger.info("Completed training with %s.", train_and_evaluate.__name__)


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
    parser.add_argument("--total-budget-gb", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--just-states-ratio", type=float, default=1.05)
    parser.add_argument("--batch-size", type=int, default=32)
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
        default="standalone",
        help="Prepare tensors for standalone IMU/MMG models or paired fusion models.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run the selected training entry point(s).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Request testing; testing is not implemented yet.",
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

    train_volunteer_count = 9 if args.train_volunteer_count is None else args.train_volunteer_count
    test_volunteer_count = 1 if args.test_volunteer_count is None else args.test_volunteer_count

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
        model_target=args.model_target,
    )

    _log_prepared_summary(prepared)

    if args.train:
        try:
            _run_selected_training(prepared)
        except Exception:
            logger.exception("Training failed; stopping execution.")
            return 1

    if args.test:
        logger.warning("Testing was requested, but testing is not implemented yet.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
