import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from data_loader import PreparedData, prepare_experiment_data, prepare_training_data


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

    if prepared.model_target == "fusion":
        logger.info("Prepared fusion arguments are available as prepared.fusion_args.")
    else:
        logger.info("Prepared standalone IMU arguments are available as prepared.imu_args.")
        logger.info("Prepared standalone MMG arguments are available as prepared.mmg_args.")


def _log_test_dispatch_hint(prepared: PreparedData) -> None:
    """Log which test entry points will be called after their bodies are refactored."""
    if prepared.model_target == "fusion":
        logger.info("Next refactor will route to tests.fusion_test.train_and_evaluate(*prepared.fusion_args).")
        return

    logger.info("Next refactor will route to tests.imu_cnn_test.train_and_evaluate(*prepared.imu_args).")
    logger.info("Next refactor will route to tests.mmg_cnn_test.train_and_evaluate(*prepared.mmg_args).")


def main() -> int:
    """Prepare train/test tensors for later model test-script execution."""
    parser = argparse.ArgumentParser(
        description="Prepare volunteer-based train/test tensors."
    )
    parser.add_argument(
        "--setup",
        choices=["separate_volunteers", "same_volunteer"],
        default="separate_volunteers",
    )
    parser.add_argument(
        "--same-volunteer-id",
        default=None,
        help="Volunteer ID for same_volunteer mode (e.g. 4 or N004).",
    )
    parser.add_argument("--train-volunteer-count", type=int, default=8)
    parser.add_argument("--test-volunteer-count", type=int, default=2)
    parser.add_argument("--total-budget-gb", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--just-states-ratio", type=float, default=1.10)
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
    args = parser.parse_args()

    experiment = prepare_experiment_data(
        setup=args.setup,
        same_volunteer_id=args.same_volunteer_id,
        train_volunteer_count=args.train_volunteer_count,
        test_volunteer_count=args.test_volunteer_count,
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
    _log_test_dispatch_hint(prepared)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
