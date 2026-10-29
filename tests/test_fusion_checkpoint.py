from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "models")]

from fusion_cnn_model import FusionCNN, FusionCNNTrainer
from fusion_cnn_window_model import FusionCNNWindow, FusionCNNWindowTrainer
from fusion_gru_model import FusionGRU, FusionGRUTrainer
from fusion_gru_window_model import FusionGRUWindow, FusionGRUWindowTrainer
from imu_cnn_model import IntentCNN
from imu_cnn_window_model import IntentCNNWindow
from mmg_cnn_model import LocomotionMMGCNN
from mmg_cnn_window_model import LocomotionMMGCNNWindow

SINGLE_IMU = dict(in_channels=6, num_classes=7, block_filters=[4, 4], kernel_pairs=[(3, 5), (3, 5)])
SINGLE_MMG = dict(
    in_channels=5, num_classes=7, block_filters=[4, 4], kernel_sizes=[5, 3],
    strides=[3, 1], dropout_rates=[0.0, 0.0], fc_hidden=8,
)
WINDOW_IMU = dict(
    in_channels=6, num_classes=7, first_conv_filters=4, first_conv_kernel_width=3,
    block_filters=[4, 4], kernel_pairs=[(3, 5), (3, 5)],
)
WINDOW_MMG = dict(
    in_channels=5, num_classes=7, first_conv_filters=4, first_conv_kernel_freq=3,
    first_conv_kernel_time=3, block_filters=[4, 4], kernel_sizes=[3, 3],
    strides=[3, 1], dropout_rates=[0.0, 0.0], fc_hidden=8,
)


def _save_backbone(path: Path, model: torch.nn.Module, config: dict) -> str:
    torch.save({"model_state_dict": model.state_dict(), "model_config": config}, path)
    return str(path)


class FusionCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _assert_one_file_roundtrip(self, build_fusion, fusion_class, parents, inputs) -> None:
        imu_path = _save_backbone(self.root / "imu.pt", *parents[0])
        mmg_path = _save_backbone(self.root / "mmg.pt", *parents[1])
        model = build_fusion(imu_path, mmg_path).eval()
        fusion_path = self.root / "fusion.pt"
        torch.save(
            {"model_state_dict": model.state_dict(), "model_config": model.config},
            fusion_path,
        )
        # Loading must not touch the backbone checkpoints.
        Path(imu_path).unlink()
        Path(mmg_path).unlink()

        loaded = fusion_class.from_checkpoint(str(fusion_path), "cpu")

        with torch.no_grad():
            expected = model.cpu()(*inputs)
            actual = loaded(*inputs)
        self.assertTrue(torch.equal(expected, actual))

    def _fine_tuning_cases(self):
        single = (
            (IntentCNN(**SINGLE_IMU), SINGLE_IMU),
            (LocomotionMMGCNN(**SINGLE_MMG), SINGLE_MMG),
        )
        window = (
            (IntentCNNWindow(**WINDOW_IMU), WINDOW_IMU),
            (LocomotionMMGCNNWindow(**WINDOW_MMG), WINDOW_MMG),
        )
        single_batch = (
            (torch.randn(4, 6, 125), torch.randn(4, 5, 40, 125)),
            torch.tensor([0, 1, 2, 3]),
        )
        window_batch = (
            (torch.randn(4, 6, 125, 4), torch.randn(4, 5, 40, 125, 4)),
            torch.tensor([0, 1, 2, 3]),
        )
        flat = lambda batch: (*batch[0], batch[1])
        return [
            (single, lambda a, b: FusionCNN(a, b, 7, 8), FusionCNNTrainer, [flat(single_batch)]),
            (single, lambda a, b: FusionGRU(a, b, 7, gru_hidden=8, fc_hidden=8),
             FusionGRUTrainer, [flat(single_batch)]),
            (window, lambda a, b: FusionCNNWindow(a, b, 7, [8]),
             FusionCNNWindowTrainer, [window_batch]),
            (window, lambda a, b: FusionGRUWindow(a, b, 7, 8, fc_hidden_dims=[8]),
             FusionGRUWindowTrainer, [window_batch]),
        ]

    def test_backbones_are_fine_tuned_with_a_reduced_learning_rate(self) -> None:
        for parents, build, trainer_class, batches in self._fine_tuning_cases():
            imu_path = _save_backbone(self.root / "imu.pt", *parents[0])
            mmg_path = _save_backbone(self.root / "mmg.pt", *parents[1])
            imu_bytes = Path(imu_path).read_bytes()
            model = build(imu_path, mmg_path)
            with self.subTest(model=type(model).__name__):
                trainer = trainer_class(
                    model,
                    {"optimizer": "Adam", "lr": 1e-2, "backbone_lr_scale": 0.1,
                     "device": "cpu"},
                )
                head_group, backbone_group = trainer.optimizer.param_groups
                self.assertAlmostEqual(backbone_group["lr"], head_group["lr"] * 0.1)

                before = {
                    name: p.detach().clone()
                    for name, p in model.backbones.named_parameters()
                    if p.requires_grad
                }
                self.assertTrue(before)
                trainer._run_epoch(batches, training=True)
                self.assertTrue(model.backbones.training)
                changed = [
                    name for name, p in model.backbones.named_parameters()
                    if name in before and not torch.equal(before[name], p)
                ]
                self.assertTrue(changed)

                trainer._run_epoch(batches, training=False)
                self.assertFalse(model.backbones.training)
                # The original backbone checkpoint is never modified.
                self.assertEqual(Path(imu_path).read_bytes(), imu_bytes)

    def test_non_positive_backbone_lr_scale_is_rejected(self) -> None:
        parents, build, trainer_class, _ = self._fine_tuning_cases()[0]
        model = build(
            _save_backbone(self.root / "imu.pt", *parents[0]),
            _save_backbone(self.root / "mmg.pt", *parents[1]),
        )
        with self.assertRaises(ValueError):
            trainer_class(model, {"lr": 1e-3, "backbone_lr_scale": 0, "device": "cpu"})

    def test_single_window_fusion_models_load_from_one_file(self) -> None:
        parents = (
            (IntentCNN(**SINGLE_IMU), SINGLE_IMU),
            (LocomotionMMGCNN(**SINGLE_MMG), SINGLE_MMG),
        )
        inputs = (torch.randn(2, 6, 125), torch.randn(2, 5, 40, 125))
        self._assert_one_file_roundtrip(
            lambda a, b: FusionCNN(a, b, 7, 8), FusionCNN, parents, inputs
        )
        self._assert_one_file_roundtrip(
            lambda a, b: FusionGRU(a, b, 7, gru_hidden=8, fc_hidden=8),
            FusionGRU, parents, inputs,
        )

    def test_windowed_fusion_models_load_from_one_file(self) -> None:
        parents = (
            (IntentCNNWindow(**WINDOW_IMU), WINDOW_IMU),
            (LocomotionMMGCNNWindow(**WINDOW_MMG), WINDOW_MMG),
        )
        inputs = (torch.randn(2, 6, 125, 4), torch.randn(2, 5, 40, 125, 4))
        self._assert_one_file_roundtrip(
            lambda a, b: FusionCNNWindow(a, b, 7, [8]), FusionCNNWindow, parents, inputs
        )
        self._assert_one_file_roundtrip(
            lambda a, b: FusionGRUWindow(a, b, 7, 8, fc_hidden_dims=[8]),
            FusionGRUWindow, parents, inputs,
        )

    def test_missing_backbone_source_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FusionCNN(None, None, 7, 8)


if __name__ == "__main__":
    unittest.main()
