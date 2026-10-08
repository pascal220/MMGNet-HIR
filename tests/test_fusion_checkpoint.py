from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "models")]

from fusion_cnn_model import FusionCNN
from fusion_cnn_window_model import FusionCNNWindow
from fusion_gru_model import FusionGRU
from fusion_gru_window_model import FusionGRUWindow
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
        self.assertFalse(any(p.requires_grad for p in loaded.backbones.parameters()))

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
