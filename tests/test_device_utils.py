from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from device_utils import device_details, normalize_device_request, resolve_device


class DeviceUtilsTests(unittest.TestCase):
    def test_normalize_accepts_supported_device_requests(self) -> None:
        self.assertEqual(normalize_device_request(None), "auto")
        self.assertEqual(normalize_device_request(" CUDA:1 "), "cuda:1")
        self.assertEqual(normalize_device_request(torch.device("cpu")), "cpu")

    def test_normalize_rejects_unsupported_device_requests(self) -> None:
        for value in ("gpu", "cuda:-1", "cuda:all", "mps"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_device_request(value)

    def test_auto_falls_back_to_cpu_when_cuda_is_unavailable(self) -> None:
        with patch("device_utils.torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("auto"), torch.device("cpu"))

    def test_explicit_cuda_rejects_cpu_only_pytorch(self) -> None:
        with (
            patch.object(torch.version, "cuda", None),
            self.assertRaisesRegex(RuntimeError, "CPU-only PyTorch build"),
        ):
            resolve_device("cuda")

    def test_indexed_cuda_is_validated(self) -> None:
        with (
            patch.object(torch.version, "cuda", "12.8"),
            patch("device_utils.torch.cuda.is_available", return_value=True),
            patch("device_utils.torch.cuda.device_count", return_value=1),
            self.assertRaisesRegex(ValueError, "detected 1 CUDA device"),
        ):
            resolve_device("cuda:1")

    def test_cpu_device_details_are_json_safe(self) -> None:
        details = device_details("cpu", torch.device("cpu"))
        self.assertEqual(details["requested"], "cpu")
        self.assertEqual(details["resolved"], "cpu")
        self.assertIsNone(details["device_name"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_real_cuda_forward_and_backward(self) -> None:
        device = resolve_device("cuda")
        model = torch.nn.Linear(4, 2).to(device)
        inputs = torch.randn(3, 4, device=device)
        loss = model(inputs).sum()
        loss.backward()

        self.assertEqual(next(model.parameters()).device.type, "cuda")
        self.assertIsNotNone(next(model.parameters()).grad)


if __name__ == "__main__":
    unittest.main()
