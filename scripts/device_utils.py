"""Shared PyTorch compute-device selection and diagnostics."""

from __future__ import annotations

import re
from typing import Any

import torch

_DEVICE_PATTERN = re.compile(r"^(?:auto|cpu|cuda(?::\d+)?)$")


def normalize_device_request(device: str | torch.device | None) -> str:
    """Normalize and validate a user-supplied compute-device request."""
    requested = "auto" if device is None else str(device).strip().lower()
    if not _DEVICE_PATTERN.fullmatch(requested):
        raise ValueError(
            "device must be 'auto', 'cpu', 'cuda', or an indexed CUDA device "
            "such as 'cuda:0'."
        )
    return requested


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """Resolve ``auto`` and validate explicit CUDA requests.

    ``auto`` uses the current CUDA device when CUDA is usable and otherwise
    falls back to CPU. An explicit CUDA request fails with actionable guidance
    rather than silently running on CPU.
    """
    requested = normalize_device_request(device)
    if requested == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if requested == "cpu":
        return torch.device("cpu")

    if torch.version.cuda is None:
        raise RuntimeError(
            f"CUDA device {requested!r} was requested, but this is a CPU-only "
            f"PyTorch build ({torch.__version__}). Install a CUDA-enabled PyTorch "
            "wheel from https://pytorch.org/get-started/locally/."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {requested!r} was requested, but CUDA is not available. "
            "Check the NVIDIA driver, GPU visibility, and PyTorch installation."
        )

    resolved = torch.device(requested)
    index = torch.cuda.current_device() if resolved.index is None else resolved.index
    device_count = torch.cuda.device_count()
    if index < 0 or index >= device_count:
        raise ValueError(
            f"CUDA device index {index} is unavailable; PyTorch detected "
            f"{device_count} CUDA device(s)."
        )
    return resolved


def device_details(
    requested: str | torch.device | None,
    resolved: torch.device,
) -> dict[str, Any]:
    """Return JSON-safe runtime details for logs and experiment manifests."""
    details: dict[str, Any] = {
        "requested": normalize_device_request(requested),
        "resolved": str(resolved),
        "type": resolved.type,
        "index": resolved.index,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "cudnn_version": torch.backends.cudnn.version(),
        "device_name": None,
    }
    if resolved.type == "cuda":
        index = torch.cuda.current_device() if resolved.index is None else resolved.index
        details["index"] = index
        details["device_name"] = torch.cuda.get_device_name(index)
    return details


def describe_device(details: dict[str, Any]) -> str:
    """Format concise compute-device diagnostics for runtime logging."""
    summary = (
        f"requested={details['requested']} | resolved={details['resolved']} | "
        f"torch={details['torch']}"
    )
    if details["type"] == "cuda":
        return (
            f"{summary} | GPU={details['device_name']} | "
            f"CUDA={details['cuda_version']} | cuDNN={details['cudnn_version']}"
        )
    if details["cuda_version"] is None:
        return f"{summary} | CUDA support is not compiled into this PyTorch build"
    return f"{summary} | CUDA runtime is currently unavailable"
