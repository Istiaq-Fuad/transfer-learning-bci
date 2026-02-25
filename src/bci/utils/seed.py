"""Reproducibility and device utilities."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Random seed set to %d", seed)


def get_device(preference: str = "auto") -> torch.device:
    """Determine the best available compute device.

    Args:
        preference: Device preference. Options:
            "auto": Use CUDA if available, then MPS, then CPU.
            "cpu": Force CPU.
            "cuda": Force CUDA (raises if not available).
            "mps": Force MPS/Apple Silicon.

    Returns:
        torch.device object.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
    elif preference == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info(
            "  GPU: %s (%.1f GB)",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_mem / 1e9,
        )

    return device
