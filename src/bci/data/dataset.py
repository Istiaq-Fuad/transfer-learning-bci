"""PyTorch Dataset classes for MI-EEG data.

Provides Dataset implementations compatible with PyTorch DataLoader:
    - SpectrogramDataset: For ViT branch (image-based)
    - EEGFeatureDataset: For math branch (CSP + Riemannian features)
    - DualBranchDataset: Combined dataset for the full pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SpectrogramDataset(Dataset):
    """PyTorch Dataset for spectrogram images.

    Serves the ViT branch. Loads pre-computed spectrogram images
    and their corresponding labels.

    Args:
        images: ndarray of shape (N, H, W, C) or (N, H, W) — uint8 images.
        labels: ndarray of shape (N,) — integer class labels.
        transform: Optional torchvision transform to apply to images.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: object | None = None,
    ) -> None:
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = self.images[idx]

        # Convert grayscale to 3-channel if needed
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        # Convert to float32 [0, 1] and to CHW format for PyTorch
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        image = torch.from_numpy(image)

        # Apply transform (e.g., torchvision normalization)
        if self.transform is not None:
            image = self.transform(image)

        label = int(self.labels[idx])
        return image, label


class EEGFeatureDataset(Dataset):
    """PyTorch Dataset for handcrafted EEG features.

    Serves the math branch with pre-extracted CSP and/or Riemannian features.

    Args:
        features: ndarray of shape (N, feature_dim) — concatenated features.
        labels: ndarray of shape (N,) — integer class labels.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.features[idx], self.labels[idx].item()


class DualBranchDataset(Dataset):
    """Combined dataset for the dual-branch model.

    Returns both spectrogram images and handcrafted features for each trial.

    Args:
        images: ndarray of shape (N, H, W, C) — spectrogram images.
        features: ndarray of shape (N, feature_dim) — handcrafted features.
        labels: ndarray of shape (N,) — integer class labels.
        transform: Optional torchvision transform for images.
    """

    def __init__(
        self,
        images: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
        transform: object | None = None,
    ) -> None:
        assert len(images) == len(features) == len(labels), (
            f"Length mismatch: images={len(images)}, features={len(features)}, "
            f"labels={len(labels)}"
        )
        self.images = images
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        # Image processing (same as SpectrogramDataset)
        image = self.images[idx]
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image)
        if self.transform is not None:
            image = self.transform(image)

        features = self.features[idx]
        label = int(self.labels[idx])

        return image, features, label
