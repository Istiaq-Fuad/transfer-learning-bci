"""Dual-branch dataset builder for Phase 2 training.

For each CV fold the builder:
    1. Fits CSP + Riemannian extractors on the training split only (no leakage).
    2. Transforms both splits to get handcrafted feature vectors.
    3. Generates CWT spectrogram images for both splits.
    4. Returns PyTorch DualBranchDatasets ready for the Trainer.

This is deliberately kept separate from the data loading code so that the same
builder works with both synthetic data and real BCI IV-2a epochs.

Usage:
    builder = DualBranchFoldBuilder(config)
    train_ds, test_ds, math_input_dim = builder.build_fold(
        X_train, y_train, X_test, y_test,
        channel_names=["C3", "Cz", "C4"], sfreq=128.0,
    )
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import TensorDataset

from bci.data.transforms import CWTSpectrogramTransform
from bci.features.csp import CSPFeatureExtractor
from bci.features.riemannian import RiemannianFeatureExtractor
from bci.utils.config import ModelConfig, SpectrogramConfig

logger = logging.getLogger(__name__)


class DualBranchFoldBuilder:
    """Builds train/test DualBranch datasets for a single CV fold.

    Encapsulates the feature extraction pipeline so each fold gets its own
    CSP and Riemannian estimators fitted exclusively on training data.

    Args:
        csp_n_components: Number of CSP components (default 6).
        csp_reg: CSP regularization (default "ledoit_wolf").
        riemann_estimator: Covariance estimator for Riemannian (default "lwf").
        riemann_metric: Riemannian metric (default "riemann").
        spec_config: CWT spectrogram configuration. Uses defaults if None.
        sfreq: EEG sampling frequency (Hz). Used for CWT.
        channel_names: EEG channel names (used to select C3/Cz/C4 channels).
    """

    def __init__(
        self,
        csp_n_components: int = 6,
        csp_reg: str = "ledoit_wolf",
        riemann_estimator: str = "lwf",
        riemann_metric: str = "riemann",
        spec_config: SpectrogramConfig | None = None,
        sfreq: float = 128.0,
        channel_names: list[str] | None = None,
    ) -> None:
        self.csp_n_components = csp_n_components
        self.csp_reg = csp_reg
        self.riemann_estimator = riemann_estimator
        self.riemann_metric = riemann_metric
        self.sfreq = sfreq
        self.channel_names = channel_names or ["C3", "Cz", "C4"]

        self.spec_config = spec_config or SpectrogramConfig(
            wavelet="morl",
            freq_min=4.0,
            freq_max=40.0,
            n_freqs=64,
            image_size=(224, 224),
            channel_mode="rgb_c3_cz_c4",
        )
        self._transform = CWTSpectrogramTransform(self.spec_config)

    def build_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[TensorDataset, TensorDataset, int]:
        """Build train and test datasets for one CV fold.

        Feature extractors are fitted ONLY on X_train to prevent data leakage.

        Args:
            X_train: Training EEG epochs (n_train, n_channels, n_times).
            y_train: Training labels (n_train,).
            X_test:  Test EEG epochs (n_test, n_channels, n_times).
            y_test:  Test labels (n_test,).

        Returns:
            (train_dataset, test_dataset, math_input_dim)
            Each dataset yields (image_tensor, feature_tensor, label_tensor).
        """
        # --- Step 1: Fit + transform handcrafted features ---
        csp = CSPFeatureExtractor(
            n_components=self.csp_n_components, reg=self.csp_reg
        )
        riemann = RiemannianFeatureExtractor(
            estimator=self.riemann_estimator, metric=self.riemann_metric
        )

        logger.info(
            "  Building fold: train=%d test=%d", len(y_train), len(y_test)
        )

        csp_train = csp.fit_transform(X_train, y_train)          # (n_train, n_csp)
        csp_test = csp.transform(X_test)                          # (n_test, n_csp)

        riemann_train = riemann.fit_transform(X_train, y_train)   # (n_train, n_riemann)
        riemann_test = riemann.transform(X_test)                  # (n_test, n_riemann)

        features_train = np.concatenate([csp_train, riemann_train], axis=1).astype(np.float32)
        features_test  = np.concatenate([csp_test,  riemann_test],  axis=1).astype(np.float32)
        math_input_dim = features_train.shape[1]

        logger.info(
            "  Handcrafted features: CSP=%d + Riemannian=%d = %d total",
            csp_train.shape[1], riemann_train.shape[1], math_input_dim,
        )

        # --- Step 2: Generate CWT spectrogram images ---
        logger.info("  Generating CWT spectrograms for train set...")
        imgs_train = self._epochs_to_images(X_train)  # (n_train, 3, 224, 224)

        logger.info("  Generating CWT spectrograms for test set...")
        imgs_test = self._epochs_to_images(X_test)    # (n_test, 3, 224, 224)

        # --- Step 3: Pack into TensorDatasets ---
        train_ds = TensorDataset(
            torch.tensor(imgs_train, dtype=torch.float32),
            torch.tensor(features_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        test_ds = TensorDataset(
            torch.tensor(imgs_test, dtype=torch.float32),
            torch.tensor(features_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        )

        return train_ds, test_ds, math_input_dim

    def _epochs_to_images(self, X: np.ndarray) -> np.ndarray:
        """Convert EEG epochs to normalised CHW float32 images.

        Args:
            X: EEG epochs (n_trials, n_channels, n_times).

        Returns:
            Float32 images of shape (n_trials, 3, H, W) in [0, 1].
        """
        images_hwc = self._transform.transform_epochs(
            X, self.channel_names, self.sfreq
        )  # (n_trials, H, W, 3) uint8
        # -> (n_trials, 3, H, W) float32 in [0, 1]
        return images_hwc.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
