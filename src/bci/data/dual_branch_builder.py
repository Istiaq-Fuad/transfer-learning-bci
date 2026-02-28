"""Dual-branch dataset builder for Phase 2 training.

For each CV fold the builder:
    1. Applies the preprocessing pipeline (notch, rejection, crop, EA, z-score).
    2. Fits CSP + Riemannian extractors on the training split only (no leakage).
    3. Transforms both splits to get handcrafted feature vectors.
    4. Generates CWT spectrogram images for both splits.
    5. Applies EEG augmentation to the TRAINING split only.
    6. Applies ImageNet normalisation (if configured).
    7. Returns PyTorch TensorDatasets ready for the Trainer.

This is deliberately kept separate from the data loading code so that the same
builder works with both synthetic data and real BCI IV-2a epochs.

BCI Competition IV-2a channel order (22 EEG channels, 0-indexed):
    Index  0  1  2  3   4   5  6  7  8   9  10  11 12  13  14  15  16  17  18  19  20  21
    Name  Fz FC3 FC1 FCz FC2 FC4 C5 C3  C1  Cz  C2  C4  C6 CP3 CP1 CPz CP2 CP4  P1  Pz  P2 POz

Key motor-cortex indices:
    C3 → index 7    (left hemisphere)
    C4 → index 11   (right hemisphere)

Usage:
    builder = DualBranchFoldBuilder()
    train_ds, test_ds, math_input_dim = builder.build_fold(
        X_train, y_train, X_test, y_test,
        channel_names=BCI_IV2A_CHANNEL_NAMES, sfreq=128.0,
    )
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import TensorDataset

from bci.data.augmentation import EEGAugmenter
from bci.data.preprocessing import apply_preprocessing_pipeline
from bci.data.transforms import CWTSpectrogramTransform, normalize_imagenet
from bci.features.csp import FBCSPFeatureExtractor
from bci.features.riemannian import FBRiemannianFeatureExtractor
from bci.utils.config import AugmentationConfig, PreprocessingConfig, SpectrogramConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Full 22-channel BCI Competition IV-2a channel list (correct order)
# ---------------------------------------------------------------------------
BCI_IV2A_CHANNEL_NAMES: list[str] = [
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
]


class DualBranchFoldBuilder:
    """Builds train/test DualBranch datasets for a single CV fold.

    Encapsulates the full pipeline: preprocessing → feature extraction →
    CWT spectrogram → augmentation → ImageNet normalisation.

    Uses FBCSPFeatureExtractor (filter bank CSP) and FBRiemannianFeatureExtractor
    (filter bank tangent space) for richer handcrafted features. All fitting is
    done exclusively on the training split to prevent data leakage.

    Args:
        csp_n_components: CSP components per band (default 4).
        csp_k_best: Feature selector k for FBCSP (default 12).
        riemann_estimator: Covariance estimator for FBRiemannian (default "oas").
        riemann_metric: Riemannian metric (default "riemann").
        riemann_n_components_pca: PCA output dim for FBRiemannian (default 128).
        spec_config: CWT spectrogram configuration. Uses defaults if None.
        prep_config: Preprocessing configuration. Uses defaults if None.
        aug_config: Augmentation configuration. Uses defaults if None.
        sfreq: EEG sampling frequency (Hz). Used for CWT and filter bank.
        channel_names: EEG channel names. Defaults to BCI IV-2a 22-channel list.
        apply_augmentation: Whether to apply EEG augmentation to training data.
        seed: Random seed for augmentation.
    """

    def __init__(
        self,
        csp_n_components: int = 4,
        csp_k_best: int | None = 12,
        riemann_estimator: str = "oas",
        riemann_metric: str = "riemann",
        riemann_n_components_pca: int | None = 128,
        spec_config: SpectrogramConfig | None = None,
        prep_config: PreprocessingConfig | None = None,
        aug_config: AugmentationConfig | None = None,
        sfreq: float = 128.0,
        channel_names: list[str] | None = None,
        apply_augmentation: bool = True,
        seed: int = 42,
    ) -> None:
        self.csp_n_components = csp_n_components
        self.csp_k_best = csp_k_best
        self.riemann_estimator = riemann_estimator
        self.riemann_metric = riemann_metric
        self.riemann_n_components_pca = riemann_n_components_pca
        self.sfreq = sfreq
        self.channel_names = channel_names or BCI_IV2A_CHANNEL_NAMES
        self.apply_augmentation = apply_augmentation

        self.prep_config = prep_config or PreprocessingConfig(
            # Trial rejection is off by default because the builder operates on
            # numpy arrays that may not be in volt scale (e.g. synthetic test data).
            # Scripts that pass real MOABB data can supply a prep_config with
            # apply_trial_rejection=True explicitly.
            apply_trial_rejection=False,
        )
        self.aug_config = aug_config or AugmentationConfig()

        self.spec_config = spec_config or SpectrogramConfig(
            wavelet="morl",
            freq_min=8.0,
            freq_max=32.0,
            n_freqs=64,
            image_size=(224, 224),
            channel_mode="multichannel",
            normalize_mode="joint",
            apply_imagenet_norm=False,
        )
        self._transform = CWTSpectrogramTransform(self.spec_config)
        self._augmenter = EEGAugmenter(self.aug_config, seed=seed)

    def build_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[TensorDataset, TensorDataset, int]:
        """Build train and test datasets for one CV fold.

        Preprocessing and feature-extractor fitting happen ONLY on X_train.
        X_test is transformed using the already-fitted estimators.

        Args:
            X_train: Training EEG epochs (n_train, n_channels, n_times).
            y_train: Training labels (n_train,).
            X_test:  Test EEG epochs (n_test, n_channels, n_times).
            y_test:  Test labels (n_test,).

        Returns:
            (train_dataset, test_dataset, math_input_dim)
            Each dataset yields (image_tensor, feature_tensor, label_tensor).
        """
        logger.info("  Building fold: train=%d test=%d", len(y_train), len(y_test))

        # --- Step 0: Preprocessing pipeline ---
        # Note: preprocessing is fit-free (no leakage), but Euclidean Alignment
        # is computed per split independently to avoid test-set statistics leaking.
        X_train, y_train = apply_preprocessing_pipeline(
            X_train, y_train, self.sfreq, self.prep_config
        )
        X_test, y_test = apply_preprocessing_pipeline(X_test, y_test, self.sfreq, self.prep_config)

        # --- Step 1: EEG augmentation on TRAINING only ---
        if self.apply_augmentation:
            X_train = self._augmenter(X_train, training=True)

        # --- Step 2: Fit + transform handcrafted features ---
        csp = FBCSPFeatureExtractor(
            n_components=self.csp_n_components,
            k_best=self.csp_k_best,
            sfreq=self.sfreq,
        )
        riemann = FBRiemannianFeatureExtractor(
            estimator=self.riemann_estimator,
            metric=self.riemann_metric,
            n_components_pca=self.riemann_n_components_pca,
            sfreq=self.sfreq,
        )

        csp_train = csp.fit_transform(X_train, y_train)  # (n_train, k_best)
        csp_test = csp.transform(X_test)  # (n_test,  k_best)

        riemann_train = riemann.fit_transform(X_train, y_train)  # (n_train, n_riemann)
        riemann_test = riemann.transform(X_test)  # (n_test,  n_riemann)

        features_train = np.concatenate([csp_train, riemann_train], axis=1).astype(np.float32)
        features_test = np.concatenate([csp_test, riemann_test], axis=1).astype(np.float32)
        math_input_dim = features_train.shape[1]

        logger.info(
            "  Handcrafted features: CSP=%d + Riemannian=%d = %d total",
            csp_train.shape[1],
            riemann_train.shape[1],
            math_input_dim,
        )

        # --- Step 3: Generate CWT spectrogram images ---
        logger.info("  Generating CWT spectrograms for train set...")
        imgs_train = self._epochs_to_images(X_train)  # (n_train, C, H, W) float32

        logger.info("  Generating CWT spectrograms for test set...")
        imgs_test = self._epochs_to_images(X_test)  # (n_test, C, H, W) float32

        # --- Step 4: Pack into TensorDatasets ---
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

        For "multichannel" mode: images are (n_trials, n_ch, H, W) in [0,1].
        For other modes (mosaic / single): images are converted to (n_trials, 1, H, W).
        ImageNet normalisation is applied on top if configured.

        Args:
            X: EEG epochs (n_trials, n_channels, n_times).

        Returns:
            Float32 images: (n_trials, C, H, W).
        """
        images = self._transform.transform_epochs(X, self.channel_names, self.sfreq)

        if self.spec_config.channel_mode == "multichannel":
            # images: already (n_trials, n_ch, H, W) float32 in [0,1]
            pass
        else:
            # mosaic / single: convert to CHW with a single channel
            images = images[:, np.newaxis, ...].astype(np.float32) / 255.0

        # Apply ImageNet normalisation if requested
        if self.spec_config.apply_imagenet_norm:
            images = normalize_imagenet(images)

        return images
