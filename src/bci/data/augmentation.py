"""EEG and spectrogram data augmentation.

Two augmenters are provided:

EEGAugmenter
    Operates on raw EEG epochs (n_trials, n_channels, n_times) before CWT.
    Techniques: Gaussian noise injection, temporal cropping + zero-pad,
    channel dropout, amplitude scaling.

SpectrogramAugmenter
    Operates on CWT magnitude spectrograms (n_trials, n_freqs, n_times)
    after CWT but before compositing into an image.
    Implements SpecAugment-style frequency and time masking.

Both augmenters are stateless callables. Pass ``training=True`` to enable
augmentation; at test time pass ``training=False`` to get the unmodified data.

Usage::

    from bci.data.augmentation import EEGAugmenter, SpectrogramAugmenter
    from bci.utils.config import AugmentationConfig

    cfg = AugmentationConfig()
    eeg_aug = EEGAugmenter(cfg, seed=42)
    spec_aug = SpectrogramAugmenter(cfg, seed=42)

    X_aug = eeg_aug(X_train, training=True)
    spec_aug_result = spec_aug(spectrogram, training=True)
"""

from __future__ import annotations

import logging

import numpy as np

from bci.utils.config import AugmentationConfig

logger = logging.getLogger(__name__)


class EEGAugmenter:
    """Augmentations applied to EEG epochs before CWT spectrogram generation.

    All augmentations are applied independently per trial.

    Args:
        config: AugmentationConfig with enable flags and hyperparameters.
        seed: Base random seed. Each call draws fresh samples.
    """

    def __init__(
        self,
        config: AugmentationConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.config = config or AugmentationConfig()
        self._rng = np.random.default_rng(seed)

    def __call__(
        self,
        X: np.ndarray,
        training: bool = True,
    ) -> np.ndarray:
        """Apply augmentation pipeline to a batch of EEG trials.

        Args:
            X: EEG epochs of shape (n_trials, n_channels, n_times).
            training: If False, return X unchanged (inference mode).

        Returns:
            Augmented array, same shape as input.
        """
        if not training:
            return X

        X = X.copy()
        cfg = self.config

        if cfg.apply_amplitude_scale:
            X = self._amplitude_scale(X)

        if cfg.apply_gaussian_noise:
            X = self._gaussian_noise(X)

        if cfg.apply_channel_dropout:
            X = self._channel_dropout(X)

        if cfg.apply_temporal_crop:
            X = self._temporal_crop(X)

        return X

    # ------------------------------------------------------------------
    # Individual augmentation methods
    # ------------------------------------------------------------------

    def _gaussian_noise(self, X: np.ndarray) -> np.ndarray:
        """Add Gaussian noise scaled to per-channel std of each trial."""
        cfg = self.config
        # Per-trial, per-channel std: shape (n_trials, n_channels, 1)
        std = X.std(axis=2, keepdims=True)
        noise = self._rng.standard_normal(X.shape).astype(X.dtype)
        X_noisy = X + noise * std * cfg.gaussian_noise_std
        return X_noisy

    def _temporal_crop(self, X: np.ndarray) -> np.ndarray:
        """Randomly zero-pad both ends of each trial (simulate crop + re-pad).

        Zeroes up to ``temporal_crop_ratio`` of the time-points from the
        start and end of each trial independently.
        """
        cfg = self.config
        n_times = X.shape[2]
        max_drop = int(n_times * cfg.temporal_crop_ratio)
        if max_drop < 1:
            return X

        X_out = X.copy()
        for i in range(X.shape[0]):
            drop_start = int(self._rng.integers(0, max_drop + 1))
            drop_end = int(self._rng.integers(0, max_drop + 1))
            if drop_start > 0:
                X_out[i, :, :drop_start] = 0.0
            if drop_end > 0:
                X_out[i, :, n_times - drop_end :] = 0.0
        return X_out

    def _channel_dropout(self, X: np.ndarray) -> np.ndarray:
        """Zero out randomly selected channels per trial."""
        cfg = self.config
        n_channels = X.shape[1]
        X_out = X.copy()
        for i in range(X.shape[0]):
            mask = self._rng.random(n_channels) < cfg.channel_dropout_prob
            X_out[i, mask, :] = 0.0
        return X_out

    def _amplitude_scale(self, X: np.ndarray) -> np.ndarray:
        """Randomly scale the amplitude of each trial."""
        cfg = self.config
        low, high = cfg.amplitude_scale_range
        scales = self._rng.uniform(low, high, size=(X.shape[0], 1, 1)).astype(X.dtype)
        return X * scales


class SpectrogramAugmenter:
    """SpecAugment-style augmentation on CWT magnitude spectrograms.

    Applied after CWT computation but before compositing into an image.

    Supported operations:
        - Frequency masking: zero a contiguous band of frequency rows.
        - Time masking: zero a contiguous block of time columns.

    Args:
        config: AugmentationConfig controlling mask widths.
        seed: Base random seed.
    """

    def __init__(
        self,
        config: AugmentationConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.config = config or AugmentationConfig()
        self._rng = np.random.default_rng(seed)

    def __call__(
        self,
        spec: np.ndarray,
        training: bool = True,
    ) -> np.ndarray:
        """Apply SpecAugment masking to a single spectrogram.

        Args:
            spec: 2D spectrogram of shape (n_freqs, n_times).
            training: If False, return spec unchanged.

        Returns:
            Augmented spectrogram, same shape.
        """
        if not training:
            return spec

        spec = spec.copy()
        cfg = self.config

        if cfg.apply_freq_mask:
            spec = self._freq_mask(spec)

        if cfg.apply_time_mask:
            spec = self._time_mask(spec)

        return spec

    def apply_batch(
        self,
        specs: np.ndarray,
        training: bool = True,
    ) -> np.ndarray:
        """Apply SpecAugment to a batch of spectrograms.

        Args:
            specs: Array of shape (n_trials, n_freqs, n_times) or
                   (n_trials, n_channels, n_freqs, n_times).
            training: If False, return specs unchanged.

        Returns:
            Augmented array, same shape.
        """
        if not training:
            return specs

        specs = specs.copy()
        if specs.ndim == 3:
            for i in range(specs.shape[0]):
                specs[i] = self(specs[i], training=True)
        elif specs.ndim == 4:
            for i in range(specs.shape[0]):
                for c in range(specs.shape[1]):
                    specs[i, c] = self(specs[i, c], training=True)
        else:
            raise ValueError(f"specs must be 3D or 4D, got shape {specs.shape}")
        return specs

    # ------------------------------------------------------------------

    def _freq_mask(self, spec: np.ndarray) -> np.ndarray:
        """Zero a random contiguous band of frequency rows."""
        n_freqs = spec.shape[0]
        max_w = min(self.config.freq_mask_max_width, n_freqs)
        if max_w < 1:
            return spec
        width = int(self._rng.integers(1, max_w + 1))
        start = int(self._rng.integers(0, n_freqs - width + 1))
        spec[start : start + width, :] = 0.0
        return spec

    def _time_mask(self, spec: np.ndarray) -> np.ndarray:
        """Zero a random contiguous block of time columns."""
        n_times = spec.shape[1]
        max_w = min(self.config.time_mask_max_width, n_times)
        if max_w < 1:
            return spec
        width = int(self._rng.integers(1, max_w + 1))
        start = int(self._rng.integers(0, n_times - width + 1))
        spec[:, start : start + width] = 0.0
        return spec
