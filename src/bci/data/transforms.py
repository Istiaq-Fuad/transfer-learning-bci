"""CWT Spectrogram Transformation for EEG signals.

Converts 1D EEG time-series into 2D time-frequency images using
Continuous Wavelet Transform (CWT) with Morlet wavelet.

These images serve as input to the Vision Transformer branch.

Channel composition mode:
    - "multichannel":  Produce one image channel per selected channel (N-channel)
                       Default: 9 motor-cortex channels → (9, H, W) float32 tensor

Normalisation modes (SpectrogramConfig.normalize_mode):
    - "per_channel":   Normalise each channel spectrogram independently to [0,1]
    - "joint":         Normalise across all selected channels jointly, preserving
                       cross-channel power ratios (laterality of motor cortex)

ImageNet normalisation (SpectrogramConfig.apply_imagenet_norm):
    After scaling to [0,1], subtract ImageNet mean and divide by std so that
    the pretrained ViT receives inputs in the expected distribution.
    For N-channel (N≠3) inputs, the first three channels use RGB stats and
    remaining channels use the green-channel stats (0.456 / 0.224).
"""

from __future__ import annotations

import logging

import numpy as np
import pywt
from PIL import Image

from bci.utils.config import SpectrogramConfig

logger = logging.getLogger(__name__)

# ImageNet normalisation constants (RGB order)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_imagenet(
    images: np.ndarray,
) -> np.ndarray:
    """Apply ImageNet mean/std normalisation to a batch of float32 images.

    Assumes images are already in [0, 1] range.

    Args:
        images: Float32 array of shape (n_trials, C, H, W) already in [0,1].

    Returns:
        Normalised float32 array of the same shape.
    """
    n_channels = images.shape[1]

    # Build per-channel mean/std arrays, extending beyond 3 channels if needed
    mean = np.empty(n_channels, dtype=np.float32)
    std = np.empty(n_channels, dtype=np.float32)
    for c in range(n_channels):
        if c < 3:
            mean[c] = _IMAGENET_MEAN[c]
            std[c] = _IMAGENET_STD[c]
        else:
            # Use green-channel stats for extra channels
            mean[c] = _IMAGENET_MEAN[1]
            std[c] = _IMAGENET_STD[1]

    # Broadcast: (C,) -> (1, C, 1, 1)
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]

    return (images - mean) / std


class CWTSpectrogramTransform:
    """Generate CWT spectrogram images from EEG epoch data.

    Usage:
        transform = CWTSpectrogramTransform(config)
        images = transform.transform_epochs(X, channel_names, sfreq)
        # images: ndarray of shape (n_trials, n_selected, H, W) float32
    """

    def __init__(self, config: SpectrogramConfig | None = None) -> None:
        self.config = config or SpectrogramConfig()

        # Pre-compute the CWT scales from frequency range
        self._scales: np.ndarray | None = None
        self._sampling_period: float | None = None

    def _compute_scales(self, sfreq: float) -> np.ndarray:
        """Compute CWT scales corresponding to the desired frequency range.

        The relationship between scale and frequency for Morlet wavelet:
            frequency = center_frequency / (scale * sampling_period)

        Args:
            sfreq: Sampling frequency of the EEG data.

        Returns:
            Array of scales sorted from high frequency to low frequency.
        """
        cfg = self.config

        # Get the center frequency of the wavelet
        center_freq = pywt.central_frequency(cfg.wavelet)

        # Create linearly spaced frequencies
        freqs = np.linspace(cfg.freq_min, cfg.freq_max, cfg.n_freqs)

        # Convert frequencies to scales
        sampling_period = 1.0 / sfreq
        scales = center_freq / (freqs * sampling_period)

        # Sort scales from high to low frequency (low to high scale)
        # so the spectrogram has low frequencies at bottom, high at top
        scales = np.sort(scales)[::-1]

        self._scales = scales
        self._sampling_period = sampling_period

        logger.info(
            "CWT scales: %d scales for %.1f-%.1f Hz (sfreq=%.1f Hz)",
            len(scales),
            cfg.freq_min,
            cfg.freq_max,
            sfreq,
        )
        return scales

    def cwt_single_channel(
        self,
        signal: np.ndarray,
        sfreq: float,
    ) -> np.ndarray:
        """Compute CWT spectrogram for a single channel.

        Args:
            signal: 1D array of shape (n_times,).
            sfreq: Sampling frequency.

        Returns:
            2D magnitude spectrogram of shape (n_freqs, n_times).
        """
        if self._scales is None or self._sampling_period != 1.0 / sfreq:
            self._compute_scales(sfreq)

        coefficients, _frequencies = pywt.cwt(
            signal,
            self._scales,
            self.config.wavelet,
            sampling_period=1.0 / sfreq,
        )

        # Take magnitude (complex -> real)
        spectrogram = np.abs(coefficients)

        return spectrogram

    def _normalize_spectrogram(self, spec: np.ndarray) -> np.ndarray:
        """Normalize a single spectrogram to [0, 255] uint8 range.

        Args:
            spec: 2D spectrogram array (n_freqs, n_times).

        Returns:
            Normalized uint8 array.
        """
        spec_min = spec.min()
        spec_max = spec.max()
        if spec_max - spec_min < 1e-10:
            return np.zeros_like(spec, dtype=np.uint8)
        normalized = (spec - spec_min) / (spec_max - spec_min)
        return (normalized * 255).astype(np.uint8)

    def _normalize_joint(self, specs: list[np.ndarray]) -> list[np.ndarray]:
        """Normalize multiple spectrograms jointly to [0, 255] uint8.

        All spectrograms share a single min/max computed across all of them,
        preserving relative power ratios between channels (e.g. C3 vs C4
        laterality for motor imagery).

        Args:
            specs: List of 2D float arrays (n_freqs, n_times).

        Returns:
            List of uint8 arrays, same order and shapes.
        """
        all_vals = np.concatenate([s.ravel() for s in specs])
        g_min = all_vals.min()
        g_max = all_vals.max()
        if g_max - g_min < 1e-10:
            return [np.zeros_like(s, dtype=np.uint8) for s in specs]
        result = []
        for s in specs:
            normalized = (s - g_min) / (g_max - g_min)
            result.append((normalized * 255).astype(np.uint8))
        return result

    def _resize_image(self, img_array: np.ndarray) -> np.ndarray:
        """Resize a spectrogram image to the target size.

        Args:
            img_array: Image array of shape (H, W) or (H, W, C).

        Returns:
            Resized image array.
        """
        target_h, target_w = self.config.image_size
        img = Image.fromarray(img_array)
        img = img.resize((target_w, target_h), Image.Resampling.BILINEAR)
        return np.array(img)

    def transform_trial_multichannel(
        self,
        trial: np.ndarray,
        channel_names: list[str],
        sfreq: float,
    ) -> np.ndarray:
        """Transform a single trial into an N-channel spectrogram image.

        Each selected channel produces one image channel (grayscale plane).
        Channels are selected via ``config.spectrogram_channels``.
        Normalisation mode follows ``config.normalize_mode``.

        Args:
            trial: 2D array of shape (n_channels, n_times).
            channel_names: List of channel names corresponding to trial rows.
            sfreq: Sampling frequency.

        Returns:
            Float32 image array of shape (n_selected, H, W) in [0, 1].
        """
        selected = self.config.spectrogram_channels
        n_selected = len(selected)
        H, W = self.config.image_size
        result = np.zeros((n_selected, H, W), dtype=np.float32)

        raw_specs: list[np.ndarray | None] = []
        for ch_name in selected:
            if ch_name in channel_names:
                ch_idx = channel_names.index(ch_name)
                raw_specs.append(self.cwt_single_channel(trial[ch_idx], sfreq))
            else:
                logger.warning(
                    "Multichannel: channel %s not found in %s; using zeros",
                    ch_name,
                    channel_names,
                )
                raw_specs.append(None)

        # Normalise
        if self.config.normalize_mode == "joint":
            valid = [s for s in raw_specs if s is not None]
            if valid:
                normed = self._normalize_joint(valid)
                norm_iter = iter(normed)
                norm_specs = [next(norm_iter) if s is not None else None for s in raw_specs]
            else:
                norm_specs = raw_specs
        else:
            norm_specs = [
                self._normalize_spectrogram(s) if s is not None else None for s in raw_specs
            ]

        for i, ns in enumerate(norm_specs):
            if ns is not None:
                resized = self._resize_image(ns)  # H×W uint8
                result[i] = resized.astype(np.float32) / 255.0

        return result  # (n_selected, H, W) in [0,1]

    def transform_epochs(
        self,
        X: np.ndarray,
        channel_names: list[str],
        sfreq: float,
    ) -> np.ndarray:
        """Transform all epochs into multichannel spectrogram images.

        Args:
            X: Epoch data of shape (n_trials, n_channels, n_times).
            channel_names: List of channel names.
            sfreq: Sampling frequency.

        Returns:
            Float32 array of shape (n_trials, n_selected, H, W) in [0, 1].
        """
        cfg = self.config
        n_trials = X.shape[0]

        if cfg.channel_mode != "multichannel":
            raise ValueError(
                f"Unsupported channel_mode '{cfg.channel_mode}'. Only 'multichannel' is supported."
            )

        logger.info(
            "Generating multichannel spectrograms for %d trials...",
            n_trials,
        )

        n_selected = len(cfg.spectrogram_channels)
        H, W = cfg.image_size
        images = np.zeros((n_trials, n_selected, H, W), dtype=np.float32)
        for i in range(n_trials):
            images[i] = self.transform_trial_multichannel(X[i], channel_names, sfreq)

        logger.info("Generated spectrogram images: shape=%s", images.shape)
        return images
