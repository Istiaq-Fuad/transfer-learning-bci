"""CWT Spectrogram Transformation for EEG signals.

Converts 1D EEG time-series into 2D time-frequency images using
Continuous Wavelet Transform (CWT) with Morlet wavelet.

These images serve as input to the Vision Transformer branch.

Supported channel composition modes:
    - "rgb_c3_cz_c4": Map C3→Red, Cz→Green, C4→Blue (3-channel image)
    - "mosaic": Tile spectrograms from multiple channels into a grid
    - "single": Generate one spectrogram per channel (grayscale)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pywt
from PIL import Image

from bci.utils.config import SpectrogramConfig

logger = logging.getLogger(__name__)


class CWTSpectrogramTransform:
    """Generate CWT spectrogram images from EEG epoch data.

    Usage:
        transform = CWTSpectrogramTransform(config)
        images = transform.transform_epochs(X, channel_names)
        # images: ndarray of shape (n_trials, H, W, 3) or (n_trials, H, W, 1)
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
            len(scales), cfg.freq_min, cfg.freq_max, sfreq,
        )
        return scales

    def cwt_single_channel(
        self, signal: np.ndarray, sfreq: float,
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
        """Normalize spectrogram to [0, 255] uint8 range.

        Args:
            spec: 2D spectrogram array.

        Returns:
            Normalized uint8 array.
        """
        # Min-max normalization
        spec_min = spec.min()
        spec_max = spec.max()
        if spec_max - spec_min < 1e-10:
            return np.zeros_like(spec, dtype=np.uint8)

        normalized = (spec - spec_min) / (spec_max - spec_min)
        return (normalized * 255).astype(np.uint8)

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

    def transform_trial_rgb(
        self,
        trial: np.ndarray,
        channel_names: list[str],
        sfreq: float,
    ) -> np.ndarray:
        """Transform a single trial into an RGB spectrogram image.

        Maps C3 -> Red, Cz -> Green, C4 -> Blue.

        Args:
            trial: 2D array of shape (n_channels, n_times).
            channel_names: List of channel names corresponding to trial rows.
            sfreq: Sampling frequency.

        Returns:
            RGB image array of shape (H, W, 3) as uint8.
        """
        rgb_channels = ["C3", "Cz", "C4"]
        rgb_image = np.zeros((*self.config.image_size, 3), dtype=np.uint8)

        for i, ch_name in enumerate(rgb_channels):
            if ch_name in channel_names:
                ch_idx = channel_names.index(ch_name)
                spec = self.cwt_single_channel(trial[ch_idx], sfreq)
                spec_norm = self._normalize_spectrogram(spec)
                rgb_image[:, :, i] = self._resize_image(spec_norm)
            else:
                logger.warning(
                    "Channel %s not found, using zeros for %s channel",
                    ch_name, ["Red", "Green", "Blue"][i],
                )

        return rgb_image

    def transform_trial_mosaic(
        self,
        trial: np.ndarray,
        channel_names: list[str],
        sfreq: float,
        max_channels: int = 9,
    ) -> np.ndarray:
        """Transform a single trial into a mosaic of channel spectrograms.

        Tiles spectrograms from multiple channels into a grid image.

        Args:
            trial: 2D array of shape (n_channels, n_times).
            channel_names: List of channel names.
            sfreq: Sampling frequency.
            max_channels: Maximum number of channels to include in the mosaic.

        Returns:
            Grayscale mosaic image array of shape (H, W) as uint8.
        """
        n_channels = min(len(channel_names), max_channels)
        grid_size = int(np.ceil(np.sqrt(n_channels)))

        target_h, target_w = self.config.image_size
        cell_h = target_h // grid_size
        cell_w = target_w // grid_size

        mosaic = np.zeros((target_h, target_w), dtype=np.uint8)

        for idx in range(n_channels):
            row = idx // grid_size
            col = idx % grid_size

            spec = self.cwt_single_channel(trial[idx], sfreq)
            spec_norm = self._normalize_spectrogram(spec)

            # Resize to cell size
            cell_img = Image.fromarray(spec_norm)
            cell_img = cell_img.resize((cell_w, cell_h), Image.Resampling.BILINEAR)

            y_start = row * cell_h
            x_start = col * cell_w
            mosaic[y_start:y_start + cell_h, x_start:x_start + cell_w] = np.array(cell_img)

        return mosaic

    def transform_epochs(
        self,
        X: np.ndarray,
        channel_names: list[str],
        sfreq: float,
    ) -> np.ndarray:
        """Transform all epochs into spectrogram images.

        Args:
            X: Epoch data of shape (n_trials, n_channels, n_times).
            channel_names: List of channel names.
            sfreq: Sampling frequency.

        Returns:
            Image array:
                - "rgb_c3_cz_c4": shape (n_trials, H, W, 3)
                - "mosaic": shape (n_trials, H, W)
                - "single": shape (n_trials * n_channels, H, W)
        """
        cfg = self.config
        n_trials = X.shape[0]

        logger.info(
            "Generating %s spectrograms for %d trials...",
            cfg.channel_mode, n_trials,
        )

        if cfg.channel_mode == "rgb_c3_cz_c4":
            images = np.zeros(
                (n_trials, *cfg.image_size, 3), dtype=np.uint8,
            )
            for i in range(n_trials):
                images[i] = self.transform_trial_rgb(X[i], channel_names, sfreq)

        elif cfg.channel_mode == "mosaic":
            images = np.zeros(
                (n_trials, *cfg.image_size), dtype=np.uint8,
            )
            for i in range(n_trials):
                images[i] = self.transform_trial_mosaic(
                    X[i], channel_names, sfreq,
                )

        elif cfg.channel_mode == "single":
            # One spectrogram per channel per trial
            n_channels = X.shape[1]
            images = np.zeros(
                (n_trials * n_channels, *cfg.image_size), dtype=np.uint8,
            )
            for i in range(n_trials):
                for j in range(n_channels):
                    spec = self.cwt_single_channel(X[i, j], sfreq)
                    spec_norm = self._normalize_spectrogram(spec)
                    images[i * n_channels + j] = self._resize_image(spec_norm)
        else:
            raise ValueError(f"Unknown channel_mode: {cfg.channel_mode}")

        logger.info("Generated spectrogram images: shape=%s", images.shape)
        return images

    def save_spectrograms(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        output_dir: str | Path | None = None,
        prefix: str = "trial",
    ) -> list[Path]:
        """Save spectrogram images to disk as PNG files.

        Args:
            images: Image array from transform_epochs().
            labels: Label array for naming files.
            output_dir: Directory to save images. Uses config default if None.
            prefix: Filename prefix.

        Returns:
            List of saved file paths.
        """
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for i in range(images.shape[0]):
            label = labels[i] if i < len(labels) else "unknown"
            filename = f"{prefix}_{i:04d}_label_{label}.png"
            filepath = output_dir / filename

            img = Image.fromarray(images[i])
            img.save(filepath)
            saved_paths.append(filepath)

        logger.info("Saved %d spectrograms to %s", len(saved_paths), output_dir)
        return saved_paths
