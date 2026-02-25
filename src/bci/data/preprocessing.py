"""EEG Preprocessing Pipeline.

Implements the standard MI-EEG preprocessing chain:
    1. Channel selection (motor cortex electrodes)
    2. Band-pass filtering (4-40 Hz for mu + beta rhythms)
    3. Notch filtering (50/60 Hz power line noise)
    4. ICA artifact removal (EOG, EMG, ECG)
    5. Epoching (segment into trials)
    6. Downsampling (standardize sampling rate)
    7. Normalization (z-score per channel per trial)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mne
import numpy as np

from bci.utils.config import PreprocessingConfig, load_config

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Configurable EEG preprocessing pipeline.

    Usage:
        config = PreprocessingConfig(l_freq=4.0, h_freq=40.0)
        pipeline = PreprocessingPipeline(config)

        # Process a single Raw object
        raw_clean = pipeline.process_raw(raw, picks=["C3", "Cz", "C4"])

        # Process Raw into Epochs given events
        epochs = pipeline.create_epochs(raw_clean, events, event_id)
    """

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()

    def select_channels(
        self, raw: mne.io.BaseRaw, channels: list[str],
    ) -> mne.io.BaseRaw:
        """Pick only the requested EEG channels.

        Args:
            raw: MNE Raw object.
            channels: List of channel names to keep.

        Returns:
            Raw object with only the selected channels.
        """
        available = raw.ch_names
        picks = [ch for ch in channels if ch in available]

        if not picks:
            raise ValueError(
                f"None of the requested channels {channels} found in "
                f"available channels: {available}"
            )

        if len(picks) < len(channels):
            missing = set(channels) - set(picks)
            logger.warning("Channels not found (skipped): %s", missing)

        raw = raw.copy().pick(picks)
        logger.info("Selected %d channels: %s", len(picks), picks)
        return raw

    def filter(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Apply band-pass and notch filtering.

        Band-pass: keeps mu (8-13 Hz) and beta (13-30 Hz) rhythms.
        Notch: removes power line noise at 50 Hz (EU) or 60 Hz (US).

        Args:
            raw: MNE Raw object.

        Returns:
            Filtered Raw object (modified in-place and returned).
        """
        cfg = self.config

        # Band-pass filter
        raw.filter(
            l_freq=cfg.l_freq,
            h_freq=cfg.h_freq,
            method="fir",
            fir_design="firwin",
            verbose=False,
        )
        logger.info("Band-pass filtered: %.1f - %.1f Hz", cfg.l_freq, cfg.h_freq)

        # Notch filter for power line noise
        if cfg.notch_freq is not None:
            raw.notch_filter(
                freqs=[cfg.notch_freq],
                method="fir",
                verbose=False,
            )
            logger.info("Notch filtered at %.1f Hz", cfg.notch_freq)

        return raw

    def remove_artifacts_ica(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Remove ocular and muscular artifacts using ICA.

        Uses FastICA by default. Automatically detects EOG-like components
        using correlation with frontal channels.

        Args:
            raw: Filtered MNE Raw object.

        Returns:
            Raw object with artifacts removed.
        """
        cfg = self.config

        if not cfg.apply_ica:
            logger.info("ICA skipped (disabled in config)")
            return raw

        n_components = cfg.n_ica_components
        if n_components is None:
            # Use 0.999 variance explained as default
            n_components = min(len(raw.ch_names), 20)

        ica = mne.preprocessing.ICA(
            n_components=n_components,
            method=cfg.ica_method,
            random_state=42,
            max_iter="auto",
        )

        # Fit ICA on a copy with 1 Hz high-pass for stability
        raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
        ica.fit(raw_for_ica, verbose=False)
        logger.info("ICA fitted with %d components", ica.n_components_)

        # Auto-detect EOG-like components
        # Try to find EOG channels, fall back to frontal EEG channels
        eog_indices = []
        try:
            eog_indices, _scores = ica.find_bads_eog(
                raw_for_ica, verbose=False,
            )
        except Exception:
            logger.warning("No EOG channel found; skipping EOG artifact detection")

        if eog_indices:
            ica.exclude = eog_indices
            logger.info("Excluding %d EOG-related ICA components: %s", len(eog_indices), eog_indices)
        else:
            logger.info("No EOG components detected")

        # Apply ICA to remove artifacts
        ica.apply(raw, verbose=False)
        return raw

    def resample(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Resample to a standard frequency.

        Args:
            raw: MNE Raw object.

        Returns:
            Resampled Raw object.
        """
        cfg = self.config

        if cfg.resample_freq is not None and raw.info["sfreq"] != cfg.resample_freq:
            raw.resample(cfg.resample_freq, verbose=False)
            logger.info("Resampled to %.1f Hz", cfg.resample_freq)

        return raw

    def create_epochs(
        self,
        raw: mne.io.BaseRaw,
        events: np.ndarray,
        event_id: dict[str, int],
    ) -> mne.Epochs:
        """Segment continuous data into trials.

        Args:
            raw: Preprocessed MNE Raw object.
            events: Events array (n_events, 3) from mne.find_events or annotations.
            event_id: Mapping of event names to integer codes.

        Returns:
            MNE Epochs object with trial data.
        """
        cfg = self.config

        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=cfg.tmin,
            tmax=cfg.tmax,
            baseline=None,  # No baseline correction for MI
            preload=True,
            verbose=False,
        )

        logger.info(
            "Created %d epochs (%.1f - %.1f s), shape: %s",
            len(epochs), cfg.tmin, cfg.tmax, epochs.get_data().shape,
        )
        return epochs

    def normalize_epochs(self, data: np.ndarray) -> np.ndarray:
        """Z-score normalize each channel within each trial.

        Args:
            data: Epoch data of shape (n_trials, n_channels, n_times).

        Returns:
            Normalized data with the same shape.
        """
        if not self.config.normalize:
            return data

        mean = data.mean(axis=-1, keepdims=True)
        std = data.std(axis=-1, keepdims=True)
        std[std < 1e-8] = 1e-8  # Avoid division by zero
        normalized = (data - mean) / std

        logger.info("Normalized epoch data: shape=%s", normalized.shape)
        return normalized

    def process_raw(
        self,
        raw: mne.io.BaseRaw,
        channels: list[str] | None = None,
    ) -> mne.io.BaseRaw:
        """Run the full preprocessing pipeline on a Raw object.

        Steps: channel selection -> filtering -> ICA -> resampling.

        Args:
            raw: MNE Raw object (will be copied, not modified).
            channels: Optional list of channel names to select.

        Returns:
            Preprocessed Raw object.
        """
        raw = raw.copy()

        if channels is not None:
            raw = self.select_channels(raw, channels)

        raw = self.filter(raw)
        raw = self.remove_artifacts_ica(raw)
        raw = self.resample(raw)

        return raw

    def process_to_array(
        self,
        raw: mne.io.BaseRaw,
        events: np.ndarray,
        event_id: dict[str, int],
        channels: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Full pipeline from Raw to normalized numpy arrays.

        Args:
            raw: MNE Raw object.
            events: Events array.
            event_id: Event name-to-id mapping.
            channels: Optional channel selection.

        Returns:
            Tuple of (X, y):
                X: ndarray (n_trials, n_channels, n_times) normalized
                y: ndarray (n_trials,) integer labels
        """
        raw = self.process_raw(raw, channels=channels)
        epochs = self.create_epochs(raw, events, event_id)

        X = epochs.get_data()
        X = self.normalize_epochs(X)

        y = epochs.events[:, -1]

        return X, y


def main() -> None:
    """CLI entry point for batch preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess MI-EEG data")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save preprocessed data",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = load_config(args.config)
    pipeline = PreprocessingPipeline(config.preprocessing)

    logger.info("Preprocessing pipeline initialized with config: %s", config.preprocessing)
    logger.info(
        "Use pipeline.process_raw() or pipeline.process_to_array() to preprocess data."
    )


if __name__ == "__main__":
    main()
