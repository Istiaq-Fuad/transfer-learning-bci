"""EEG Preprocessing Pipeline.

Implements the standard MI-EEG preprocessing chain:
    1. Channel selection (motor cortex electrodes)
    2. Band-pass filtering (4-40 Hz for mu + beta rhythms)
    3. Notch filtering (50/60 Hz power line noise)
    4. ICA artifact removal (EOG, EMG, ECG)
    5. Epoching (segment into trials)
    6. Downsampling (standardize sampling rate)
    7. Normalization (z-score per channel per trial)

Functional API (operates on numpy arrays, no MNE dependency):
    notch_filter_epochs()       — remove power-line noise after epoching
    reject_bad_trials()         — discard trials with excessive amplitude
    euclidean_align()           — re-center covariance per subject
    crop_time_window()          — trim epoch to [tmin, tmax] in samples
    normalize_epochs()          — z-score per channel per trial
    apply_preprocessing_pipeline() — run all steps in order
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mne
import numpy as np
from scipy.signal import iirnotch, sosfiltfilt, tf2sos

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
        self,
        raw: mne.io.BaseRaw,
        channels: list[str],
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
                raw_for_ica,
                verbose=False,
            )
        except Exception:
            logger.warning("No EOG channel found; skipping EOG artifact detection")

        if eog_indices:
            ica.exclude = eog_indices
            logger.info(
                "Excluding %d EOG-related ICA components: %s", len(eog_indices), eog_indices
            )
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
            len(epochs),
            cfg.tmin,
            cfg.tmax,
            epochs.get_data().shape,
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


# ---------------------------------------------------------------------------
# Functional API — operates on (n_trials, n_channels, n_times) numpy arrays
# ---------------------------------------------------------------------------


def notch_filter_epochs(
    X: np.ndarray,
    sfreq: float,
    notch_freq: float = 50.0,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Apply a notch (band-stop) IIR filter to remove power-line noise.

    Applied per-trial using zero-phase forward-backward filtering.

    Args:
        X: EEG data of shape (n_trials, n_channels, n_times).
        sfreq: Sampling frequency in Hz.
        notch_freq: Frequency to remove (50 Hz EU / 60 Hz US).
        quality_factor: Q factor controlling notch width (higher = narrower).

    Returns:
        Filtered array, same shape as input.
    """
    b, a = iirnotch(w0=notch_freq, Q=quality_factor, fs=sfreq)
    # Convert to second-order sections for numerical stability
    sos = tf2sos(b, a)
    # Apply along time axis (axis=2) for each trial
    X_filtered = sosfiltfilt(sos, X, axis=2)
    logger.debug(
        "Notch filter applied at %.1f Hz (Q=%.1f) to %d trials",
        notch_freq,
        quality_factor,
        X.shape[0],
    )
    return X_filtered.astype(X.dtype)


def reject_bad_trials(
    X: np.ndarray,
    y: np.ndarray,
    threshold_uv: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discard trials whose peak-to-peak amplitude exceeds a threshold.

    The threshold is in microvolts (µV). MOABB epochs are already in volts,
    so the threshold is converted internally (100 µV = 100e-6 V).

    Args:
        X: EEG data of shape (n_trials, n_channels, n_times).
        y: Labels of shape (n_trials,).
        threshold_uv: Peak-to-peak amplitude threshold in µV.

    Returns:
        Tuple (X_clean, y_clean, kept_mask) where kept_mask is a boolean
        array of shape (n_trials,) indicating which trials were retained.
    """
    threshold_v = threshold_uv * 1e-6  # convert µV → V

    # Peak-to-peak per channel per trial
    ptp = X.max(axis=2) - X.min(axis=2)  # (n_trials, n_channels)
    max_ptp = ptp.max(axis=1)  # (n_trials,)
    kept_mask = max_ptp <= threshold_v

    n_rejected = int((~kept_mask).sum())
    if n_rejected > 0:
        logger.info(
            "Trial rejection: removed %d/%d trials (threshold=%.0f µV)",
            n_rejected,
            len(y),
            threshold_uv,
        )

    return X[kept_mask], y[kept_mask], kept_mask


def euclidean_align(X: np.ndarray) -> np.ndarray:
    """Re-centre the data using Euclidean Alignment (EA).

    EA divides each trial by the square root of the mean covariance matrix
    of the subject, so that the average covariance becomes the identity
    matrix. This reduces inter-session/inter-subject non-stationarity.

    Reference: He & Wu, "Transfer Learning for Brain-Computer Interfaces:
    A Euclidean Space Data Alignment Approach", IEEE TNSRE 2020.

    If the input is empty or the mean covariance is degenerate (NaN/Inf),
    the function returns X unchanged with a warning (graceful degradation).

    Args:
        X: EEG epochs of shape (n_trials, n_channels, n_times).

    Returns:
        Aligned epochs of shape (n_trials, n_channels, n_times).
    """
    n_trials, n_channels, n_times = X.shape

    # Guard: cannot align with zero trials
    if n_trials == 0:
        logger.warning("euclidean_align: no trials to align — returning X unchanged")
        return X

    # Guard: need at least 2 time points for a valid covariance estimate
    if n_times < 2:
        logger.warning("euclidean_align: too few time points (%d) — returning X unchanged", n_times)
        return X

    # Compute per-trial covariance matrices  (n_trials, n_ch, n_ch)
    covs = np.einsum("tci,tdi->tcd", X, X) / (n_times - 1)

    # Mean covariance across trials
    mean_cov = covs.mean(axis=0)  # (n_ch, n_ch)

    # Guard: degenerate mean covariance (all-NaN or Inf — e.g. from synthetic
    # data whose amplitudes are far outside the expected volt range)
    if not np.isfinite(mean_cov).all():
        logger.warning(
            "euclidean_align: mean covariance contains NaN/Inf "
            "(data may not be in volts) — skipping alignment"
        )
        return X

    # Matrix square root via eigendecomposition
    try:
        eigvals, eigvecs = np.linalg.eigh(mean_cov)
    except np.linalg.LinAlgError as exc:
        logger.warning(
            "euclidean_align: eigendecomposition failed (%s) — returning X unchanged", exc
        )
        return X

    eigvals = np.maximum(eigvals, 1e-10)  # clamp numerical negatives
    sqrt_inv = eigvecs @ np.diag(eigvals**-0.5) @ eigvecs.T  # (n_ch, n_ch)

    # Apply whitening: X_aligned[t] = sqrt_inv @ X[t]
    # einsum "cd,tds->tcs": matrix-multiply sqrt_inv (c,d) with X (t,d,s) → (t,c,s)
    X_aligned = np.einsum("cd,tds->tcs", sqrt_inv, X)

    logger.debug("Euclidean alignment applied to %d trials (%d channels)", n_trials, n_channels)
    return X_aligned.astype(X.dtype)


def crop_time_window(
    X: np.ndarray,
    sfreq: float,
    tmin: float = 0.5,
    tmax: float = 3.5,
    epoch_tmin: float = 0.0,
) -> np.ndarray:
    """Crop epochs to a fixed time window relative to cue onset.

    Args:
        X: EEG epochs of shape (n_trials, n_channels, n_times).
        sfreq: Sampling frequency in Hz.
        tmin: Start of desired window in seconds (relative to cue onset).
        tmax: End of desired window in seconds (relative to cue onset).
        epoch_tmin: Start time of the epoch in seconds (from MOABB, usually 0.0).

    Returns:
        Cropped array of shape (n_trials, n_channels, n_new_times).
    """
    start_sample = int(round((tmin - epoch_tmin) * sfreq))
    end_sample = int(round((tmax - epoch_tmin) * sfreq))
    start_sample = max(0, start_sample)
    end_sample = min(X.shape[2], end_sample)
    logger.debug(
        "Cropping time window [%.2f, %.2f]s → samples [%d, %d]",
        tmin,
        tmax,
        start_sample,
        end_sample,
    )
    return X[:, :, start_sample:end_sample]


def normalize_epochs(X: np.ndarray) -> np.ndarray:
    """Z-score normalize each channel within each trial independently.

    Args:
        X: EEG data of shape (n_trials, n_channels, n_times).

    Returns:
        Normalized array of the same shape.
    """
    mean = X.mean(axis=2, keepdims=True)  # (n_trials, n_ch, 1)
    std = X.std(axis=2, keepdims=True)  # (n_trials, n_ch, 1)
    std = np.where(std < 1e-8, 1e-8, std)
    X_norm = (X - mean) / std
    logger.debug("Z-score normalization applied: shape=%s", X.shape)
    return X_norm.astype(X.dtype)


def apply_preprocessing_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    config: PreprocessingConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the full preprocessing pipeline on a set of EEG epochs.

    Pipeline order:
        1. Notch filter      (if config.apply_notch)
        2. Trial rejection   (if config.apply_trial_rejection)
        3. Time-window crop  (if tmin_crop / tmax_crop set)
        4. Euclidean align   (if config.apply_euclidean_alignment)
        5. Z-score normalise (if config.apply_zscore)

    CSD is intentionally omitted here because it requires electrode positions
    (handled upstream with MNE if needed).

    Args:
        X: EEG epochs (n_trials, n_channels, n_times).
        y: Labels (n_trials,).
        sfreq: Sampling frequency in Hz.
        config: PreprocessingConfig. Uses defaults if None.

    Returns:
        (X_processed, y_processed) — y may be shorter than input after rejection.
    """
    cfg = config or PreprocessingConfig()

    if cfg.apply_notch:
        X = notch_filter_epochs(X, sfreq, notch_freq=cfg.notch_freq)

    if cfg.apply_trial_rejection:
        X, y, _ = reject_bad_trials(X, y, threshold_uv=cfg.rejection_threshold_uv)
        if len(y) == 0:
            logger.warning(
                "apply_preprocessing_pipeline: all trials rejected — "
                "returning empty arrays (check data units; threshold=%.0f µV)",
                cfg.rejection_threshold_uv,
            )
            return X, y

    # Crop time window: only if the window differs from the epoch boundaries
    if cfg.tmin_crop != cfg.tmin or cfg.tmax_crop != cfg.tmax:
        X = crop_time_window(X, sfreq, tmin=cfg.tmin_crop, tmax=cfg.tmax_crop, epoch_tmin=cfg.tmin)

    if cfg.apply_euclidean_alignment:
        X = euclidean_align(X)

    if cfg.apply_zscore:
        X = normalize_epochs(X)

    logger.info(
        "Preprocessing pipeline done: %d trials remaining (shape=%s)",
        len(y),
        X.shape,
    )
    return X, y


def main() -> None:
    """CLI entry point for batch preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess MI-EEG data")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save preprocessed data",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = load_config(args.config)
    pipeline = PreprocessingPipeline(config.preprocessing)

    logger.info("Preprocessing pipeline initialized with config: %s", config.preprocessing)
    logger.info("Use pipeline.process_raw() or pipeline.process_to_array() to preprocess data.")


if __name__ == "__main__":
    main()
