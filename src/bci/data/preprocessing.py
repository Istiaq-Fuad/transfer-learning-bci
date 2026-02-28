"""EEG Preprocessing Pipeline.

Functional API (operates on numpy arrays, no MNE dependency):
    notch_filter_epochs()       — remove power-line noise after epoching
    reject_bad_trials()         — discard trials with excessive amplitude
    euclidean_align()           — re-center covariance per subject
    crop_time_window()          — trim epoch to [tmin, tmax] in samples
    normalize_epochs()          — z-score per channel per trial
    apply_preprocessing_pipeline() — run all steps in order
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import iirnotch, sosfiltfilt, tf2sos

from bci.utils.config import PreprocessingConfig

logger = logging.getLogger(__name__)


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
