"""Filter Bank Common Spatial Patterns (FBCSP) feature extraction.

FBCSP decomposes EEG into multiple frequency sub-bands, applies CSP to each
sub-band independently, and then selects the most discriminative features
across all bands using mutual information.

This addresses a key limitation of standard CSP: the reliance on a single
broadband signal. Motor imagery signals have subject-specific frequency
characteristics (mu: ~8-13 Hz, beta: ~13-30 Hz), and FBCSP automatically
identifies which frequency bands are most informative for each subject.

References:
    - Ang et al., "Filter Bank Common Spatial Pattern (FBCSP) in
      Brain-Computer Interface" (IJCNN 2008)
    - Ang et al., "Filter Bank Common Spatial Pattern Algorithm on BCI
      Competition IV Datasets 2a and 2b" (Front. Neurosci., 2012)
"""

from __future__ import annotations

import logging

import numpy as np
from mne.decoding import CSP
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif

logger = logging.getLogger(__name__)


def _design_filter_bank(
    freq_min: float = 4.0,
    freq_max: float = 40.0,
    band_width: float = 4.0,
    overlap: float = 0.0,
    sfreq: float = 128.0,
    order: int = 5,
) -> list[tuple[float, float, np.ndarray]]:
    """Design a bank of Butterworth bandpass filters.

    Args:
        freq_min: Lowest frequency of the first band (Hz).
        freq_max: Highest frequency of the last band (Hz).
        band_width: Width of each sub-band (Hz).
        overlap: Overlap between adjacent bands (Hz).
        sfreq: Sampling frequency (Hz).
        order: Butterworth filter order.

    Returns:
        List of (low_freq, high_freq, sos_coefficients) tuples.
    """
    nyquist = sfreq / 2.0
    filters = []
    low = freq_min
    step = band_width - overlap

    while low + band_width <= freq_max + 1e-6:
        high = min(low + band_width, freq_max)
        # Ensure we don't exceed Nyquist
        if high >= nyquist:
            high = nyquist - 1.0
        if low >= high:
            break

        sos = butter(order, [low / nyquist, high / nyquist], btype="band", output="sos")
        filters.append((low, high, sos))
        low += step

    return filters


def _apply_filter_bank(
    X: np.ndarray,
    filter_bank: list[tuple[float, float, np.ndarray]],
) -> list[np.ndarray]:
    """Apply each filter in the bank to the EEG data.

    Args:
        X: EEG data of shape (n_trials, n_channels, n_times).
        filter_bank: List of (low, high, sos) from _design_filter_bank.

    Returns:
        List of filtered data arrays, each of shape (n_trials, n_channels, n_times).
    """
    filtered = []
    for _low, _high, sos in filter_bank:
        # sosfiltfilt applies zero-phase filtering along the last axis
        X_filt = sosfiltfilt(sos, X, axis=-1).astype(np.float32)
        filtered.append(X_filt)
    return filtered


class FBCSPFeatureExtractor(BaseEstimator, TransformerMixin):
    """Filter Bank CSP (FBCSP) feature extractor.

    Decomposes EEG into frequency sub-bands, applies CSP per band,
    concatenates all CSP features, and optionally selects the top-k
    features by mutual information with the labels.

    Args:
        n_components: Number of CSP components per sub-band.
        freq_min: Lowest frequency of the filter bank (Hz).
        freq_max: Highest frequency of the filter bank (Hz).
        band_width: Width of each sub-band (Hz).
        overlap: Overlap between adjacent bands (Hz).
        sfreq: Sampling frequency of the input data (Hz).
        filter_order: Butterworth filter order.
        n_features_select: Number of features to select via MIBIF.
            If None or >= total features, all features are kept.
        reg: CSP regularization parameter.

    Usage:
        fbcsp = FBCSPFeatureExtractor(sfreq=128.0)
        fbcsp.fit(X_train, y_train)
        features = fbcsp.transform(X_test)
    """

    def __init__(
        self,
        n_components: int = 4,
        freq_min: float = 4.0,
        freq_max: float = 40.0,
        band_width: float = 4.0,
        overlap: float = 0.0,
        sfreq: float = 128.0,
        filter_order: int = 5,
        n_features_select: int | None = None,
        reg: str | float | None = "ledoit_wolf",
    ) -> None:
        self.n_components = n_components
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.band_width = band_width
        self.overlap = overlap
        self.sfreq = sfreq
        self.filter_order = filter_order
        self.n_features_select = n_features_select
        self.reg = reg

        self._filter_bank: list[tuple[float, float, np.ndarray]] = []
        self._csp_list: list[CSP] = []
        self._selected_indices: np.ndarray | None = None
        self._is_fitted = False
        self._n_bands: int = 0
        self._n_total_features: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> FBCSPFeatureExtractor:
        """Fit the FBCSP pipeline: design filters, fit CSP per band, select features.

        Args:
            X: EEG epoch data of shape (n_trials, n_channels, n_times).
            y: Binary labels of shape (n_trials,).

        Returns:
            self
        """
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(
                f"FBCSP requires exactly 2 classes, got {len(unique_classes)}: {unique_classes}"
            )

        # Design filter bank
        self._filter_bank = _design_filter_bank(
            freq_min=self.freq_min,
            freq_max=self.freq_max,
            band_width=self.band_width,
            overlap=self.overlap,
            sfreq=self.sfreq,
            order=self.filter_order,
        )
        self._n_bands = len(self._filter_bank)

        if self._n_bands == 0:
            raise ValueError(
                f"No valid filter bands for freq_min={self.freq_min}, "
                f"freq_max={self.freq_max}, band_width={self.band_width}, "
                f"sfreq={self.sfreq}"
            )

        # Apply filter bank and fit CSP per band
        filtered_data = _apply_filter_bank(X, self._filter_bank)
        self._csp_list = []
        all_features = []

        for band_idx, X_band in enumerate(filtered_data):
            low, high, _ = self._filter_bank[band_idx]
            csp = CSP(
                n_components=self.n_components,
                reg=self.reg,
                log=True,
                norm_trace=False,
            )
            feats = csp.fit_transform(X_band, y)
            self._csp_list.append(csp)
            all_features.append(feats)
            logger.debug(
                "  Band %d (%.0f-%.0f Hz): CSP fitted, %d features",
                band_idx,
                low,
                high,
                feats.shape[1],
            )

        # Concatenate all band features
        all_features_concat = np.hstack(all_features)
        self._n_total_features = all_features_concat.shape[1]

        # Feature selection via mutual information (MIBIF)
        if self.n_features_select is not None and self.n_features_select < self._n_total_features:
            mi_scores = mutual_info_classif(all_features_concat, y, random_state=42)
            self._selected_indices = np.argsort(mi_scores)[::-1][: self.n_features_select]
            self._selected_indices = np.sort(self._selected_indices)
        else:
            self._selected_indices = np.arange(self._n_total_features)

        self._is_fitted = True

        n_selected = len(self._selected_indices)
        band_ranges = [f"{low:.0f}-{high:.0f}" for low, high, _ in self._filter_bank]
        logger.info(
            "FBCSP fitted: %d bands (%s Hz), %d CSP/band, %d/%d features selected",
            self._n_bands,
            ", ".join(band_ranges),
            self.n_components,
            n_selected,
            self._n_total_features,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract FBCSP features from epoch data.

        Args:
            X: EEG epoch data of shape (n_trials, n_channels, n_times).

        Returns:
            Feature matrix of shape (n_trials, n_selected_features).
        """
        if not self._is_fitted:
            raise RuntimeError("FBCSPFeatureExtractor must be fitted before transform")

        filtered_data = _apply_filter_bank(X, self._filter_bank)
        all_features = []

        for band_idx, X_band in enumerate(filtered_data):
            feats = self._csp_list[band_idx].transform(X_band)
            all_features.append(feats)

        all_features_concat = np.hstack(all_features)

        # Apply feature selection
        selected = all_features_concat[:, self._selected_indices]
        logger.info("FBCSP features extracted: shape=%s", selected.shape)
        return selected

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            X: EEG epoch data.
            y: Labels.

        Returns:
            FBCSP feature matrix.
        """
        return self.fit(X, y).transform(X)

    @property
    def n_bands(self) -> int:
        """Number of frequency sub-bands."""
        return self._n_bands

    @property
    def n_features(self) -> int | None:
        """Number of output features (after selection, if any)."""
        if not self._is_fitted:
            return None
        return len(self._selected_indices)

    @property
    def band_ranges(self) -> list[tuple[float, float]]:
        """Frequency ranges of each sub-band."""
        return [(low, high) for low, high, _ in self._filter_bank]
