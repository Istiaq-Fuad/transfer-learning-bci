"""Common Spatial Patterns (CSP) feature extraction.

CSP finds spatial filters that maximize the variance ratio between two classes.
For Left vs Right Hand MI:
    - Maximizes variance for left hand trials
    - Minimizes variance for right hand trials (and vice versa)

The log-variance of CSP-filtered signals serves as a discriminative feature vector.

Also provides:
    - FBCSPFeatureExtractor: Filter Bank CSP across multiple frequency sub-bands.
    - EnsembleCSPClassifier: Multi-window majority-vote CSP ensemble.

References:
    - Blankertz et al., "Optimizing Spatial Filters for Robust EEG Single-Trial Analysis"
    - Ang et al., "Filter Bank Common Spatial Pattern (FBCSP) in Brain-Computer Interface"
      (IJCNN 2008)
    - MNE-Python CSP implementation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from mne.decoding import CSP
from mne.filter import filter_data
from scipy.stats import mode
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Default sub-bands for Filter Bank CSP (mu + beta rhythm decomposition, 8–32 Hz)
_DEFAULT_BANDS: list[tuple[float, float]] = [
    (8.0, 12.0),  # mu / lower-alpha
    (12.0, 16.0),  # lower-beta
    (16.0, 20.0),  # mid-beta
    (20.0, 24.0),  # mid-beta
    (24.0, 28.0),  # upper-beta
    (28.0, 32.0),  # upper-beta
]

# Default time windows for Ensemble CSP (seconds, relative to epoch start)
_DEFAULT_WINDOWS: list[tuple[float, float]] = [
    (0.0, 2.0),
    (0.5, 2.5),
    (1.0, 3.0),
]


class CSPFeatureExtractor(BaseEstimator, TransformerMixin):
    """CSP-based feature extractor compatible with scikit-learn pipelines.

    Wraps MNE's CSP implementation and outputs log-variance features.

    Args:
        n_components: Number of CSP components to extract.
            MNE returns exactly n_components log-variance features per trial.
            Common values: 4, 6, 8 (larger = more features, more risk of overfitting).
        reg: Regularization parameter for covariance estimation.
            Useful for small sample sizes. Options: None, "ledoit_wolf", "oas", float.
        log: If True, return log-variance features. If False, return raw variance.
        norm_trace: If True, normalize the covariance trace before CSP.

    Usage:
        csp = CSPFeatureExtractor(n_components=6)
        csp.fit(X_train, y_train)
        features = csp.transform(X_test)
        # features.shape == (n_trials, 6)
    """

    def __init__(
        self,
        n_components: int = 6,
        reg: str | float | None = "ledoit_wolf",
        log: bool = True,
        norm_trace: bool = False,
    ) -> None:
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.norm_trace = norm_trace

        self._csp = CSP(
            n_components=n_components,
            reg=reg,
            log=log,
            norm_trace=norm_trace,
        )
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> CSPFeatureExtractor:
        """Fit CSP spatial filters on training data.

        Args:
            X: EEG epoch data of shape (n_trials, n_channels, n_times).
            y: Labels of shape (n_trials,). Must be binary (2 classes).

        Returns:
            self
        """
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(
                f"CSP requires exactly 2 classes, got {len(unique_classes)}: {unique_classes}"
            )

        self._csp.fit(X, y)
        self._is_fitted = True

        logger.info(
            "CSP fitted: %d components, input shape %s, classes %s",
            self.n_components,
            X.shape,
            unique_classes,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract CSP features from epoch data.

        Args:
            X: EEG epoch data of shape (n_trials, n_channels, n_times).

        Returns:
            Feature matrix of shape (n_trials, n_components).
            If log=True, these are log-variance features.
        """
        if not self._is_fitted:
            raise RuntimeError("CSPFeatureExtractor must be fitted before transform")

        features = self._csp.transform(X)
        logger.info("CSP features extracted: shape=%s", features.shape)
        return features

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            X: EEG epoch data.
            y: Labels.

        Returns:
            CSP feature matrix.
        """
        return self.fit(X, y).transform(X)

    def get_spatial_patterns(self) -> np.ndarray:
        """Return the CSP spatial patterns (for topographic visualization).

        Returns:
            Patterns matrix of shape (n_components, n_channels).
        """
        if not self._is_fitted:
            raise RuntimeError("CSP must be fitted first")
        return self._csp.patterns_

    def get_spatial_filters(self) -> np.ndarray:
        """Return the CSP spatial filters.

        Returns:
            Filters matrix of shape (n_components, n_channels).
        """
        if not self._is_fitted:
            raise RuntimeError("CSP must be fitted first")
        return self._csp.filters_


class FBCSPFeatureExtractor(BaseEstimator, TransformerMixin):
    """Filter Bank CSP (FBCSP) feature extractor.

    Applies CSP independently across multiple frequency sub-bands, then
    concatenates the per-band log-variance feature vectors.  A mutual-
    information feature selector reduces the concatenated vector to the
    ``k_best`` most discriminative features, preventing the classifier from
    being overwhelmed by irrelevant band/component combinations.

    Args:
        bands: List of (l_freq, h_freq) tuples defining the sub-bands.
            Default: 6 bands covering 8–32 Hz in 4 Hz steps.
        sfreq: Sampling frequency of the input data in Hz. Default: 128.0.
        n_components: CSP components per band. Default: 4.
        reg: Covariance regularization for each per-band CSP. Default: "ledoit_wolf".
        k_best: Number of features to keep after SelectKBest MI selection.
            If None, all features are kept (n_bands × n_components).
            Default: 12 (= 6 bands × 2 components, retaining ~half).

    Output dimensionality:
        ``k_best`` if set, else ``len(bands) * n_components``.

    Usage::

        fbcsp = FBCSPFeatureExtractor()
        fbcsp.fit(X_train, y_train)
        features = fbcsp.transform(X_test)   # shape (n_trials, k_best)
    """

    def __init__(
        self,
        bands: list[tuple[float, float]] | None = None,
        sfreq: float = 128.0,
        n_components: int = 4,
        reg: str | float | None = "ledoit_wolf",
        k_best: int | None = 12,
    ) -> None:
        self.bands = bands if bands is not None else list(_DEFAULT_BANDS)
        self.sfreq = sfreq
        self.n_components = n_components
        self.reg = reg
        self.k_best = k_best

        self._csps: list[CSP] = []
        self._selector: SelectKBest | None = None
        self._is_fitted = False

    def _filter_band(self, X: np.ndarray, l_freq: float, h_freq: float) -> np.ndarray:
        """Band-pass filter X with MNE filter_data (zero-phase FIR)."""
        return filter_data(
            X.astype(np.float64),
            sfreq=self.sfreq,
            l_freq=l_freq,
            h_freq=h_freq,
            method="fir",
            verbose=False,
        ).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FBCSPFeatureExtractor":
        """Fit one CSP per sub-band and optionally fit the feature selector.

        Args:
            X: EEG epoch data of shape (n_trials, n_channels, n_times).
            y: Class labels of shape (n_trials,). Must be binary.

        Returns:
            self
        """
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(
                f"FBCSP requires exactly 2 classes, got {len(unique_classes)}: {unique_classes}"
            )

        self._csps = []
        all_feats: list[np.ndarray] = []

        for l_freq, h_freq in self.bands:
            X_band = self._filter_band(X, l_freq, h_freq)
            csp = CSP(n_components=self.n_components, reg=self.reg, log=True)
            feats = csp.fit_transform(X_band, y)  # (n_trials, n_components)
            self._csps.append(csp)
            all_feats.append(feats)

        concat = np.hstack(all_feats)  # (n_trials, n_bands * n_components)

        if self.k_best is not None and self.k_best < concat.shape[1]:
            self._selector = SelectKBest(mutual_info_classif, k=self.k_best)
            self._selector.fit(concat, y)
        else:
            self._selector = None

        self._is_fitted = True
        n_out = self.k_best if (self._selector is not None) else concat.shape[1]
        logger.info(
            "FBCSP fitted: %d bands × %d components → %d features selected",
            len(self.bands),
            self.n_components,
            n_out,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract FBCSP features from epoch data.

        Args:
            X: EEG epoch data of shape (n_trials, n_channels, n_times).

        Returns:
            Feature matrix of shape (n_trials, k_best) or
            (n_trials, n_bands * n_components) if k_best is None.
        """
        if not self._is_fitted:
            raise RuntimeError("FBCSPFeatureExtractor must be fitted before transform")

        all_feats: list[np.ndarray] = []
        for (l_freq, h_freq), csp in zip(self.bands, self._csps):
            X_band = self._filter_band(X, l_freq, h_freq)
            all_feats.append(csp.transform(X_band))

        concat = np.hstack(all_feats)

        if self._selector is not None:
            concat = self._selector.transform(concat)

        logger.info("FBCSP features extracted: shape=%s", concat.shape)
        return concat

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:  # type: ignore[override]
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    @property
    def n_features_out(self) -> int | None:
        """Number of output features (available after fitting)."""
        if not self._is_fitted:
            return None
        if self._selector is not None:
            return int(self._selector.k)
        return len(self.bands) * self.n_components


class EnsembleCSPClassifier(BaseEstimator, ClassifierMixin):
    """Ensemble CSP classifier using majority vote across time windows.

    For each time window a fresh FBCSP + LDA pipeline is trained, then
    predictions are aggregated by majority vote (labels) and probability
    averaging (soft vote).

    Args:
        windows: List of (tmin, tmax) pairs in *seconds* defining time windows
            relative to the epoch start. Default: three overlapping 2-second
            windows covering 0–3 s.
        sfreq: Sampling frequency in Hz. Default: 128.0.
        fbcsp_bands: Sub-bands forwarded to FBCSPFeatureExtractor. Default: 6
            bands from 8–32 Hz.
        n_components: CSP components per band. Default: 4.
        k_best: Feature selector k forwarded to FBCSPFeatureExtractor.
        lda_shrinkage: Shrinkage for LDA. "auto" uses Ledoit-Wolf analytic
            formula (recommended). Default: "auto".

    Usage::

        clf = EnsembleCSPClassifier()
        clf.fit(X_train, y_train)
        y_pred, y_prob = clf.predict(X_test), clf.predict_proba(X_test)
    """

    def __init__(
        self,
        windows: list[tuple[float, float]] | None = None,
        sfreq: float = 128.0,
        fbcsp_bands: list[tuple[float, float]] | None = None,
        n_components: int = 4,
        k_best: int | None = 12,
        lda_shrinkage: str | float = "auto",
    ) -> None:
        self.windows = windows if windows is not None else list(_DEFAULT_WINDOWS)
        self.sfreq = sfreq
        self.fbcsp_bands = fbcsp_bands
        self.n_components = n_components
        self.k_best = k_best
        self.lda_shrinkage = lda_shrinkage

        self._pipelines: list[tuple[FBCSPFeatureExtractor, LinearDiscriminantAnalysis]] = []
        self._classes: np.ndarray | None = None
        self._is_fitted = False

    def _slice_window(self, X: np.ndarray, tmin: float, tmax: float) -> np.ndarray:
        """Slice epoch array to the given time window."""
        i_start = int(tmin * self.sfreq)
        i_end = int(tmax * self.sfreq)
        # Clamp to valid range
        i_end = min(i_end, X.shape[2])
        return X[:, :, i_start:i_end]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleCSPClassifier":
        """Fit one FBCSP + LDA pipeline per time window.

        Args:
            X: EEG epoch data of shape (n_trials, n_channels, n_times).
            y: Class labels of shape (n_trials,). Must be binary.

        Returns:
            self
        """
        self._classes = np.unique(y)
        self._pipelines = []

        for tmin, tmax in self.windows:
            X_win = self._slice_window(X, tmin, tmax)
            fbcsp = FBCSPFeatureExtractor(
                bands=self.fbcsp_bands,
                sfreq=self.sfreq,
                n_components=self.n_components,
                k_best=self.k_best,
            )
            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=self.lda_shrinkage)
            feats = fbcsp.fit_transform(X_win, y)
            lda.fit(feats, y)
            self._pipelines.append((fbcsp, lda))
            logger.info("Ensemble window [%.1f–%.1f s] fitted", tmin, tmax)

        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return averaged class probabilities across all window pipelines.

        Args:
            X: EEG epoch data of shape (n_trials, n_channels, n_times).

        Returns:
            Probability matrix of shape (n_trials, n_classes).
        """
        if not self._is_fitted:
            raise RuntimeError("EnsembleCSPClassifier must be fitted before predict")

        probs: list[np.ndarray] = []
        for (tmin, tmax), (fbcsp, lda) in zip(self.windows, self._pipelines):
            X_win = self._slice_window(X, tmin, tmax)
            feats = fbcsp.transform(X_win)
            probs.append(lda.predict_proba(feats))

        return np.mean(probs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return majority-vote class predictions across all window pipelines.

        Args:
            X: EEG epoch data of shape (n_trials, n_channels, n_times).

        Returns:
            Label array of shape (n_trials,).
        """
        if not self._is_fitted:
            raise RuntimeError("EnsembleCSPClassifier must be fitted before predict")

        all_preds: list[np.ndarray] = []
        for (tmin, tmax), (fbcsp, lda) in zip(self.windows, self._pipelines):
            X_win = self._slice_window(X, tmin, tmax)
            feats = fbcsp.transform(X_win)
            all_preds.append(lda.predict(feats))

        stacked = np.array(all_preds)  # (n_windows, n_trials)
        majority, _ = mode(stacked, axis=0)
        return majority.ravel()
