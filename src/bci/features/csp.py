"""Common Spatial Patterns (CSP) feature extraction.

CSP finds spatial filters that maximize the variance ratio between two classes.
For Left vs Right Hand MI:
    - Maximizes variance for left hand trials
    - Minimizes variance for right hand trials (and vice versa)

The log-variance of CSP-filtered signals serves as a discriminative feature vector.

References:
    - Blankertz et al., "Optimizing Spatial Filters for Robust EEG Single-Trial Analysis"
    - MNE-Python CSP implementation
"""

from __future__ import annotations

import logging

import numpy as np
from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


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
        # features.shape == (n_trials, 12)
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
            self.n_components, X.shape, unique_classes,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract CSP features from epoch data.

        Args:
            X: EEG epoch data of shape (n_trials, n_channels, n_times).

        Returns:
            Feature matrix of shape (n_trials, 2 * n_components).
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
