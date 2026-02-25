"""Riemannian geometry feature extraction for EEG covariance matrices.

EEG covariance matrices are Symmetric Positive-Definite (SPD) and live on a
Riemannian manifold. Standard Euclidean operations (mean, distance) don't apply
properly. Instead, we use:

    1. Covariance estimation: Compute SPD covariance matrices per trial
    2. Riemannian alignment: Align covariance matrices to a reference
    3. Tangent space projection: Project SPD matrices to Euclidean tangent space
    4. Use tangent vectors as feature vectors for classification

References:
    - Barachant et al., "Multiclass Brain-Computer Interface Classification
      by Riemannian Geometry" (IEEE TBME, 2012)
    - pyRiemann documentation
"""

from __future__ import annotations

import logging

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class RiemannianFeatureExtractor(BaseEstimator, TransformerMixin):
    """Riemannian tangent space feature extractor.

    Computes covariance matrices from EEG epochs, then projects them into
    the tangent space at the geometric mean, yielding Euclidean feature vectors.

    Args:
        estimator: Covariance estimator. Options:
            "lwf" (Ledoit-Wolf, default), "scm" (sample), "oas".
        metric: Riemannian metric for tangent space. Options:
            "riemann" (default), "logeuclid", "euclid".
        tsupdate: If True, re-estimate the reference point during transform.

    Usage:
        riemann = RiemannianFeatureExtractor()
        riemann.fit(X_train, y_train)
        features = riemann.transform(X_test)
        # features.shape == (n_trials, n_channels * (n_channels + 1) / 2)
    """

    def __init__(
        self,
        estimator: str = "lwf",
        metric: str = "riemann",
        tsupdate: bool = False,
    ) -> None:
        self.estimator = estimator
        self.metric = metric
        self.tsupdate = tsupdate

        self._pipeline = Pipeline([
            ("covariances", Covariances(estimator=estimator)),
            ("tangent_space", TangentSpace(metric=metric, tsupdate=tsupdate)),
        ])
        self._is_fitted = False
        self._n_features: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> RiemannianFeatureExtractor:
        """Fit the Riemannian pipeline (estimate reference point).

        Args:
            X: EEG epoch data of shape (n_trials, n_channels, n_times).
            y: Labels (optional, not used for fitting but kept for API compatibility).

        Returns:
            self
        """
        self._pipeline.fit(X, y)
        self._is_fitted = True

        n_channels = X.shape[1]
        self._n_features = n_channels * (n_channels + 1) // 2

        logger.info(
            "Riemannian fitted: %d channels -> %d tangent space features "
            "(estimator=%s, metric=%s)",
            n_channels, self._n_features, self.estimator, self.metric,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract Riemannian tangent space features.

        Args:
            X: EEG epoch data of shape (n_trials, n_channels, n_times).

        Returns:
            Feature matrix of shape (n_trials, n_channels * (n_channels + 1) / 2).
        """
        if not self._is_fitted:
            raise RuntimeError("RiemannianFeatureExtractor must be fitted before transform")

        features = self._pipeline.transform(X)
        logger.info("Riemannian features extracted: shape=%s", features.shape)
        return features

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            X: EEG epoch data.
            y: Labels (optional).

        Returns:
            Tangent space feature matrix.
        """
        return self.fit(X, y).transform(X)

    @property
    def n_features(self) -> int | None:
        """Number of output features (after fitting)."""
        return self._n_features

    @staticmethod
    def compute_covariances(
        X: np.ndarray, estimator: str = "lwf",
    ) -> np.ndarray:
        """Compute covariance matrices without tangent space projection.

        Useful for visualization or when you need the raw SPD matrices.

        Args:
            X: EEG epoch data of shape (n_trials, n_channels, n_times).
            estimator: Covariance estimator type.

        Returns:
            Covariance matrices of shape (n_trials, n_channels, n_channels).
        """
        cov = Covariances(estimator=estimator)
        covmats = cov.fit_transform(X)
        logger.info("Covariance matrices computed: shape=%s", covmats.shape)
        return covmats
