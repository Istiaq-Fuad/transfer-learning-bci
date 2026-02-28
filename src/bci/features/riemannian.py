"""Riemannian geometry feature extraction for EEG covariance matrices.

EEG covariance matrices are Symmetric Positive-Definite (SPD) and live on a
Riemannian manifold. Standard Euclidean operations (mean, distance) don't apply
properly. Instead, we use:

    1. Covariance estimation: Compute SPD covariance matrices per trial
    2. Riemannian alignment: Align covariance matrices to a reference
    3. Tangent space projection: Project SPD matrices to Euclidean tangent space
    4. Use tangent vectors as feature vectors for classification

Also provides:
    - FBRiemannianFeatureExtractor: Filter Bank Tangent Space across multiple
      frequency sub-bands, concatenated and optionally reduced via PCA.
    - riemannian_recenter(): Riemannian Alignment — re-center both train and
      test covariance distributions to the identity matrix using the geometric
      mean of the training set.

References:
    - Barachant et al., "Multiclass Brain-Computer Interface Classification
      by Riemannian Geometry" (IEEE TBME, 2012)
    - Zanini et al., "Transfer Learning: A Riemannian Geometry Framework with
      Applications to Brain-Computer Interfaces" (IEEE TBME, 2018)
    - pyRiemann documentation
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.linalg
from mne.filter import filter_data
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from bci.utils.config import FILTER_BANK_BANDS

logger = logging.getLogger(__name__)


def riemannian_recenter(
    X_train: np.ndarray,
    X_test: np.ndarray,
    estimator: str = "oas",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Riemannian Alignment (RA) to EEG epoch data.

    Re-centres the covariance distribution of each split to the identity matrix
    using the geometric mean of the *training* split.  This reduces between-
    session and between-subject non-stationarity without requiring labelled
    target data.

    The whitening transform is::

        X_aligned = R^{-1/2} @ X_raw

    where R is the Riemannian mean covariance matrix estimated on X_train.
    The same R is applied to both splits to avoid test-set leakage.

    Args:
        X_train: Training EEG epochs of shape (n_train, n_channels, n_times).
        X_test:  Test EEG epochs of shape (n_test, n_channels, n_times).
        estimator: Covariance estimator — "oas" | "lwf" | "scm".

    Returns:
        Tuple (X_train_aligned, X_test_aligned) with the same shape as inputs.
    """
    cov_est = Covariances(estimator=estimator)
    covs_train = cov_est.fit_transform(X_train.astype(np.float64))  # (n_train, C, C)

    # Riemannian mean of training covariances
    R_mean = mean_riemann(covs_train)  # (C, C)

    # Whitening matrix: R_mean^{-1/2}
    R_sqrt = scipy.linalg.sqrtm(R_mean)
    R_inv_sqrt = np.linalg.inv(R_sqrt).real  # (C, C)

    def _whiten(X: np.ndarray) -> np.ndarray:
        # X: (n_trials, C, T) → apply R_inv_sqrt @ x per trial
        return np.einsum("ij,njt->nit", R_inv_sqrt, X.astype(np.float64)).astype(np.float32)

    logger.info(
        "Riemannian alignment: computed mean cov from %d training trials (estimator=%s)",
        len(X_train),
        estimator,
    )
    return _whiten(X_train), _whiten(X_test)


class RiemannianFeatureExtractor(BaseEstimator, TransformerMixin):
    """Riemannian tangent space feature extractor.

    Computes covariance matrices from EEG epochs, then projects them into
    the tangent space at the geometric mean, yielding Euclidean feature vectors.

    Args:
        estimator: Covariance estimator. Options:
            "oas" (Oracle Approximating Shrinkage, default), "lwf" (Ledoit-Wolf), "scm" (sample).
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
        estimator: str = "oas",
        metric: str = "riemann",
        tsupdate: bool = False,
    ) -> None:
        self.estimator = estimator
        self.metric = metric
        self.tsupdate = tsupdate

        self._pipeline = Pipeline(
            [
                ("covariances", Covariances(estimator=estimator)),
                ("tangent_space", TangentSpace(metric=metric, tsupdate=tsupdate)),
            ]
        )
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
            "Riemannian fitted: %d channels -> %d tangent space features (estimator=%s, metric=%s)",
            n_channels,
            self._n_features,
            self.estimator,
            self.metric,
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
        X: np.ndarray,
        estimator: str = "lwf",
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


class FBRiemannianFeatureExtractor(BaseEstimator, TransformerMixin):
    """Filter Bank Riemannian (FBTS) feature extractor.

    Applies Riemannian tangent space feature extraction independently across
    multiple frequency sub-bands, then concatenates the per-band feature
    vectors.  Optional PCA reduces the high-dimensional concatenated vector
    (e.g. 22-ch × 6 bands = 1518 features) to ``n_components_pca`` dimensions.

    Args:
        bands: List of (l_freq, h_freq) tuples defining the sub-bands.
            Default: 6 bands covering 8–32 Hz in 4 Hz steps (from FILTER_BANK_BANDS).
        sfreq: Sampling frequency of the input data in Hz. Default: 128.0.
        estimator: Covariance estimator — "oas" | "lwf" | "scm". Default: "oas".
        metric: Riemannian metric for tangent space. Default: "riemann".
        n_components_pca: If not None, apply PCA to reduce the concatenated
            features to this many dimensions. Default: 128.

    Output dimensionality:
        ``n_components_pca`` if set, else
        ``len(bands) × n_channels × (n_channels + 1) // 2``.

    Usage::

        fbriem = FBRiemannianFeatureExtractor()
        fbriem.fit(X_train, y_train)
        features = fbriem.transform(X_test)   # shape (n_trials, 128)
    """

    def __init__(
        self,
        bands: list[tuple[float, float]] | None = None,
        sfreq: float = 128.0,
        estimator: str = "oas",
        metric: str = "riemann",
        n_components_pca: int | None = 128,
    ) -> None:
        self.bands = bands if bands is not None else list(FILTER_BANK_BANDS)
        self.sfreq = sfreq
        self.estimator = estimator
        self.metric = metric
        self.n_components_pca = n_components_pca

        self._extractors: list[RiemannianFeatureExtractor] = []
        self._pca: PCA | None = None
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

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "FBRiemannianFeatureExtractor":
        """Fit one Riemannian extractor per sub-band and optionally fit PCA.

        Args:
            X: EEG epoch data of shape (n_trials, n_channels, n_times).
            y: Labels (optional, passed to each extractor for API compatibility).

        Returns:
            self
        """
        self._extractors = []
        all_feats: list[np.ndarray] = []

        for l_freq, h_freq in self.bands:
            X_band = self._filter_band(X, l_freq, h_freq)
            extractor = RiemannianFeatureExtractor(
                estimator=self.estimator,
                metric=self.metric,
            )
            feats = extractor.fit_transform(X_band, y)  # (n_trials, n_ts_features)
            self._extractors.append(extractor)
            all_feats.append(feats)

        concat = np.hstack(all_feats)  # (n_trials, n_bands * n_ts_features)

        n_components = self.n_components_pca
        if n_components is not None and n_components < concat.shape[1]:
            # Clamp to available samples/features
            n_components = min(n_components, concat.shape[0] - 1, concat.shape[1])
            self._pca = PCA(n_components=n_components)
            self._pca.fit(concat)
        else:
            self._pca = None

        self._is_fitted = True
        n_out = (
            int(self._pca.n_components_)  # type: ignore[union-attr]
            if self._pca is not None
            else concat.shape[1]
        )
        logger.info(
            "FBRiemannian fitted: %d bands × %d TS features → %d features (PCA=%s)",
            len(self.bands),
            concat.shape[1] // len(self.bands),
            n_out,
            self._pca is not None,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract FBTS features from epoch data.

        Args:
            X: EEG epoch data of shape (n_trials, n_channels, n_times).

        Returns:
            Feature matrix of shape (n_trials, n_components_pca) or
            (n_trials, n_bands * n_ts_features) if n_components_pca is None.
        """
        if not self._is_fitted:
            raise RuntimeError("FBRiemannianFeatureExtractor must be fitted before transform")

        all_feats: list[np.ndarray] = []
        for (l_freq, h_freq), extractor in zip(self.bands, self._extractors):
            X_band = self._filter_band(X, l_freq, h_freq)
            all_feats.append(extractor.transform(X_band))

        concat = np.hstack(all_feats)

        if self._pca is not None:
            concat = self._pca.transform(concat)

        logger.info("FBRiemannian features extracted: shape=%s", concat.shape)
        return concat

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:  # type: ignore[override]
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    @property
    def n_features_out(self) -> int | None:
        """Number of output features (available after fitting)."""
        if not self._is_fitted:
            return None
        if self._pca is not None:
            return int(self._pca.n_components_)
        return len(self.bands) * (self._extractors[0].n_features or 0)
