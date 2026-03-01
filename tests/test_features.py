"""Feature extractor tests (FBCSP, FBRiemannian, recenter)."""

from __future__ import annotations

import logging

import numpy as np
import pytest

logging.disable(logging.CRITICAL)


def _make_eeg_data(
    n_trials: int = 40,
    n_channels: int = 8,
    n_times: int = 256,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_per_class = n_trials // 2
    X_list = []
    y_list = []
    for cls in range(2):
        X_cls = rng.standard_normal((n_per_class, n_channels, n_times)).astype(np.float32)
        t = np.linspace(0, 2, n_times)
        signal = np.sin(2 * np.pi * 10 * t) * 3.0
        if cls == 0:
            X_cls[:, 0, :] += signal
        else:
            X_cls[:, 1, :] += signal
        X_list.append(X_cls)
        y_list.append(np.full(n_per_class, cls, dtype=np.int64))
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


class TestFBCSPFeatureExtractor:
    def test_output_shape_default_k_best(self):
        from bci.features.csp import FBCSPFeatureExtractor

        X, y = _make_eeg_data(n_trials=40, n_channels=8, n_times=256)
        fbcsp = FBCSPFeatureExtractor(sfreq=128.0, k_best=12)
        feats = fbcsp.fit_transform(X, y)
        assert feats.shape == (40, 12)

    def test_output_shape_no_selector(self):
        from bci.features.csp import FBCSPFeatureExtractor

        X, y = _make_eeg_data(n_trials=40, n_channels=8, n_times=256)
        fbcsp = FBCSPFeatureExtractor(sfreq=128.0, n_components=4, k_best=None)
        feats = fbcsp.fit_transform(X, y)
        assert feats.shape == (40, 24)

    def test_transform_test_set_shape(self):
        from bci.features.csp import FBCSPFeatureExtractor

        X_train, y_train = _make_eeg_data(n_trials=40, n_channels=8, n_times=256)
        X_test, _ = _make_eeg_data(n_trials=10, n_channels=8, n_times=256, seed=7)
        fbcsp = FBCSPFeatureExtractor(sfreq=128.0, k_best=12)
        fbcsp.fit(X_train, y_train)
        feats_test = fbcsp.transform(X_test)
        assert feats_test.shape == (10, 12)

    def test_raises_before_fit(self):
        from bci.features.csp import FBCSPFeatureExtractor

        X, _ = _make_eeg_data(n_trials=10, n_channels=8, n_times=256)
        fbcsp = FBCSPFeatureExtractor()
        with pytest.raises(RuntimeError, match="fitted"):
            fbcsp.transform(X)

    def test_n_features_out_property(self):
        from bci.features.csp import FBCSPFeatureExtractor

        X, y = _make_eeg_data(n_trials=40, n_channels=8, n_times=256)
        fbcsp = FBCSPFeatureExtractor(sfreq=128.0, k_best=8)
        assert fbcsp.n_features_out is None
        fbcsp.fit(X, y)
        assert fbcsp.n_features_out == 8


class TestEnsembleCSPClassifier:
    def test_predict_output_shape(self):
        from bci.features.csp import EnsembleCSPClassifier

        X_train, y_train = _make_eeg_data(n_trials=40, n_channels=8, n_times=512)
        X_test, _ = _make_eeg_data(n_trials=10, n_channels=8, n_times=512, seed=5)
        clf = EnsembleCSPClassifier(sfreq=128.0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        assert y_pred.shape == (10,)
        assert set(y_pred).issubset({0, 1})

    def test_predict_proba_output_shape(self):
        from bci.features.csp import EnsembleCSPClassifier

        X_train, y_train = _make_eeg_data(n_trials=40, n_channels=8, n_times=512)
        X_test, _ = _make_eeg_data(n_trials=10, n_channels=8, n_times=512, seed=5)
        clf = EnsembleCSPClassifier(sfreq=128.0)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)
        assert y_prob.shape == (10, 2)
        assert np.allclose(y_prob.sum(axis=1), 1.0, atol=1e-5)

    def test_raises_before_fit(self):
        from bci.features.csp import EnsembleCSPClassifier

        X, _ = _make_eeg_data(n_trials=10, n_channels=8, n_times=512)
        clf = EnsembleCSPClassifier()
        with pytest.raises(RuntimeError, match="fitted"):
            clf.predict(X)


class TestFBRiemannianFeatureExtractor:
    def test_output_shape_with_pca(self):
        from bci.features.riemannian import FBRiemannianFeatureExtractor

        X, y = _make_eeg_data(n_trials=40, n_channels=8, n_times=256)
        fbriem = FBRiemannianFeatureExtractor(sfreq=128.0, n_components_pca=32)
        feats = fbriem.fit_transform(X, y)
        assert feats.shape == (40, 32)

    def test_output_shape_no_pca(self):
        from bci.features.riemannian import FBRiemannianFeatureExtractor

        X, y = _make_eeg_data(n_trials=40, n_channels=4, n_times=256)
        fbriem = FBRiemannianFeatureExtractor(sfreq=128.0, n_components_pca=None)
        feats = fbriem.fit_transform(X, y)
        assert feats.shape == (40, 60)

    def test_transform_test_set_shape(self):
        from bci.features.riemannian import FBRiemannianFeatureExtractor

        X_train, y_train = _make_eeg_data(n_trials=40, n_channels=8, n_times=256)
        X_test, _ = _make_eeg_data(n_trials=10, n_channels=8, n_times=256, seed=9)
        fbriem = FBRiemannianFeatureExtractor(sfreq=128.0, n_components_pca=16)
        fbriem.fit(X_train, y_train)
        feats_test = fbriem.transform(X_test)
        assert feats_test.shape == (10, 16)

    def test_raises_before_fit(self):
        from bci.features.riemannian import FBRiemannianFeatureExtractor

        X, _ = _make_eeg_data(n_trials=10, n_channels=8, n_times=256)
        fbriem = FBRiemannianFeatureExtractor()
        with pytest.raises(RuntimeError, match="fitted"):
            fbriem.transform(X)

    def test_n_features_out_property(self):
        from bci.features.riemannian import FBRiemannianFeatureExtractor

        X, y = _make_eeg_data(n_trials=40, n_channels=8, n_times=256)
        fbriem = FBRiemannianFeatureExtractor(sfreq=128.0, n_components_pca=20)
        assert fbriem.n_features_out is None
        fbriem.fit(X, y)
        assert fbriem.n_features_out == 20


class TestRiemannianRecenter:
    def test_output_shapes_preserved(self):
        from bci.features.riemannian import riemannian_recenter

        X_train, _ = _make_eeg_data(n_trials=40, n_channels=8, n_times=256)
        X_test, _ = _make_eeg_data(n_trials=10, n_channels=8, n_times=256, seed=3)
        X_tr_aligned, X_te_aligned = riemannian_recenter(X_train, X_test)
        assert X_tr_aligned.shape == X_train.shape
        assert X_te_aligned.shape == X_test.shape

    def test_dtype_float32(self):
        from bci.features.riemannian import riemannian_recenter

        X_train, _ = _make_eeg_data(n_trials=40, n_channels=8, n_times=256)
        X_test, _ = _make_eeg_data(n_trials=10, n_channels=8, n_times=256, seed=3)
        X_tr_aligned, X_te_aligned = riemannian_recenter(X_train, X_test)
        assert X_tr_aligned.dtype == np.float32
        assert X_te_aligned.dtype == np.float32

    def test_recenters_mean_covariance(self):
        from bci.features.riemannian import riemannian_recenter
        from pyriemann.estimation import Covariances
        from pyriemann.utils.mean import mean_riemann

        X_train, _ = _make_eeg_data(n_trials=60, n_channels=4, n_times=256)
        X_test, _ = _make_eeg_data(n_trials=10, n_channels=4, n_times=256, seed=4)
        X_tr_aligned, _ = riemannian_recenter(X_train, X_test, estimator="scm")

        covs = Covariances(estimator="scm").fit_transform(X_tr_aligned.astype(np.float64))
        mean_cov = mean_riemann(covs)
        assert np.allclose(mean_cov, np.eye(4), atol=0.15)
