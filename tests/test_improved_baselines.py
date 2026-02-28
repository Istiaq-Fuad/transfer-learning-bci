"""Unit tests for improved baselines (FBCSP, MDM, TS+SVM, ensemble).

Tests cover:
    - FBCSPFeatureExtractor: filter bank design, fit/transform, feature selection
    - MDM predict_fn: shape and value checks
    - TS+SVM predict_fn: shape and value checks
    - TS+LogReg predict_fn: shape and value checks
    - CSP+SVM predict_fn: shape and value checks
    - Ensemble predict_fn: shape and value checks

Run with:
    uv run pytest tests/test_improved_baselines.py -v
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

logging.disable(logging.CRITICAL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eeg_data(
    n_trials: int = 40,
    n_channels: int = 8,
    n_times: int = 512,
    sfreq: float = 128.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate discriminative synthetic EEG data.

    Uses 512 timepoints at 128 Hz (4 seconds) to ensure sufficient
    data for bandpass filtering in FBCSP.
    """
    rng = np.random.default_rng(seed)
    n_per_class = n_trials // 2
    X_list = []
    y_list = []
    t = np.linspace(0, n_times / sfreq, n_times)

    for cls in range(2):
        X_cls = rng.standard_normal((n_per_class, n_channels, n_times)).astype(np.float32)
        # Class 0: 10 Hz mu rhythm on channel 0
        # Class 1: 10 Hz mu rhythm on channel 1
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


# ---------------------------------------------------------------------------
# FBCSP tests
# ---------------------------------------------------------------------------


class TestFBCSPFeatureExtractor:
    def test_fit_transform_shape(self):
        from bci.features.fbcsp import FBCSPFeatureExtractor

        X, y = _make_eeg_data(n_trials=40, n_channels=8, n_times=512)
        fbcsp = FBCSPFeatureExtractor(
            n_components=4,
            sfreq=128.0,
            n_features_select=None,
        )
        features = fbcsp.fit_transform(X, y)
        # 9 bands x 4 components = 36 features (no selection)
        assert features.shape[0] == 40
        assert features.shape[1] == fbcsp.n_bands * 4

    def test_feature_selection_reduces_dimensions(self):
        from bci.features.fbcsp import FBCSPFeatureExtractor

        X, y = _make_eeg_data(n_trials=40, n_channels=8, n_times=512)
        fbcsp = FBCSPFeatureExtractor(
            n_components=4,
            sfreq=128.0,
            n_features_select=12,
        )
        features = fbcsp.fit_transform(X, y)
        assert features.shape == (40, 12)

    def test_transform_without_fit_raises(self):
        from bci.features.fbcsp import FBCSPFeatureExtractor

        X, _ = _make_eeg_data(n_trials=10, n_channels=8, n_times=512)
        fbcsp = FBCSPFeatureExtractor(sfreq=128.0)
        with pytest.raises(RuntimeError):
            fbcsp.transform(X)

    def test_requires_binary_classes(self):
        from bci.features.fbcsp import FBCSPFeatureExtractor

        X = np.random.randn(30, 8, 512).astype(np.float32)
        y = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int64)
        fbcsp = FBCSPFeatureExtractor(sfreq=128.0)
        with pytest.raises(ValueError, match="2 classes"):
            fbcsp.fit(X, y)

    def test_n_bands_property(self):
        from bci.features.fbcsp import FBCSPFeatureExtractor

        X, y = _make_eeg_data(n_trials=40, n_channels=8, n_times=512)
        fbcsp = FBCSPFeatureExtractor(
            freq_min=4.0,
            freq_max=40.0,
            band_width=4.0,
            sfreq=128.0,
        )
        fbcsp.fit(X, y)
        # 4-40 Hz in 4 Hz bands = 9 bands
        assert fbcsp.n_bands == 9

    def test_band_ranges_property(self):
        from bci.features.fbcsp import FBCSPFeatureExtractor

        X, y = _make_eeg_data(n_trials=40, n_channels=8, n_times=512)
        fbcsp = FBCSPFeatureExtractor(
            freq_min=8.0,
            freq_max=24.0,
            band_width=4.0,
            sfreq=128.0,
        )
        fbcsp.fit(X, y)
        ranges = fbcsp.band_ranges
        assert len(ranges) == 4  # 8-12, 12-16, 16-20, 20-24
        assert ranges[0] == (8.0, 12.0)
        assert ranges[-1] == (20.0, 24.0)

    def test_transform_shape_matches_fit(self):
        from bci.features.fbcsp import FBCSPFeatureExtractor

        X_train, y_train = _make_eeg_data(n_trials=30, n_channels=8, n_times=512, seed=0)
        X_test, _ = _make_eeg_data(n_trials=10, n_channels=8, n_times=512, seed=99)
        fbcsp = FBCSPFeatureExtractor(
            n_components=4,
            sfreq=128.0,
            n_features_select=8,
        )
        fbcsp.fit(X_train, y_train)
        features = fbcsp.transform(X_test)
        assert features.shape == (10, 8)

    def test_n_features_property(self):
        from bci.features.fbcsp import FBCSPFeatureExtractor

        X, y = _make_eeg_data(n_trials=40, n_channels=8, n_times=512)
        fbcsp = FBCSPFeatureExtractor(
            n_components=4,
            sfreq=128.0,
            n_features_select=10,
        )
        assert fbcsp.n_features is None  # before fit
        fbcsp.fit(X, y)
        assert fbcsp.n_features == 10  # after fit


# ---------------------------------------------------------------------------
# Filter bank design tests
# ---------------------------------------------------------------------------


class TestFilterBankDesign:
    def test_default_bands(self):
        from bci.features.fbcsp import _design_filter_bank

        filters = _design_filter_bank(
            freq_min=4.0,
            freq_max=40.0,
            band_width=4.0,
            sfreq=128.0,
        )
        assert len(filters) == 9
        # Check first and last bands
        assert filters[0][0] == 4.0
        assert filters[0][1] == 8.0
        assert filters[-1][0] == 36.0
        assert filters[-1][1] == 40.0

    def test_overlap_increases_bands(self):
        from bci.features.fbcsp import _design_filter_bank

        no_overlap = _design_filter_bank(
            freq_min=4.0,
            freq_max=40.0,
            band_width=4.0,
            overlap=0.0,
            sfreq=128.0,
        )
        with_overlap = _design_filter_bank(
            freq_min=4.0,
            freq_max=40.0,
            band_width=4.0,
            overlap=2.0,
            sfreq=128.0,
        )
        assert len(with_overlap) > len(no_overlap)

    def test_respects_nyquist(self):
        from bci.features.fbcsp import _design_filter_bank

        # sfreq=64 Hz -> nyquist=32 Hz, so 32-36 band should not appear
        filters = _design_filter_bank(
            freq_min=4.0,
            freq_max=40.0,
            band_width=4.0,
            sfreq=64.0,
        )
        for low, high, _ in filters:
            assert high < 32.0  # all below Nyquist


# ---------------------------------------------------------------------------
# Improved baseline predict_fn tests
# ---------------------------------------------------------------------------


class TestMDMPredictFn:
    def test_output_shape(self):
        from scripts.baseline_improved import make_mdm_predict_fn

        predict_fn = make_mdm_predict_fn()
        X_train, y_train = _make_eeg_data(n_trials=30, seed=0)
        X_test, _ = _make_eeg_data(n_trials=10, seed=99)
        y_pred, y_prob = predict_fn(X_train, y_train, X_test)
        assert y_pred.shape == (10,)
        assert set(y_pred).issubset({0, 1})

    def test_binary_predictions(self):
        from scripts.baseline_improved import make_mdm_predict_fn

        predict_fn = make_mdm_predict_fn()
        X_train, y_train = _make_eeg_data(n_trials=40, seed=0)
        X_test, _ = _make_eeg_data(n_trials=20, seed=99)
        y_pred, _ = predict_fn(X_train, y_train, X_test)
        assert all(p in (0, 1) for p in y_pred)


class TestTSSVMPredictFn:
    def test_output_shape(self):
        from scripts.baseline_improved import make_ts_svm_predict_fn

        predict_fn = make_ts_svm_predict_fn()
        X_train, y_train = _make_eeg_data(n_trials=30, seed=0)
        X_test, _ = _make_eeg_data(n_trials=10, seed=99)
        y_pred, y_prob = predict_fn(X_train, y_train, X_test)
        assert y_pred.shape == (10,)
        assert y_prob.shape == (10, 2)
        assert set(y_pred).issubset({0, 1})

    def test_probabilities_sum_to_one(self):
        from scripts.baseline_improved import make_ts_svm_predict_fn

        predict_fn = make_ts_svm_predict_fn()
        X_train, y_train = _make_eeg_data(n_trials=30, seed=0)
        X_test, _ = _make_eeg_data(n_trials=10, seed=99)
        _, y_prob = predict_fn(X_train, y_train, X_test)
        np.testing.assert_allclose(y_prob.sum(axis=1), 1.0, atol=1e-5)


class TestTSLogRegPredictFn:
    def test_output_shape(self):
        from scripts.baseline_improved import make_ts_logreg_predict_fn

        predict_fn = make_ts_logreg_predict_fn()
        X_train, y_train = _make_eeg_data(n_trials=30, seed=0)
        X_test, _ = _make_eeg_data(n_trials=10, seed=99)
        y_pred, y_prob = predict_fn(X_train, y_train, X_test)
        assert y_pred.shape == (10,)
        assert y_prob.shape == (10, 2)
        assert set(y_pred).issubset({0, 1})


class TestFBCSPSVMPredictFn:
    def test_output_shape(self):
        from scripts.baseline_improved import make_fbcsp_svm_predict_fn

        predict_fn = make_fbcsp_svm_predict_fn(sfreq=128.0)
        X_train, y_train = _make_eeg_data(n_trials=30, seed=0)
        X_test, _ = _make_eeg_data(n_trials=10, seed=99)
        y_pred, y_prob = predict_fn(X_train, y_train, X_test)
        assert y_pred.shape == (10,)
        assert y_prob.shape == (10, 2)
        assert set(y_pred).issubset({0, 1})


class TestCSPSVMPredictFn:
    def test_output_shape(self):
        from scripts.baseline_improved import make_csp_svm_predict_fn

        predict_fn = make_csp_svm_predict_fn()
        X_train, y_train = _make_eeg_data(n_trials=30, seed=0)
        X_test, _ = _make_eeg_data(n_trials=10, seed=99)
        y_pred, y_prob = predict_fn(X_train, y_train, X_test)
        assert y_pred.shape == (10,)
        assert y_prob.shape == (10, 2)
        assert set(y_pred).issubset({0, 1})


class TestEnsemblePredictFn:
    def test_output_shape(self):
        from scripts.baseline_improved import make_ensemble_predict_fn

        predict_fn = make_ensemble_predict_fn(n_csp_components=4)
        X_train, y_train = _make_eeg_data(n_trials=30, seed=0)
        X_test, _ = _make_eeg_data(n_trials=10, seed=99)
        y_pred, y_prob = predict_fn(X_train, y_train, X_test)
        assert y_pred.shape == (10,)
        assert y_prob.shape == (10, 2)
        assert set(y_pred).issubset({0, 1})

    def test_ensemble_binary_only(self):
        from scripts.baseline_improved import make_ensemble_predict_fn

        predict_fn = make_ensemble_predict_fn(n_csp_components=4)
        X_train, y_train = _make_eeg_data(n_trials=40, seed=0)
        X_test, _ = _make_eeg_data(n_trials=20, seed=99)
        y_pred, _ = predict_fn(X_train, y_train, X_test)
        assert all(p in (0, 1) for p in y_pred)


# ---------------------------------------------------------------------------
# Integration test: run within_subject_cv with improved pipelines
# ---------------------------------------------------------------------------


class TestIntegrationWithCV:
    def test_mdm_within_subject_cv(self):
        """MDM pipeline works end-to-end with within_subject_cv."""
        from scripts.baseline_improved import make_mdm_predict_fn
        from bci.training.cross_validation import within_subject_cv

        X, y = _make_eeg_data(n_trials=40, n_channels=8, n_times=512, seed=42)
        predict_fn = make_mdm_predict_fn()
        result = within_subject_cv(X, y, predict_fn, n_folds=2)
        assert len(result.folds) == 2
        assert 0.0 <= result.mean_accuracy <= 100.0

    def test_fbcsp_within_subject_cv(self):
        """FBCSP pipeline works end-to-end with within_subject_cv."""
        from scripts.baseline_improved import make_fbcsp_svm_predict_fn
        from bci.training.cross_validation import within_subject_cv

        X, y = _make_eeg_data(n_trials=40, n_channels=8, n_times=512, seed=42)
        predict_fn = make_fbcsp_svm_predict_fn(sfreq=128.0)
        result = within_subject_cv(X, y, predict_fn, n_folds=2)
        assert len(result.folds) == 2
        assert 0.0 <= result.mean_accuracy <= 100.0
