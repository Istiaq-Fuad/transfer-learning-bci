"""Cross-validation and splits tests."""

from __future__ import annotations

import logging

import numpy as np
import pytest

logging.disable(logging.CRITICAL)


def _make_eeg_data(
    n_trials: int = 40,
    n_channels: int = 8,
    n_times: int = 128,
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


class TestSyntheticData:
    def test_shape(self):
        from bci.training.cross_validation import make_synthetic_subject_data

        data = make_synthetic_subject_data(n_subjects=3, n_trials_per_subject=40)
        assert len(data) == 3
        for _, (X, y) in data.items():
            assert X.shape == (40, 22, 512)
            assert y.shape == (40,)
            assert set(np.unique(y)) == {0, 1}

    def test_balanced_classes(self):
        from bci.training.cross_validation import make_synthetic_subject_data

        data = make_synthetic_subject_data(n_subjects=2, n_trials_per_subject=40)
        for _, (_, y) in data.items():
            assert np.sum(y == 0) == 20
            assert np.sum(y == 1) == 20


class TestCrossValidation:
    def _dummy_predict_fn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        majority = int(np.bincount(y_train).argmax())
        y_pred = np.full(len(X_test), majority, dtype=np.int64)
        n_classes = len(np.unique(y_train))
        y_prob = np.zeros((len(X_test), n_classes), dtype=np.float32)
        y_prob[:, majority] = 1.0
        return y_pred, y_prob

    def test_within_subject_cv_returns_correct_n_folds(self):
        from bci.training.cross_validation import within_subject_cv

        X, y = _make_eeg_data(n_trials=30)
        result = within_subject_cv(X, y, self._dummy_predict_fn, n_folds=3)
        assert len(result.folds) == 3

    def test_within_subject_cv_all_subjects(self):
        from bci.training.cross_validation import make_synthetic_subject_data, within_subject_cv_all

        data = make_synthetic_subject_data(n_subjects=2, n_trials_per_subject=20)
        result = within_subject_cv_all(data, self._dummy_predict_fn, n_folds=2)
        assert len(result.folds) == 4
        assert result.strategy == "within_subject"

    def test_loso_cv_n_folds_equals_n_subjects(self):
        from bci.training.cross_validation import loso_cv, make_synthetic_subject_data

        data = make_synthetic_subject_data(n_subjects=4, n_trials_per_subject=20)
        result = loso_cv(data, self._dummy_predict_fn)
        assert len(result.folds) == 4
        assert result.strategy == "loso"

    def test_loso_cv_per_subject_accuracy_keys(self):
        from bci.training.cross_validation import loso_cv, make_synthetic_subject_data

        data = make_synthetic_subject_data(n_subjects=3, n_trials_per_subject=20)
        result = loso_cv(data, self._dummy_predict_fn)
        assert set(result.per_subject_accuracy.keys()) == {1, 2, 3}

    def test_cv_result_summary_str(self):
        from bci.training.cross_validation import within_subject_cv

        X, y = _make_eeg_data(n_trials=30)
        result = within_subject_cv(X, y, self._dummy_predict_fn, n_folds=2)
        summary = result.summary()
        assert "Accuracy" in summary
        assert "Kappa" in summary

    def test_fold_result_counts(self):
        from bci.training.cross_validation import within_subject_cv

        X, y = _make_eeg_data(n_trials=30)
        result = within_subject_cv(X, y, self._dummy_predict_fn, n_folds=3)
        for fold in result.folds:
            assert fold.n_train + fold.n_test == len(y)


class TestSplitManager:
    def test_get_or_create_splits(self, tmp_path):
        from bci.training.splits import get_or_create_splits

        subject_data = {1: _make_eeg_data(n_trials=20), 2: _make_eeg_data(n_trials=20, seed=1)}
        spec = get_or_create_splits(tmp_path, "bci_iv2a", subject_data, n_folds=3, seed=42)
        assert spec.n_folds == 3
        assert spec.seed == 42
        assert set(spec.within_subject.keys()) == {1, 2}
        assert len(spec.loso_subjects) == 2

        spec_reload = get_or_create_splits(tmp_path, "bci_iv2a", subject_data, n_folds=3, seed=42)
        assert spec_reload.n_folds == 3
        assert spec_reload.seed == 42
