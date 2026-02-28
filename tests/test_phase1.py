"""Unit tests for Phase 1 components.

Tests cover:
    - Cross-validation infrastructure (within-subject, LOSO)
    - Trainer (training loop, early stopping, predict)
    - Baseline A predict_fn (CSP+LDA)
    - Baseline B predict_fn (Riemannian+LDA)

Run with:
    uv run pytest tests/test_phase1.py -v
    uv run pytest tests/test_phase1.py -v -k "test_cv"
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

logging.disable(logging.CRITICAL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eeg_data(
    n_trials: int = 40,
    n_channels: int = 8,
    n_times: int = 128,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate discriminative synthetic EEG data."""
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


# ---------------------------------------------------------------------------
# Cross-validation tests
# ---------------------------------------------------------------------------


class TestSyntheticData:
    def test_shape(self):
        from bci.training.cross_validation import make_synthetic_subject_data

        data = make_synthetic_subject_data(n_subjects=3, n_trials_per_subject=40)
        assert len(data) == 3
        for subj_id, (X, y) in data.items():
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
        """Always predicts the majority class."""
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
        from bci.training.cross_validation import (
            make_synthetic_subject_data,
            within_subject_cv_all,
        )

        data = make_synthetic_subject_data(n_subjects=2, n_trials_per_subject=20)
        result = within_subject_cv_all(data, self._dummy_predict_fn, n_folds=2)
        # 2 subjects x 2 folds
        assert len(result.folds) == 4
        assert result.strategy == "within_subject"

    def test_loso_cv_n_folds_equals_n_subjects(self):
        from bci.training.cross_validation import (
            loso_cv,
            make_synthetic_subject_data,
        )

        data = make_synthetic_subject_data(n_subjects=4, n_trials_per_subject=20)
        result = loso_cv(data, self._dummy_predict_fn)
        assert len(result.folds) == 4
        assert result.strategy == "loso"

    def test_loso_cv_per_subject_accuracy_keys(self):
        from bci.training.cross_validation import (
            loso_cv,
            make_synthetic_subject_data,
        )

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

    def test_csp_lda_predict_fn_output_shape(self):
        """predict_fn from Baseline A returns correct shapes."""
        from scripts.pipeline.stage_02_baseline_a import make_predict_fn  # type: ignore[import]

        predict_fn = make_predict_fn(n_components=4)
        X_train, y_train = _make_eeg_data(n_trials=30)
        X_test, _ = _make_eeg_data(n_trials=10, seed=99)
        y_pred, y_prob = predict_fn(X_train, y_train, X_test)
        assert y_pred.shape == (10,)
        assert y_prob.shape == (10, 2)
        assert set(y_pred).issubset({0, 1})

    def test_riemannian_lda_predict_fn_output_shape(self):
        """predict_fn from Baseline B returns correct shapes."""
        from scripts.pipeline.stage_03_baseline_b import make_predict_fn  # type: ignore[import]

        predict_fn = make_predict_fn(estimator="scm", metric="riemann")
        X_train, y_train = _make_eeg_data(n_trials=30)
        X_test, _ = _make_eeg_data(n_trials=10, seed=99)
        y_pred, y_prob = predict_fn(X_train, y_train, X_test)
        assert y_pred.shape == (10,)
        assert y_prob.shape == (10, 2)
        assert set(y_pred).issubset({0, 1})


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------


class TestTrainer:
    def _make_tiny_model(self) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
        )

    def _make_tiny_dataset(self, n: int = 40, seed: int = 0):
        from torch.utils.data import TensorDataset

        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, 4)).astype(np.float32)
        y = (rng.random(n) > 0.5).astype(np.int64)
        return TensorDataset(torch.tensor(X), torch.tensor(y))

    def test_trainer_fit_returns_result(self):
        from bci.training.trainer import Trainer

        model = self._make_tiny_model()
        trainer = Trainer(model, device="cpu", epochs=3, batch_size=8, patience=5, seed=0)
        dataset = self._make_tiny_dataset()
        result = trainer.fit(dataset)
        assert result.final_epoch <= 3
        assert 0.0 <= result.best_val_accuracy <= 100.0

    def test_trainer_predict_output_shapes(self):
        from torch.utils.data import DataLoader, TensorDataset

        from bci.training.trainer import Trainer

        model = self._make_tiny_model()
        trainer = Trainer(model, device="cpu", epochs=2, batch_size=8, seed=0)
        dataset = self._make_tiny_dataset(n=20)
        trainer.fit(dataset)

        # Create a test loader
        rng = np.random.default_rng(1)
        X_test = torch.tensor(rng.standard_normal((10, 4)).astype(np.float32))
        y_test = torch.zeros(10, dtype=torch.long)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=4)

        y_pred, y_prob = trainer.predict(test_loader)
        assert y_pred.shape == (10,)
        assert y_prob.shape == (10, 2)
        assert set(y_pred).issubset({0, 1})
        assert np.allclose(y_prob.sum(axis=-1), 1.0, atol=1e-5)

    def test_early_stopping_triggers(self):
        """With patience=1 and min_delta=1.0, early stopping should trigger quickly.

        Setting min_delta=1.0 means val_acc must improve by >1% each epoch,
        which is very unlikely for a random tiny dataset. With patience=2 and
        50 max epochs the trainer stops well before 50 epochs.
        """
        from bci.training.trainer import Trainer

        model = self._make_tiny_model()
        trainer = Trainer(
            model,
            device="cpu",
            epochs=50,
            batch_size=8,
            patience=2,
            min_delta=100.0,  # impossible improvement threshold
            seed=0,
        )
        dataset = self._make_tiny_dataset(n=30)
        result = trainer.fit(dataset)
        # Should have stopped well before 50 epochs
        assert result.final_epoch <= 10

    def test_trainer_history_length(self):
        from bci.training.trainer import Trainer

        model = self._make_tiny_model()
        trainer = Trainer(model, device="cpu", epochs=5, batch_size=8, patience=100, seed=0)
        dataset = self._make_tiny_dataset()
        result = trainer.fit(dataset)
        assert len(result.history) == result.final_epoch

    def test_trainer_history_epoch_numbers(self):
        from bci.training.trainer import Trainer

        model = self._make_tiny_model()
        trainer = Trainer(model, device="cpu", epochs=4, batch_size=8, patience=100, seed=0)
        dataset = self._make_tiny_dataset()
        result = trainer.fit(dataset)
        epochs = [r.epoch for r in result.history]
        assert epochs == list(range(1, result.final_epoch + 1))


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluation:
    def test_compute_metrics_perfect(self):
        from bci.training.evaluation import compute_metrics

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == pytest.approx(100.0)
        assert m["kappa"] == pytest.approx(1.0)
        assert m["f1_macro"] == pytest.approx(1.0)

    def test_compute_metrics_chance(self):
        from bci.training.evaluation import compute_metrics

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])  # all wrong
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == pytest.approx(0.0)
        assert m["kappa"] == pytest.approx(-1.0)

    def test_compute_metrics_with_proba(self):
        from bci.training.evaluation import compute_metrics

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
        m = compute_metrics(y_true, y_pred, y_prob)
        assert "auc_roc" in m
        assert m["auc_roc"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Filter Bank CSP tests
# ---------------------------------------------------------------------------


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
        # 6 bands × 4 components = 24 features when k_best=None
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
        assert fbcsp.n_features_out is None  # before fit
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


# ---------------------------------------------------------------------------
# Filter Bank Riemannian + riemannian_recenter tests
# ---------------------------------------------------------------------------


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
        # 4-ch tangent space = 4*5//2 = 10 features per band; 6 bands = 60
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
        assert fbriem.n_features_out is None  # before fit
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
        """After alignment the mean covariance of train set should be ~identity."""
        from bci.features.riemannian import riemannian_recenter
        from pyriemann.estimation import Covariances
        from pyriemann.utils.mean import mean_riemann

        X_train, _ = _make_eeg_data(n_trials=60, n_channels=4, n_times=256)
        X_test, _ = _make_eeg_data(n_trials=10, n_channels=4, n_times=256, seed=4)
        X_tr_aligned, _ = riemannian_recenter(X_train, X_test, estimator="scm")

        covs = Covariances(estimator="scm").fit_transform(X_tr_aligned.astype(np.float64))
        mean_cov = mean_riemann(covs)
        # Mean should be close to identity
        assert np.allclose(mean_cov, np.eye(4), atol=0.15)
