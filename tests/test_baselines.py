"""Baseline model tests (CSP+LDA and Riemannian+LDA)."""

from __future__ import annotations

import logging

import numpy as np

logging.disable(logging.CRITICAL)


def _make_eeg_data(
    n_trials: int = 30,
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


class TestBaselines:
    def test_csp_lda_predict_fn_output_shape(self):
        from scripts.pipeline.stage_02_baseline_a import make_predict_fn

        predict_fn = make_predict_fn(n_components=4)
        X_train, y_train = _make_eeg_data(n_trials=30)
        X_test, _ = _make_eeg_data(n_trials=10, seed=99)
        y_pred, y_prob = predict_fn(X_train, y_train, X_test)
        assert y_pred.shape == (10,)
        assert y_prob.shape == (10, 2)
        assert set(y_pred).issubset({0, 1})

    def test_riemannian_lda_predict_fn_output_shape(self):
        from scripts.pipeline.stage_03_baseline_b import make_predict_fn

        predict_fn = make_predict_fn(estimator="scm", metric="riemann")
        X_train, y_train = _make_eeg_data(n_trials=30)
        X_test, _ = _make_eeg_data(n_trials=10, seed=99)
        y_pred, y_prob = predict_fn(X_train, y_train, X_test)
        assert y_pred.shape == (10,)
        assert y_prob.shape == (10, 2)
        assert set(y_pred).issubset({0, 1})
