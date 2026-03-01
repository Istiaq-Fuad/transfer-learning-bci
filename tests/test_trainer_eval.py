"""Trainer and evaluation tests."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

logging.disable(logging.CRITICAL)


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
        from bci.training.trainer import Trainer

        model = self._make_tiny_model()
        trainer = Trainer(
            model,
            device="cpu",
            epochs=50,
            batch_size=8,
            patience=2,
            min_delta=100.0,
            seed=0,
        )
        dataset = self._make_tiny_dataset(n=30)
        result = trainer.fit(dataset)
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
        y_pred = np.array([1, 1, 0, 0])
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
