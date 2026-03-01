"""Dual-branch builder and model tests."""

from __future__ import annotations

import logging

import numpy as np
import torch

logging.disable(logging.CRITICAL)


def _make_eeg_data(
    n_trials: int = 20,
    n_channels: int = 22,
    n_times: int = 128,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_per = n_trials // 2
    X_list, y_list = [], []
    for cls in range(2):
        X = rng.standard_normal((n_per, n_channels, n_times)).astype(np.float32)
        t = np.linspace(0, 1, n_times)
        X[:, cls, :] += np.sin(2 * np.pi * 10 * t) * 3.0
        X_list.append(X)
        y_list.append(np.full(n_per, cls, dtype=np.int64))
    Xc = np.concatenate(X_list)
    yc = np.concatenate(y_list)
    idx = rng.permutation(len(yc))
    return Xc[idx], yc[idx]


class TestDualBranchFoldBuilder:
    def _make_builder(self):
        from bci.data.dual_branch_builder import DualBranchFoldBuilder
        from bci.utils.config import SpectrogramConfig

        spec_cfg = SpectrogramConfig(
            n_freqs=8,
            image_size=(224, 224),
            channel_mode="multichannel",
        )
        return DualBranchFoldBuilder(
            csp_n_components=4,
            csp_k_best=12,
            riemann_estimator="oas",
            riemann_metric="logeuclid",
            riemann_n_components_pca=32,
            spec_config=spec_cfg,
            sfreq=128.0,
            channel_names=["C3", "Cz", "C4"],
        )

    def test_build_fold_returns_correct_types(self):
        builder = self._make_builder()
        X_train, y_train = _make_eeg_data(n_trials=16, n_channels=22)
        X_test, y_test = _make_eeg_data(n_trials=8, n_channels=22, seed=1)
        train_ds, test_ds, math_dim = builder.build_fold(X_train, y_train, X_test, y_test)
        from torch.utils.data import TensorDataset

        assert isinstance(train_ds, TensorDataset)
        assert isinstance(test_ds, TensorDataset)
        assert isinstance(math_dim, int)
        assert math_dim > 0

    def test_build_fold_train_length(self):
        builder = self._make_builder()
        X_train, y_train = _make_eeg_data(n_trials=16)
        X_test, y_test = _make_eeg_data(n_trials=8, seed=1)
        train_ds, test_ds, _ = builder.build_fold(X_train, y_train, X_test, y_test)
        assert len(train_ds) == 16
        assert len(test_ds) == 8

    def test_build_fold_item_shapes(self):
        builder = self._make_builder()
        X_train, y_train = _make_eeg_data(n_trials=16)
        X_test, y_test = _make_eeg_data(n_trials=8, seed=1)
        train_ds, test_ds, math_dim = builder.build_fold(X_train, y_train, X_test, y_test)

        img, feat, lbl = train_ds[0]
        assert img.shape == (9, 224, 224)
        assert feat.shape == (math_dim,)
        assert lbl.shape == ()

    def test_build_fold_math_dim_is_csp_plus_riemannian(self):
        builder = self._make_builder()
        X_train, y_train = _make_eeg_data(n_trials=16, n_channels=22)
        X_test, y_test = _make_eeg_data(n_trials=8, n_channels=22, seed=1)
        train_ds, test_ds, math_dim = builder.build_fold(X_train, y_train, X_test, y_test)

        assert math_dim > 0
        _, feat_train, _ = train_ds[0]
        _, feat_test, _ = test_ds[0]
        assert feat_train.shape == (math_dim,)
        assert feat_test.shape == (math_dim,)

    def test_build_fold_images_normalised(self):
        builder = self._make_builder()
        X_train, y_train = _make_eeg_data(n_trials=16)
        X_test, y_test = _make_eeg_data(n_trials=8, seed=1)
        train_ds, _, _ = builder.build_fold(X_train, y_train, X_test, y_test)
        img, _, _ = train_ds[0]
        assert img.dtype == torch.float32
        assert torch.isfinite(img).all()
        assert float(img.max()) < 10.0


class TestDualBranchModel:
    def _make_model(self, fusion: str = "attention", math_input_dim: int = 16):
        from bci.models.dual_branch import DualBranchModel
        from bci.utils.config import ModelConfig

        cfg = ModelConfig(
            vit_pretrained=False,
            fusion_method=fusion,
            fused_dim=32,
            classifier_hidden_dim=16,
            n_classes=2,
        )
        return DualBranchModel(math_input_dim=math_input_dim, config=cfg)

    def test_forward_pass_shape(self):
        model = self._make_model()
        images = torch.randn(2, 9, 224, 224)
        features = torch.randn(2, 16)
        with torch.no_grad():
            logits = model(images, features)
        assert logits.shape == (2, 2)

    def test_all_fusion_methods_run(self):
        for fusion in ["attention", "concat", "gated"]:
            model = self._make_model(fusion=fusion)
            logits = model(torch.randn(2, 9, 224, 224), torch.randn(2, 16))
            assert logits.shape == (2, 2)

    def test_get_branch_features_shapes(self):
        model = self._make_model()
        images = torch.randn(2, 9, 224, 224)
        features = torch.randn(2, 16)
        with torch.no_grad():
            vit_f, math_f, fused_f = model.get_branch_features(images, features)
        assert vit_f.shape == (2, 192)
        assert math_f.shape == (2, 128)
        assert fused_f.shape == (2, 32)

    def test_freeze_vit_backbone(self):
        model = self._make_model()
        model.freeze_vit_backbone(unfreeze_last_n_blocks=1)
        frozen = sum(1 for p in model.vit_branch.parameters() if not p.requires_grad)
        total = sum(1 for p in model.vit_branch.parameters())
        assert frozen > 0
        assert frozen < total

    def test_param_count_reasonable(self):
        model = self._make_model(math_input_dim=259)
        n_params = sum(p.numel() for p in model.parameters())
        assert 3_000_000 < n_params < 7_000_000


class TestDualBranchIntegration:
    def test_single_fold_training_runs(self):
        from torch.utils.data import DataLoader

        from bci.data.dual_branch_builder import DualBranchFoldBuilder
        from bci.models.dual_branch import DualBranchModel
        from bci.training.evaluation import compute_metrics
        from bci.training.trainer import Trainer
        from bci.utils.config import ModelConfig, SpectrogramConfig

        spec_cfg = SpectrogramConfig(n_freqs=8, image_size=(224, 224))
        builder = DualBranchFoldBuilder(
            csp_n_components=4,
            csp_k_best=12,
            riemann_estimator="oas",
            riemann_metric="logeuclid",
            riemann_n_components_pca=32,
            spec_config=spec_cfg,
            sfreq=128.0,
        )
        X_tr, y_tr = _make_eeg_data(n_trials=16, n_channels=22, n_times=384)
        X_te, y_te = _make_eeg_data(n_trials=8, n_channels=22, n_times=384, seed=1)

        train_ds, test_ds, math_dim = builder.build_fold(X_tr, y_tr, X_te, y_te)

        cfg = ModelConfig(vit_pretrained=False, fused_dim=32, n_classes=2)
        model = DualBranchModel(math_input_dim=math_dim, config=cfg)

        def dual_fwd(batch):
            imgs, feats, labels = batch
            return model(imgs, feats), labels

        trainer = Trainer(model, device="cpu", epochs=2, batch_size=4, patience=100, seed=0)
        result = trainer.fit(train_ds, forward_fn=dual_fwd)
        assert result.final_epoch == 2

        test_loader = DataLoader(test_ds, batch_size=4)
        y_pred, y_prob = trainer.predict(test_loader, forward_fn=dual_fwd)
        assert y_pred.shape == (8,)
        assert y_prob.shape == (8, 2)
        assert np.allclose(y_prob.sum(axis=1), 1.0, atol=1e-4)

        m = compute_metrics(y_te, y_pred, y_prob)
        assert "accuracy" in m
        assert "kappa" in m
        assert "auc_roc" in m
