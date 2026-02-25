"""Unit tests for Phase 2 components.

Tests cover:
    - DualBranchFoldBuilder (no-leakage feature extraction, output shapes)
    - DualBranchModel (forward pass, branch feature extraction)
    - Fusion modules (ConcatFusion, AttentionFusion, GatedFusion)
    - Full fold training integration (tiny synthetic, CPU, 2 epochs)

Run with:
    uv run pytest tests/test_phase2.py -v
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
    n_trials: int = 20,
    n_channels: int = 22,
    n_times: int = 128,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimal discriminative EEG for fast testing."""
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


# ---------------------------------------------------------------------------
# DualBranchFoldBuilder tests
# ---------------------------------------------------------------------------

class TestDualBranchFoldBuilder:
    def _make_builder(self):
        from bci.data.dual_branch_builder import DualBranchFoldBuilder
        from bci.utils.config import SpectrogramConfig

        spec_cfg = SpectrogramConfig(
            n_freqs=8,
            image_size=(224, 224),
            channel_mode="rgb_c3_cz_c4",
        )
        return DualBranchFoldBuilder(
            csp_n_components=4,
            csp_reg="ledoit_wolf",
            riemann_estimator="scm",
            riemann_metric="riemann",
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
        assert img.shape == (3, 224, 224), f"Expected (3,224,224), got {img.shape}"
        assert feat.shape == (math_dim,), f"Expected ({math_dim},), got {feat.shape}"
        assert lbl.shape == ()

    def test_build_fold_math_dim_is_csp_plus_riemannian(self):
        """math_dim should equal n_csp_components + n_channels*(n_channels+1)//2."""
        n_ch = 22
        n_csp = 4
        expected_riemann = n_ch * (n_ch + 1) // 2  # 253
        expected_total = n_csp + expected_riemann

        builder = self._make_builder()
        X_train, y_train = _make_eeg_data(n_trials=16, n_channels=n_ch)
        X_test, y_test = _make_eeg_data(n_trials=8, n_channels=n_ch, seed=1)
        _, _, math_dim = builder.build_fold(X_train, y_train, X_test, y_test)
        assert math_dim == expected_total

    def test_build_fold_images_normalised(self):
        """Images should be float32 in [0, 1]."""
        builder = self._make_builder()
        X_train, y_train = _make_eeg_data(n_trials=16)
        X_test, y_test = _make_eeg_data(n_trials=8, seed=1)
        train_ds, _, _ = builder.build_fold(X_train, y_train, X_test, y_test)
        img, _, _ = train_ds[0]
        assert img.dtype == torch.float32
        assert float(img.min()) >= 0.0
        assert float(img.max()) <= 1.0


# ---------------------------------------------------------------------------
# Fusion module tests
# ---------------------------------------------------------------------------

class TestFusionModules:
    def test_concat_fusion_no_projection(self):
        from bci.models.fusion import ConcatFusion

        f = ConcatFusion(dim_a=10, dim_b=8)
        a = torch.randn(4, 10)
        b = torch.randn(4, 8)
        out = f(a, b)
        assert out.shape == (4, 18)

    def test_concat_fusion_with_projection(self):
        from bci.models.fusion import ConcatFusion

        f = ConcatFusion(dim_a=10, dim_b=8, output_dim=16)
        out = f(torch.randn(4, 10), torch.randn(4, 8))
        assert out.shape == (4, 16)

    def test_attention_fusion_output_shape(self):
        from bci.models.fusion import AttentionFusion

        f = AttentionFusion(dim_a=192, dim_b=128, output_dim=128)
        out = f(torch.randn(4, 192), torch.randn(4, 128))
        assert out.shape == (4, 128)

    def test_attention_fusion_weights_sum_to_one(self):
        """Attention weights must sum to 1 (Softmax)."""
        from bci.models.fusion import AttentionFusion

        f = AttentionFusion(dim_a=8, dim_b=8, output_dim=8)
        # Extract attention weights manually
        proj_a = f.proj_a(torch.randn(2, 8))
        proj_b = f.proj_b(torch.randn(2, 8))
        combined = torch.cat([proj_a, proj_b], dim=-1)
        weights = f.attention(combined)
        assert weights.shape == (2, 2)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5)

    def test_gated_fusion_output_shape(self):
        from bci.models.fusion import GatedFusion

        f = GatedFusion(dim_a=192, dim_b=128, output_dim=128)
        out = f(torch.randn(4, 192), torch.randn(4, 128))
        assert out.shape == (4, 128)

    def test_create_fusion_factory(self):
        from bci.models.fusion import create_fusion
        from bci.utils.config import ModelConfig

        for method in ["attention", "concat", "gated"]:
            cfg = ModelConfig(fusion_method=method, fused_dim=64)
            fusion = create_fusion(dim_a=32, dim_b=24, config=cfg)
            out = fusion(torch.randn(2, 32), torch.randn(2, 24))
            assert out.shape == (2, 64), f"Failed for method={method}: {out.shape}"


# ---------------------------------------------------------------------------
# DualBranchModel tests
# ---------------------------------------------------------------------------

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
        images = torch.randn(2, 3, 224, 224)
        features = torch.randn(2, 16)
        with torch.no_grad():
            logits = model(images, features)
        assert logits.shape == (2, 2)

    def test_all_fusion_methods_run(self):
        for fusion in ["attention", "concat", "gated"]:
            model = self._make_model(fusion=fusion)
            logits = model(torch.randn(2, 3, 224, 224), torch.randn(2, 16))
            assert logits.shape == (2, 2), f"Failed for fusion={fusion}"

    def test_get_branch_features_shapes(self):
        model = self._make_model()
        images = torch.randn(2, 3, 224, 224)
        features = torch.randn(2, 16)
        with torch.no_grad():
            vit_f, math_f, fused_f = model.get_branch_features(images, features)
        assert vit_f.shape == (2, 192)   # ViT-Tiny output
        assert math_f.shape == (2, 128)  # MathBranch output (last hidden dim)
        assert fused_f.shape == (2, 32)  # fused_dim from config

    def test_freeze_vit_backbone(self):
        model = self._make_model()
        model.freeze_vit_backbone(unfreeze_last_n_blocks=1)
        # After freezing, some params should be frozen
        frozen = sum(1 for p in model.vit_branch.parameters() if not p.requires_grad)
        total  = sum(1 for p in model.vit_branch.parameters())
        assert frozen > 0, "Expected some frozen parameters after freeze_vit_backbone"
        assert frozen < total, "Expected not all parameters frozen"

    def test_param_count_reasonable(self):
        """Model should have roughly 5-6 million parameters."""
        model = self._make_model(math_input_dim=259)
        n_params = sum(p.numel() for p in model.parameters())
        assert 5_000_000 < n_params < 7_000_000, f"Unexpected param count: {n_params}"


# ---------------------------------------------------------------------------
# Integration test: train one fold with DualBranchModel
# ---------------------------------------------------------------------------

class TestDualBranchIntegration:
    def test_single_fold_training_runs(self):
        """Train the dual-branch model on one tiny fold for 2 epochs."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from bci.data.dual_branch_builder import DualBranchFoldBuilder
        from bci.models.dual_branch import DualBranchModel
        from bci.training.evaluation import compute_metrics
        from bci.training.trainer import Trainer
        from bci.utils.config import ModelConfig, SpectrogramConfig

        spec_cfg = SpectrogramConfig(n_freqs=8, image_size=(224, 224))
        builder = DualBranchFoldBuilder(
            csp_n_components=4,
            csp_reg="ledoit_wolf",
            riemann_estimator="scm",
            spec_config=spec_cfg,
            sfreq=128.0,
        )
        X_tr, y_tr = _make_eeg_data(n_trials=16, n_channels=22)
        X_te, y_te = _make_eeg_data(n_trials=8,  n_channels=22, seed=1)

        train_ds, test_ds, math_dim = builder.build_fold(X_tr, y_tr, X_te, y_te)

        cfg = ModelConfig(vit_pretrained=False, fused_dim=32, n_classes=2)
        model = DualBranchModel(math_input_dim=math_dim, config=cfg)

        def dual_fwd(batch):
            imgs, feats, labels = batch
            return model(imgs, feats), labels

        trainer = Trainer(model, device="cpu", epochs=2, batch_size=4,
                          patience=100, seed=0)
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
