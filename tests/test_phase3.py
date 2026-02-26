"""Phase 3 Unit Tests.

Tests cover:
    1. ViT pretraining pipeline (image dataset building, model saving/loading)
    2. Transfer learning: checkpoint load into DualBranchModel
    3. Reduced-data subsampling logic
    4. End-to-end: pretrain -> finetune mini-run
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Helpers shared by multiple tests
# ---------------------------------------------------------------------------

def _make_eeg_data(
    n_trials: int = 16,
    n_channels: int = 3,
    n_times: int = 512,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_trials, n_channels, n_times)).astype(np.float32)
    y = np.array([i % 2 for i in range(n_trials)], dtype=np.int64)
    return X, y


def _make_image_dataset(n: int = 12) -> TensorDataset:
    imgs = torch.rand(n, 3, 224, 224)
    labels = torch.randint(0, 2, (n,))
    return TensorDataset(imgs, labels)


# ---------------------------------------------------------------------------
# 1. ViT pretraining pipeline
# ---------------------------------------------------------------------------

class TestViTPretraining:
    def test_vit_branch_as_classifier_forward(self):
        """ViTBranch with as_feature_extractor=False outputs (B, 2) logits."""
        from bci.models.vit_branch import ViTBranch
        from bci.utils.config import ModelConfig

        cfg = ModelConfig(vit_pretrained=False, n_classes=2)
        model = ViTBranch(config=cfg, as_feature_extractor=False)
        imgs = torch.rand(4, 3, 224, 224)
        out = model(imgs)
        assert out.shape == (4, 2), f"Expected (4,2), got {out.shape}"

    def test_vit_branch_as_feature_extractor(self):
        """ViTBranch with as_feature_extractor=True outputs (B, 1280) features (EfficientNet-B0)."""
        from bci.models.vit_branch import ViTBranch
        from bci.utils.config import ModelConfig

        cfg = ModelConfig(vit_pretrained=False)
        model = ViTBranch(config=cfg, as_feature_extractor=True)
        imgs = torch.rand(4, 3, 224, 224)
        out = model(imgs)
        assert out.shape == (4, 1280), f"Expected (4,1280), got {out.shape}"

    def test_pretrain_saves_checkpoint(self):
        """pretrain_vit saves a .pt file that can be loaded."""
        from bci.training.cross_validation import make_synthetic_subject_data
        from bci.utils.config import SpectrogramConfig
        from scripts.pretrain_physionet import pretrain_vit

        subject_data = make_synthetic_subject_data(n_subjects=2)
        spec_cfg = SpectrogramConfig(n_freqs=8, image_size=(224, 224))

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "vit_test.pt"
            metrics = pretrain_vit(
                subject_data=subject_data,
                channel_names=["C3", "Cz", "C4"],
                sfreq=128.0,
                spec_config=spec_cfg,
                device="cpu",
                epochs=2,
                lr=1e-4,
                batch_size=4,
                warmup_epochs=1,
                patience=100,
                use_imagenet_pretrained=False,
                val_fraction=0.2,
                seed=0,
                checkpoint_path=ckpt_path,
            )
            assert ckpt_path.exists(), "Checkpoint file not created"
            assert "val_accuracy" in metrics
            assert 0.0 <= metrics["val_accuracy"] <= 100.0

    def test_checkpoint_contains_expected_keys(self):
        """Saved checkpoint has ViTBranch backbone keys."""
        from bci.training.cross_validation import make_synthetic_subject_data
        from bci.utils.config import SpectrogramConfig
        from scripts.pretrain_physionet import pretrain_vit

        subject_data = make_synthetic_subject_data(n_subjects=1)
        spec_cfg = SpectrogramConfig(n_freqs=8, image_size=(224, 224))

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "vit_keys_test.pt"
            pretrain_vit(
                subject_data=subject_data,
                channel_names=["C3", "Cz", "C4"],
                sfreq=128.0,
                spec_config=spec_cfg,
                device="cpu",
                epochs=1,
                lr=1e-4,
                batch_size=4,
                warmup_epochs=1,
                patience=100,
                use_imagenet_pretrained=False,
                val_fraction=0.2,
                seed=0,
                checkpoint_path=ckpt_path,
            )
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            # Checkpoint is ViTBranch state_dict — should have backbone keys
            backbone_keys = [k for k in ckpt if k.startswith("backbone.")]
            assert len(backbone_keys) > 0, "No backbone keys in checkpoint"


# ---------------------------------------------------------------------------
# 2. Transfer learning: checkpoint loading into DualBranchModel
# ---------------------------------------------------------------------------

class TestTransferLearning:
    def _make_checkpoint(self, tmpdir: str) -> Path:
        """Create a minimal valid ViTBranch checkpoint."""
        from bci.models.vit_branch import ViTBranch
        from bci.utils.config import ModelConfig

        cfg = ModelConfig(vit_pretrained=False, n_classes=2)
        vit = ViTBranch(config=cfg, as_feature_extractor=False)
        ckpt_path = Path(tmpdir) / "vit.pt"
        torch.save(vit.state_dict(), ckpt_path)
        return ckpt_path

    def test_scratch_model_all_params_trainable(self):
        """Scratch condition: all DualBranchModel params are trainable."""
        from scripts.finetune_bci_iv2a import _build_model

        model = _build_model(
            condition="scratch",
            math_input_dim=10,
            checkpoint_path=None,
            unfreeze_last_n=2,
            fusion="attention",
        )
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        assert n_trainable == n_total, "Scratch: expected all params trainable"

    def test_imagenet_condition_freezes_most_vit(self):
        """ImageNet condition: most ViT backbone params are frozen."""
        from scripts.finetune_bci_iv2a import _build_model

        model = _build_model(
            condition="imagenet",
            math_input_dim=10,
            checkpoint_path=None,
            unfreeze_last_n=2,
            fusion="attention",
        )
        frozen = sum(
            p.numel() for p in model.vit_branch.backbone.parameters()
            if not p.requires_grad
        )
        total_vit = sum(p.numel() for p in model.vit_branch.backbone.parameters())
        # Most ViT params should be frozen (>50%)
        assert frozen > total_vit * 0.5, \
            f"Expected >50% frozen, got {frozen}/{total_vit}"

    def test_transfer_loads_checkpoint(self):
        """Transfer condition loads checkpoint without errors."""
        from scripts.finetune_bci_iv2a import _build_model

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = self._make_checkpoint(tmpdir)
            model = _build_model(
                condition="transfer",
                math_input_dim=10,
                checkpoint_path=ckpt_path,
                unfreeze_last_n=2,
                fusion="attention",
            )
            assert model is not None

    def test_transfer_missing_checkpoint_raises(self):
        """Transfer condition raises FileNotFoundError when checkpoint missing."""
        from scripts.finetune_bci_iv2a import _build_model

        with pytest.raises(FileNotFoundError):
            _build_model(
                condition="transfer",
                math_input_dim=10,
                checkpoint_path=Path("/nonexistent/path.pt"),
                unfreeze_last_n=2,
                fusion="attention",
            )

    def test_transfer_forward_pass(self):
        """Transfer-initialised model can do a forward pass."""
        from scripts.finetune_bci_iv2a import _build_model

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = self._make_checkpoint(tmpdir)
            model = _build_model(
                condition="transfer",
                math_input_dim=20,
                checkpoint_path=ckpt_path,
                unfreeze_last_n=2,
                fusion="attention",
            )
            model.eval()
            with torch.no_grad():
                imgs = torch.rand(2, 3, 224, 224)
                feats = torch.rand(2, 20)
                out = model(imgs, feats)
            assert out.shape == (2, 2), f"Expected (2,2), got {out.shape}"


# ---------------------------------------------------------------------------
# 3. Reduced-data subsampling
# ---------------------------------------------------------------------------

class TestReducedDataSubsampling:
    def test_subsample_respects_fraction(self):
        """StratifiedShuffleSplit gives approximately the right number of samples."""
        from sklearn.model_selection import StratifiedShuffleSplit

        n_total = 100
        X = np.zeros((n_total, 3))
        y = np.array([i % 2 for i in range(n_total)])

        for frac in [0.10, 0.25, 0.50]:
            n_keep = max(2, int(n_total * frac))
            sss = StratifiedShuffleSplit(n_splits=1, train_size=n_keep, random_state=0)
            keep_idx, _ = next(sss.split(X, y))
            assert len(keep_idx) == n_keep, \
                f"frac={frac}: expected {n_keep}, got {len(keep_idx)}"

    def test_subsample_stays_balanced(self):
        """Stratified subsampling preserves class balance."""
        from sklearn.model_selection import StratifiedShuffleSplit

        n_total = 200
        y = np.array([i % 2 for i in range(n_total)])
        X = np.zeros((n_total, 3))

        n_keep = 50
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_keep, random_state=0)
        keep_idx, _ = next(sss.split(X, y))
        y_sub = y[keep_idx]
        assert np.sum(y_sub == 0) == np.sum(y_sub == 1), \
            "Stratified split should be balanced"

    def test_fraction_1_uses_all_data(self):
        """At fraction=1.0, all training data is used."""
        from sklearn.model_selection import StratifiedShuffleSplit

        n_total = 80
        y = np.array([i % 2 for i in range(n_total)])
        X = np.zeros((n_total, 3))

        fraction = 1.0
        if fraction < 1.0:
            n_keep = max(2, int(n_total * fraction))
            sss = StratifiedShuffleSplit(n_splits=1, train_size=n_keep, random_state=0)
            keep_idx, _ = next(sss.split(X, y))
            X_sub = X[keep_idx]
        else:
            X_sub = X
        assert len(X_sub) == n_total


# ---------------------------------------------------------------------------
# 4. End-to-end: mini pretrain -> finetune
# ---------------------------------------------------------------------------

class TestEndToEndPhase3:
    def test_pretrain_then_transfer_finetune(self):
        """Full pipeline: pretrain ViT, save, load, fine-tune dual-branch."""
        from bci.data.dual_branch_builder import DualBranchFoldBuilder
        from bci.training.cross_validation import make_synthetic_subject_data
        from bci.utils.config import SpectrogramConfig
        from scripts.finetune_bci_iv2a import _build_model, _train_and_eval_fold
        from scripts.pretrain_physionet import pretrain_vit

        spec_cfg = SpectrogramConfig(n_freqs=8, image_size=(224, 224))
        source_data = make_synthetic_subject_data(n_subjects=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "vit_pretrained.pt"

            # Step 1: Pretrain
            metrics = pretrain_vit(
                subject_data=source_data,
                channel_names=["C3", "Cz", "C4"],
                sfreq=128.0,
                spec_config=spec_cfg,
                device="cpu",
                epochs=2,
                lr=1e-4,
                batch_size=4,
                warmup_epochs=1,
                patience=100,
                use_imagenet_pretrained=False,
                val_fraction=0.2,
                seed=0,
                checkpoint_path=ckpt_path,
            )
            assert ckpt_path.exists()

            # Step 2: Fine-tune with transfer
            rng = np.random.default_rng(0)
            X = rng.standard_normal((20, 3, 512)).astype(np.float32)
            y = np.array([i % 2 for i in range(20)], dtype=np.int64)

            builder = DualBranchFoldBuilder(
                csp_n_components=4,
                sfreq=128.0,
                channel_names=["C3", "Cz", "C4"],
                spec_config=spec_cfg,
            )
            result = _train_and_eval_fold(
                fold_idx=0,
                subject_id=1,
                X_train=X[:16], y_train=y[:16],
                X_test=X[16:],  y_test=y[16:],
                condition="transfer",
                checkpoint_path=ckpt_path,
                builder=builder,
                device="cpu",
                epochs=2,
                lr=1e-4,
                batch_size=4,
                warmup_epochs=1,
                patience=100,
                unfreeze_last_n=2,
                fusion="attention",
                seed=0,
            )
            assert 0.0 <= result.accuracy <= 100.0

    def test_reduced_data_one_trial_runs(self):
        """_run_one_trial completes for both scratch and transfer conditions."""
        from bci.data.dual_branch_builder import DualBranchFoldBuilder
        from bci.training.cross_validation import make_synthetic_subject_data
        from bci.utils.config import SpectrogramConfig
        from scripts.pretrain_physionet import pretrain_vit
        from scripts.reduced_data_experiment import _run_one_trial

        spec_cfg = SpectrogramConfig(n_freqs=8, image_size=(224, 224))
        source_data = make_synthetic_subject_data(n_subjects=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "vit.pt"
            pretrain_vit(
                subject_data=source_data,
                channel_names=["C3", "Cz", "C4"],
                sfreq=128.0,
                spec_config=spec_cfg,
                device="cpu",
                epochs=1,
                lr=1e-4,
                batch_size=4,
                warmup_epochs=1,
                patience=100,
                use_imagenet_pretrained=False,
                val_fraction=0.2,
                seed=0,
                checkpoint_path=ckpt_path,
            )

            rng = np.random.default_rng(0)
            X = rng.standard_normal((20, 3, 512)).astype(np.float32)
            y = np.array([i % 2 for i in range(20)], dtype=np.int64)

            builder = DualBranchFoldBuilder(
                csp_n_components=4,
                sfreq=128.0,
                channel_names=["C3", "Cz", "C4"],
                spec_config=spec_cfg,
            )

            for condition in ("scratch", "transfer"):
                ckpt = ckpt_path if condition == "transfer" else None
                acc = _run_one_trial(
                    condition=condition,
                    X_train_full=X[:16], y_train_full=y[:16],
                    X_test=X[16:],       y_test=y[16:],
                    fraction=0.5,
                    builder=builder,
                    checkpoint_path=ckpt,
                    device="cpu",
                    epochs=2,
                    lr=1e-4,
                    batch_size=4,
                    warmup_epochs=1,
                    patience=100,
                    unfreeze_last_n=2,
                    seed=0,
                )
                assert 0.0 <= acc <= 100.0, \
                    f"Unexpected accuracy for {condition}: {acc}"
