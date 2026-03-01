"""Transfer learning and reduced-data tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


class TestTransferLearning:
    def _make_checkpoint(self, tmpdir: str) -> Path:
        from bci.models.vit_branch import ViTBranch
        from bci.utils.config import ModelConfig

        cfg = ModelConfig(vit_pretrained=False, n_classes=2)
        vit = ViTBranch(config=cfg, as_feature_extractor=False)
        ckpt_path = Path(tmpdir) / "vit.pt"
        torch.save(vit.backbone.state_dict(), ckpt_path)
        return ckpt_path

    def test_transfer_loads_checkpoint(self):
        from scripts.pipeline.stage_06_dual_branch import _build_model

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
        from scripts.pipeline.stage_06_dual_branch import _build_model

        with pytest.raises(FileNotFoundError):
            _build_model(
                condition="transfer",
                math_input_dim=10,
                checkpoint_path=Path("/nonexistent/path.pt"),
                unfreeze_last_n=2,
                fusion="attention",
            )

    def test_transfer_forward_pass(self):
        from scripts.pipeline.stage_06_dual_branch import _build_model

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
                imgs = torch.rand(2, 9, 224, 224)
                feats = torch.rand(2, 20)
                out = model(imgs, feats)
            assert out.shape == (2, 2)


class TestReducedDataSubsampling:
    def test_subsample_respects_fraction(self):
        from sklearn.model_selection import StratifiedShuffleSplit

        n_total = 100
        X = np.zeros((n_total, 3))
        y = np.array([i % 2 for i in range(n_total)])

        for frac in [0.10, 0.25, 0.50]:
            n_keep = max(2, int(n_total * frac))
            sss = StratifiedShuffleSplit(n_splits=1, train_size=n_keep, random_state=0)
            keep_idx, _ = next(sss.split(X, y))
            assert len(keep_idx) == n_keep

    def test_subsample_stays_balanced(self):
        from sklearn.model_selection import StratifiedShuffleSplit

        n_total = 200
        y = np.array([i % 2 for i in range(n_total)])
        X = np.zeros((n_total, 3))

        n_keep = 50
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_keep, random_state=0)
        keep_idx, _ = next(sss.split(X, y))
        y_sub = y[keep_idx]
        assert np.sum(y_sub == 0) == np.sum(y_sub == 1)

    def test_fraction_1_uses_all_data(self):
        n_total = 80
        y = np.array([i % 2 for i in range(n_total)])
        X = np.zeros((n_total, 3))

        fraction = 1.0
        if fraction < 1.0:
            n_keep = max(2, int(n_total * fraction))
            from sklearn.model_selection import StratifiedShuffleSplit

            sss = StratifiedShuffleSplit(n_splits=1, train_size=n_keep, random_state=0)
            keep_idx, _ = next(sss.split(X, y))
            X_sub = X[keep_idx]
        else:
            X_sub = X
        assert len(X_sub) == n_total

    def test_reduced_data_one_trial_runs(self):
        from bci.data.dual_branch_builder import DualBranchFoldBuilder
        from bci.training.cross_validation import make_synthetic_subject_data
        from bci.utils.config import SpectrogramConfig
        from scripts.pipeline.stage_04_pretrain_vit import pretrain_vit
        from scripts.pipeline.stage_07_reduced_data import _run_one_trial

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

            acc = _run_one_trial(
                X_train_full=X[:16],
                y_train_full=y[:16],
                X_test=X[16:],
                y_test=y[16:],
                fraction=0.5,
                builder=builder,
                checkpoint_path=ckpt_path,
                device="cpu",
                epochs=2,
                lr=1e-4,
                batch_size=4,
                warmup_epochs=1,
                patience=100,
                seed=0,
            )
            assert 0.0 <= acc <= 100.0
