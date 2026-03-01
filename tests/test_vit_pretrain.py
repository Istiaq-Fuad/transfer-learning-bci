"""ViT pretraining and checkpoint tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch


class TestViTPretraining:
    def test_vit_branch_as_classifier_forward(self):
        from bci.models.vit_branch import ViTBranch
        from bci.utils.config import ModelConfig

        cfg = ModelConfig(vit_pretrained=False, n_classes=2)
        model = ViTBranch(config=cfg, as_feature_extractor=False)
        imgs = torch.rand(4, 9, 224, 224)
        out = model(imgs)
        assert out.shape == (4, 2)

    def test_vit_branch_as_feature_extractor(self):
        from bci.models.vit_branch import ViTBranch
        from bci.utils.config import ModelConfig

        cfg = ModelConfig(vit_pretrained=False)
        model = ViTBranch(config=cfg, as_feature_extractor=True)
        imgs = torch.rand(4, 9, 224, 224)
        out = model(imgs)
        assert out.shape == (4, 192)

    def test_pretrain_saves_checkpoint(self):
        from bci.training.cross_validation import make_synthetic_subject_data
        from bci.utils.config import SpectrogramConfig
        from scripts.pipeline.stage_04_pretrain_vit import pretrain_vit

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
            assert ckpt_path.exists()
            assert "val_accuracy" in metrics
            assert 0.0 <= metrics["val_accuracy"] <= 100.0

    def test_checkpoint_contains_expected_keys(self):
        from bci.training.cross_validation import make_synthetic_subject_data
        from bci.utils.config import SpectrogramConfig
        from scripts.pipeline.stage_04_pretrain_vit import pretrain_vit

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
            assert len(ckpt) > 0
            assert any("patch_embed" in k for k in ckpt)
