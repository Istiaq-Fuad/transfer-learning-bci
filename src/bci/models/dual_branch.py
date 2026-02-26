"""Dual-Branch Model: EfficientNet-B0 + Math features with fusion.

The complete architecture for MI-EEG classification:
    Branch A: CWT Spectrogram -> EfficientNet-B0 -> 1280-dim features
    Branch B: CSP + Riemannian features -> MLP -> 128-dim features
    Fusion: Attention-based fusion -> 256-dim
    Classifier: MLP(256 -> 128 -> 2) -> Softmax
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from bci.models.fusion import create_fusion
from bci.models.math_branch import MathBranch
from bci.models.vit_branch import ViTBranch
from bci.utils.config import ModelConfig

logger = logging.getLogger(__name__)


class ClassifierHead(nn.Module):
    """Lightweight classification head.

    Takes fused features and produces class logits.

    Args:
        input_dim: Dimension of fused features.
        hidden_dim: Hidden layer dimension.
        n_classes: Number of output classes.
        drop_rate: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_classes: int = 2,
        drop_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class DualBranchModel(nn.Module):
    """Full dual-branch model for MI-EEG classification.

    Combines a ViT-based spectrogram branch with a handcrafted feature
    branch, fuses them with attention, and classifies.

    Args:
        math_input_dim: Dimensionality of the handcrafted feature vector
            (CSP features + Riemannian features).
        config: Model configuration.
    """

    def __init__(
        self,
        math_input_dim: int,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        # Branch A: ViT for spectrograms
        self.vit_branch = ViTBranch(
            config=self.config, as_feature_extractor=True,
        )

        # Branch B: MLP for handcrafted features
        self.math_branch = MathBranch(
            input_dim=math_input_dim, config=self.config,
        )

        # Fusion layer
        self.fusion = create_fusion(
            dim_a=self.vit_branch.feature_dim,
            dim_b=self.math_branch.output_dim,
            config=self.config,
        )

        # Classification head
        self.classifier = ClassifierHead(
            input_dim=self.config.fused_dim,
            hidden_dim=self.config.classifier_hidden_dim,
            n_classes=self.config.n_classes,
            drop_rate=self.config.vit_drop_rate,
        )

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "DualBranchModel: %d total params (%d trainable)",
            total_params, trainable_params,
        )

    def forward(
        self,
        images: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the dual-branch model.

        Args:
            images: Spectrogram images of shape (batch, 3, 224, 224).
            features: Handcrafted features of shape (batch, math_input_dim).

        Returns:
            Class logits of shape (batch, n_classes).
        """
        # Branch A: ViT features
        vit_features = self.vit_branch(images)

        # Branch B: Math features
        math_features = self.math_branch(features)

        # Fusion
        fused = self.fusion(vit_features, math_features)

        # Classification
        logits = self.classifier(fused)

        return logits

    def freeze_vit_backbone(self, unfreeze_last_n_blocks: int = 2) -> None:
        """Freeze the ViT backbone for transfer learning.

        Args:
            unfreeze_last_n_blocks: Number of ViT blocks to keep trainable.
        """
        self.vit_branch.freeze_backbone(unfreeze_last_n_blocks)

    def get_branch_features(
        self,
        images: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract intermediate features from both branches (for analysis).

        Args:
            images: Spectrogram images.
            features: Handcrafted features.

        Returns:
            Tuple of (vit_features, math_features, fused_features).
        """
        vit_features = self.vit_branch(images)
        math_features = self.math_branch(features)
        fused = self.fusion(vit_features, math_features)
        return vit_features, math_features, fused
