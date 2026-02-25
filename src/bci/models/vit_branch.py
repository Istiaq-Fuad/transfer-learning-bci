"""Vision Transformer (ViT) branch for spectrogram-based classification.

Uses a pretrained ViT-Tiny from the `timm` library as a feature extractor.
The ViT processes CWT spectrogram images of EEG signals, capturing
spectral-temporal patterns through self-attention.

Architecture:
    Input: (batch, 3, 224, 224) spectrogram images
    -> ViT-Tiny (pretrained on ImageNet, fine-tuned)
    -> Feature vector of dim 192
"""

from __future__ import annotations

import logging

import timm
import torch
import torch.nn as nn

from bci.utils.config import ModelConfig

logger = logging.getLogger(__name__)


class ViTBranch(nn.Module):
    """ViT-based feature extractor for EEG spectrograms.

    Loads a pretrained ViT-Tiny and replaces the classification head
    with either an identity (for feature extraction) or a new classifier.

    Args:
        config: Model configuration with ViT parameters.
        as_feature_extractor: If True, removes the classification head
            and returns the feature vector. If False, adds a new head
            for n_classes classification.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        as_feature_extractor: bool = True,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        # Load pretrained ViT from timm
        self.backbone = timm.create_model(
            self.config.vit_model_name,
            pretrained=self.config.vit_pretrained,
            drop_rate=self.config.vit_drop_rate,
        )

        # Get the feature dimension from the model
        self.feature_dim = self.backbone.head.in_features

        if as_feature_extractor:
            # Remove classification head -> output is feature vector
            self.backbone.head = nn.Identity()
            logger.info(
                "ViT branch (feature extractor): %s, feature_dim=%d",
                self.config.vit_model_name, self.feature_dim,
            )
        else:
            # Replace with new classification head
            self.backbone.head = nn.Linear(
                self.feature_dim, self.config.n_classes,
            )
            logger.info(
                "ViT branch (classifier): %s, n_classes=%d",
                self.config.vit_model_name, self.config.n_classes,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ViT backbone.

        Args:
            x: Input tensor of shape (batch, 3, 224, 224).

        Returns:
            Feature vector of shape (batch, feature_dim) if feature extractor,
            or logits of shape (batch, n_classes) if classifier.
        """
        return self.backbone(x)

    def freeze_backbone(self, unfreeze_last_n_blocks: int = 2) -> None:
        """Freeze ViT backbone parameters for transfer learning.

        Keeps the last N transformer blocks and the head unfrozen.

        Args:
            unfreeze_last_n_blocks: Number of transformer blocks from the end
                to keep trainable. Default is 2.
        """
        # Freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last N blocks
        blocks = list(self.backbone.blocks)
        for block in blocks[-unfreeze_last_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        # Always unfreeze the head and norm
        if hasattr(self.backbone, "head"):
            for param in self.backbone.head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

        frozen = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
        total = sum(1 for p in self.backbone.parameters())
        logger.info(
            "Frozen %d/%d parameters (unfroze last %d blocks + head)",
            frozen, total, unfreeze_last_n_blocks,
        )

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count model parameters.

        Args:
            trainable_only: If True, count only trainable parameters.

        Returns:
            Number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
