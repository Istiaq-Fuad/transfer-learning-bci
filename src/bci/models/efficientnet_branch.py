"""EfficientNet-B0 image backbone branch for spectrogram-based classification.

EfficientNet-B0 processes CWT spectrogram images of EEG signals and produces
a 1280-dimensional feature vector used by the DualBranchModel.

Architecture (feature-extractor mode):
    Input: (batch, 3, 224, 224) spectrogram images
    -> EfficientNet-B0 (pretrained ImageNet)
    -> 1280-dim feature vector

This module is the EfficientNet-specific counterpart to vit_branch.py.
The class interface is intentionally identical so that DualBranchModel can
swap between the two without changes to the fusion or classifier layers.

Canonical architecture dimensions for EfficientNet-B0
(used by pipeline stages and run_all.sh to avoid magic numbers):

    FEATURE_DIM          = 1280   # backbone output before fusion
    DEFAULT_FUSED_DIM    = 256    # after AttentionFusion
    DEFAULT_CLS_HIDDEN   = 128    # ClassifierHead hidden layer
    BACKBONE_SHORT       = "efficientnet"  # short tag for filenames
"""

from __future__ import annotations

import logging

import timm
import torch
import torch.nn as nn

from bci.utils.config import ModelConfig
from bci.models.vit_branch import (
    _get_classifier_attr,
    _get_feature_dim,
    _get_block_list,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical architecture constants for EfficientNet-B0
# Import these in pipeline scripts instead of duplicating the dicts.
# ---------------------------------------------------------------------------

#: Short tag used in output filenames (e.g. "real_dual_branch_attention_efficientnet.json")
BACKBONE_SHORT: str = "efficientnet"

#: EfficientNet-B0 backbone output dimension (before fusion)
FEATURE_DIM: int = 1280

#: Default fused feature dimension after AttentionFusion
DEFAULT_FUSED_DIM: int = 256

#: Default ClassifierHead hidden layer dimension
DEFAULT_CLS_HIDDEN: int = 128

#: The timm model name this module is designed for
MODEL_NAME: str = "efficientnet_b0"


# ---------------------------------------------------------------------------
# EfficientNetBranch
# ---------------------------------------------------------------------------

class EfficientNetBranch(nn.Module):
    """EfficientNet-B0 image feature extractor for EEG spectrograms.

    Loads a pretrained EfficientNet-B0 backbone and replaces the
    classification head with either an identity (for feature extraction) or
    a new linear classifier.

    The public interface is identical to ViTBranch so that DualBranchModel
    can use either branch transparently.

    Args:
        config: Model configuration. ``config.vit_model_name`` should be
            ``"efficientnet_b0"``, but any EfficientNet variant supported by
            timm will also work.
        as_feature_extractor: If True, removes the classification head and
            returns the raw 1280-dim feature vector. If False, attaches a new
            linear head for ``n_classes`` classification (used during
            standalone pretraining on PhysioNet).
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        as_feature_extractor: bool = True,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig(vit_model_name=MODEL_NAME)

        # Load backbone from timm
        self.backbone = timm.create_model(
            self.config.vit_model_name,
            pretrained=self.config.vit_pretrained,
            drop_rate=self.config.vit_drop_rate,
        )

        # Detect the classification head attribute and feature dimension
        head_attr = _get_classifier_attr(self.backbone)
        if head_attr is None:
            raise ValueError(
                f"Cannot find a classification head in timm model "
                f"'{self.config.vit_model_name}'. "
                "Expected one of: head, classifier, fc, head_fc."
            )
        self._head_attr = head_attr
        self.feature_dim = _get_feature_dim(self.backbone, head_attr)

        if as_feature_extractor:
            # Replace head with identity -> output is feature vector
            setattr(self.backbone, head_attr, nn.Identity())
            logger.info(
                "EfficientNet branch (feature extractor): %s [head=%s], feature_dim=%d",
                self.config.vit_model_name, head_attr, self.feature_dim,
            )
        else:
            # Attach a new classification head for standalone training
            setattr(
                self.backbone, head_attr,
                nn.Linear(self.feature_dim, self.config.n_classes),
            )
            logger.info(
                "EfficientNet branch (classifier): %s [head=%s], n_classes=%d",
                self.config.vit_model_name, head_attr, self.config.n_classes,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the EfficientNet backbone.

        Args:
            x: Input tensor of shape (batch, 3, 224, 224).

        Returns:
            Feature vector of shape (batch, feature_dim) if feature extractor,
            or logits of shape (batch, n_classes) if classifier.
        """
        return self.backbone(x)

    def freeze_backbone(self, unfreeze_last_n_blocks: int = 2) -> None:
        """Freeze backbone parameters for transfer learning.

        Keeps the last N MBConv stages and the classification head unfrozen.

        Args:
            unfreeze_last_n_blocks: Number of backbone stages from the end
                to keep trainable. Default is 2.
        """
        # Freeze everything first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last N blocks (EfficientNet exposes backbone.blocks)
        blocks = _get_block_list(self.backbone)
        if blocks:
            for block in blocks[-unfreeze_last_n_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
        else:
            logger.warning(
                "freeze_backbone: could not find block list for '%s'; "
                "only the head will be unfrozen.",
                self.config.vit_model_name,
            )

        # Always unfreeze the classification head
        head = getattr(self.backbone, self._head_attr, None)
        if head is not None:
            for param in head.parameters():
                param.requires_grad = True

        # EfficientNet-specific: unfreeze bn2 / conv_head after last stage
        for attr in ("conv_head", "bn2"):
            if hasattr(self.backbone, attr):
                for param in getattr(self.backbone, attr).parameters():
                    param.requires_grad = True

        frozen = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
        total  = sum(1 for p in self.backbone.parameters())
        logger.info(
            "Frozen %d/%d parameters (unfroze last %d blocks + head)",
            frozen, total, unfreeze_last_n_blocks,
        )

    def get_backbone_params(self) -> list[nn.Parameter]:
        """Return parameters that belong to the frozen/pretrained backbone."""
        head = getattr(self.backbone, self._head_attr, None)
        head_ids = {id(p) for p in (head.parameters() if head else [])}
        return [p for p in self.backbone.parameters() if id(p) not in head_ids]

    def get_head_params(self) -> list[nn.Parameter]:
        """Return parameters that belong to the classification head."""
        head = getattr(self.backbone, self._head_attr, None)
        if head is None:
            return []
        return list(head.parameters())

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
