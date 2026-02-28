"""Image backbone branch for spectrogram-based classification.

Supports any timm model as a feature extractor or standalone classifier.
The backbone processes CWT spectrogram images of EEG signals.

Architecture (feature-extractor mode):
    Input: (batch, 3, 224, 224) spectrogram images
    -> timm backbone (pretrained)
    -> Feature vector of dim `feature_dim`
       - ViT-Tiny (default)  -> 192

Canonical architecture constants for ViT-Tiny
(used by pipeline stages and run_all.sh to avoid magic numbers):

    FEATURE_DIM          = 192    # ViT-Tiny backbone output before fusion
    DEFAULT_FUSED_DIM    = 128    # after AttentionFusion
    DEFAULT_CLS_HIDDEN   = 64     # ClassifierHead hidden layer
    BACKBONE_SHORT       = "vit"  # short tag for filenames
"""

from __future__ import annotations

import logging

import timm
import torch
import torch.nn as nn

from bci.utils.config import ModelConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical architecture constants for ViT-Tiny
# Import these in pipeline scripts instead of duplicating the dicts.
# ---------------------------------------------------------------------------

#: Short tag used in output filenames (e.g. "real_dual_branch_attention_vit.json")
BACKBONE_SHORT: str = "vit"

#: ViT-Tiny backbone output dimension (before fusion)
FEATURE_DIM: int = 192

#: Default fused feature dimension after AttentionFusion
DEFAULT_FUSED_DIM: int = 128

#: Default ClassifierHead hidden layer dimension
DEFAULT_CLS_HIDDEN: int = 64

#: The timm model name this module is designed for
MODEL_NAME: str = "vit_tiny_patch16_224"

# ---------------------------------------------------------------------------
# Helpers to detect the classification head and feature dim in a timm model
# in a model-agnostic way.
# ---------------------------------------------------------------------------


def _get_classifier_attr(backbone: nn.Module) -> str | None:
    """Return the name of the classification head attribute, or None."""
    for attr in ("head", "classifier", "fc", "head_fc"):
        if hasattr(backbone, attr):
            layer = getattr(backbone, attr)
            # Must be an actual layer (not None / Identity) to be "real"
            if isinstance(layer, (nn.Linear, nn.Sequential)):
                return attr
    return None


def _get_feature_dim(backbone: nn.Module, head_attr: str) -> int:
    """Return in_features of the classifier head."""
    head = getattr(backbone, head_attr)
    if isinstance(head, nn.Linear):
        return head.in_features
    if isinstance(head, nn.Sequential):
        # Walk to the last Linear in the Sequential
        for layer in reversed(list(head.children())):
            if isinstance(layer, nn.Linear):
                return layer.in_features
    raise ValueError(f"Cannot determine feature_dim from head attribute '{head_attr}'")


def _get_block_list(backbone: nn.Module) -> list[nn.Module]:
    """Return the list of backbone blocks for partial un-freezing.

    - ViT-style models expose `backbone.blocks` (ModuleList of transformer blocks)
    - Falls back to `backbone.features` then an empty list if not found.
    """
    if hasattr(backbone, "blocks"):
        blocks_attr = backbone.blocks
        if isinstance(blocks_attr, (nn.ModuleList, nn.Sequential)):
            return list(blocks_attr.children())
    if hasattr(backbone, "features") and isinstance(backbone.features, nn.Sequential):
        return list(backbone.features.children())
    return []


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------


class ViTBranch(nn.Module):
    """Timm-based image feature extractor for EEG spectrograms.

    Loads a pretrained backbone and replaces the classification head with
    either an identity (for feature extraction) or a new linear classifier.

    Args:
        config: Model configuration with image-branch parameters.
        as_feature_extractor: If True, removes the classification head and
            returns the raw feature vector. If False, attaches a new head
            for `n_classes` classification.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        as_feature_extractor: bool = True,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        # Load backbone from timm
        self.backbone = timm.create_model(
            self.config.vit_model_name,
            pretrained=self.config.vit_pretrained,
            drop_rate=self.config.vit_drop_rate,
            in_chans=self.config.in_chans,
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
                "Image branch (feature extractor): %s [head=%s], feature_dim=%d",
                self.config.vit_model_name,
                head_attr,
                self.feature_dim,
            )
        else:
            # Attach a new classification head
            setattr(
                self.backbone,
                head_attr,
                nn.Linear(self.feature_dim, self.config.n_classes),
            )
            logger.info(
                "Image branch (classifier): %s [head=%s], n_classes=%d",
                self.config.vit_model_name,
                head_attr,
                self.config.n_classes,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone.

        Args:
            x: Input tensor of shape (batch, 3, 224, 224).

        Returns:
            Feature vector of shape (batch, feature_dim) if feature extractor,
            or logits of shape (batch, n_classes) if classifier.
        """
        return self.backbone(x)

    def freeze_backbone(self, unfreeze_last_n_blocks: int = 2) -> None:
        """Freeze backbone parameters for transfer learning.

        Keeps the last N blocks and the classification head unfrozen.

        Args:
            unfreeze_last_n_blocks: Number of backbone blocks from the end
                to keep trainable. Default is 2.
        """
        # Freeze everything first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last N blocks
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

        # ViT-specific: unfreeze the final layer norm
        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

        frozen = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
        total = sum(1 for p in self.backbone.parameters())
        logger.info(
            "Frozen %d/%d parameters (unfroze last %d blocks + head)",
            frozen,
            total,
            unfreeze_last_n_blocks,
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
