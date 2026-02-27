"""Model architectures for MI-EEG classification."""

from bci.models.vit_branch import ViTBranch
from bci.models.efficientnet_branch import EfficientNetBranch
from bci.models.math_branch import MathBranch
from bci.models.fusion import AttentionFusion
from bci.models.dual_branch import DualBranchModel

__all__ = [
    "ViTBranch",
    "EfficientNetBranch",
    "MathBranch",
    "AttentionFusion",
    "DualBranchModel",
]
