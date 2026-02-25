"""Mathematical feature branch (MLP for handcrafted features).

Processes concatenated CSP + Riemannian tangent space features
through a small MLP to produce a fixed-dimensional feature vector
for fusion with the ViT branch.

Architecture:
    Input: (batch, csp_dim + riemannian_dim)
    -> Linear(input_dim, 256) -> BatchNorm -> ReLU -> Dropout
    -> Linear(256, 128) -> BatchNorm -> ReLU -> Dropout
    -> Output: (batch, 128)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from bci.utils.config import ModelConfig

logger = logging.getLogger(__name__)


class MathBranch(nn.Module):
    """MLP for processing handcrafted EEG features.

    Takes concatenated CSP and Riemannian features as input
    and produces a fixed-dimensional embedding.

    Args:
        input_dim: Dimensionality of the input feature vector
            (csp_features + riemannian_features).
        config: Model configuration with MLP parameters.
    """

    def __init__(
        self,
        input_dim: int,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.input_dim = input_dim

        hidden_dims = self.config.math_hidden_dims
        drop_rate = self.config.math_drop_rate

        # Build MLP layers dynamically from config
        layers: list[nn.Module] = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop_rate),
            ])
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim

        logger.info(
            "MathBranch: input_dim=%d -> hidden=%s -> output_dim=%d",
            input_dim, hidden_dims, self.output_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Feature vector of shape (batch, output_dim).
        """
        return self.mlp(x)
