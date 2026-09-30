"""
Stage 1: Connectivity Prediction.

Binary classifier on edge features h_ij to predict P(bond exists | i, j).
Edges with P < 0.5 are pruned; remaining edges proceed to Stage 2/3.
"""

import torch
from torch import nn


class ConnectivityPredictor(nn.Module):
    """
    Lightweight MLP binary classifier for bond existence.

    Takes per-edge invariant features h_ij from the PaiNN backbone
    and predicts a scalar logit for each candidate edge.

    Args:
        hidden_size: Dimension of input edge features (must match backbone output).
        mlp_hidden: Hidden dimension of the classifier MLP.
        dropout: Dropout rate.
    """

    def __init__(self, hidden_size: int = 128, mlp_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, edge_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_feat: (E, H) per-edge invariant features from backbone.

        Returns:
            logits: (E,) binary logits for bond existence.
        """
        return self.classifier(edge_feat).squeeze(-1)

    def predict(self, edge_feat: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Binary prediction with threshold.

        Returns:
            mask: (E,) boolean tensor — True where bond is predicted.
        """
        logits = self.forward(edge_feat)
        probs = torch.sigmoid(logits)
        return probs >= threshold
