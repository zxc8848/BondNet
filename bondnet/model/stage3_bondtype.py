"""
Stage 3: Bond Type Classification.

The continuous BO regression head has been removed. This module keeps the
historical tuple return signature by returning an empty tensor placeholder
for code paths that still expect a second value.
"""

import torch
from torch import nn


class BondTypePredictor(nn.Module):
    """
    Discrete bond type prediction head.

    Args:
        hidden_size: Dimension of input edge features (from backbone).
        mlp_hidden: Hidden dimension for MLP heads.
        num_bond_types: Number of discrete bond types (default 3: single/double/triple).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        mlp_hidden: int = 64,
        num_bond_types: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_bond_types = num_bond_types

        # Shared feature transform
        self.shared_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Discrete classification head
        self.cls_head = nn.Sequential(
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_bond_types),
        )

    def forward(self, edge_feat: torch.Tensor):
        """
        Args:
            edge_feat: (E', H) edge features for non-aromatic bonded edges.

        Returns:
            cls_logits: (E', num_bond_types) classification logits.
            bo_pred: (0,) empty placeholder for backward compatibility.
        """
        shared = self.shared_mlp(edge_feat)
        cls_logits = self.cls_head(shared)
        bo_pred = torch.zeros(0, dtype=shared.dtype, device=shared.device)
        return cls_logits, bo_pred

    def predict(self, edge_feat: torch.Tensor):
        """
        Discrete bond type prediction.

        Returns:
            bond_types: (E',) int tensor — 0=single, 1=double, 2=triple.
            bo_pred: (0,) empty placeholder for backward compatibility.
        """
        cls_logits, bo_pred = self.forward(edge_feat)
        bond_types = cls_logits.argmax(dim=-1)
        return bond_types, bo_pred
