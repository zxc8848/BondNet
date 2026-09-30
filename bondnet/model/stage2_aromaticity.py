"""
Legacy experimental ring-level aromaticity detector (not part of BondNet v2).

The trained two-stage pipeline uses ``bond_type_gnn.BondTypeGNN`` for Stage 2.
This module is retained only for historical reproducibility and is not imported
by the training or inference commands in the revised manuscript.

Given the predicted connectivity graph from Stage 1, detect rings and classify
each ring as aromatic or non-aromatic. If a ring is aromatic, all its edges
are labeled "aromatic" — completely bypassing Kekulé ambiguity.

Features for ring classification:
  - Mean and variance of bond lengths within the ring
  - Planarity score (RMSD from best-fit plane)
  - Ring size (one-hot encoded, sizes 3–8)
  - Atom-type composition (bag-of-atoms encoding)
"""

import torch
from torch import nn
import numpy as np


class RingAromaticityPredictor(nn.Module):
    """
    Ring-level aromaticity classifier.

    Takes ring-level features (aggregated from atoms & edges in each ring)
    and predicts P(aromatic | ring).

    Args:
        hidden_size: Hidden dimension from backbone edge features.
        max_ring_size: Maximum ring size to encode (default 8).
        num_atom_types: Number of atom types for bag-of-atoms encoding.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        max_ring_size: int = 8,
        min_ring_size: int = 3,
        num_atom_types: int = 10,
    ):
        super().__init__()
        self.max_ring_size = max_ring_size
        self.min_ring_size = min_ring_size
        self.num_ring_sizes = max_ring_size - min_ring_size + 1

        # Ring feature dimension:
        #   2 (mean/var bond length) + 1 (planarity) +
        #   num_ring_sizes (ring size one-hot) +
        #   num_atom_types (atom composition) +
        #   hidden_size (mean of edge features in ring)
        ring_feat_dim = 2 + 1 + self.num_ring_sizes + num_atom_types + hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(ring_feat_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def compute_ring_features(
        self,
        ring_atoms: list,
        ring_edges: list,
        coord: torch.Tensor,
        edge_dist: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_index: torch.Tensor,
        elems: torch.Tensor,
        num_atom_types: int = 10,
    ) -> torch.Tensor:
        """
        Compute feature vector for a single ring.

        Args:
            ring_atoms: List of atom indices in the ring.
            ring_edges: List of edge indices (into edge_index) belonging to the ring.
            coord: (N, 3) atom positions.
            edge_dist: (E,) interatomic distances.
            edge_feat: (E, H) edge features from backbone.
            edge_index: (E, 2) edge index tensor.
            elems: (N,) atomic numbers.
            num_atom_types: Number of atom types for bag-of-atoms.

        Returns:
            feat: (D,) ring feature vector.
        """
        device = coord.device

        # Bond length statistics
        ring_dists = edge_dist[ring_edges]
        mean_dist = ring_dists.mean()
        var_dist = ring_dists.var() if len(ring_edges) > 1 else torch.tensor(0.0, device=device)

        # Planarity: RMSD from best-fit plane
        ring_coords = coord[ring_atoms]  # (R, 3)
        centroid = ring_coords.mean(dim=0)
        centered = ring_coords - centroid
        # SVD to find normal of best-fit plane
        try:
            _, S, Vt = torch.linalg.svd(centered)
            normal = Vt[-1]  # smallest singular value direction
            distances_to_plane = (centered @ normal).abs()
            planarity = distances_to_plane.pow(2).mean().sqrt()
        except Exception:
            planarity = torch.tensor(0.0, device=device)

        # Ring size one-hot
        ring_size = len(ring_atoms)
        size_idx = min(max(ring_size - self.min_ring_size, 0), self.num_ring_sizes - 1)
        ring_size_onehot = torch.zeros(self.num_ring_sizes, device=device)
        ring_size_onehot[size_idx] = 1.0

        # Atom-type composition (bag of atoms, capped at num_atom_types)
        atom_comp = torch.zeros(num_atom_types, device=device)
        for atom_idx in ring_atoms:
            z = elems[atom_idx].item()
            z_capped = min(z, num_atom_types - 1)
            atom_comp[z_capped] += 1.0
        atom_comp = atom_comp / max(len(ring_atoms), 1)

        # Mean edge feature in ring
        mean_edge_feat = edge_feat[ring_edges].mean(dim=0)  # (H,)

        # Concatenate all
        feat = torch.cat([
            mean_dist.unsqueeze(0),
            var_dist.unsqueeze(0),
            planarity.unsqueeze(0),
            ring_size_onehot,
            atom_comp,
            mean_edge_feat,
        ])
        return feat

    def forward(self, ring_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ring_features: (R, D) stacked ring feature vectors.

        Returns:
            logits: (R,) aromaticity logits per ring.
        """
        return self.classifier(ring_features).squeeze(-1)

    def predict(self, ring_features: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Returns:
            mask: (R,) boolean tensor — True if ring is aromatic.
        """
        logits = self.forward(ring_features)
        return torch.sigmoid(logits) >= threshold
