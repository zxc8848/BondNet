"""
Valence constraint loss for BondNet.

Penalizes deviations from expected atomic valence:

  L_valence = (1/N) * Σ_i ( Σ_j BO_ij  −  v_i )²

where BO_ij is the predicted continuous bond order for bond (i,j),
and v_i is the expected valence for atom i.

Both directions of each undirected bond are present in edge_index, so we
aggregate over edges where each atom is the *source* (edge_index[:, 0]).
This correctly sums each bond once per endpoint atom.
"""

import torch
from torch import nn
from typing import Optional


class ValenceLoss(nn.Module):
    """
    Soft valence constraint.

    Args:
        reduction: 'mean' (average over atoms) or 'sum'.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        bo_pred: torch.Tensor,            # (E_bond,)  predicted bond orders
        edge_index_bond: torch.Tensor,    # (E_bond, 2) edge indices for bonded edges
        expected_valence: torch.Tensor,   # (N,)  target valence per atom
        num_atoms: int,
    ) -> torch.Tensor:
        """
        Compute valence loss.

        edge_index_bond is expected to contain **both** directions (i→j) and
        (j→i) for each undirected bond.  We aggregate bo_pred over source atoms
        (edge_index_bond[:, 0]) — this gives the correct per-atom valence sum.

        Args:
            bo_pred:          Predicted continuous bond order for each directed edge.
            edge_index_bond:  (E_bond, 2) in the convention edge[i,0]=receiver,
                              edge[i,1]=sender used by the backbone. We take
                              edge[:,1] as the source atom to accumulate valence.
            expected_valence: (N,) float tensor of target valences.
            num_atoms:        Number of atoms N.

        Returns:
            Scalar valence loss.
        """
        if bo_pred.numel() == 0:
            return bo_pred.sum() * 0.0

        # Accumulate predicted BO at each atom (treat atom as bond source)
        bo_sum = torch.zeros(num_atoms, device=bo_pred.device, dtype=bo_pred.dtype)
        # edge_index_bond[:, 1] = sender atom j
        # edge_index_bond[:, 0] = receiver atom i
        # Since both (i,j) and (j,i) exist, summing over senders == summing over receivers
        # Either direction gives the correct "contribution of atom x to valence"
        sources = edge_index_bond[:, 0]   # receiver atom (convention in backbone)
        bo_sum.index_add_(0, sources, bo_pred)

        diff = bo_sum - expected_valence.to(bo_pred.dtype)
        loss_per_atom = diff.pow(2)

        if self.reduction == 'mean':
            return loss_per_atom.mean()
        return loss_per_atom.sum()
