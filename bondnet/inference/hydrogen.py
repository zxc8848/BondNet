"""
Hydrogen handling utilities for BondNet inference.

At inference time, if the 3D structure includes explicit H atoms, this module
computes geometry-based atom features for ALL atoms that match the 4-dim
features used during training:

    [num_hs/4, formal_charge/2, degree/4, (degree+num_hs)/6]

All four are geometry-computable when H atoms are present in the xyz:
    - num_hs:          count Z=1 atoms within covalent-radius cutoff
    - formal_charge:   assume 0 for neutral organic molecules
    - degree:          count ALL bonded atoms (H + heavy) within cutoff
    - (degree+num_hs): mirrors GetDegree()+GetTotalNumHs() from training

Usage:
    from bondnet.inference.hydrogen import atom_features_from_xyz

    feat = atom_features_from_xyz(xyz, elems)
    # feat: (N_total, 4) — pass as batch['atom_features']
"""

import torch
from typing import Optional

# Covalent radii (Å) for common elements — same table as featurizer.py
_COVALENT_RADII = {
    1: 0.31,   # H
    6: 0.76,   # C
    7: 0.71,   # N
    8: 0.66,   # O
    9: 0.57,   # F
    15: 1.07,  # P
    16: 1.05,  # S
    17: 1.02,  # Cl
    35: 1.20,  # Br
    53: 1.39,  # I
}
_DEFAULT_COVALENT_RADIUS = 1.00
_BOND_MARGIN = 0.30   # Å tolerance added to sum of covalent radii


def atom_features_from_xyz(
    xyz: torch.Tensor,                          # (N, 3) all atoms including H
    elems: torch.Tensor,                        # (N,) atomic numbers
    formal_charges: Optional[torch.Tensor] = None,  # (N,) optional, defaults to 0
) -> torch.Tensor:
    """
    Compute 4-dim atom features for ALL atoms from geometry.

    Matches the training-time featurizer (mol with explicit H atoms):
        feat[i] = [num_hs/4, formal_charge/2, degree/4, (degree+num_hs)/6]

    Args:
        xyz:           (N, 3) all atom positions (including H).
        elems:         (N,) atomic numbers.
        formal_charges:(N,) formal charges; if None, all assumed zero.

    Returns:
        (N, 4) float32 tensor.
    """
    N = xyz.shape[0]
    device = xyz.device

    # Covalent radius per atom
    radii = torch.tensor(
        [_COVALENT_RADII.get(int(z), _DEFAULT_COVALENT_RADIUS) for z in elems.tolist()],
        dtype=torch.float32, device=device,
    )

    # Bond cutoff matrix: r_i + r_j + margin
    cutoff = radii.unsqueeze(1) + radii.unsqueeze(0) + _BOND_MARGIN   # (N, N)

    # Pairwise distances
    diff = xyz.unsqueeze(1) - xyz.unsqueeze(0)   # (N, N, 3)
    dists = torch.norm(diff, dim=-1)              # (N, N)

    # Bonded pairs: within cutoff, exclude self
    bonded = (dists < cutoff) & (dists > 0.01)   # (N, N)

    is_h = (elems == 1)                           # (N,)

    # num_hs[i] = count of H neighbors of atom i
    num_hs = (bonded & is_h.unsqueeze(0)).sum(dim=-1).float()   # (N,)

    # degree[i] = total bonded neighbors (H + heavy)
    degree = bonded.sum(dim=-1).float()           # (N,)

    # Formal charge
    if formal_charges is None:
        fc = torch.zeros(N, dtype=torch.float32, device=device)
    else:
        fc = formal_charges.float().to(device)

    return torch.stack([
        num_hs / 4.0,
        fc / 2.0,
        degree / 4.0,
        (degree + num_hs) / 6.0,
    ], dim=-1)   # (N, 4)
