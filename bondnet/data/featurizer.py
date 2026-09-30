"""
MoleculeFeaturizer: converts RDKit molecules to BondNet tensor format.

Output dict keys
----------------
elems            (N,)       atomic numbers
coord            (N, 3)     3D atom positions
edge_index       (E, 2)     ALL candidate edges from radius graph (both directions)
edge_diff        (E, 3)     displacement vectors  coord[j] - coord[i]
edge_dist        (E,)       interatomic distances
bond_exists      (E,)       binary — 1 if pair is a real bond, 0 otherwise
bond_mask        (E,)       same as bond_exists (bool version)
train_edge_mask  (E,)       candidate edges learned by Stage 1
train_bond_mask  (E,)       true heavy-heavy bonded edges learned by Stage 3
h_candidate_edge_mask (E,)  geometry-derived H-X candidate edges
edge_index_bond  (E_b, 2)   edge_index rows where bond_exists == 1
bond_type        (E'', 3-cls) discrete label {0=single, 1=double, 2=triple}
                             only for non-aromatic bonded edges
bond_is_aromatic (E_b,)     True for aromatic bonds
wiberg_bo        (E_b,)     Wiberg bond order (0.0 when not available)
is_resonance     (E_b,)     True if xTB BO in [1.2, 1.8]
bond_type_train  (E_train,) 4-class labels for learned heavy-heavy bonds only
expected_valence (N,)       target valence per atom
num_atoms        int
ring_atoms       list[list[int]]   (filled later if ring detection is run)
ring_edge_indices list[list[int]]  (filled later)
ring_is_aromatic list[bool]        (filled later)
ring_features    None              (filled later)
"""

import torch
import numpy as np
from typing import Dict, Optional, List, Tuple

from ..utils.chemistry import get_expected_valence

# Aromatic bond is RDKIT bond type 12
_RDKIT_BOND_TYPE_TO_INT = {}  # populated lazily to avoid rdkit import at module level

# ------------------------------------------------------------------ #
# Proxy bond order from geometry (used when xTB labels not available)  #
# ------------------------------------------------------------------ #
# Reference bond lengths (Å) for (Z_i, Z_j, bond_type) combinations.
# Interpolation gives continuous BO values so the resonance mask
# [1.2, 1.8] fires correctly on conjugated / amide-type bonds.
_REF_SINGLE = {(6, 6): 1.54, (6, 7): 1.47, (6, 8): 1.43,
               (7, 7): 1.45, (7, 8): 1.44, (8, 8): 1.48}
_REF_DOUBLE = {(6, 6): 1.34, (6, 7): 1.29, (6, 8): 1.20,
               (7, 7): 1.25, (7, 8): 1.21, (8, 8): 1.21}
_REF_TRIPLE = {(6, 6): 1.20, (6, 7): 1.16, (6, 8): 1.13}
_FALLBACK_SINGLE, _FALLBACK_DOUBLE, _FALLBACK_TRIPLE = 1.50, 1.30, 1.20


def _dist_to_proxy_bo(dist: float, zi: int, zj: int, btype: int) -> float:
    """
    Estimate a continuous Wiberg-like bond order from interatomic distance.

    For clearly discrete bonds (far from the single/double boundary) this
    returns values close to 1.0/2.0/3.0.  For short single bonds (e.g.
    amide C-N at 1.34 Å) it returns ~1.3-1.4, which activates the
    resonance mask and prevents contradictory discrete-loss supervision.
    """
    pair = (min(zi, zj), max(zi, zj))
    if btype == 3:          # aromatic
        return 1.5
    if btype == 2:          # nominal triple
        d_single = _REF_SINGLE.get(pair, _FALLBACK_SINGLE)
        d_triple = _REF_TRIPLE.get(pair, _FALLBACK_TRIPLE)
        return 1.0 + 2.0 * max(0.0, min(1.0, (d_single - dist) / (d_single - d_triple)))
    if btype == 1:          # nominal double
        d_single = _REF_SINGLE.get(pair, _FALLBACK_SINGLE)
        d_double = _REF_DOUBLE.get(pair, _FALLBACK_DOUBLE)
        t = max(0.0, min(1.0, (d_single - dist) / max(d_single - d_double, 1e-3)))
        return 1.0 + t      # range [1.0, 2.0]
    # nominal single
    d_single = _REF_SINGLE.get(pair, _FALLBACK_SINGLE)
    d_double = _REF_DOUBLE.get(pair, _FALLBACK_DOUBLE)
    # short single bonds get BO > 1.0 (up to ~1.5 for very short ones)
    t = max(0.0, min(1.0, (d_single - dist) / max(d_single - d_double, 1e-3)))
    return 1.0 + 0.5 * t   # range [1.0, 1.5]


def _get_bond_map():
    global _RDKIT_BOND_TYPE_TO_INT
    if not _RDKIT_BOND_TYPE_TO_INT:
        from rdkit.Chem import BondType  # type: ignore
        _RDKIT_BOND_TYPE_TO_INT = {
            BondType.SINGLE:   0,
            BondType.DOUBLE:   1,
            BondType.TRIPLE:   2,
            BondType.AROMATIC: 3,
        }
    return _RDKIT_BOND_TYPE_TO_INT


class MoleculeFeaturizer:
    """
    Converts a single RDKit molecule with 3D coordinates into BondNet tensors.

    Args:
        cutoff:            Radius cutoff for candidate edges (Å).
        resonance_lo/hi:   BO range flagging a bond as resonant.
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        h_cutoff: float = 2.0,
        resonance_lo: float = 1.2,
        resonance_hi: float = 1.8,
        explicit_h: bool = False,
    ):
        self.cutoff = cutoff
        self.h_cutoff = h_cutoff
        self.resonance_lo = resonance_lo
        self.resonance_hi = resonance_hi
        self.explicit_h = explicit_h

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def featurize(
        self,
        mol,                              # rdkit.Chem.Mol (with conformer)
        wiberg_bo_matrix: Optional[np.ndarray] = None,  # (N, N) or None
        conf_id: int = 0,
    ) -> Dict:
        """
        Convert an RDKit molecule to a BondNet feature dict.

        Args:
            mol:              RDKit Mol with a 3D conformer.
            wiberg_bo_matrix: Symmetric (N, N) Wiberg bond order matrix from xTB.
                              If None, BOs are set to integer bond orders.
            conf_id:          Conformer index to use.

        Returns:
            Feature dict (all torch tensors).
        """
        from rdkit import Chem  # type: ignore

        if mol is None:
            raise ValueError("mol is None — RDKit conversion failed for this molecule")

        if not self.explicit_h:
            mol = Chem.RemoveHs(mol)  # work on heavy atoms only
        conf = mol.GetConformer(conf_id)
        N = mol.GetNumAtoms()

        # Atomic numbers and positions
        elems = torch.tensor(
            [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(N)],
            dtype=torch.long,
        )
        pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)  # (N, 3)

        # Build ground-truth bond lookup from RDKit  {(i,j): (bond_type_int, is_aromatic)}
        bond_lookup: Dict[Tuple[int, int], Tuple[int, bool]] = {}
        bond_map = _get_bond_map()
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            btype = bond_map.get(bond.GetBondType(), 0)
            is_arom = bond.GetIsAromatic()
            bond_lookup[(i, j)] = (btype, is_arom)
            bond_lookup[(j, i)] = (btype, is_arom)

        if self.explicit_h:
            edge_index, edge_diff, edge_dist, train_edge_mask, h_candidate_edge_mask = (
                self._candidate_graph(pos, elems, self.cutoff, self.h_cutoff)
            )
        else:
            edge_index, edge_diff, edge_dist = self._radius_graph(pos, self.cutoff)
            train_edge_mask = torch.ones(edge_index.shape[0], dtype=torch.bool)
            h_candidate_edge_mask = torch.zeros(edge_index.shape[0], dtype=torch.bool)
        E = edge_index.shape[0]

        # Build per-edge labels
        bond_exists = torch.zeros(E, dtype=torch.float32)
        bond_type_full = torch.zeros(E, dtype=torch.long)     # 0/1/2/3
        bond_is_aromatic = torch.zeros(E, dtype=torch.bool)
        wiberg_bo_full = torch.zeros(E, dtype=torch.float32)

        for e, (i, j) in enumerate(edge_index.tolist()):
            key = (i, j)
            if key in bond_lookup:
                btype, is_arom = bond_lookup[key]
                bond_exists[e] = 1.0
                bond_type_full[e] = btype
                bond_is_aromatic[e] = is_arom
                if wiberg_bo_matrix is not None:
                    wiberg_bo_full[e] = float(wiberg_bo_matrix[i, j])
                else:
                    # Integer approximation: single=1.0, double=2.0, triple=3.0, arom=1.5.
                    # Proxy BO (bond-length interpolation) was tried but falsely flags
                    # short single bonds (e.g. conjugated C-C at 1.45Å → proxy BO=1.22)
                    # as resonance, removing them from focal-loss supervision and hurting
                    # F1-double.  Proxy BO is only safe with real xTB labels.
                    wiberg_bo_full[e] = [1.0, 2.0, 3.0, 1.5][btype]

        bond_mask = bond_exists.bool()
        h_edge_mask = (elems[edge_index[:, 0]] == 1) | (elems[edge_index[:, 1]] == 1)
        train_bond_mask = bond_mask & train_edge_mask & ~h_edge_mask
        edge_index_bond = edge_index[bond_mask]      # (E_b, 2)
        bond_type_bond = bond_type_full[bond_mask]   # (E_b,)  0/1/2/3
        is_arom_bond = bond_is_aromatic[bond_mask]   # (E_b,)
        bo_bond = wiberg_bo_full[bond_mask]               # (E_b,)
        bond_type_train = bond_type_full[train_bond_mask]

        # Resonance mask for bonded edges
        # is_resonance_all: True for bonds with intermediate BO that have ambiguous
        # discrete labels (e.g. amide C-N labeled single but BO≈1.4).
        # Aromatic bonds (label 3) are explicitly excluded — their RDKit label is
        # unambiguous even though their BO≈1.5 falls in the resonance range.
        is_resonance_all = (
            (bo_bond >= self.resonance_lo) & (bo_bond <= self.resonance_hi)
            & ~is_arom_bond
        )

        # Non-aromatic bond labels (kept for backward-compat / ablation scripts)
        non_arom_mask = ~is_arom_bond
        bond_type_non_arom = bond_type_bond[non_arom_mask]  # {0,1,2} only

        # Expected valence per atom
        expected_valence = torch.tensor(
            [get_expected_valence(int(z)) for z in elems.tolist()],
            dtype=torch.float32,
        )

        return {
            'elems':             elems,
            'coord':             pos,
            'edge_index':        edge_index,
            'edge_diff':         edge_diff,
            'edge_dist':         edge_dist,
            'bond_exists':       bond_exists,
            'bond_mask':         bond_mask,
            'train_edge_mask':   train_edge_mask,
            'train_bond_mask':   train_bond_mask,
            'h_candidate_edge_mask': h_candidate_edge_mask,
            'edge_index_bond':   edge_index_bond,
            'bond_type':         bond_type_non_arom,     # legacy: non-arom only
            'bond_type_full':    bond_type_bond,         # 4-class {0,1,2,3} all bonds
            'bond_type_train':   bond_type_train,        # learned heavy-heavy bonds only
            'bond_is_aromatic':  is_arom_bond,
            'wiberg_bo':         bo_bond[non_arom_mask], # legacy: non-arom only
            'wiberg_bo_all':     bo_bond,                # BO for all bonded edges
            'is_resonance':      is_resonance_all[non_arom_mask],  # legacy
            'is_resonance_all':  is_resonance_all,       # for 4-class cls loss
            'non_aromatic_mask': non_arom_mask,
            'expected_valence':  expected_valence,
            'num_atoms':         N,
            # Ring features are computed separately (ring detection is graph-level)
            'ring_atoms':        [],
            'ring_edge_indices': [],
            'ring_is_aromatic':  None,
            'ring_features':     None,
        }

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _radius_graph(
        pos: torch.Tensor,
        cutoff: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build a directed radius graph (both i→j and j→i for each pair within cutoff).

        Fully vectorized: O(N²) memory but no Python loops → ~100x faster than
        the naive loop version for typical molecule sizes (N ≤ 100).

        Returns:
            edge_index: (E, 2)  [receiver i, sender j] convention (matches PaiNN)
            edge_diff:  (E, 3)  pos[j] - pos[i]
            edge_dist:  (E,)    ||pos[j] - pos[i]||
        """
        N = pos.shape[0]
        if N == 0:
            return (torch.zeros((0, 2), dtype=torch.long),
                    torch.zeros((0, 3), dtype=pos.dtype),
                    torch.zeros((0,), dtype=pos.dtype))

        # (N, N, 3) all pairwise displacement vectors
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)   # diff[i,j] = pos[j] - pos[i]
        dist = diff.norm(dim=-1)                      # (N, N)

        # Mask: within cutoff and not self-loop
        mask = (dist < cutoff) & (dist > 0)
        idx_i, idx_j = torch.where(mask)             # receiver i, sender j

        edge_index = torch.stack([idx_i, idx_j], dim=1)  # (E, 2)
        edge_diff  = diff[idx_i, idx_j]                   # pos[j] - pos[i]
        edge_dist  = dist[idx_i, idx_j]

        return edge_index, edge_diff, edge_dist

    @staticmethod
    def _candidate_graph(
        pos: torch.Tensor,
        elems: torch.Tensor,
        cutoff: float,
        h_cutoff: float = 2.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build the candidate graph used by BondNet.

        Heavy-heavy pairs use the standard radius graph (cutoff).
        Hydrogen-heavy pairs use a tighter cutoff (h_cutoff) since H-X bonds
        are all shorter than 1.7A; 2.0A covers them with margin.
        """
        device = pos.device
        dtype = pos.dtype
        heavy_mask = elems != 1
        heavy_idx = torch.where(heavy_mask)[0]
        hydrogen_idx = torch.where(~heavy_mask)[0]

        edge_parts = []
        diff_parts = []
        dist_parts = []
        train_mask_parts = []
        h_candidate_mask_parts = []

        if heavy_idx.numel() > 0:
            heavy_pos = pos[heavy_idx]
            hh_edge_local, hh_diff, hh_dist = MoleculeFeaturizer._radius_graph(heavy_pos, cutoff)
            if hh_edge_local.numel() > 0:
                hh_edge = heavy_idx[hh_edge_local]
                edge_parts.append(hh_edge)
                diff_parts.append(hh_diff)
                dist_parts.append(hh_dist)
                train_mask_parts.append(torch.ones(hh_edge.shape[0], dtype=torch.bool, device=device))
                h_candidate_mask_parts.append(torch.zeros(hh_edge.shape[0], dtype=torch.bool, device=device))

        if heavy_idx.numel() > 0 and hydrogen_idx.numel() > 0:
            hydrogen_pos = pos[hydrogen_idx]
            pair_dist = torch.cdist(hydrogen_pos, pos[heavy_idx])
            valid_h_pairs = pair_dist <= h_cutoff
            h_local_idx, heavy_local_idx = torch.where(valid_h_pairs)

            if h_local_idx.numel() > 0:
                h_atom_idx = hydrogen_idx[h_local_idx]
                heavy_atom_idx = heavy_idx[heavy_local_idx]
                h_to_heavy = torch.stack([h_atom_idx, heavy_atom_idx], dim=1)
                heavy_to_h = torch.stack([heavy_atom_idx, h_atom_idx], dim=1)
                h_edges = torch.cat([h_to_heavy, heavy_to_h], dim=0)
                h_diff = pos[h_edges[:, 1]] - pos[h_edges[:, 0]]
                h_dist = h_diff.norm(dim=-1)

                edge_parts.append(h_edges)
                diff_parts.append(h_diff)
                dist_parts.append(h_dist)
                train_mask_parts.append(torch.ones(h_edges.shape[0], dtype=torch.bool, device=device))
                h_candidate_mask_parts.append(torch.ones(h_edges.shape[0], dtype=torch.bool, device=device))

        if not edge_parts:
            return (
                torch.zeros((0, 2), dtype=torch.long, device=device),
                torch.zeros((0, 3), dtype=dtype, device=device),
                torch.zeros((0,), dtype=dtype, device=device),
                torch.zeros((0,), dtype=torch.bool, device=device),
                torch.zeros((0,), dtype=torch.bool, device=device),
            )

        return (
            torch.cat(edge_parts, dim=0),
            torch.cat(diff_parts, dim=0),
            torch.cat(dist_parts, dim=0),
            torch.cat(train_mask_parts, dim=0),
            torch.cat(h_candidate_mask_parts, dim=0),
        )
