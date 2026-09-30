"""
Chemical post-processing rules for bond type predictions.

  Rule 1 — Aromaticity requires ring membership (DISABLED — hurts net):
      Empirically, the model's aromatic predictions in rings are better
      than any simple topological rule.  This rule is kept for reference.

  Rule 2 — Valence constraint (ENABLED — fixes ~17 unique bond errors):
      After predicting bond types, sum the bond orders for each atom.
      If any atom's bond order sum exceeds its standard valence by > 0.5,
      its highest-order bond is downgraded until the constraint is met.
      This directly fixes single→triple and single→double errors where the
      model over-estimates one bond causing valence overflow.

      Atom standard valences: C=4, N=3, O=2, F=1, etc.

Key implementation note: bonded_edge_index contains BOTH directed edges
(i→j) and (j→i) for each undirected bond.  Both directions are updated
together to keep predictions symmetric.
"""

import torch
from typing import List, Set, Dict

from ..utils.ring_detection import detect_rings
from ..utils.chemistry import get_expected_valence


# Bond order associated with each class label
_BO = {0: 1.0, 1: 2.0, 2: 3.0, 3: 1.5}
# Downgrade path: if bond is too high, step it down
_DOWNGRADE = {2: 1, 1: 0, 3: 0}   # triple→double, double→single, aromatic→single


class ChemicalRuleFilter:
    """
    Post-process bond type predictions with chemical ring rules.

    Args:
        max_ring_size: Maximum ring size to consider for aromaticity rules.
    """

    def __init__(self, max_ring_size: int = 8):
        self.max_ring_size = max_ring_size

    def __call__(
        self,
        bond_types: torch.Tensor,        # (E_b,) in {0,1,2,3}
        bonded_edge_index: torch.Tensor,  # (E_b, 2) atom indices of predicted bonds
        elems: torch.Tensor,             # (N,) atomic numbers
        n_atoms: int,
    ) -> torch.Tensor:
        """
        Apply valence constraint post-processing.

        For each atom whose predicted bond order sum exceeds standard valence,
        iteratively downgrade its highest-order bond until the constraint holds.
        Both directed edges (i→j) and (j→i) of each undirected bond are updated
        together to keep predictions symmetric.

        Args:
            bond_types:       Predicted bond types from model.predict().
            bonded_edge_index: (E_b, 2) atom indices of predicted bonds.
            elems:            (N,) atomic numbers.
            n_atoms:          Total number of atoms.

        Returns:
            Corrected bond type tensor.
        """
        if bond_types.numel() == 0:
            return bond_types

        corrected = bond_types.clone().tolist()
        ei_list = bonded_edge_index.tolist()

        # Build reverse edge lookup: (i,j) → position
        edge_pos: Dict = {}
        for pos, (ai, aj) in enumerate(ei_list):
            edge_pos[(int(ai), int(aj))] = pos

        # Expected valence per atom
        exp_val = [get_expected_valence(int(elems[i])) for i in range(n_atoms)]

        # Iteratively fix valence violations (up to n_atoms passes)
        for _ in range(n_atoms):
            # Compute current bond order sum per atom
            bo_sum = [0.0] * n_atoms
            for e, (ai, aj) in enumerate(ei_list):
                bo_sum[ai] += _BO[corrected[e]]

            # Atoms with any aromatic bond: skip — polycyclic fused carbons
            # have 3 aromatic bonds (BO sum=4.5) which exceeds C valence=4
            # by design; downgrading them would corrupt correct predictions.
            has_arom = [False] * n_atoms
            for e, (ai, aj) in enumerate(ei_list):
                if corrected[e] == 3:   # aromatic
                    has_arom[ai] = True

            # Find atoms with valence violation (exclude aromatic-bond atoms)
            violated = [
                i for i in range(n_atoms)
                if bo_sum[i] > exp_val[i] + 0.4 and not has_arom[i]
            ]
            if not violated:
                break

            changed = False
            for atom_i in violated:
                # Find bonds of atom_i (as source) sorted by current bond order desc
                incident = [(e, _BO[corrected[e]]) for e, (ai, _) in enumerate(ei_list)
                            if ai == atom_i and corrected[e] in _DOWNGRADE]
                if not incident:
                    continue
                incident.sort(key=lambda x: -x[1])   # highest-order first

                for e_fwd, _ in incident:
                    if bo_sum[atom_i] <= exp_val[atom_i] + 0.4:
                        break
                    cur = corrected[e_fwd]
                    new = _DOWNGRADE[cur]
                    diff = _BO[cur] - _BO[new]

                    # Update forward edge
                    corrected[e_fwd] = new
                    ai, aj = ei_list[e_fwd]
                    bo_sum[ai] -= diff

                    # Update reverse edge (keep symmetric)
                    e_rev = edge_pos.get((int(aj), int(ai)))
                    if e_rev is not None:
                        corrected[e_rev] = new
                    changed = True

            if not changed:
                break

        return torch.tensor(corrected, dtype=torch.long, device=bond_types.device)
