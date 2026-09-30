"""
Constrained bond-order discretization at inference time (Section 5 of proposal).

Solves per-molecule integer program:

  min  Σ_{(i,j)} (b_ij − ŵ_ij)²
  s.t. Σ_j b_ij = v_i    for all atoms i
       b_ij ∈ {1, 2, 3}

where ŵ_ij is the predicted continuous bond order and v_i is the expected
atomic valence.  Because molecules have at most ~tens of bonds, this is
solved exactly in milliseconds by enumeration or scipy's milp solver.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple

from ..utils.chemistry import get_expected_valence


class ConstrainedDiscretizer:
    """
    Post-processing step that assigns final integer bond orders subject to
    valence constraints.

    Usage:
        disc = ConstrainedDiscretizer()
        final_types = disc(bond_orders, edge_index, elems)
    """

    def __init__(
        self,
        allowed_orders: Tuple[int, ...] = (1, 2, 3),
        max_bonds_for_exact: int = 30,
    ):
        """
        Args:
            allowed_orders:       Allowed discrete bond order values.
            max_bonds_for_exact:  Molecules with more bonds than this use greedy
                                  rounding instead of exact ILP.
        """
        self.allowed_orders = allowed_orders
        self.max_bonds_for_exact = max_bonds_for_exact

    # ------------------------------------------------------------------ #
    # Main interface                                                        #
    # ------------------------------------------------------------------ #

    def __call__(
        self,
        bond_orders: torch.Tensor,     # (E_b,) continuous predicted BO
        edge_index_bond: torch.Tensor, # (E_b, 2) atom indices of bonded edges
        elems: torch.Tensor,           # (N,) atomic numbers
        resonance_mask: Optional[torch.Tensor] = None,  # (E_b,) True = resonance bond
    ) -> torch.Tensor:
        """
        Discretize continuous bond orders to integers under valence constraints.

        Args:
            bond_orders:      Predicted continuous BO per bond (one direction).
            edge_index_bond:  (E_b, 2) [i, j] in the direction used by backbone.
            elems:            (N,) atomic numbers.
            resonance_mask:   If True for a bond, that bond may get 1 or 2.

        Returns:
            disc_bo: (E_b,) integer bond orders.
        """
        n_atoms = elems.shape[0]
        bo_np = bond_orders.detach().cpu().numpy()
        ei_np = edge_index_bond.detach().cpu().numpy()  # (E_b, 2)
        z_np = elems.detach().cpu().numpy()

        expected = np.array([get_expected_valence(int(z)) for z in z_np], dtype=np.float32)

        # Deduplicate: for undirected bonds, each appears as both (i,j) and (j,i)
        # Group directed edges into undirected bonds
        undirected_bonds, directed_to_undirected = _build_undirected_bonds(ei_np)
        n_bonds = len(undirected_bonds)
        # Average BO across both directions for robustness
        bo_undirected = _average_directed_bo(bo_np, undirected_bonds, directed_to_undirected, len(bo_np))

        if n_bonds <= self.max_bonds_for_exact:
            disc_undirected = self._solve_exact(
                bo_undirected, undirected_bonds, expected, n_atoms
            )
        else:
            disc_undirected = self._solve_greedy(
                bo_undirected, undirected_bonds, expected, n_atoms
            )

        # Map back to directed edges
        disc_directed = np.ones(len(bo_np), dtype=np.int64)
        for d_idx, u_idx in enumerate(directed_to_undirected):
            disc_directed[d_idx] = disc_undirected[u_idx]

        return torch.tensor(disc_directed, dtype=torch.long)

    # ------------------------------------------------------------------ #
    # Solvers                                                               #
    # ------------------------------------------------------------------ #

    def _solve_exact(
        self,
        bo_target: np.ndarray,   # (n_bonds,) target BO for each undirected bond
        bonds: List[Tuple[int, int]],
        expected_valence: np.ndarray,
        n_atoms: int,
    ) -> np.ndarray:
        """
        Solve the ILP exactly using scipy.milp (available since scipy 1.9).
        Falls back to greedy if milp fails.
        """
        try:
            return self._solve_milp(bo_target, bonds, expected_valence, n_atoms)
        except Exception:
            return self._solve_greedy(bo_target, bonds, expected_valence, n_atoms)

    def _solve_milp(
        self,
        bo_target: np.ndarray,
        bonds: List[Tuple[int, int]],
        expected_valence: np.ndarray,
        n_atoms: int,
    ) -> np.ndarray:
        """
        Integer linear program via scipy.milp.

        Decision variables:  x_{bond,order}  ∈ {0,1}
        One-hot constraint:  Σ_k x_{b,k} = 1  for each bond b
        Valence constraint:  Σ_b Σ_k k * x_{b,k} = v_i  for each atom i
        Objective:           min Σ_b Σ_k (k - ŵ_b)² * x_{b,k}

        This is a QIP (quadratic); we linearize by precomputing (k - ŵ_b)².
        """
        from scipy.optimize import milp, LinearConstraint, Bounds  # type: ignore
        from scipy.sparse import lil_matrix, csr_matrix

        n_bonds = len(bonds)
        orders = np.array(self.allowed_orders, dtype=float)
        n_orders = len(orders)
        n_vars = n_bonds * n_orders

        # Objective: (k - ŵ_b)^2 for each (bond, order) variable
        c = np.zeros(n_vars)
        for b, bo in enumerate(bo_target):
            for k_idx, k in enumerate(orders):
                c[b * n_orders + k_idx] = (k - bo) ** 2

        # Constraint 1: one-hot per bond — Σ_k x_{b,k} = 1
        A_onehot = lil_matrix((n_bonds, n_vars))
        for b in range(n_bonds):
            for k_idx in range(n_orders):
                A_onehot[b, b * n_orders + k_idx] = 1.0

        # Constraint 2: valence per atom — Σ_{bonds incident to i} Σ_k k * x_{b,k} = v_i
        A_valence = lil_matrix((n_atoms, n_vars))
        for b, (i, j) in enumerate(bonds):
            for k_idx, k in enumerate(orders):
                A_valence[i, b * n_orders + k_idx] += k
                A_valence[j, b * n_orders + k_idx] += k

        n_constraints = n_bonds + n_atoms
        A_full = lil_matrix((n_constraints, n_vars))
        A_full[:n_bonds, :] = A_onehot
        A_full[n_bonds:, :] = A_valence
        A_csr = csr_matrix(A_full)

        b_lower = np.ones(n_constraints)
        b_upper = np.ones(n_constraints)
        b_lower[n_bonds:] = expected_valence
        b_upper[n_bonds:] = expected_valence

        constraints = LinearConstraint(A_csr, b_lower, b_upper)
        integrality = np.ones(n_vars)  # all vars are integer (0 or 1)
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))

        result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)

        if result.success:
            x = result.x.reshape(n_bonds, n_orders)
            disc = orders[np.argmax(x, axis=1)].astype(np.int64)
            return disc

        # milp failed (infeasible) — fall back to greedy
        return self._solve_greedy(bo_target, bonds, expected_valence, n_atoms)

    def _solve_greedy(
        self,
        bo_target: np.ndarray,
        bonds: List[Tuple[int, int]],
        expected_valence: np.ndarray,
        n_atoms: int,
    ) -> np.ndarray:
        """
        Greedy discretization: round each bond order to nearest allowed value,
        then fix valence violations by bumping/reducing bonds.
        """
        n_bonds = len(bonds)
        orders = np.array(self.allowed_orders)

        # Initial assignment: nearest allowed order
        disc = np.array([
            orders[np.argmin(np.abs(orders - bo))]
            for bo in bo_target
        ], dtype=np.float32)

        # Iteratively correct valence violations (up to 10 passes)
        for _ in range(10):
            valence_sum = np.zeros(n_atoms)
            for b, (i, j) in enumerate(bonds):
                valence_sum[i] += disc[b]
                valence_sum[j] += disc[b]

            excess = valence_sum - expected_valence
            if np.allclose(excess, 0, atol=0.5):
                break

            for atom_i in range(n_atoms):
                if abs(excess[atom_i]) < 0.5:
                    continue
                # Find bonds incident to atom_i, sorted by |disc[b] - target[b]|
                incident = [(b, i, j) for b, (i, j) in enumerate(bonds)
                            if i == atom_i or j == atom_i]
                incident.sort(key=lambda x: abs(disc[x[0]] - bo_target[x[0]]))

                for b, _, _ in incident:
                    cur = disc[b]
                    if excess[atom_i] > 0.5 and cur > orders[0]:
                        disc[b] = cur - 1
                        excess[atom_i] -= 1
                        # Update the other endpoint
                        other = bonds[b][1] if bonds[b][0] == atom_i else bonds[b][0]
                        excess[other] -= 1
                    elif excess[atom_i] < -0.5 and cur < orders[-1]:
                        disc[b] = cur + 1
                        excess[atom_i] += 1
                        other = bonds[b][1] if bonds[b][0] == atom_i else bonds[b][0]
                        excess[other] += 1
                    if abs(excess[atom_i]) < 0.5:
                        break

        return disc.astype(np.int64)


# ------------------------------------------------------------------ #
# Helper functions                                                      #
# ------------------------------------------------------------------ #

def _build_undirected_bonds(
    edge_index: np.ndarray,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Convert directed edge_index (E_b, 2) to a list of unique undirected bonds.

    Returns:
        undirected_bonds:       List of (i, j) with i < j.
        directed_to_undirected: (E_b,) array mapping each directed edge to its
                                undirected bond index.
    """
    bond_map: Dict[Tuple[int, int], int] = {}
    directed_to_undirected = np.zeros(len(edge_index), dtype=np.int64)
    undirected_bonds: List[Tuple[int, int]] = []

    for e_idx, (i, j) in enumerate(edge_index.tolist()):
        key = (min(i, j), max(i, j))
        if key not in bond_map:
            bond_map[key] = len(undirected_bonds)
            undirected_bonds.append(key)
        directed_to_undirected[e_idx] = bond_map[key]

    return undirected_bonds, directed_to_undirected


def _average_directed_bo(
    bo_directed: np.ndarray,
    undirected_bonds: List[Tuple[int, int]],
    directed_to_undirected: np.ndarray,
    n_directed: int,
) -> np.ndarray:
    """Average BO over both directed versions of each undirected bond."""
    n_bonds = len(undirected_bonds)
    bo_sum = np.zeros(n_bonds)
    count = np.zeros(n_bonds)
    for d_idx in range(n_directed):
        u_idx = directed_to_undirected[d_idx]
        bo_sum[u_idx] += bo_directed[d_idx]
        count[u_idx] += 1
    count = np.maximum(count, 1)
    return bo_sum / count
