"""Regression check for the inference-only candidate-envelope audit."""

import torch

from scripts.audit_revision_v3_candidate_envelope import add_missing_active_edges


def test_add_missing_active_edges_preserves_existing_bonds_and_adds_both_directions():
    coord = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
    old_edges = torch.tensor([[0, 1], [1, 0]])
    batch = {
        "coord": coord,
        "elems": torch.tensor([6, 6, 1]),
        "num_atoms_per_mol": torch.tensor([3]),
        "edge_index": old_edges,
        "edge_diff": coord[old_edges[:, 1]] - coord[old_edges[:, 0]],
        "edge_dist": torch.tensor([1.5, 1.5]),
        "edge_mol_idx": torch.tensor([0, 0]),
        "edge_index_bond": old_edges,
        "bond_exists": torch.tensor([True, True]),
        "bond_mask": torch.tensor([True, True]),
        "train_edge_mask": torch.tensor([True, True]),
        "train_bond_mask": torch.tensor([True, True]),
        "h_candidate_edge_mask": torch.tensor([False, False]),
    }
    expanded, n_new = add_missing_active_edges(batch, cutoff=2.5)
    assert n_new == 2
    assert expanded["edge_index"].tolist() == [[0, 1], [1, 0], [1, 2], [2, 1]]
    assert expanded["bond_mask"].tolist() == [True, True, False, False]
    assert expanded["h_candidate_edge_mask"].tolist() == [False, False, True, True]
    assert torch.allclose(expanded["edge_diff"],
                          coord[expanded["edge_index"][:, 1]] - coord[expanded["edge_index"][:, 0]])
    assert batch["edge_index"].tolist() == [[0, 1], [1, 0]]
