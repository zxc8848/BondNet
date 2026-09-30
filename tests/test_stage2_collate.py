import torch

from bondnet.data.dataset import collate_stage2_fn


def _sample(mol_id, n_atoms, edges, bond_mask):
    coord = torch.zeros((n_atoms, 3), dtype=torch.float32)
    edge_index = torch.tensor(edges, dtype=torch.long)
    return {
        'elems': torch.full((n_atoms,), 6, dtype=torch.long),
        'coord': coord,
        'edge_index': edge_index,
        'edge_diff': torch.zeros((len(edges), 3), dtype=torch.float32),
        'edge_dist': torch.ones(len(edges), dtype=torch.float32),
        'bond_exists': torch.tensor(bond_mask, dtype=torch.float32),
        'bond_mask': torch.tensor(bond_mask, dtype=torch.bool),
        'bond_type_full': torch.zeros(sum(bond_mask), dtype=torch.long),
        'num_atoms': n_atoms,
        'geom_mol_idx': mol_id,
    }


def test_stage2_collate_preserves_per_molecule_edge_and_bond_indices():
    batch = collate_stage2_fn([
        _sample(11, 2, [[0, 1], [1, 0]], [1, 1]),
        _sample(29, 3, [[0, 1], [1, 0], [1, 2], [2, 1]], [1, 1, 0, 0]),
    ])
    assert batch['sample_ids'].tolist() == [11, 29]
    assert batch['edge_mol_idx'].tolist() == [0, 0, 1, 1, 1, 1]
    assert batch['bond_mol_idx'].tolist() == [0, 0, 1, 1]
