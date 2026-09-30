import torch

from evaluate import corrupt_explicit_hydrogens


def _sample():
    elems = torch.tensor([6, 8, 1, 1])
    coord = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        [-0.9, 0.0, 0.0],
        [2.1, 0.0, 0.0],
    ])
    edge_index = torch.tensor([[0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1]])
    edge_diff = coord[edge_index[:, 1]] - coord[edge_index[:, 0]]
    edge_dist = edge_diff.norm(dim=-1)
    bond_mask = torch.ones(6, dtype=torch.bool)
    return {
        'elems': elems,
        'coord': coord,
        'edge_index': edge_index,
        'edge_diff': edge_diff,
        'edge_dist': edge_dist,
        'bond_exists': torch.ones(6),
        'bond_mask': bond_mask,
        'train_edge_mask': torch.ones(6, dtype=torch.bool),
        'train_bond_mask': torch.tensor([True, True, False, False, False, False]),
        'h_candidate_edge_mask': torch.tensor([False, False, True, True, True, True]),
        'edge_index_bond': edge_index.clone(),
        'bond_type_full': torch.zeros(6, dtype=torch.long),
        'bond_is_aromatic': torch.zeros(6, dtype=torch.bool),
        'wiberg_bo_all': torch.ones(6),
        'is_resonance_all': torch.zeros(6, dtype=torch.bool),
        'non_aromatic_mask': torch.ones(6, dtype=torch.bool),
        'bond_type': torch.zeros(2, dtype=torch.long),
        'bond_type_train': torch.zeros(2, dtype=torch.long),
        'wiberg_bo': torch.ones(2),
        'is_resonance': torch.zeros(2, dtype=torch.bool),
        'expected_valence': torch.tensor([4.0, 2.0, 1.0, 1.0]),
        'num_atoms': 4,
        'ring_atoms': [],
        'ring_edge_indices': [],
        'geom_mol_idx': 7,
    }


def test_drop_all_hydrogens_preserves_heavy_heavy_targets():
    out = corrupt_explicit_hydrogens(_sample(), drop_fraction=1.0, seed=5)
    assert out['elems'].tolist() == [6, 8]
    assert out['num_atoms'] == 2
    assert out['edge_index'].tolist() == [[0, 1], [1, 0]]
    assert out['bond_mask'].tolist() == [True, True]
    assert out['bond_type_full'].tolist() == [0, 0]
    assert out['edge_index_bond'].tolist() == [[0, 1], [1, 0]]


def test_hydrogen_noise_moves_only_hydrogens_deterministically():
    sample = _sample()
    a = corrupt_explicit_hydrogens(sample, hydrogen_noise_sigma=0.2, seed=11)
    b = corrupt_explicit_hydrogens(sample, hydrogen_noise_sigma=0.2, seed=11)
    assert torch.equal(a['coord'], b['coord'])
    assert torch.equal(a['coord'][:2], sample['coord'][:2])
    assert not torch.equal(a['coord'][2:], sample['coord'][2:])
    assert torch.equal(a['elems'], sample['elems'])
