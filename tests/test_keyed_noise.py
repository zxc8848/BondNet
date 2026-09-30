import torch

from bondnet.data.noise_augment import GaussianNoiseAugment
from scripts.p0d_unified_robustness import _torch_keyed_noise


def _batch(counts, sample_ids):
    n_atoms = sum(counts)
    coord = torch.arange(n_atoms * 3, dtype=torch.float32).reshape(n_atoms, 3) / 10
    return {
        'coord': coord,
        'edge_index': torch.empty((0, 2), dtype=torch.long),
        'edge_diff': torch.empty((0, 3), dtype=torch.float32),
        'edge_dist': torch.empty((0,), dtype=torch.float32),
        'num_atoms_per_mol': torch.tensor(counts, dtype=torch.long),
        'sample_ids': torch.tensor(sample_ids, dtype=torch.long),
    }


def _molecule_slices(tensor, counts):
    result = []
    start = 0
    for count in counts:
        result.append(tensor[start:start + count])
        start += count
    return result


def test_keyed_noise_pairs_explicit_h_and_heavy_only_heavy_atoms():
    # Explicit-H stores heavy atoms first within each molecule.
    explicit_counts = [4, 3]
    heavy_counts = [2, 1]
    sample_ids = [17, 29]
    explicit = _batch(explicit_counts, sample_ids)
    heavy = _batch(heavy_counts, sample_ids)
    # Give the shared heavy atoms exactly the same starting coordinates.
    explicit_mols = _molecule_slices(explicit['coord'], explicit_counts)
    heavy_mols = _molecule_slices(heavy['coord'], heavy_counts)
    for explicit_mol, heavy_mol in zip(explicit_mols, heavy_mols):
        explicit_mol[:heavy_mol.shape[0]] = heavy_mol

    augment = GaussianNoiseAugment(sigma=0.1, sigma_min=0.1, sigma_max=0.1)
    explicit_out = augment.augment_batch(explicit, base_seed=20260921, epoch=3)
    heavy_out = augment.augment_batch(heavy, base_seed=20260921, epoch=3)

    explicit_out_mols = _molecule_slices(explicit_out['coord'], explicit_counts)
    heavy_out_mols = _molecule_slices(heavy_out['coord'], heavy_counts)
    for explicit_mol, heavy_mol in zip(explicit_out_mols, heavy_out_mols):
        assert torch.equal(explicit_mol[:heavy_mol.shape[0]], heavy_mol)


def test_keyed_noise_is_independent_of_batch_order():
    augment = GaussianNoiseAugment(sigma=0.1, sigma_min=0.0, sigma_max=0.15)
    a = _batch([2, 3], [17, 29])
    b = _batch([3, 2], [29, 17])
    # Match each molecule's clean coordinates despite reversed batch order.
    a_mols = _molecule_slices(a['coord'], [2, 3])
    b_mols = _molecule_slices(b['coord'], [3, 2])
    b_mols[0].copy_(a_mols[1])
    b_mols[1].copy_(a_mols[0])

    a_out = augment.augment_batch(a, base_seed=44, epoch=7)
    b_out = augment.augment_batch(b, base_seed=44, epoch=7)
    a_out_mols = _molecule_slices(a_out['coord'], [2, 3])
    b_out_mols = _molecule_slices(b_out['coord'], [3, 2])
    assert torch.equal(a_out_mols[0], b_out_mols[1])
    assert torch.equal(a_out_mols[1], b_out_mols[0])


def test_materialized_keyed_noise_matches_model_evaluation_protocol():
    sample = {'coord': torch.zeros((5, 3)), 'geom_mol_idx': 17}
    batch = _batch([5], [17])
    batch['coord'].zero_()
    sigma = 0.1
    eval_seed = 20260921
    model_noise = GaussianNoiseAugment(
        sigma=sigma, sigma_min=sigma, sigma_max=sigma
    ).augment_batch(
        batch,
        base_seed=eval_seed + round(sigma * 10000),
    )['coord']
    materialized_noise = torch.from_numpy(
        _torch_keyed_noise([sample], [sigma], eval_seed)[sigma][0]
    )
    assert torch.equal(model_noise, materialized_noise)
