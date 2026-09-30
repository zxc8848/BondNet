"""Checks for the post-hoc shared-coordinate hydrogen sensitivity analysis."""

import torch

from bondnet.data.noise_augment import GaussianNoiseAugment
from scripts.evaluate_paired_hydrogen_noise import (
    PairedHeavyDataset,
    atom_mapping,
    full_explicit_noise,
)


class _Samples:
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def test_shared_noise_maps_stereo_retained_hydrogen_and_recomputes_vectors():
    # Heavy-only retains the last explicit H; the other two H atoms are removed.
    explicit = {
        "geom_mol_idx": 17,
        "elems": torch.tensor([6, 7, 1, 1, 1]),
        "coord": torch.tensor([[0., 0., 0.], [1., 0., 0.],
                               [0., 1., 0.], [1., 1., 0.], [2., 1., 0.]]),
    }
    heavy = {
        "geom_mol_idx": 17,
        "elems": torch.tensor([6, 7, 1]),
        "coord": explicit["coord"][[0, 1, 4]].clone(),
        "edge_index": torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]]),
    }
    mapping = atom_mapping(explicit, heavy)
    assert mapping.tolist() == [0, 1, 4]

    sigma, seed = 0.2, 20260921
    dataset = PairedHeavyDataset(_Samples([explicit]), _Samples([heavy]), sigma, seed)
    paired = dataset[0]
    source_noise = full_explicit_noise(explicit, sigma, seed)
    assert torch.equal(paired["coord"] - heavy["coord"], source_noise[mapping])
    i, j = paired["edge_index"].unbind(-1)
    assert torch.equal(paired["edge_diff"], paired["coord"][j] - paired["coord"][i])
    assert torch.equal(paired["edge_dist"], paired["edge_diff"].norm(dim=-1))


def test_explicit_noise_matches_original_keyed_evaluator_draw():
    sample = {
        "geom_mol_idx": 29,
        "coord": torch.arange(63, dtype=torch.float32).reshape(21, 3) / 10,
    }
    sigma, seed = 0.1, 20260921
    batch = {
        "coord": sample["coord"],
        "edge_index": torch.empty((0, 2), dtype=torch.long),
        "num_atoms_per_mol": torch.tensor([21]),
        "sample_ids": torch.tensor([29]),
    }
    original = GaussianNoiseAugment(
        sigma=sigma, sigma_min=sigma, sigma_max=sigma, cutoff=2.5,
    ).augment_batch(
        batch, per_molecule=True, base_seed=seed + round(sigma * 10000), epoch=0,
    )["coord"]
    assert torch.equal(sample["coord"] + full_explicit_noise(sample, sigma, seed), original)
