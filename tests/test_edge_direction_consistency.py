"""Regression tests for sender-minus-receiver edge vectors and old caches."""

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from bondnet.data.dataset import (
    CACHE_FORMAT_VERSION,
    STAGE3_EXCLUDES_H,
    CachedBondNetDataset,
    ShardedCachedBondNetDataset,
    collate_connectivity_fn,
    collate_fn,
    collate_stage2_fn,
)
from bondnet.data.featurizer import MoleculeFeaturizer


def _sample():
    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    sample = MoleculeFeaturizer(
        cutoff=2.5, h_cutoff=2.5, explicit_h=True
    ).featurize(mol)
    sample["geom_mol_idx"] = 7
    return sample


def _assert_geometry(sample):
    edge_index = sample["edge_index"]
    expected = sample["coord"][edge_index[:, 1]] - sample["coord"][edge_index[:, 0]]
    torch.testing.assert_close(sample["edge_diff"], expected)
    torch.testing.assert_close(sample["edge_dist"], expected.norm(dim=-1))
    heavy = sample["elems"][edge_index[:, 0]].ne(1) & sample["elems"][edge_index[:, 1]].ne(1)
    assert heavy.any() and (~heavy).any()


def _legacy_sample():
    sample = _sample()
    edge_index = sample["edge_index"]
    heavy = sample["elems"][edge_index[:, 0]].ne(1) & sample["elems"][edge_index[:, 1]].ne(1)
    sample["edge_diff"] = sample["edge_diff"].clone()
    sample["edge_diff"][heavy] *= -1
    return sample


def test_featurizer_uses_same_direction_for_heavy_and_hydrogen_edges():
    _assert_geometry(_sample())
    # The heavy-only radius graph uses the same sender-minus-receiver rule.
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    edges, diff, _ = MoleculeFeaturizer._radius_graph(pos, 2.5)
    torch.testing.assert_close(diff, pos[edges[:, 1]] - pos[edges[:, 0]])


def test_old_monolithic_cache_is_corrected_without_mutating_file(tmp_path):
    old = _legacy_sample()
    cache = tmp_path / "legacy.pt"
    torch.save({
        "metadata": {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "stage3_excludes_h": STAGE3_EXCLUDES_H,
        },
        "samples": [old],
        "diversity_weights": torch.ones(1),
    }, cache)
    dataset = CachedBondNetDataset(str(cache))
    fixed = dataset[0]
    _assert_geometry(fixed)
    assert not torch.equal(old["edge_diff"], fixed["edge_diff"])
    torch.testing.assert_close(dataset.samples[0]["edge_diff"], old["edge_diff"])
    for collate in (collate_fn, collate_connectivity_fn, collate_stage2_fn):
        _assert_geometry(collate([fixed]))


def test_old_sharded_cache_is_corrected_on_read(tmp_path):
    old = _legacy_sample()
    torch.save({"samples": [old]}, tmp_path / "shard_000.pt")
    torch.save({
        "metadata": {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "stage3_excludes_h": STAGE3_EXCLUDES_H,
            "n_molecules": 1,
        },
        "shards": [{"path": "shard_000.pt", "start": 0}],
        "n_samples": 1,
    }, tmp_path / ShardedCachedBondNetDataset.MANIFEST_NAME)
    dataset = ShardedCachedBondNetDataset(str(tmp_path))
    _assert_geometry(dataset[0])
