import tempfile
import unittest
from pathlib import Path

import torch

from bondnet.data.dataset import (
    CACHE_FORMAT_VERSION,
    STAGE3_EXCLUDES_H,
    CachedBondNetDataset,
    ShardedCachedBondNetDataset,
)


def _metadata(n_molecules=10):
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "stage3_excludes_h": STAGE3_EXCLUDES_H,
        "n_molecules": n_molecules,
    }


def _samples():
    labels = ["train"] * 6 + ["val"] * 2 + ["test"] * 2
    return [
        {"geom_mol_idx": idx, "split": label, "sample_id": idx}
        for idx, label in enumerate(labels)
    ]


class FixedSplitTests(unittest.TestCase):
    def test_monolithic_cache_never_uses_test_as_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.pt"
            samples = [
                {"geom_mol_idx": idx, "split": "train" if idx < 8 else "test", "sample_id": idx}
                for idx in range(10)
            ]
            torch.save(
                {
                    "metadata": _metadata(),
                    "samples": samples,
                    "diversity_weights": torch.ones(10),
                },
                cache,
            )
            with self.assertRaisesRegex(ValueError, "Refusing to use the test split"):
                CachedBondNetDataset.split(str(cache), split_key="geom_mol_idx", seed=42)
            with self.assertRaisesRegex(ValueError, "Refusing to use the test split"):
                CachedBondNetDataset.split(str(cache), seed=42)

    def test_monolithic_cache_fixed_labels_override_split_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.pt"
            torch.save(
                {
                    "metadata": _metadata(),
                    "samples": _samples(),
                    "diversity_weights": torch.ones(10),
                },
                cache,
            )

            train, val = CachedBondNetDataset.split(
                str(cache), split_key="geom_mol_idx", seed=42
            )
            self.assertEqual([s["sample_id"] for s in train.samples], list(range(6)))
            self.assertEqual([s["sample_id"] for s in val.samples], [6, 7])
            self.assertFalse({8, 9} & {s["sample_id"] for s in train.samples})

            legacy_train, legacy_val = CachedBondNetDataset.split(
                str(cache),
                split_key="geom_mol_idx",
                seed=42,
                respect_fixed_split=False,
            )
            self.assertEqual(len(legacy_train) + len(legacy_val), 10)

    def test_sharded_cache_fixed_labels_override_split_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = _samples()
            torch.save({"samples": samples}, root / "shard_000.pt")
            torch.save(
                {
                    "metadata": _metadata(),
                    "shards": [{"path": "shard_000.pt", "start": 0}],
                    "n_samples": 10,
                    "sample_meta": samples,
                    "diversity_weights": torch.ones(10),
                },
                root / ShardedCachedBondNetDataset.MANIFEST_NAME,
            )

            train, val = ShardedCachedBondNetDataset.split(
                str(root), split_key="geom_mol_idx", seed=42
            )
            self.assertEqual(train.indices, list(range(6)))
            self.assertEqual(val.indices, [6, 7])
            self.assertFalse({8, 9} & set(train.indices))

    def test_sharded_cache_never_uses_test_as_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = [
                {"geom_mol_idx": idx, "split": "train" if idx < 8 else "test", "sample_id": idx}
                for idx in range(10)
            ]
            torch.save({"samples": samples}, root / "shard_000.pt")
            torch.save(
                {
                    "metadata": _metadata(),
                    "shards": [{"path": "shard_000.pt", "start": 0}],
                    "n_samples": 10,
                    "sample_meta": samples,
                    "diversity_weights": torch.ones(10),
                },
                root / ShardedCachedBondNetDataset.MANIFEST_NAME,
            )
            with self.assertRaisesRegex(ValueError, "Refusing to use the test split"):
                ShardedCachedBondNetDataset.split(
                    str(root), split_key="geom_mol_idx", seed=42
                )
            with self.assertRaisesRegex(ValueError, "Refusing to use the test split"):
                ShardedCachedBondNetDataset.split(str(root), seed=42)


if __name__ == "__main__":
    unittest.main()
