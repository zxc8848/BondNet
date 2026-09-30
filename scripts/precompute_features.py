"""
Pre-compute BondNet features for all molecules in an SDF file and save to disk.

This eliminates RDKit featurization from the training hot-path, allowing the
DataLoader to saturate the GPU by returning O(1) cached tensor dicts instead of
running expensive graph construction on each call.

Usage
-----
# Build cache (once):
python scripts/precompute_features.py \
    --data_path E:/zxcproject/voxelnet/qm9_raw/raw/gdb9.sdf \
    --output data/qm9_cache.pt

# With pre-computed Wiberg bond orders (xTB):
python scripts/precompute_features.py \
    --data_path E:/zxcproject/voxelnet/qm9_raw/raw/gdb9.sdf \
    --wiberg_npz data/qm9_xtb_bo.npz \
    --output data/qm9_cache_xtb.pt

# Debug / partial cache:
python scripts/precompute_features.py \
    --data_path E:/zxcproject/voxelnet/qm9_raw/raw/gdb9.sdf \
    --output data/qm9_cache_1k.pt \
    --max_mols 1000

Then train with:
python train.py --cache_path data/qm9_cache.pt \
    --batch_size 512 --num_workers 4 \
    --hidden_size 256 --num_interactions 6 ...
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure bondnet package is importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(
        description='Pre-compute BondNet features and save to disk cache.'
    )
    p.add_argument('--dataset', choices=['sdf', 'qm9'], default='sdf',
                   help='Dataset type')
    p.add_argument('--data_path', default=None,
                   help='Path to .sdf file (required when --dataset sdf)')
    p.add_argument('--pyg_root', default='./data/qm9',
                   help='Root dir for PyG QM9 download (used when --dataset qm9)')
    p.add_argument('--output', required=True,
                   help='Output .pt cache file path (e.g. data/qm9_cache.pt)')
    p.add_argument('--wiberg_npz', default=None,
                   help='Optional .npz with pre-computed Wiberg BOs (key: bo_matrices)')
    p.add_argument('--max_mols', type=int, default=None,
                   help='Limit number of molecules (for debugging)')
    p.add_argument('--cutoff', type=float, default=3.0,
                   help='Radius cutoff for candidate edges (A)')
    p.add_argument('--h_cutoff', type=float, default=None,
                   help='Radius cutoff for H-X candidate edges (A). Default: same as --cutoff.')
    p.add_argument('--explicit_h', action='store_true',
                   help='Keep explicit hydrogens and build H-aware candidate edges')
    p.add_argument('--num_workers', type=int, default=0,
                   help='Parallel workers for cache building')
    p.add_argument('--split_seed', type=int, default=42,
                   help='Seed for deterministic train/val/test split labels in cache')
    p.add_argument('--split_val_frac', type=float, default=0.1,
                   help='Validation fraction for fixed split labels')
    p.add_argument('--split_test_frac', type=float, default=0.1,
                   help='Test fraction for fixed split labels')
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    from bondnet.data.dataset import (
        BondNetDataset,
        CACHE_FORMAT_VERSION,
        STAGE3_EXCLUDES_H,
        _assign_split_label,
    )
    from bondnet.data.featurizer import MoleculeFeaturizer

    h_cutoff = args.cutoff if args.h_cutoff is None else args.h_cutoff
    featurizer = MoleculeFeaturizer(
        cutoff=args.cutoff,
        h_cutoff=h_cutoff,
        explicit_h=args.explicit_h,
    )

    t_load = time.time()
    if args.dataset == 'qm9':
        print(f"Loading QM9 from PyG root: {args.pyg_root}")
        ds = BondNetDataset.from_pyg_qm9(
            root=args.pyg_root,
            split=None,
            max_mols=args.max_mols,
            featurizer=featurizer,
        )
    else:
        if not args.data_path:
            print("ERROR: --data_path is required when --dataset sdf")
            sys.exit(1)
        print(f"Loading molecules from: {args.data_path}")
        if args.wiberg_npz:
            print(f"Wiberg BOs: {args.wiberg_npz}")
        ds = BondNetDataset.from_sdf(
            sdf_path=args.data_path,
            wiberg_npz=args.wiberg_npz,
            max_mols=args.max_mols,
            featurizer=featurizer,
        )
    n_total = len(ds)
    print(f"Loaded {n_total} molecules in {time.time() - t_load:.1f}s")

    # ── Featurize all molecules ──────────────────────────────────── #
    print(f"Featurizing {n_total} molecules ...")
    t_feat = time.time()

    samples = []
    valid_indices = []
    n_invalid = 0

    for i in range(n_total):
        if i % 10_000 == 0 and i > 0:
            elapsed = time.time() - t_feat
            rate = i / elapsed
            eta = (n_total - i) / max(rate, 1e-9)
            print(f"  {i:>6}/{n_total}  {rate:>6.0f} mol/s  ETA {eta:>5.0f}s")

        s = ds[i]
        if s.get('_invalid', False):
            n_invalid += 1
            continue

        # Remove mol_idx (will be reassigned dynamically in __getitem__)
        s.pop('mol_idx', None)
        s['split'] = _assign_split_label(
            s,
            default_idx=len(samples),
            val_frac=args.split_val_frac,
            test_frac=args.split_test_frac,
            seed=args.split_seed,
        )
        samples.append(s)
        valid_indices.append(i)

    elapsed_feat = time.time() - t_feat
    n_valid = len(samples)
    print(f"  Featurized {n_valid} valid molecules "
          f"({n_invalid} skipped) in {elapsed_feat:.1f}s "
          f"({n_valid / max(elapsed_feat, 1e-9):.0f} mol/s)")

    # ── Build diversity weights for the valid subset ─────────────── #
    import torch
    full_weights = ds._diversity_weights                    # (n_total,)
    valid_idx_tensor = torch.tensor(valid_indices, dtype=torch.long)
    subset_weights = full_weights[valid_idx_tensor]         # (n_valid,)

    # ── Save ─────────────────────────────────────────────────────── #
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving cache to: {out_path} ...")
    torch.save(
        {
            'metadata': {
                'cache_format_version': CACHE_FORMAT_VERSION,
                'cutoff': args.cutoff,
                'h_cutoff': h_cutoff,
                'has_wiberg_npz': bool(args.wiberg_npz),
                'has_local_geom': True,
                'bo_fallback': 'integer_by_bond_type',
                'stage3_excludes_h': STAGE3_EXCLUDES_H,
                'explicit_h': args.explicit_h,
                'h_candidate_policy': 'explicit_h_learned_edges',
                'fixed_split': {
                    'method': 'stable_hash',
                    'seed': int(args.split_seed),
                    'val_frac': float(args.split_val_frac),
                    'test_frac': float(args.split_test_frac),
                },
            },
            'samples': samples,
            'diversity_weights': subset_weights,
        },
        str(out_path),
    )
    size_mb = out_path.stat().st_size / 1_000_000
    print(f"Cache saved: {size_mb:.1f} MB  ({n_valid} molecules)")
    print()
    print("To train with this cache, add:  --cache_path", str(out_path))


if __name__ == '__main__':
    main()
