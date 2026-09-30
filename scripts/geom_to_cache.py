"""
Build a BondNet feature cache directly from GEOM drugs_crude.msgpack(.tar.gz).

This skips the intermediate SDF file. It streams GEOM molecules/conformers,
constructs RDKit molecules with the GEOM coordinates, featurizes them, and saves
a cache compatible with train.py / train_stage2.py.

Example:
    python scripts/geom_to_cache.py ^
        --input data/drugs_crude.msgpack.tar.gz ^
        --output data/geom_drugs_allconf_c3_explicit_h.pt ^
        --cutoff 3.0 ^
        --explicit_h ^
        --conformers_per_mol -1
"""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geom_to_sdf import (
    geom_mol_to_rdkit, iter_msgpack_entries, pick_conformers, _boltzmann_weights,
)


def parse_args():
    p = argparse.ArgumentParser(description='Build BondNet cache from GEOM msgpack')
    p.add_argument('--input', required=True, help='Path to drugs_crude.msgpack or .tar.gz')
    p.add_argument('--output', required=True, help='Output .pt cache path')
    p.add_argument('--cutoff', type=float, default=3.0)
    p.add_argument('--explicit_h', action='store_true')
    p.add_argument('--conformer', choices=['lowest_energy', 'first', 'random', 'boltzmann'],
                   default='lowest_energy')
    p.add_argument('--conformers_per_mol', type=int, default=-1,
                   help='Conformers per molecule. Use -1 for all conformers.')
    p.add_argument('--max_rel_energy', type=float, default=None,
                   help='Exclude conformers with relativeenergy (kcal/mol) above this. '
                        'Recommended: 5.0 for boltzmann/lowest, 10.0 for random.')
    p.add_argument('--skip_mols', type=int, default=0)
    p.add_argument('--max_mols', type=int, default=None,
                   help='Maximum unique input molecules to process')
    p.add_argument('--max_samples', type=int, default=None,
                   help='Maximum conformer samples to write')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--progress_every', type=int, default=5000)
    p.add_argument('--split_seed', type=int, default=42,
                   help='Seed for stable train/val/test split labels')
    p.add_argument('--split_val_frac', type=float, default=0.1,
                   help='Validation fraction for stable split labels')
    p.add_argument('--split_test_frac', type=float, default=0.1,
                   help='Test fraction for stable split labels')
    p.add_argument('--keep_split', choices=['all', 'train', 'val', 'test'], default='all',
                   help='Optionally write only one split to the output cache')
    return p.parse_args()


def _diversity_weight(sample):
    bt = sample.get('bond_type_full')
    if bt is None:
        return 1.0
    if (bt == 2).any().item():
        return 4.0
    if (bt == 1).any().item():
        return 2.0
    return 1.0


def main():
    args = parse_args()
    random.seed(args.seed)

    import torch
    from bondnet.data.dataset import (
        CACHE_FORMAT_VERSION,
        STAGE3_EXCLUDES_H,
        _assign_split_label,
    )
    from bondnet.data.featurizer import MoleculeFeaturizer

    featurizer = MoleculeFeaturizer(cutoff=args.cutoff, explicit_h=args.explicit_h)

    samples = []
    weights = []
    n_seen = 0
    n_processed_mols = 0
    n_failed_mols = 0
    n_failed_confs = 0
    t0 = time.time()

    print(f"Streaming GEOM data from: {args.input}")
    for smiles, mol_data in iter_msgpack_entries(args.input):
        n_seen += 1
        if n_seen <= args.skip_mols:
            continue
        if args.max_mols is not None and n_processed_mols >= args.max_mols:
            break
        if args.max_samples is not None and len(samples) >= args.max_samples:
            break
        n_processed_mols += 1

        if not isinstance(mol_data, dict):
            n_failed_mols += 1
            continue

        conformers = mol_data.get('conformers', [])
        if args.max_rel_energy is not None:
            conformers = [
                c for c in conformers
                if c.get('relativeenergy') is None
                or float(c.get('relativeenergy', 0)) <= args.max_rel_energy
            ]
        selected = pick_conformers(conformers, args.conformer, args.conformers_per_mol)
        if not selected:
            n_failed_mols += 1
            continue

        for conf_rank, conf_data in enumerate(selected):
            if args.max_samples is not None and len(samples) >= args.max_samples:
                break

            mol = geom_mol_to_rdkit(str(smiles), conf_data)
            if mol is None:
                n_failed_confs += 1
                continue

            try:
                sample = featurizer.featurize(mol)
            except Exception:
                n_failed_confs += 1
                continue

            geom_id = conf_data.get('geom_id', conf_rank)
            sample['geom_mol_idx'] = int(n_seen - 1)
            try:
                sample['geom_conf_idx'] = int(geom_id)
            except Exception:
                sample['geom_conf_idx'] = str(geom_id)
            sample['geom_conf_rank'] = int(conf_rank)
            sample['geom_smiles'] = str(smiles)
            sample['split'] = _assign_split_label(
                sample,
                default_idx=len(samples),
                val_frac=args.split_val_frac,
                test_frac=args.split_test_frac,
                seed=args.split_seed,
            )

            if args.keep_split != 'all' and sample['split'] != args.keep_split:
                continue

            samples.append(sample)
            weights.append(_diversity_weight(sample))

        if args.progress_every > 0 and n_processed_mols % args.progress_every == 0:
            elapsed = time.time() - t0
            rate = len(samples) / max(elapsed, 1e-9)
            print(
                f"  mols={n_processed_mols} samples={len(samples)} "
                f"failed_mols={n_failed_mols} failed_confs={n_failed_confs} "
                f"{rate:.0f} samples/s"
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'metadata': {
            'cache_format_version': CACHE_FORMAT_VERSION,
            'n_molecules': len(samples),
            'source': str(args.input),
            'dataset': 'geom_drugs_crude',
            'cutoff': args.cutoff,
            'explicit_h': args.explicit_h,
            'stage3_excludes_h': STAGE3_EXCLUDES_H,
            'h_candidate_policy': 'explicit_h_learned_edges',
            'conformer': args.conformer,
            'conformers_per_mol': args.conformers_per_mol,
            'max_rel_energy': args.max_rel_energy,
            'has_group_key': 'geom_mol_idx',
            'fixed_split': {
                'method': 'stable_hash',
                'seed': int(args.split_seed),
                'val_frac': float(args.split_val_frac),
                'test_frac': float(args.split_test_frac),
            },
            'kept_split': args.keep_split,
        },
        'samples': samples,
        'diversity_weights': torch.tensor(weights, dtype=torch.float32),
    }
    torch.save(payload, out_path)

    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / 1_000_000
    unique_mols = len({s['geom_mol_idx'] for s in samples})
    print(
        f"\nDone: unique_mols={unique_mols}, samples={len(samples)}, "
        f"failed_mols={n_failed_mols}, failed_confs={n_failed_confs}, {elapsed:.1f}s"
    )
    print(f"Cache saved: {out_path} ({size_mb:.1f} MB)")
    print("Train split example: --split_key geom_mol_idx --train_groups 900")


if __name__ == '__main__':
    main()
