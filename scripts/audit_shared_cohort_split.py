"""Audit fixed-split overlap for the archived 5,000-molecule YuelBond SDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from bondnet.data.dataset import _assign_split_label


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_ids(path: Path) -> list[int]:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    ids = []
    for idx, mol in enumerate(supplier):
        if mol is None or not mol.HasProp('geom_mol_idx'):
            raise ValueError(f'Missing GEOM molecule identity at {path}:{idx}')
        ids.append(int(mol.GetProp('geom_mol_idx')))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean', type=Path, default=Path('data/shared_bench_clean.sdf'))
    parser.add_argument('--noisy', type=Path, default=Path('data/shared_bench_noisy.sdf'))
    parser.add_argument('--output', type=Path,
                        default=Path('results/p0c/joint_v3/shared_fixed_split_audit.json'))
    args = parser.parse_args()
    clean_ids = read_ids(args.clean)
    noisy_ids = read_ids(args.noisy)
    if clean_ids != noisy_ids or len(clean_ids) != 5000 or len(set(clean_ids)) != 5000:
        raise ValueError('The shared SDFs differ in GEOM molecule identity/order or size')
    labels = [_assign_split_label({'geom_mol_idx': mol_id}, idx)
              for idx, mol_id in enumerate(clean_ids)]
    counts = Counter(labels)
    result = {
        'n_molecules': len(clean_ids),
        'clean_sdf_sha256': sha256(args.clean),
        'noisy_sdf_sha256': sha256(args.noisy),
        'split_method': 'BondNet fixed GEOM FNV-1a by geom_mol_idx, seed 42, 80/10/10',
        'split_counts': dict(counts),
        'fixed_test_sdf_indices': [i for i, label in enumerate(labels) if label == 'test'],
        'fixed_test_geom_mol_idx': [mol_id for mol_id, label in zip(clean_ids, labels) if label == 'test'],
        'interpretation': 'Only the fixed-test subset is training-disjoint for corrected BondNet; YuelBond split remains unverified.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(counts)
    print(args.output)


if __name__ == '__main__':
    main()
