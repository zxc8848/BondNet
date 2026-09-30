"""Summarize the exploratory 478-molecule shared-cohort fixed-test subset."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path('results/p0c/joint_v3')
SEEDS = (42, 43, 44)
CONDITIONS = ('clean', 'noisy')
METHODS = ('yuelbond', 'rdkit', 'openbabel')
FIELDS = ('f1_macro_pipeline', 'true_bond_exact_type_rate',
          'full_graph_exact_match', 'extra_bonds_per_successful_molecule')


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def check(row: dict, sha: str, path: Path) -> None:
    if row['n_molecules'] != 478 or row['fixed_split_subset'] != 'test':
        raise ValueError(f'Wrong fixed-test subset: {path}')
    if row['sdf']['sdf_sha256'] != sha:
        raise ValueError(f'Wrong SDF fingerprint: {path}')
    if not (0 <= row['success_rate'] <= 1):
        raise ValueError(f'Invalid success rate: {path}')


def main() -> None:
    audit = read(ROOT / 'shared_fixed_split_audit.json')
    if audit['split_counts'] != {'train': 3994, 'val': 528, 'test': 478}:
        raise ValueError('Unexpected shared-cohort split counts')
    output = {'scope': 'Exploratory fixed-test overlap of the archived shared GEOM cohort, not independent external validation',
              'n_molecules': 478, 'split_audit': str(ROOT / 'shared_fixed_split_audit.json'),
              'conditions': {}}
    for condition in CONDITIONS:
        sha = audit[f'{condition}_sdf_sha256']
        seed_rows = []
        for seed in SEEDS:
            path = ROOT / f'seed{seed}_{condition}_score.json'
            row = read(path)
            check(row, sha, path)
            if row['success_rate'] != 1:
                raise ValueError(f'Joint inference omitted molecules: {path}')
            seed_rows.append(row)
        joint = {'method': 'joint explicit-H', 'n_seeds': 3, 'success_rate': 1.0}
        for field in FIELDS:
            values = [float(row[field]) for row in seed_rows]
            joint[field + '_mean'] = statistics.mean(values)
            joint[field + '_sample_sd'] = statistics.stdev(values)
            joint[field + '_by_seed'] = dict(zip(map(str, SEEDS), values))
        baselines = []
        for method in METHODS:
            path = ROOT / f'{method}_{condition}_fixedtest478_score.json'
            row = read(path)
            check(row, sha, path)
            baselines.append({key: row[key] for key in
                              ('name', 'success_rate', *FIELDS)})
        output['conditions'][condition] = {'joint': joint, 'baselines': baselines}
        print(f'{condition}: joint F1 {joint["f1_macro_pipeline_mean"]:.6f} '
              f'(SD {joint["f1_macro_pipeline_sample_sd"]:.6f}), '
              f'HH {100 * joint["full_graph_exact_match_mean"]:.3f}% '
              f'(SD {100 * joint["full_graph_exact_match_sample_sd"]:.3f} pp)')
        for row in baselines:
            print(f'  {row["name"]:<10} F1 {row["f1_macro_pipeline"]:.6f} '
                  f'HH {100 * row["full_graph_exact_match"]:.3f}%')
    destination = ROOT / 'fixedtest478_summary.json'
    destination.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(destination)


if __name__ == '__main__':
    main()
