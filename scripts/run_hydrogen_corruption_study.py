"""Run deterministic missing/misplaced-hydrogen evaluation for revision M6."""

import argparse
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--python', default='python')
    p.add_argument(
        '--cache_path',
        default='data/fixed_split_caches/geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed_test.pt',
    )
    p.add_argument('--output_root', default='results/revision_hydrogen_corruption')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--generation', choices=['v1', 'v2'], default='v1')
    p.add_argument('--skip_completed', action='store_true')
    return p.parse_args()


def checkpoints(seed, generation):
    if generation == 'v2':
        return (
            ROOT / 'checkpoints' / f'revision_v2_seed{seed}' / 'explicit_stage1' / 'best_e2e.pt',
            ROOT / 'checkpoints' / f'revision_v2_seed{seed}' / 'explicit_stage2' / 'best_e2e.pt',
        )
    return (
        ROOT / 'checkpoints' / f'revision_matched_seed{seed}' / 'two_stage_stage1' / 'best.pt',
        ROOT / 'checkpoints' / f'revision_predicted_topology_seed{seed}' / 'best.pt',
    )


CONDITIONS = [
    ('intact', 0.0, 0.0),
    ('drop025', 0.25, 0.0),
    ('drop050', 0.50, 0.0),
    ('drop100', 1.00, 0.0),
    ('hnoise010', 0.0, 0.10),
    ('hnoise020', 0.0, 0.20),
    ('hnoise030', 0.0, 0.30),
]


def main():
    args = parse_args()
    output_root = (ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for seed in args.seeds:
        stage1, stage2 = checkpoints(seed, args.generation)
        if not stage1.exists() or not stage2.exists():
            raise FileNotFoundError(f'Missing checkpoint for seed {seed}: {stage1}, {stage2}')
        for name, drop_fraction, h_noise in CONDITIONS:
            out_dir = output_root / f'seed{seed}' / name
            cmd = [
                args.python, str(ROOT / 'evaluate.py'),
                '--checkpoint', str(stage1),
                '--stage2_ckpt', str(stage2),
                '--allow_stage2_cache_mismatch',
                '--cache_path', str(ROOT / args.cache_path),
                '--explicit_h', '--cutoff', '2.5', '--h_cutoff', '2.5',
                '--eval_split', 'test', '--noise_levels', '0',
                '--output_dir', str(out_dir),
                '--batch_size', str(args.batch_size), '--num_workers', '0',
                '--device', 'auto', '--split_key', 'geom_mol_idx',
                '--conn_threshold', '0.5', '--eval_noise_seed', '20260921',
                '--hydrogen_drop_fraction', str(drop_fraction),
                '--hydrogen_noise_sigma', str(h_noise),
                '--hydrogen_corruption_seed', '20260925',
            ]
            if args.skip_completed and (out_dir / 'results.json').exists():
                print(f'[SKIP] seed={seed} condition={name}', flush=True)
            else:
                print(f'[RUN] seed={seed} condition={name}', flush=True)
                subprocess.run(cmd, cwd=ROOT, check=True)
            result = json.loads((out_dir / 'results.json').read_text())['BondNet'][0]
            rows.append({
                'seed': seed,
                'condition': name,
                'hydrogen_drop_fraction': drop_fraction,
                'hydrogen_noise_sigma': h_noise,
                'reference_pair_f1': result['f1_macro'],
                'pipeline_f1': result['f1_macro_pipeline'],
                'hh_graph_exact': result['full_graph_exact_match'],
                'conn_f1': result['conn_f1'],
                'reference_bond_exact': result['mol_validity'],
                'n_molecules': result['n_molecules'],
            })

    fields = list(rows[0])
    with (output_root / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_root / 'protocol.json').write_text(json.dumps({
        'seeds': args.seeds,
        'conditions': [
            {'name': n, 'hydrogen_drop_fraction': d, 'hydrogen_noise_sigma': h}
            for n, d, h in CONDITIONS
        ],
        'global_coordinate_noise_sigma': 0.0,
        'hydrogen_corruption_seed': 20260925,
        'test_cache': args.cache_path,
        'checkpoint_generation': args.generation,
        'checkpoint_paths': {
            str(seed): [str(path) for path in checkpoints(seed, args.generation)]
            for seed in args.seeds
        },
    }, indent=2))
    print(f'[DONE] wrote {output_root / "summary.csv"}', flush=True)


if __name__ == '__main__':
    main()
