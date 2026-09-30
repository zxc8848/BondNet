"""Paired molecule bootstrap for final undirected joint explicit-H vs heavy-only scores.

Intervals condition on the fitted training seeds and the repeatedly inspected
fixed GEOM cohort. They do not substitute for across-seed uncertainty or an
independent test set.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np


ROOT = Path('results/revision_v3_undirected_audit')
SEEDS = (42, 43, 44)
SIGMAS = (0.0, 0.1, 0.2)


def tag(sigma: float) -> str:
    return f'{round(100 * sigma):03d}'


def macro_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float),
                     where=denom > 0).mean(axis=-1)


def load(sigma: float) -> dict:
    records = {}
    ids = None
    for model in ('joint', 'heavy'):
        for seed in SEEDS:
            path = ROOT / model / f'seed{seed}' / f'sigma_{tag(sigma)}.npz'
            with np.load(path) as archive:
                row = {key: archive[key] for key in
                       ('mol_idx', 'u_pipe_tp', 'u_pipe_fp', 'u_pipe_fn',
                        'u_hh_graph_exact')}
            if ids is None:
                ids = row['mol_idx']
            elif not np.array_equal(ids, row['mol_idx']):
                raise ValueError(f'Molecule identity/order mismatch in {path}')
            records[model, seed] = row
    if ids is None or len(ids) != 27240:
        raise ValueError('Unexpected fixed-test cohort size')
    return records


def point(row: dict, metric: str) -> float:
    if metric == 'undirected_pipeline_macro_f1':
        return float(macro_f1(row['u_pipe_tp'].sum(0), row['u_pipe_fp'].sum(0),
                              row['u_pipe_fn'].sum(0)))
    return float(row['u_hh_graph_exact'].mean())


def replicas(row: dict, metric: str, index: np.ndarray) -> np.ndarray:
    if metric == 'undirected_pipeline_macro_f1':
        return macro_f1(row['u_pipe_tp'][index].sum(1),
                        row['u_pipe_fp'][index].sum(1),
                        row['u_pipe_fn'][index].sum(1))
    return row['u_hh_graph_exact'][index].mean(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicates', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=20260929)
    parser.add_argument('--chunk-size', type=int, default=20)
    parser.add_argument('--output', type=Path,
                        default=Path('results/revision_v3_undirected_audit/hydrogen_bootstrap.json'))
    args = parser.parse_args()
    if args.replicates <= 0 or args.chunk_size <= 0:
        raise ValueError('replicates and chunk-size must be positive')
    rng = np.random.default_rng(args.bootstrap_seed)
    rows = []
    for sigma in SIGMAS:
        data = load(sigma)
        for metric in ('undirected_pipeline_macro_f1', 'undirected_hh_graph_exact_rate'):
            points = [point(data['joint', seed], metric) - point(data['heavy', seed], metric)
                      for seed in SEEDS]
            values = []
            remaining = args.replicates
            while remaining:
                size = min(args.chunk_size, remaining)
                index = rng.integers(0, 27240, size=(size, 27240), dtype=np.int32)
                differences = [replicas(data['joint', seed], metric, index) -
                               replicas(data['heavy', seed], metric, index)
                               for seed in SEEDS]
                values.extend(np.mean(differences, axis=0).tolist())
                remaining -= size
            low, high = np.quantile(values, (0.025, 0.975))
            result = {
                'sigma': sigma, 'metric': metric,
                'comparison': 'joint_explicit_minus_joint_heavy',
                'n_molecules': 27240, 'n_training_seeds': 3,
                'replicates': args.replicates,
                'point': statistics.mean(points),
                'training_seed_sample_sd_of_difference': statistics.stdev(points),
                'molecule_bootstrap_95_percentile_interval': [float(low), float(high)],
            }
            rows.append(result)
            print(f'{metric} sigma={sigma:.2f}: {result["point"]:.6f} '
                  f'[{low:.6f}, {high:.6f}]', flush=True)
    summary = json.loads((ROOT / 'three_seed_summary.json').read_text(encoding='utf-8'))
    for row in rows:
        expected = next(item for item in summary['paired_joint_minus_heavy']
                        if item['sigma'] == row['sigma'])
        if abs(row['point'] - expected[row['metric'] + '_mean_difference']) > 1e-12:
            raise ValueError(f'Point estimate differs from final main summary: {row}')
    output = {
        'protocol': {
            'unit': 'molecule', 'paired_across_models_and_training_seeds': True,
            'bootstrap_seed': args.bootstrap_seed, 'replicates': args.replicates,
            'interpretation': 'Conditional molecule-sampling interval on repeatedly inspected GEOM test; separate from training-seed sample SD and not an independent confirmation.',
        },
        'rows': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
