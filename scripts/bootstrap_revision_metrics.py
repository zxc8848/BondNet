"""Paired molecule bootstrap for the three-seed architecture comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stats_root', default='results/revision_bootstrap/stats')
    p.add_argument('--output_dir', default='results/revision_bootstrap')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--noise_levels', type=float, nargs='+', default=[0.0, 0.1, 0.2])
    p.add_argument('--replicates', type=int, default=2000)
    p.add_argument('--bootstrap_seed', type=int, default=20260925)
    p.add_argument('--chunk_size', type=int, default=20)
    return p.parse_args()


def tag(sigma):
    return f'{round(100 * float(sigma)):03d}'


def load_all(root, seeds, sigma):
    output = {}
    for seed in seeds:
        output[seed] = {}
        for method in ('one_stage', 'two_stage'):
            path = root / f'seed{seed}' / method / f'sigma_{tag(sigma)}.npz'
            if not path.exists():
                raise FileNotFoundError(path)
            output[seed][method] = dict(np.load(path))
    ids = output[seeds[0]]['one_stage']['mol_idx']
    for seed in seeds:
        for method in ('one_stage', 'two_stage'):
            if not np.array_equal(output[seed][method]['mol_idx'], ids):
                raise ValueError(f'molecule order mismatch: seed={seed} method={method}')
    return output, len(ids)


def f1_macro(tp, fp, fn):
    denom = 2 * tp + fp + fn
    per = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    return per.mean(axis=-1)


def point_metric(stats, metric):
    if metric == 'reference_pair_macro_f1':
        return float(f1_macro(stats['ref_tp'].sum(0), stats['ref_fp'].sum(0), stats['ref_fn'].sum(0)))
    if metric == 'pipeline_macro_f1':
        return float(f1_macro(stats['pipe_tp'].sum(0), stats['pipe_fp'].sum(0), stats['pipe_fn'].sum(0)))
    field = {
        'true_pair_exact_rate': 'true_pair_exact',
        'hh_graph_exact_rate': 'hh_graph_exact',
        'conn_exact_rate': 'conn_exact',
    }[metric]
    return float(stats[field].mean())


def bootstrap_metric(stats, metric, indices):
    if metric == 'reference_pair_macro_f1':
        tp = stats['ref_tp'][indices].sum(axis=1)
        fp = stats['ref_fp'][indices].sum(axis=1)
        fn = stats['ref_fn'][indices].sum(axis=1)
        return f1_macro(tp, fp, fn)
    if metric == 'pipeline_macro_f1':
        tp = stats['pipe_tp'][indices].sum(axis=1)
        fp = stats['pipe_fp'][indices].sum(axis=1)
        fn = stats['pipe_fn'][indices].sum(axis=1)
        return f1_macro(tp, fp, fn)
    field = {
        'true_pair_exact_rate': 'true_pair_exact',
        'hh_graph_exact_rate': 'hh_graph_exact',
        'conn_exact_rate': 'conn_exact',
    }[metric]
    return stats[field][indices].mean(axis=1)


def ci(values):
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)


def main():
    args = parse_args()
    stats_root = (ROOT / args.stats_root).resolve()
    output = (ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.bootstrap_seed)
    metrics = [
        'reference_pair_macro_f1', 'pipeline_macro_f1',
        'true_pair_exact_rate', 'hh_graph_exact_rate', 'conn_exact_rate',
    ]
    rows = []

    for sigma in args.noise_levels:
        data, n = load_all(stats_root, args.seeds, sigma)
        reps = {
            metric: {
                seed: {method: [] for method in ('one_stage', 'two_stage')}
                for seed in args.seeds
            }
            for metric in metrics
        }
        remaining = args.replicates
        while remaining > 0:
            chunk = min(args.chunk_size, remaining)
            indices = rng.integers(0, n, size=(chunk, n), dtype=np.int32)
            for metric in metrics:
                for seed in args.seeds:
                    for method in ('one_stage', 'two_stage'):
                        vals = bootstrap_metric(data[seed][method], metric, indices)
                        reps[metric][seed][method].append(vals)
            remaining -= chunk
        for metric in metrics:
            for seed in args.seeds:
                for method in ('one_stage', 'two_stage'):
                    reps[metric][seed][method] = np.concatenate(reps[metric][seed][method])
                one_point = point_metric(data[seed]['one_stage'], metric)
                two_point = point_metric(data[seed]['two_stage'], metric)
                for method, point in (('one_stage', one_point), ('two_stage', two_point)):
                    lo, hi = ci(reps[metric][seed][method])
                    rows.append({
                        'sigma': sigma, 'metric': metric, 'scope': f'seed{seed}',
                        'comparison': method, 'point': point, 'ci_low': lo, 'ci_high': hi,
                        'seed_sd': '',
                    })
                diff_rep = reps[metric][seed]['two_stage'] - reps[metric][seed]['one_stage']
                lo, hi = ci(diff_rep)
                rows.append({
                    'sigma': sigma, 'metric': metric, 'scope': f'seed{seed}',
                    'comparison': 'two_minus_one', 'point': two_point - one_point,
                    'ci_low': lo, 'ci_high': hi, 'seed_sd': '',
                })

            one_points = np.array([point_metric(data[s]['one_stage'], metric) for s in args.seeds])
            two_points = np.array([point_metric(data[s]['two_stage'], metric) for s in args.seeds])
            one_rep = np.mean([reps[metric][s]['one_stage'] for s in args.seeds], axis=0)
            two_rep = np.mean([reps[metric][s]['two_stage'] for s in args.seeds], axis=0)
            for method, points, values in (
                ('one_stage', one_points, one_rep), ('two_stage', two_points, two_rep)
            ):
                lo, hi = ci(values)
                rows.append({
                    'sigma': sigma, 'metric': metric, 'scope': 'three_seed_mean',
                    'comparison': method, 'point': float(points.mean()),
                    'ci_low': lo, 'ci_high': hi,
                    'seed_sd': float(points.std(ddof=1)),
                })
            diff_points = two_points - one_points
            diff_rep = two_rep - one_rep
            lo, hi = ci(diff_rep)
            rows.append({
                'sigma': sigma, 'metric': metric, 'scope': 'three_seed_mean',
                'comparison': 'two_minus_one', 'point': float(diff_points.mean()),
                'ci_low': lo, 'ci_high': hi,
                'seed_sd': float(diff_points.std(ddof=1)),
            })
        print(f'[BOOT] sigma={sigma:g} complete ({args.replicates} replicates)', flush=True)

    fields = list(rows[0])
    with (output / 'bootstrap_summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    (output / 'bootstrap_summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    (output / 'bootstrap_protocol.json').write_text(json.dumps({
        'unit': 'molecule', 'paired': True, 'n_molecules': n,
        'replicates': args.replicates, 'bootstrap_seed': args.bootstrap_seed,
        'seeds': args.seeds, 'noise_levels': args.noise_levels,
        'interpretation': (
            'Percentile intervals quantify fixed-test molecule-sampling uncertainty '
            'conditional on the fitted seed set. Across-training-seed variation is '
            'reported separately as sample SD and is not absorbed into the bootstrap CI.'
        ),
    }, indent=2), encoding='utf-8')
    print(f'[DONE] wrote {output / "bootstrap_summary.csv"}', flush=True)


if __name__ == '__main__':
    main()
