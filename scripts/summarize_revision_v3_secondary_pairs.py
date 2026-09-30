"""Summarize three-seed secondary diagnostics under undirected HH-pair scoring."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path('results/revision_v3_secondary_pair_audit')
MAIN = Path('results/revision_v3_undirected_audit/joint')
OLD_H = Path('results/revision_v3_joint_h_corruption')
OLD_STRUCTURED = Path('results/revision_v3_joint_structured_directed')
SEEDS = (42, 43, 44)
CONDITIONS = ('intact', 'drop025', 'drop050', 'drop100',
              'hnoise010', 'hnoise020', 'hnoise030')
DISTORTIONS = ('clean', 'scale_1p05', 'scale_1p10',
               'anisotropic', 'sparse_outlier')
FIELDS = ('undirected_pipeline_macro_f1', 'undirected_hh_graph_exact_rate')


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def check_close(actual: float, expected: float, label: str, tol=1e-5):
    if abs(actual - expected) > tol:
        raise ValueError(f'{label}: observed {actual}, expected {expected}')


def summarize(rows: dict[int, dict], name: str) -> dict:
    out = {'condition': name, 'n_seeds': 3, 'n_molecules': 27240}
    for field in FIELDS:
        values = [float(rows[seed][field]) for seed in SEEDS]
        out[field + '_mean'] = statistics.mean(values)
        out[field + '_sample_sd'] = statistics.stdev(values)
        out[field + '_by_seed'] = {str(seed): rows[seed][field] for seed in SEEDS}
    return out


def main() -> None:
    h_summary = []
    for condition in CONDITIONS:
        rows = {}
        for seed in SEEDS:
            data = read(ROOT / 'h_corruption' / f'seed{seed}' / condition / 'aggregate.json')
            if len(data) != 1 or data[0]['sigma'] != 0 or data[0]['n_molecules'] != 27240:
                raise ValueError(f'Unexpected H-cohort: {condition}, seed {seed}')
            row = data[0]
            old = read(OLD_H / f'seed{seed}' / condition / 'results.json')['BondNet'][0]
            check_close(row['pipeline_macro_f1'], old['f1_macro_pipeline'], 'H directed F1')
            check_close(row['hh_graph_exact_rate'], old['full_graph_exact_match'], 'H directed exact', 1e-12)
            if condition == 'intact':
                main = read(MAIN / f'seed{seed}' / 'aggregate.json')[0]
                for field in FIELDS:
                    check_close(row[field], main[field], f'intact H {field}', 1e-12)
            rows[seed] = row
        h_summary.append(summarize(rows, condition))

    structured = {name: {} for name in DISTORTIONS}
    for seed in SEEDS:
        new = read(ROOT / 'structured' / f'seed{seed}' / 'results.json')
        old = read(OLD_STRUCTURED / f'seed{seed}' / 'results.json')
        if [row['distortion'] for row in new] != list(DISTORTIONS):
            raise ValueError(f'Unexpected distortion protocol: seed {seed}')
        for row, previous in zip(new, old):
            name = row['distortion']
            if previous['distortion'] != name or row['n_molecules'] != 27240:
                raise ValueError(f'Unexpected distortion cohort: seed {seed}, {name}')
            check_close(row['pipeline_macro_f1'], previous['pipeline_macro_f1'], 'structured directed F1', 1e-12)
            check_close(row['hh_graph_exact_rate'], previous['hh_graph_exact_rate'], 'structured directed exact', 1e-12)
            if name == 'clean':
                main = read(MAIN / f'seed{seed}' / 'aggregate.json')[0]
                for field in FIELDS:
                    check_close(row[field], main[field], f'clean structured {field}', 1e-12)
            structured[name][seed] = row

    structured_summary = []
    for name in DISTORTIONS:
        item = summarize(structured[name], name)
        values = [structured[name][seed]['mean_abs_reference_bond_length_change_A'] for seed in SEEDS]
        item['mean_abs_reference_bond_length_change_A_mean'] = statistics.mean(values)
        item['mean_abs_reference_bond_length_change_A_sample_sd'] = statistics.stdev(values)
        structured_summary.append(item)

    (ROOT / 'h_corruption_three_seed.json').write_text(json.dumps(h_summary, indent=2) + '\n', encoding='utf-8')
    (ROOT / 'structured_three_seed.json').write_text(json.dumps(structured_summary, indent=2) + '\n', encoding='utf-8')
    for family, items in (('H', h_summary), ('structured', structured_summary)):
        for row in items:
            print(f"{family:<10} {row['condition']:<18} "
                  f"F1 {row['undirected_pipeline_macro_f1_mean']:.6f} "
                  f"HH {100 * row['undirected_hh_graph_exact_rate_mean']:.3f}%")


if __name__ == '__main__':
    main()
