"""Audit element-pair distributions and stratified BondNet errors.

The fixed training split is used only to define empirical support. Predictions
are made on the untouched fixed-test cache. Each undirected heavy-heavy pair is
counted once. Pipeline confusion matrices include the no-bond class, exposing
missed and spurious connectivity alongside bond-typing errors.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bondnet.data.dataset import CachedBondNetDataset, collate_fn
from export_per_molecule_stats import load_stage2, predict
import evaluate as ev


BOND_NAMES = ('single', 'double', 'triple', 'aromatic')
PIPELINE_NAMES = ('no_bond',) + BOND_NAMES


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train_cache', default=None,
                   help='Monolithic cache with fixed split labels (training support is counted '
                        'from its train split). Not needed with --train_distribution_csv.')
    p.add_argument('--train_distribution_csv', default=None,
                   help='Reuse train_pair_bond_distribution.csv from an earlier run on the same '
                        'fixed training split instead of re-scanning the training cache.')
    p.add_argument('--test_cache', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--stage2_ckpt', default=None,
                   help='Stage-2 checkpoint for the hard two-stage model; omit for the joint model')
    p.add_argument('--pair_policy', choices=['undirected_or', 'directed_src_lt_dst'],
                   default='undirected_or',
                   help='undirected_or (default) matches the main table: an unordered HH pair is '
                        'predicted bonded if either directed record is, and the two type labels '
                        'must agree. directed_src_lt_dst reproduces the earlier v2 strata files.')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--noise_levels', type=float, nargs='+', default=[0.0, 0.1, 0.2])
    p.add_argument('--eval_noise_seed', type=int, default=20260921)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--cutoff', type=float, default=2.5)
    p.add_argument('--h_cutoff', type=float, default=2.5)
    p.add_argument('--conn_threshold', type=float, default=0.5)
    p.add_argument('--device', default='auto')
    return p.parse_args()


def pair_name(a: int, b: int) -> str:
    lo, hi = sorted((int(a), int(b)))
    return f'{lo}-{hi}'


def iter_true_hh_bonds(sample):
    elems = sample['elems']
    edge = sample['edge_index_bond']
    types = sample['bond_type_full'].long()
    keep = (edge[:, 0] < edge[:, 1]) & (elems[edge[:, 0]] != 1) & (elems[edge[:, 1]] != 1)
    for (u, v), bond_type in zip(edge[keep].tolist(), types[keep].tolist()):
        yield pair_name(elems[u], elems[v]), int(bond_type)


def training_distribution(path: Path):
    ds = CachedBondNetDataset(str(path))
    pair_counts = Counter()
    type_counts = Counter()
    joint_counts = Counter()
    n_train = 0
    for sample in ds.samples:
        if sample.get('split') != 'train':
            continue
        n_train += 1
        for pair, bond_type in iter_true_hh_bonds(sample):
            pair_counts[pair] += 1
            type_counts[bond_type] += 1
            joint_counts[(pair, bond_type)] += 1
        if n_train % 50000 == 0:
            print(f'[TRAIN] {n_train} molecules scanned', flush=True)
    return n_train, pair_counts, type_counts, joint_counts


def training_distribution_from_csv(path: Path):
    pair_counts, type_counts, joint_counts = Counter(), Counter(), Counter()
    with path.open(encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            pair, bond_type, count = row['element_pair'], BOND_NAMES.index(row['bond_type']), int(row['count'])
            pair_counts[pair] += count
            type_counts[bond_type] += count
            joint_counts[(pair, bond_type)] += count
    return None, pair_counts, type_counts, joint_counts


def undirected_or_classes(edge, true_class, pred_class, elems):
    """One record per unordered HH pair; OR of the two directed predictions."""
    src = edge[:, 0].detach().cpu().numpy().astype(np.int64)
    dst = edge[:, 1].detach().cpu().numpy().astype(np.int64)
    el = elems.detach().cpu().numpy()
    tc_all = true_class.detach().cpu().numpy()
    pc_all = pred_class.detach().cpu().numpy()
    hh = (el[src] != 1) & (el[dst] != 1)
    n = int(max(src.max(initial=0), dst.max(initial=0))) + 1
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    key = lo * n + hi
    idx = np.where(hh)[0]
    order = idx[np.argsort(key[idx], kind='stable')]
    keys_sorted = key[order]
    starts = np.flatnonzero(np.r_[True, keys_sorted[1:] != keys_sorted[:-1]])
    ends = np.r_[starts[1:], len(order)]
    rep_idx, tc, pc = [], [], []
    for a, b in zip(starts, ends):
        members = order[a:b]
        t = set(int(x) for x in tc_all[members])
        if len(t) != 1:
            raise ValueError('conflicting reference labels for one unordered pair')
        p = {int(x) for x in pc_all[members] if int(x) > 0}
        if len(p) > 1:
            raise ValueError('conflicting predicted types for one unordered pair')
        rep_idx.append(int(members[0]))
        tc.append(t.pop())
        pc.append(p.pop() if p else 0)
    rep_idx = np.asarray(rep_idx, dtype=np.int64)
    lo_hi = np.stack([lo[rep_idx], hi[rep_idx]], axis=1)
    return rep_idx, lo_hi, np.asarray(tc, dtype=np.int64), np.asarray(pc, dtype=np.int64)


def support_stratum(n: int) -> str:
    if n == 0:
        return 'unseen'
    if n < 100:
        return 'rare_1_99'
    if n < 1000:
        return 'medium_100_999'
    return 'common_ge_1000'


def size_stratum(n_heavy: int) -> str:
    if n_heavy <= 20:
        return 'heavy_le_20'
    if n_heavy <= 30:
        return 'heavy_21_30'
    return 'heavy_ge_31'


def metrics_from_confusion(confusion):
    cm = np.asarray(confusion, dtype=np.int64)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    support = cm.sum(axis=1)
    present = support > 0
    return {
        'n_pairs': int(cm.sum()),
        'accuracy': float(tp.sum() / cm.sum()) if cm.sum() else None,
        'f1_per_class': f1.tolist(),
        'macro_f1_present_classes': float(f1[present].mean()) if present.any() else None,
        'support_per_class': support.tolist(),
        'confusion_matrix_true_rows_pred_columns': cm.tolist(),
    }


def true_bond_summary(rows):
    n = sum(r['n'] for r in rows)
    correct = sum(r['correct'] for r in rows)
    missed = sum(r['missed'] for r in rows)
    detected = n - missed
    return {
        'n_true_bonds': int(n),
        'correct_pipeline': int(correct),
        'missed_connectivity': int(missed),
        'pipeline_bond_accuracy': float(correct / n) if n else None,
        'missed_connectivity_rate': float(missed / n) if n else None,
        'typing_accuracy_given_detected': float(correct / detected) if detected else None,
    }


@torch.no_grad()
def evaluate_sigma(model, stage2, dataset, device, args, sigma, train_pair, train_joint):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    pipeline_cm = np.zeros((5, 5), dtype=np.int64)
    reference_cm = np.zeros((4, 5), dtype=np.int64)
    pair_stats = defaultdict(lambda: {'n': 0, 'correct': 0, 'missed': 0})
    joint_stats = defaultdict(lambda: {'n': 0, 'correct': 0, 'missed': 0})
    size_cms = defaultdict(lambda: np.zeros((5, 5), dtype=np.int64))
    size_true = defaultdict(lambda: {'n': 0, 'correct': 0, 'missed': 0})
    n_done = 0

    for batch in loader:
        batch, pred_conn, pred_type = predict(model, stage2, batch, device, args, sigma)
        edge = batch['edge_index']
        elems = batch['elems']
        src, dst = edge[:, 0], edge[:, 1]
        unique_hh = (src < dst) & (elems[src] != 1) & (elems[dst] != 1)

        true_class = torch.zeros(edge.shape[0], dtype=torch.long, device=device)
        true_pos = torch.where(batch['bond_mask'].bool())[0]
        true_class[true_pos] = batch['bond_type_full'].long() + 1
        pred_class = torch.where(pred_type >= 0, pred_type + 1, torch.zeros_like(pred_type))

        if args.pair_policy == 'undirected_or':
            sel_np, pair_nodes, tc, pc = undirected_or_classes(edge, true_class, pred_class, elems)
            sel = torch.as_tensor(sel_np, dtype=torch.long, device=edge.device)
        else:
            sel = torch.where(unique_hh)[0]
            tc = true_class[sel].detach().cpu().numpy()
            pc = pred_class[sel].detach().cpu().numpy()
        np.add.at(pipeline_cm, (tc, pc), 1)

        edge_mol = batch['edge_mol_idx'][sel].detach().cpu().numpy()
        n_atoms = batch['num_atoms_per_mol'].detach().cpu().tolist()
        atom_starts = np.cumsum([0] + n_atoms[:-1]).tolist()
        elem_cpu = elems.detach().cpu()
        edge_cpu = edge[sel].detach().cpu()
        heavy_counts = [int((elem_cpu[start:start + count] != 1).sum())
                        for start, count in zip(atom_starts, n_atoms)]

        for mol_i in range(len(n_atoms)):
            mask = edge_mol == mol_i
            if not mask.any():
                continue
            stratum = size_stratum(heavy_counts[mol_i])
            np.add.at(size_cms[stratum], (tc[mask], pc[mask]), 1)

        for j in np.where(tc > 0)[0].tolist():
            u, v = edge_cpu[j].tolist()
            pair = pair_name(elem_cpu[u], elem_cpu[v])
            bond_type = int(tc[j] - 1)
            pred = int(pc[j])
            correct = int(pred == int(tc[j]))
            missed = int(pred == 0)
            pair_stats[pair]['n'] += 1
            pair_stats[pair]['correct'] += correct
            pair_stats[pair]['missed'] += missed
            joint_stats[(pair, bond_type)]['n'] += 1
            joint_stats[(pair, bond_type)]['correct'] += correct
            joint_stats[(pair, bond_type)]['missed'] += missed
            reference_cm[bond_type, pred] += 1
            mol_i = int(edge_mol[j])
            stratum = size_stratum(heavy_counts[mol_i])
            size_true[stratum]['n'] += 1
            size_true[stratum]['correct'] += correct
            size_true[stratum]['missed'] += missed

        n_done += int(batch['num_mols'])
        if n_done % 5120 == 0:
            print(f'[TEST] sigma={sigma:g} {n_done}/{len(dataset)}', flush=True)

    pair_rows = []
    for pair, stats in sorted(pair_stats.items()):
        row = {'element_pair': pair, 'train_count': int(train_pair[pair]), **stats}
        row.update(true_bond_summary([stats]))
        pair_rows.append(row)

    joint_rows = []
    for (pair, bond_type), stats in sorted(joint_stats.items()):
        support = int(train_joint[(pair, bond_type)])
        row = {
            'element_pair': pair,
            'bond_type': BOND_NAMES[bond_type],
            'train_count': support,
            'support_stratum': support_stratum(support),
            **stats,
        }
        row.update(true_bond_summary([stats]))
        joint_rows.append(row)

    support_rows = []
    for name in ('unseen', 'rare_1_99', 'medium_100_999', 'common_ge_1000'):
        members = [r for r in joint_rows if r['support_stratum'] == name]
        support_rows.append({'support_stratum': name, **true_bond_summary(members)})

    size_rows = []
    for name in ('heavy_le_20', 'heavy_21_30', 'heavy_ge_31'):
        size_rows.append({
            'size_stratum': name,
            'pipeline_candidate_metrics': metrics_from_confusion(size_cms[name]),
            'true_bond_metrics': true_bond_summary([size_true[name]]),
        })

    return {
        'sigma': sigma,
        'n_molecules': len(dataset),
        'pipeline_labels': list(PIPELINE_NAMES),
        'pipeline_candidate_metrics': metrics_from_confusion(pipeline_cm),
        # Same quantity as the main-table pipeline macro-F1 (bond classes only;
        # no-bond predictions count as misses / no-bond truths as extras).
        'bond_class_pipeline_macro_f1': float(np.mean(
            metrics_from_confusion(pipeline_cm)['f1_per_class'][1:])),
        'pair_policy': args.pair_policy,
        'reference_bond_true_labels': list(BOND_NAMES),
        'reference_bond_pred_labels': list(PIPELINE_NAMES),
        'reference_bond_confusion_true_rows_pred_columns': reference_cm.tolist(),
        'reference_bond_metrics': true_bond_summary(list(joint_stats.values())),
        'support_strata': support_rows,
        'molecule_size_strata': size_rows,
        'pair_rows': pair_rows,
        'joint_rows': joint_rows,
    }


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys and not isinstance(row[key], (dict, list)):
                keys.append(key)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in keys} for row in rows)


def main():
    args = parse_args()
    output = (ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.train_distribution_csv:
        n_train, train_pair, train_type, train_joint = training_distribution_from_csv(
            (ROOT / args.train_distribution_csv).resolve())
    elif args.train_cache:
        n_train, train_pair, train_type, train_joint = training_distribution(
            (ROOT / args.train_cache).resolve())
    else:
        raise SystemExit('provide --train_cache or --train_distribution_csv')
    write_csv(output / 'train_pair_bond_distribution.csv', [
        {'element_pair': pair, 'bond_type': BOND_NAMES[bond_type], 'count': count}
        for (pair, bond_type), count in sorted(train_joint.items())
    ])

    device = ev._get_device(args.device)
    model = ev.load_model(str((ROOT / args.checkpoint).resolve()), device)
    model.eval()
    stage2 = (load_stage2(str((ROOT / args.stage2_ckpt).resolve()), model, device)
              if args.stage2_ckpt else None)
    dataset = CachedBondNetDataset(str((ROOT / args.test_cache).resolve()))

    summaries = []
    for sigma in args.noise_levels:
        sigma = float(sigma)
        seed = int(args.eval_noise_seed + round(sigma * 10000.0))
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32 - 1))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        result = evaluate_sigma(model, stage2, dataset, device, args, sigma,
                                train_pair, train_joint)
        tag = f'{round(100 * sigma):03d}'
        write_csv(output / f'pair_performance_sigma_{tag}.csv', result.pop('pair_rows'))
        joint_rows = result.pop('joint_rows')
        write_csv(output / f'pair_type_performance_sigma_{tag}.csv', joint_rows)
        if sigma == 0.0:
            test_distribution = {
                (r['element_pair'], BOND_NAMES.index(r['bond_type'])): int(r['n'])
                for r in joint_rows
            }
            comparison = []
            for key in sorted(set(train_joint) | set(test_distribution)):
                pair, bond_type = key
                comparison.append({
                    'element_pair': pair,
                    'bond_type': BOND_NAMES[bond_type],
                    'train_count': int(train_joint[key]),
                    'test_count': int(test_distribution.get(key, 0)),
                    'support_stratum': support_stratum(int(train_joint[key])),
                })
            write_csv(output / 'train_test_pair_bond_distribution.csv', comparison)
        (output / f'analysis_sigma_{tag}.json').write_text(
            json.dumps(result, indent=2), encoding='utf-8'
        )
        summaries.append(result)
        print(
            f'[RESULT] sigma={sigma:g} pipeline_macro='
            f'{result["pipeline_candidate_metrics"]["macro_f1_present_classes"]:.6f} '
            f'bond_class_macro={result["bond_class_pipeline_macro_f1"]:.6f} '
            f'true_bond_accuracy={result["reference_bond_metrics"]["pipeline_bond_accuracy"]:.6f}',
            flush=True,
        )

    (output / 'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    (output / 'protocol.json').write_text(json.dumps({
        'train_cache': args.train_cache,
        'test_cache': args.test_cache,
        'checkpoint': args.checkpoint,
        'stage2_ckpt': args.stage2_ckpt,
        'model_family': 'joint' if not args.stage2_ckpt else 'hard two-stage',
        'pair_policy': args.pair_policy,
        'train_distribution_csv': args.train_distribution_csv,
        'fixed_train_molecules': n_train,
        'fixed_test_molecules': len(dataset),
        'noise_levels': args.noise_levels,
        'eval_noise_seed': args.eval_noise_seed,
        'noise_protocol': 'molecule-keyed Gaussian noise generated on CPU before device transfer',
        'undirected_heavy_heavy_pairs_counted_once': True,
        'candidate_confusion_scope': '3.0-A clean cache envelope; bond-class F1 excludes no-bond class',
        'support_definition': 'joint element-pair and bond-type count in fixed training split',
    }, indent=2), encoding='utf-8')
    print(f'[DONE] wrote {output}', flush=True)


if __name__ == '__main__':
    main()
