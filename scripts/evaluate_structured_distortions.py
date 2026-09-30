"""Evaluate BondNet under controlled non-Gaussian coordinate distortions.

These are diagnostic stress tests, not substitutes for outputs from a molecular
generator.  They isolate systematic scale bias, anisotropic coordinate strain,
and sparse local coordinate outliers on the fixed test cohort.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluate as ev
from bondnet.data.dataset import CachedBondNetDataset, collate_fn
from export_per_molecule_stats import (
    class_counts, load_stage2, predict, undirected_hh_labels,
)


DISTORTIONS = ('clean', 'scale_1p05', 'scale_1p10', 'anisotropic', 'sparse_outlier')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--stage2_ckpt', default=None,
                   help='Optional hard two-stage bond-type checkpoint; omit for the joint model.')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--distortions', nargs='+', choices=DISTORTIONS, default=list(DISTORTIONS))
    p.add_argument('--seed', type=int, default=20260921)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--cutoff', type=float, default=2.5)
    p.add_argument('--h_cutoff', type=float, default=2.5)
    p.add_argument('--conn_threshold', type=float, default=0.5)
    p.add_argument('--device', default='auto')
    return p.parse_args()


def recompute_geometry(batch, coord):
    edge = batch['edge_index']
    diff = coord[edge[:, 1]] - coord[edge[:, 0]]
    out = dict(batch)
    out['coord'] = coord
    out['edge_diff'] = diff
    out['edge_dist'] = diff.norm(dim=-1)
    # The revision checkpoints do not consume cached local_geom. Dropping it is
    # safer than retaining geometry from the undistorted coordinates.
    out.pop('local_geom', None)
    return out


def molecule_centers(coord, counts):
    centers = []
    offset = 0
    for count in counts:
        centers.append(coord[offset:offset + count].mean(dim=0))
        offset += count
    return torch.repeat_interleave(torch.stack(centers), torch.as_tensor(counts, device=coord.device), dim=0)


def apply_distortion(batch, name):
    if name == 'clean':
        return batch
    coord = batch['coord']
    counts = [int(x) for x in batch['num_atoms_per_mol'].detach().cpu().tolist()]
    centers = molecule_centers(coord, counts)
    centered = coord - centers
    if name.startswith('scale_'):
        factor = 1.05 if name == 'scale_1p05' else 1.10
        new_coord = centers + factor * centered
    elif name == 'anisotropic':
        # Fixed laboratory-frame calibration strain. Dataset orientations are
        # arbitrary, so this probes direction-dependent rather than rigid motion.
        factors = torch.tensor([1.15, 0.90, 1.00], device=coord.device, dtype=coord.dtype)
        new_coord = centers + centered * factors
    elif name == 'sparse_outlier':
        new_coord = coord.clone()
        elems = batch['elems']
        offset = 0
        for count in counts:
            local = torch.arange(offset, offset + count, device=coord.device)
            heavy = local[elems[local] != 1]
            n_move = max(1, int(math.ceil(0.10 * heavy.numel())))
            chosen = heavy[torch.randperm(heavy.numel(), device=coord.device)[:n_move]]
            direction = torch.randn((n_move, 3), device=coord.device, dtype=coord.dtype)
            direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
            new_coord[chosen] += 0.40 * direction
            offset += count
    else:
        raise ValueError(name)
    return recompute_geometry(batch, new_coord)


def prf(confusion):
    cm = np.asarray(confusion, dtype=np.int64)
    values = []
    for cls in range(1, 5):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        denom = 2 * tp + fp + fn
        values.append(float(2 * tp / denom) if denom else 0.0)
    return values, float(np.mean(values))


@torch.no_grad()
def evaluate_distortion(model, stage2, dataset, device, args, name):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    cm = np.zeros((5, 5), dtype=np.int64)
    u_tp = np.zeros(4, dtype=np.int64)
    u_fp = np.zeros(4, dtype=np.int64)
    u_fn = np.zeros(4, dtype=np.int64)
    u_true_exact = []
    u_graph_exact = []
    conn = np.zeros(3, dtype=np.int64)  # tp, fp, fn
    true_exact = []
    graph_exact = []
    displacement_sq = 0.0
    displacement_n = 0
    bond_delta = 0.0
    bond_n = 0
    n_done = 0

    for batch in loader:
        batch = ev._to_device(batch, device)
        original_coord = batch['coord'].clone()
        original_dist = batch['edge_dist'].clone()
        batch = apply_distortion(batch, name)
        displacement_sq += float(((batch['coord'] - original_coord) ** 2).sum().item())
        displacement_n += int(batch['coord'].shape[0])
        true_pos_all = torch.where(batch['bond_mask'].bool())[0]
        true_edge = batch['edge_index'][true_pos_all]
        true_unique = true_edge[:, 0] < true_edge[:, 1]
        unique_true_pos = true_pos_all[true_unique]
        bond_delta += float((batch['edge_dist'][unique_true_pos] - original_dist[unique_true_pos]).abs().sum().item())
        bond_n += int(unique_true_pos.numel())

        batch, pred_conn, pred_type = predict(model, stage2, batch, device, args, sigma=0.0)
        edge = batch['edge_index']
        elems = batch['elems']
        src, dst = edge[:, 0], edge[:, 1]
        valid = ~((elems[src] == 1) & (elems[dst] == 1))
        true_conn = batch['bond_mask'].bool()
        conn[0] += int((pred_conn & true_conn & valid).sum().item())
        conn[1] += int((pred_conn & ~true_conn & valid).sum().item())
        conn[2] += int((~pred_conn & true_conn & valid).sum().item())

        true_class = torch.zeros(edge.shape[0], dtype=torch.long, device=device)
        true_class[true_pos_all] = batch['bond_type_full'].long() + 1
        pred_class = torch.where(pred_type >= 0, pred_type + 1, torch.zeros_like(pred_type))
        # Retain the historical directed-cache scorer for an exact regression
        # gate; independently accumulate the final one-pair-per-HH-bond score.
        hh_records = (elems[src] != 1) & (elems[dst] != 1)
        sel = torch.where(hh_records)[0]
        tc = true_class[sel].detach().cpu().numpy()
        pc = pred_class[sel].detach().cpu().numpy()
        np.add.at(cm, (tc, pc), 1)

        edge_mol = batch['edge_mol_idx'][sel].detach().cpu().numpy()
        hh_edge = edge[sel].detach().cpu().numpy()
        mol_ids = batch['mol_idx'].detach().cpu().numpy()
        for mol_i in range(int(batch['num_mols'])):
            m = edge_mol == mol_i
            tm, pm = tc[m], pc[m]
            bonded = tm > 0
            true_exact.append(bool(np.all(tm[bonded] == pm[bonded])))
            graph_exact.append(bool(np.array_equal(tm, pm)))
            u_true, u_pred = undirected_hh_labels(
                hh_edge[m], tm - 1, pm - 1, int(mol_ids[mol_i])
            )
            tp, fp, fn = class_counts(u_true, u_pred, include_extra=True)
            u_tp += tp; u_fp += fp; u_fn += fn
            u_bonded = u_true >= 0
            u_true_exact.append(bool(np.array_equal(u_true[u_bonded], u_pred[u_bonded])))
            u_graph_exact.append(bool(np.array_equal(u_true, u_pred)))

        n_done += int(batch['num_mols'])
        if n_done % 5120 == 0:
            print(f'[DISTORT] {name} {n_done}/{len(dataset)}', flush=True)

    f1s, macro = prf(cm)
    u_denom = 2 * u_tp + u_fp + u_fn
    u_f1s = np.divide(2 * u_tp, u_denom, out=np.zeros(4, dtype=float), where=u_denom > 0)
    u_macro = float(u_f1s.mean())
    tp, fp, fn = conn.tolist()
    cp = tp / (tp + fp) if tp + fp else 0.0
    cr = tp / (tp + fn) if tp + fn else 0.0
    cf1 = 2 * cp * cr / (cp + cr) if cp + cr else 0.0
    return {
        'distortion': name,
        'n_molecules': len(dataset),
        'coordinate_rms_displacement_A': math.sqrt(displacement_sq / displacement_n),
        'mean_abs_reference_bond_length_change_A': bond_delta / bond_n,
        'connectivity_precision': cp,
        'connectivity_recall': cr,
        'connectivity_f1': cf1,
        'pipeline_f1_per_bond_class': dict(zip(('single', 'double', 'triple', 'aromatic'), f1s)),
        'pipeline_macro_f1': macro,
        'true_bond_exact_rate': float(np.mean(true_exact)),
        'hh_graph_exact_rate': float(np.mean(graph_exact)),
        'undirected_pipeline_f1_per_bond_class': dict(zip(('single', 'double', 'triple', 'aromatic'), u_f1s.tolist())),
        'undirected_pipeline_macro_f1': u_macro,
        'undirected_true_bond_exact_rate': float(np.mean(u_true_exact)),
        'undirected_hh_graph_exact_rate': float(np.mean(u_graph_exact)),
        'pipeline_confusion_labels': ['no_bond', 'single', 'double', 'triple', 'aromatic'],
        'pipeline_confusion_true_rows_pred_columns': cm.tolist(),
    }


def main():
    args = parse_args()
    device = ev._get_device(args.device)
    model = ev.load_model(str((ROOT / args.checkpoint).resolve()), device)
    model.eval()
    stage2 = load_stage2(str((ROOT / args.stage2_ckpt).resolve()), model, device) if args.stage2_ckpt else None
    dataset = CachedBondNetDataset(str((ROOT / args.cache).resolve()))
    output = (ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    results = []
    for index, name in enumerate(args.distortions):
        seed = args.seed + index * 100003
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32 - 1))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        result = evaluate_distortion(model, stage2, dataset, device, args, name)
        results.append(result)
        print(f'[RESULT] {name} macro={result["pipeline_macro_f1"]:.6f} '
              f'hh_exact={result["hh_graph_exact_rate"]:.6f}', flush=True)

    (output / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    (output / 'protocol.json').write_text(json.dumps({
        'cache': args.cache,
        'checkpoint': args.checkpoint,
        'stage2_ckpt': args.stage2_ckpt,
        'distortions': args.distortions,
        'seed': args.seed,
        'batch_size': args.batch_size,
        'device': str(device),
        'scope_caveat': 'Controlled synthetic stress tests; not actual molecular-generator outputs.',
        'definitions': {
            'scale_1p05': 'uniform 1.05x expansion about each molecular centroid',
            'scale_1p10': 'uniform 1.10x expansion about each molecular centroid',
            'anisotropic': 'laboratory-frame scale factors (1.15, 0.90, 1.00) about each centroid',
            'sparse_outlier': '10% of heavy atoms (ceil, at least one) displaced by exactly 0.40 A in random directions',
        },
    }, indent=2), encoding='utf-8')
    print(f'[DONE] wrote {output}', flush=True)


if __name__ == '__main__':
    main()
