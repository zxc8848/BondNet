"""Export per-molecule sufficient statistics for paired bootstrap analysis."""

from __future__ import annotations

import argparse
import json
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
from bondnet.data.noise_augment import GaussianNoiseAugment, apply_dynamic_candidate_mask
from bondnet.model.bond_type_gnn import BondTypeGNN, compute_bonded_geometry
from bondnet.model.bondnet import _symmetrize_logits


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--stage2_ckpt')
    p.add_argument('--cache', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--noise_levels', type=float, nargs='+', default=[0.0, 0.1, 0.2])
    p.add_argument('--eval_noise_seed', type=int, default=20260921)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--cutoff', type=float, default=2.5)
    p.add_argument('--h_cutoff', type=float, default=2.5)
    p.add_argument('--conn_threshold', type=float, default=0.5)
    p.add_argument('--device', default='auto')
    p.add_argument('--cpu_threads', type=int, default=None)
    p.add_argument('--max_mols', type=int, default=None,
                   help='Smoke-test on the first N molecules; omit for final statistics')
    p.add_argument('--hydrogen_drop_fraction', type=float, default=0.0)
    p.add_argument('--hydrogen_noise_sigma', type=float, default=0.0)
    p.add_argument('--hydrogen_corruption_seed', type=int, default=20260925)
    return p.parse_args()


def load_stage2(path, model, device):
    if not path:
        return None
    ck = torch.load(path, map_location='cpu', weights_only=False)
    a = ck.get('args', {})
    stage2 = BondTypeGNN(
        hidden_size=a.get('hidden_size', model.backbone.hidden_state_size),
        num_layers=a.get('num_layers', 4),
        edge_embedding_size=a.get('edge_embedding_size', 16),
        bond_cutoff=a.get('bond_cutoff', 3.0),
        random_node_feat_dim=a.get('random_node_feat_dim', 0),
        random_node_feat_std=a.get('random_node_feat_std', 1.0),
    ).to(device)
    stage2.load_state_dict(ck['stage2_state_dict'])
    stage2.eval()
    return stage2


@torch.no_grad()
def predict(model, stage2, batch, device, args, sigma):
    if sigma > 0:
        aug = GaussianNoiseAugment(
            sigma=sigma, sigma_min=sigma, sigma_max=sigma, cutoff=args.cutoff
        )
        noise_seed = int(args.eval_noise_seed) + int(round(sigma * 10000.0))
        batch = aug.augment_batch(
            batch, per_molecule=True, base_seed=noise_seed, epoch=0
        )
    batch = apply_dynamic_candidate_mask(batch, args.cutoff, args.h_cutoff)
    batch = ev._to_device(batch, device)
    elems, coord = batch['elems'], batch['coord']
    edge_index = batch['edge_index']
    _, _, edge_feat = model.backbone(
        elems, coord, edge_index, batch['edge_diff'], batch['edge_dist'],
        num_atoms_per_mol=batch.get('num_atoms_per_mol'),
    )
    pred_conn = (torch.sigmoid(model.stage1(edge_feat)) >= args.conn_threshold)
    pred_conn = pred_conn & batch['train_edge_mask']
    bonded_pos = torch.where(pred_conn)[0]
    pred_type = torch.full((edge_index.shape[0],), -1, dtype=torch.long, device=device)
    if bonded_pos.numel():
        bonded_ei = edge_index[bonded_pos]
        hh = (elems[bonded_ei[:, 0]] != 1) & (elems[bonded_ei[:, 1]] != 1)
        readout_pos = bonded_pos[hh]
        readout_ei = bonded_ei[hh]
        if readout_ei.numel():
            if stage2 is None:
                logits, _ = model.stage3(edge_feat[readout_pos])
            else:
                bdiff, bdist = compute_bonded_geometry(coord, bonded_ei)
                rdiff, rdist = compute_bonded_geometry(coord, readout_ei)
                logits = stage2(elems, bonded_ei, bdiff, bdist,
                                readout_ei, rdiff, rdist)
            logits = _symmetrize_logits(logits, readout_ei)
            pred_type[readout_pos] = logits.argmax(dim=-1)
    return batch, pred_conn, pred_type


def class_counts(true, pred, include_extra):
    tp = np.zeros(4, dtype=np.int64)
    fp = np.zeros(4, dtype=np.int64)
    fn = np.zeros(4, dtype=np.int64)
    for c in range(4):
        tp[c] = int(((true == c) & (pred == c)).sum())
        fn[c] = int(((true == c) & (pred != c)).sum())
        fp[c] = int(((pred == c) & (true != c)).sum())
    if not include_extra:
        # Caller passes only reference bonds, so this flag documents the
        # intended denominator rather than changing the calculation.
        pass
    return tp, fp, fn


def undirected_hh_labels(edge_index, true_labels, predicted_labels, mol_id):
    """Collapse directed HH records using the exported-graph OR policy."""
    pair_true = {}
    pair_pred = {}
    for (src_i, dst_i), t, p in zip(edge_index, true_labels, predicted_labels):
        pair = (min(int(src_i), int(dst_i)), max(int(src_i), int(dst_i)))
        if t >= 0:
            if pair in pair_true and pair_true[pair] != int(t):
                raise ValueError(f'Conflicting reference HH labels: {mol_id}, {pair}')
            pair_true[pair] = int(t)
        if p >= 0:
            if pair in pair_pred and pair_pred[pair] != int(p):
                raise ValueError(f'Conflicting predicted HH labels: {mol_id}, {pair}')
            pair_pred[pair] = int(p)
    pair_keys = sorted(set(pair_true) | set(pair_pred))
    true = np.asarray([pair_true.get(pair, -1) for pair in pair_keys], dtype=np.int64)
    pred = np.asarray([pair_pred.get(pair, -1) for pair in pair_keys], dtype=np.int64)
    return true, pred


def collect_sigma(model, stage2, dataset, device, args, sigma):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    fields = {
        'mol_idx': [], 'ref_tp': [], 'ref_fp': [], 'ref_fn': [],
        'pipe_tp': [], 'pipe_fp': [], 'pipe_fn': [],
        'u_pipe_tp': [], 'u_pipe_fp': [], 'u_pipe_fn': [],
        'conn_tp': [], 'conn_fp': [], 'conn_fn': [],
        'true_pair_exact': [], 'hh_graph_exact': [], 'conn_exact': [],
        'u_true_pair_exact': [], 'u_hh_graph_exact': [],
    }
    n_done = 0
    for batch in loader:
        batch, pred_conn, pred_type = predict(model, stage2, batch, device, args, sigma)
        edge_index = batch['edge_index']
        elems = batch['elems']
        src, dst = edge_index[:, 0], edge_index[:, 1]
        hh = (elems[src] != 1) & (elems[dst] != 1)
        valid_conn = ~((elems[src] == 1) & (elems[dst] == 1))
        true_conn = batch['bond_mask'].bool()
        true_type_full = torch.full_like(pred_type, -1)
        true_pos_all = torch.where(true_conn)[0]
        true_ei = edge_index[true_pos_all]
        true_hh = (elems[true_ei[:, 0]] != 1) & (elems[true_ei[:, 1]] != 1)
        true_type_full[true_pos_all[true_hh]] = batch['bond_type_full'][true_hh].long()
        edge_mol = batch['edge_mol_idx']
        mol_ids = batch['mol_idx'].detach().cpu().tolist()

        for mol_i, mol_id in enumerate(mol_ids):
            em = edge_mol == mol_i
            hm = em & hh
            cm = em & valid_conn
            true_pipe = true_type_full[hm].detach().cpu().numpy()
            pred_pipe = pred_type[hm].detach().cpu().numpy()
            ptp, pfp, pfn = class_counts(true_pipe, pred_pipe, include_extra=True)

            # Rule-tool scoring represents each unordered HH atom pair once.
            # Decode a learned pair if either directed record predicts it;
            # Stage-3 logits are symmetrized, so two predicted labels agree.
            hh_edge = edge_index[hm].detach().cpu().numpy()
            u_true, u_pred = undirected_hh_labels(hh_edge, true_pipe, pred_pipe, mol_id)
            utp, ufp, ufn = class_counts(u_true, u_pred, include_extra=True)
            u_ref_mask = u_true >= 0

            ref_mask = true_pipe >= 0
            true_ref = true_pipe[ref_mask]
            pred_ref = pred_pipe[ref_mask].copy()
            missed = pred_ref < 0
            pred_ref[missed] = np.where(true_ref[missed] == 0, 1, 0)
            rtp, rfp, rfn = class_counts(true_ref, pred_ref, include_extra=False)

            tc = true_conn[cm].detach().cpu().numpy().astype(bool)
            pc = pred_conn[cm].detach().cpu().numpy().astype(bool)
            ctp = int(np.logical_and(tc, pc).sum())
            cfp = int(np.logical_and(~tc, pc).sum())
            cfn = int(np.logical_and(tc, ~pc).sum())

            fields['mol_idx'].append(int(mol_id))
            fields['ref_tp'].append(rtp); fields['ref_fp'].append(rfp); fields['ref_fn'].append(rfn)
            fields['pipe_tp'].append(ptp); fields['pipe_fp'].append(pfp); fields['pipe_fn'].append(pfn)
            fields['u_pipe_tp'].append(utp); fields['u_pipe_fp'].append(ufp); fields['u_pipe_fn'].append(ufn)
            fields['conn_tp'].append(ctp); fields['conn_fp'].append(cfp); fields['conn_fn'].append(cfn)
            fields['true_pair_exact'].append(int(np.array_equal(true_ref, pred_ref)))
            fields['hh_graph_exact'].append(int(np.array_equal(true_pipe, pred_pipe)))
            fields['u_true_pair_exact'].append(int(np.array_equal(u_true[u_ref_mask], u_pred[u_ref_mask])))
            fields['u_hh_graph_exact'].append(int(np.array_equal(u_true, u_pred)))
            fields['conn_exact'].append(int(np.array_equal(tc, pc)))
            n_done += 1
        if n_done % 5120 == 0:
            print(f'[STATS] sigma={sigma:g} {n_done}/{len(dataset)}', flush=True)
    return {key: np.asarray(value) for key, value in fields.items()}


def f1_from_stats(tp, fp, fn):
    tp = tp.sum(axis=0)
    fp = fp.sum(axis=0)
    fn = fn.sum(axis=0)
    denom = 2 * tp + fp + fn
    per_class = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    return per_class, float(per_class.mean())


def main():
    args = parse_args()
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
    device = ev._get_device(args.device)
    model = ev.load_model(str((ROOT / args.checkpoint).resolve()), device)
    model.eval()
    stage2_path = str((ROOT / args.stage2_ckpt).resolve()) if args.stage2_ckpt else None
    stage2 = load_stage2(stage2_path, model, device)
    dataset = CachedBondNetDataset(str((ROOT / args.cache).resolve()))
    if args.max_mols is not None:
        from torch.utils.data import Subset
        dataset = Subset(dataset, range(min(args.max_mols, len(dataset))))
    dataset = ev.apply_requested_hydrogen_corruption(dataset, args)
    output = (ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    aggregate = []
    for sigma in args.noise_levels:
        sigma = float(sigma)
        seed = int(args.eval_noise_seed) + int(round(sigma * 10000.0))
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32 - 1))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        stats = collect_sigma(model, stage2, dataset, device, args, sigma)
        tag = f'{round(100 * sigma):03d}'
        np.savez_compressed(output / f'sigma_{tag}.npz', **stats)
        ref_pc, ref_macro = f1_from_stats(stats['ref_tp'], stats['ref_fp'], stats['ref_fn'])
        pipe_pc, pipe_macro = f1_from_stats(stats['pipe_tp'], stats['pipe_fp'], stats['pipe_fn'])
        u_pipe_pc, u_pipe_macro = f1_from_stats(stats['u_pipe_tp'], stats['u_pipe_fp'], stats['u_pipe_fn'])
        aggregate.append({
            'sigma': sigma, 'n_molecules': len(stats['mol_idx']),
            'reference_pair_f1_per_class': ref_pc.tolist(),
            'reference_pair_macro_f1': ref_macro,
            'pipeline_f1_per_class': pipe_pc.tolist(),
            'pipeline_macro_f1': pipe_macro,
            'undirected_pipeline_f1_per_class': u_pipe_pc.tolist(),
            'undirected_pipeline_macro_f1': u_pipe_macro,
            'true_pair_exact_rate': float(stats['true_pair_exact'].mean()),
            'hh_graph_exact_rate': float(stats['hh_graph_exact'].mean()),
            'undirected_true_pair_exact_rate': float(stats['u_true_pair_exact'].mean()),
            'undirected_hh_graph_exact_rate': float(stats['u_hh_graph_exact'].mean()),
            'conn_exact_rate': float(stats['conn_exact'].mean()),
        })
        print(f'[RESULT] sigma={sigma:g} ref={ref_macro:.6f} pipe={pipe_macro:.6f}', flush=True)
    (output / 'aggregate.json').write_text(json.dumps(aggregate, indent=2), encoding='utf-8')
    (output / 'protocol.json').write_text(json.dumps({
        'checkpoint': args.checkpoint, 'stage2_ckpt': args.stage2_ckpt,
        'cache': args.cache, 'noise_levels': args.noise_levels,
        'eval_noise_seed': args.eval_noise_seed, 'batch_size': args.batch_size,
        'cutoff': args.cutoff, 'h_cutoff': args.h_cutoff,
        'conn_threshold': args.conn_threshold,
        'hydrogen_drop_fraction': args.hydrogen_drop_fraction,
        'hydrogen_noise_sigma': args.hydrogen_noise_sigma,
        'hydrogen_corruption_seed': args.hydrogen_corruption_seed,
        'undirected_decoding': 'OR across directed connectivity records; Stage-3 labels required to agree',
        'device': str(device), 'max_mols': args.max_mols,
    }, indent=2), encoding='utf-8')
    print(f'[DONE] wrote {output}', flush=True)


if __name__ == '__main__':
    main()
