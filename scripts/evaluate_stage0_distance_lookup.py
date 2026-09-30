"""Evaluate the rule-only Stage 0 and a train-split distance lookup baseline.

The lookup is deliberately non-neural.  It maps
  (sorted element pair, 0.05-A distance bin, candidate degrees)
to the majority of {no bond, single, double, triple, aromatic}, using only the
fixed training split.  Sparse cells back off to pair+distance, then element pair,
then the global class distribution.
"""

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bondnet.data.dataset import CachedBondNetDataset, collate_fn
from bondnet.data.noise_augment import GaussianNoiseAugment, apply_dynamic_candidate_mask
from bondnet.utils.metrics import BondNetMetrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_cache', required=True)
    p.add_argument('--test_cache', required=True)
    p.add_argument('--lookup_path', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--noise_levels', type=float, nargs='+', default=[0.0, 0.1, 0.2])
    p.add_argument('--cutoff', type=float, default=2.5)
    p.add_argument('--distance_bin_width', type=float, default=0.05)
    p.add_argument('--degree_clip', type=int, default=12)
    p.add_argument('--fit_batch_size', type=int, default=512)
    p.add_argument('--eval_batch_size', type=int, default=128)
    p.add_argument('--fit_max_mols', type=int, default=None)
    p.add_argument('--eval_max_mols', type=int, default=None)
    p.add_argument('--eval_noise_seed', type=int, default=20260921)
    p.add_argument('--noise_device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--force_refit', action='store_true')
    return p.parse_args()


def _pair_features(batch, active, cutoff, bin_width, degree_clip):
    elems = batch['elems']
    edge_index = batch['edge_index']
    src, dst = edge_index[:, 0], edge_index[:, 1]
    valid_pair = ~((elems[src] == 1) & (elems[dst] == 1))
    active = active & valid_pair
    degree = torch.bincount(src[active], minlength=elems.shape[0]).clamp(max=degree_clip)

    z_src, z_dst = elems[src], elems[dst]
    swap = (z_src > z_dst) | ((z_src == z_dst) & (degree[src] > degree[dst]))
    z_lo = torch.where(swap, z_dst, z_src)
    z_hi = torch.where(swap, z_src, z_dst)
    d_lo = torch.where(swap, degree[dst], degree[src])
    d_hi = torch.where(swap, degree[src], degree[dst])
    pair_code = z_lo.long() * 128 + z_hi.long()
    n_bins = int(np.ceil(cutoff / bin_width)) + 1
    dist_bin = torch.floor(batch['edge_dist'] / bin_width).long().clamp(0, n_bins - 1)
    return active, valid_pair, pair_code, dist_bin, d_lo.long(), d_hi.long(), n_bins


def fit_lookup(args):
    cache_path = str((ROOT / args.train_cache).resolve())
    ds = CachedBondNetDataset(cache_path)
    train_idx = [i for i, s in enumerate(ds.samples) if s.get('split') == 'train']
    if args.fit_max_mols is not None:
        train_idx = train_idx[:args.fit_max_mols]
    loader = DataLoader(
        Subset(ds, train_idx), batch_size=args.fit_batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    tables = {}
    global_counts = np.zeros(5, dtype=np.uint64)
    t0 = time.time()
    n_seen = 0
    for batch_i, batch in enumerate(loader, 1):
        active_mask = apply_dynamic_candidate_mask(batch, args.cutoff, args.cutoff)['train_edge_mask']
        active, _, pair, dist_bin, d_lo, d_hi, n_bins = _pair_features(
            batch, active_mask, args.cutoff, args.distance_bin_width, args.degree_clip
        )
        labels = torch.zeros(batch['edge_index'].shape[0], dtype=torch.long)
        true_pos = torch.where(batch['bond_mask'].bool())[0]
        labels[true_pos] = batch['bond_type_full'].long() + 1

        sel = torch.where(active)[0]
        p = pair[sel].numpy()
        b = dist_bin[sel].numpy()
        dl = d_lo[sel].numpy()
        dh = d_hi[sel].numpy()
        y = labels[sel].numpy()
        global_counts += np.bincount(y, minlength=5).astype(np.uint64)
        for code in np.unique(p):
            m = p == code
            table = tables.setdefault(
                int(code),
                np.zeros((n_bins, args.degree_clip + 1, args.degree_clip + 1, 5), dtype=np.uint32),
            )
            np.add.at(table, (b[m], dl[m], dh[m], y[m]), 1)
        n_seen += int(batch.get('num_mols', 0))
        if batch_i % 50 == 0:
            print(f'[FIT] {n_seen}/{len(train_idx)} molecules; pairs={len(tables)}', flush=True)

    payload = {
        # Older project PyTorch builds cannot serialize uint32/uint64 storages.
        'tables': {k: torch.from_numpy(v.astype(np.int64)) for k, v in tables.items()},
        'global_counts': torch.from_numpy(global_counts.astype(np.int64)),
        'cutoff': args.cutoff,
        'distance_bin_width': args.distance_bin_width,
        'degree_clip': args.degree_clip,
        'n_train_molecules': len(train_idx),
        'elapsed_s': time.time() - t0,
    }
    lookup_path = ROOT / args.lookup_path
    lookup_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, lookup_path)
    print(f'[FIT] wrote {lookup_path} in {payload["elapsed_s"]:.1f}s', flush=True)
    return payload


def load_or_fit(args):
    lookup_path = ROOT / args.lookup_path
    if lookup_path.exists() and not args.force_refit:
        return torch.load(lookup_path, map_location='cpu', weights_only=False)
    return fit_lookup(args)


def lookup_predict(payload, pair, dist_bin, d_lo, d_hi, active):
    pred = torch.zeros(pair.shape[0], dtype=torch.long)
    sel = torch.where(active)[0]
    p = pair[sel].numpy()
    b = dist_bin[sel].numpy()
    dl = d_lo[sel].numpy()
    dh = d_hi[sel].numpy()
    out = np.zeros(sel.shape[0], dtype=np.int64)
    global_counts = payload['global_counts'].numpy()

    for code in np.unique(p):
        m = p == code
        table_t = payload['tables'].get(int(code))
        if table_t is None:
            out[m] = int(global_counts.argmax())
            continue
        table = table_t.numpy()
        exact = table[b[m], dl[m], dh[m]]
        pair_bin = table.sum(axis=(1, 2), dtype=np.uint64)[b[m]]
        pair_all = table.sum(axis=(0, 1, 2), dtype=np.uint64)
        chosen = exact.astype(np.uint64)
        empty = chosen.sum(axis=1) == 0
        chosen[empty] = pair_bin[empty]
        empty = chosen.sum(axis=1) == 0
        chosen[empty] = pair_all
        empty = chosen.sum(axis=1) == 0
        chosen[empty] = global_counts
        out[m] = chosen.argmax(axis=1)
    pred[sel] = torch.from_numpy(out)
    return pred


def evaluate(args, payload, sigma):
    ds = CachedBondNetDataset(str((ROOT / args.test_cache).resolve()))
    if args.eval_max_mols is not None:
        ds = Subset(ds, range(min(args.eval_max_mols, len(ds))))
    loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    lookup_metrics = BondNetMetrics()
    stage0_metrics = BondNetMetrics()
    stage0_conn_exact = []
    lookup_conn_exact = []
    seed = int(args.eval_noise_seed + round(sigma * 10000.0))
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    aug = GaussianNoiseAugment(sigma=sigma, sigma_min=sigma, sigma_max=sigma, cutoff=args.cutoff)
    n_mols = 0
    t0 = time.time()

    for batch in loader:
        if sigma > 0:
            if args.noise_device == 'cuda':
                if not torch.cuda.is_available():
                    raise RuntimeError('--noise_device cuda requested but CUDA is unavailable')
                batch = {
                    key: value.cuda() if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
            batch = aug.augment_batch(batch, per_molecule=True, base_seed=seed, epoch=0)
            if args.noise_device == 'cuda':
                batch = {
                    key: value.cpu() if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
        masked = apply_dynamic_candidate_mask(batch, args.cutoff, args.cutoff)
        active_mask = masked['train_edge_mask']
        active, valid_pair, pair, dist_bin, d_lo, d_hi, _ = _pair_features(
            batch, active_mask, args.cutoff, args.distance_bin_width, args.degree_clip
        )
        true_conn = batch['bond_mask'].bool()
        stage0_conn = active
        pred_class = lookup_predict(payload, pair, dist_bin, d_lo, d_hi, active)
        lookup_conn = pred_class > 0
        stage0_metrics.update_connectivity(stage0_conn[valid_pair], true_conn[valid_pair])
        lookup_metrics.update_connectivity(lookup_conn[valid_pair], true_conn[valid_pair])

        true_pos_all = torch.where(true_conn)[0]
        true_ei = batch['edge_index'][true_pos_all]
        true_hh = (batch['elems'][true_ei[:, 0]] != 1) & (batch['elems'][true_ei[:, 1]] != 1)
        true_pos = true_pos_all[true_hh]
        true_type = batch['bond_type_full'][true_hh].long()
        predicted_type = pred_class[true_pos] - 1
        missed = predicted_type < 0
        predicted_type[missed] = torch.where(
            true_type[missed] == 0,
            torch.ones_like(true_type[missed]),
            torch.zeros_like(true_type[missed]),
        )
        if true_type.numel() > 0:
            lookup_metrics.update_bond_types(predicted_type, true_type)

        edge_mol = batch['edge_mol_idx']
        bond_mol = batch['bond_mol_idx'][true_hh]
        correct_type = predicted_type == true_type
        for mol_i in range(int(batch['num_mols'])):
            em = (edge_mol == mol_i) & valid_pair
            stage0_conn_exact.append(bool((stage0_conn[em] == true_conn[em]).all().item()))
            lookup_conn_exact.append(bool((lookup_conn[em] == true_conn[em]).all().item()))
            bm = bond_mol == mol_i
            lookup_metrics.update_molecule_validity(
                bool(correct_type[bm].all().item()) if bm.any() else True
            )
        n_mols += int(batch['num_mols'])

    stage0 = stage0_metrics.compute()
    lookup = lookup_metrics.compute()
    stage0['conn_exact'] = float(np.mean(stage0_conn_exact))
    lookup['conn_exact'] = float(np.mean(lookup_conn_exact))
    stage0['n_molecules'] = n_mols
    lookup['n_molecules'] = n_mols
    stage0['sigma'] = sigma
    lookup['sigma'] = sigma
    stage0['elapsed_s'] = time.time() - t0
    lookup['elapsed_s'] = stage0['elapsed_s']
    return stage0, lookup


def main():
    args = parse_args()
    payload = load_or_fit(args)
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    results = {'Stage0': [], 'DistanceLookup': []}
    for sigma in args.noise_levels:
        print(f'[EVAL] sigma={sigma}', flush=True)
        stage0, lookup = evaluate(args, payload, float(sigma))
        results['Stage0'].append(stage0)
        results['DistanceLookup'].append(lookup)
        print(
            f'[RESULT] sigma={sigma} stage0_conn_f1={stage0["conn_f1"]:.4f} '
            f'lookup_conn_f1={lookup["conn_f1"]:.4f} lookup_macro_f1={lookup.get("f1_macro", 0):.4f}',
            flush=True,
        )
    (output / 'results.json').write_text(json.dumps(results, indent=2))
    (output / 'protocol.json').write_text(json.dumps({
        'train_cache': args.train_cache,
        'test_cache': args.test_cache,
        'lookup_path': args.lookup_path,
        'fixed_train_only': True,
        'cutoff': args.cutoff,
        'distance_bin_width': args.distance_bin_width,
        'degree_clip': args.degree_clip,
        'noise_levels': args.noise_levels,
        'eval_noise_seed': args.eval_noise_seed,
        'noise_device': args.noise_device,
        'noise_protocol': f'{args.noise_device.upper()} molecule-keyed by persistent sample id, epoch 0',
        'n_train_molecules': payload['n_train_molecules'],
    }, indent=2))
    print(f'[DONE] wrote {output / "results.json"}', flush=True)


if __name__ == '__main__':
    main()
