"""Matched runtime benchmark for the revision fixed-test cohort.

The benchmark uses the same materialized SDFs for BondNet, RDKit, and
OpenBabel. BondNet timing separates SDF read/sanitization, lazy graph
featurization+batch collation, host-to-device transfer, and GPU forward passes.
It also reports size-stratified throughput and peak allocated GPU memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluate as ev
from bondnet.data.dataset import collate_fn
from bondnet.model.bond_type_gnn import BondTypeGNN, compute_bonded_geometry
from bondnet.model.bondnet import _symmetrize_logits
from scripts.evaluate_rule_baselines import _openbabel_predict, _rdkit_predict


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default=None,
                   help='Joint one-stage checkpoint.')
    p.add_argument('--stage1', default=None,
                   help='Hard two-stage connectivity checkpoint.')
    p.add_argument('--stage2', default=None,
                   help='Hard two-stage bond-type checkpoint.')
    p.add_argument('--sdf', action='append', nargs=2, metavar=('SIGMA', 'PATH'), required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--device', default='auto')
    p.add_argument('--cutoff', type=float, default=2.5)
    p.add_argument('--h_cutoff', type=float, default=2.5)
    p.add_argument('--conn_threshold', type=float, default=0.5)
    p.add_argument('--num_workers', type=int, default=0)
    return p.parse_args()


def load_stage2(path, model, device):
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


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def forward_batch(model, stage2, batch, threshold):
    elems, coord = batch['elems'], batch['coord']
    edge_index = batch['edge_index']
    _, _, edge_feat = model.backbone(
        elems, coord, edge_index, batch['edge_diff'], batch['edge_dist'],
        num_atoms_per_mol=batch.get('num_atoms_per_mol'),
    )
    conn = torch.sigmoid(model.stage1(edge_feat)) >= threshold
    if batch.get('train_edge_mask') is not None:
        conn = conn & batch['train_edge_mask']
    bonded_pos = torch.where(conn)[0]
    n_typed = 0
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
                logits = stage2(elems, bonded_ei, bdiff, bdist, readout_ei, rdiff, rdist)
            _symmetrize_logits(logits, readout_ei).argmax(dim=-1)
            n_typed = int(readout_ei.shape[0])
    return int(bonded_pos.shape[0]), n_typed


def benchmark_model(dataset, indices, model, stage2, device, args, label):
    loader = DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    feature_s = transfer_s = forward_s = 0.0
    n_molecules = n_batches = n_pred_edges = n_typed = 0
    iterator = iter(loader)
    while True:
        t0 = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        feature_s += time.perf_counter() - t0
        if not batch:
            continue
        t0 = time.perf_counter()
        batch = ev._to_device(batch, device)
        synchronize(device)
        transfer_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        npe, nt = forward_batch(model, stage2, batch, args.conn_threshold)
        synchronize(device)
        forward_s += time.perf_counter() - t0
        n_pred_edges += npe
        n_typed += nt
        n_molecules += int(batch['num_mols'])
        n_batches += 1
    measured_s = feature_s + transfer_s + forward_s
    peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
    return {
        'method': 'BondNet', 'stratum': label, 'n_molecules': n_molecules,
        'n_batches': n_batches, 'feature_and_collate_s': feature_s,
        'host_to_device_s': transfer_s, 'forward_s': forward_s,
        'measured_inference_s': measured_s,
        'throughput_mol_s': n_molecules / measured_s,
        'forward_throughput_mol_s': n_molecules / forward_s,
        'ms_per_molecule': 1000.0 * measured_s / n_molecules,
        'peak_gpu_memory_bytes': peak, 'predicted_directed_edges': n_pred_edges,
        'typed_directed_hh_edges': n_typed,
    }


def size_strata(dataset):
    bins = {
        'heavy_le_20': [],
        'heavy_21_30': [],
        'heavy_ge_31': [],
    }
    for idx, mol in enumerate(dataset.mols):
        heavy = sum(atom.GetAtomicNum() > 1 for atom in mol.GetAtoms())
        if heavy <= 20:
            bins['heavy_le_20'].append(idx)
        elif heavy <= 30:
            bins['heavy_21_30'].append(idx)
        else:
            bins['heavy_ge_31'].append(idx)
    return bins


def benchmark_rule(method, mols):
    predictor = _rdkit_predict if method == 'RDKit' else _openbabel_predict
    bins = {
        'heavy_le_20': [0.0, 0, 0],
        'heavy_21_30': [0.0, 0, 0],
        'heavy_ge_31': [0.0, 0, 0],
    }
    ok = failed = 0
    t_all = time.perf_counter()
    for mol in mols:
        heavy = sum(atom.GetAtomicNum() > 1 for atom in mol.GetAtoms())
        label = 'heavy_le_20' if heavy <= 20 else ('heavy_21_30' if heavy <= 30 else 'heavy_ge_31')
        t0 = time.perf_counter()
        try:
            predictor(mol)
            ok += 1
            bins[label][1] += 1
        except Exception:
            failed += 1
            bins[label][2] += 1
        bins[label][0] += time.perf_counter() - t0
    elapsed = time.perf_counter() - t_all
    rows = [{
        'method': method, 'stratum': 'all', 'n_molecules': len(mols),
        'n_success': ok, 'n_failed': failed, 'prediction_s': elapsed,
        'throughput_mol_s': len(mols) / elapsed,
        'ms_per_molecule': 1000.0 * elapsed / len(mols),
    }]
    for label, (seconds, n_ok, n_failed) in bins.items():
        n = n_ok + n_failed
        rows.append({
            'method': method, 'stratum': label, 'n_molecules': n,
            'n_success': n_ok, 'n_failed': n_failed, 'prediction_s': seconds,
            'throughput_mol_s': n / seconds if seconds else 0.0,
            'ms_per_molecule': 1000.0 * seconds / n if n else 0.0,
        })
    return rows


def load_sdf_dataset(path, args):
    ds_args = SimpleNamespace(
        cache_path=None, dataset='sdf', data_path=str(path), data_dir=None,
        pyg_root=None, wiberg_npz=None, max_mols=None, shard_cache_size=2,
        split_key=None, train_groups=None, eval_split='all', explicit_h=True,
        cutoff=args.cutoff, h_cutoff=args.h_cutoff,
    )
    t0 = time.perf_counter()
    dataset = ev.load_test_dataset(ds_args, sigma=0.0)
    return dataset, time.perf_counter() - t0


def main():
    from rdkit import RDLogger, rdBase
    from openbabel import openbabel as ob

    args = parse_args()
    if bool(args.checkpoint) == bool(args.stage1):
        raise ValueError('Specify either --checkpoint or --stage1, but not both.')
    if args.stage1 and not args.stage2:
        raise ValueError('--stage2 is required with --stage1.')
    if args.checkpoint and args.stage2:
        raise ValueError('--stage2 must not be used with --checkpoint.')
    RDLogger.DisableLog('rdApp.error')
    RDLogger.DisableLog('rdApp.warning')
    try:
        ob.obErrorLog.SetOutputLevel(0)
    except Exception:
        pass
    device = ev._get_device(args.device)
    model_path = args.checkpoint if args.checkpoint else args.stage1
    model = ev.load_model(str((ROOT / model_path).resolve()), device)
    model.eval()
    stage2 = load_stage2(str((ROOT / args.stage2).resolve()), model, device) if args.stage2 else None

    # Warm up kernels once; this is excluded from reported timings.
    warm_path = (ROOT / args.sdf[0][1]).resolve()
    warm_ds, _ = load_sdf_dataset(warm_path, args)
    warm_loader = DataLoader(Subset(warm_ds, range(min(128, len(warm_ds)))), batch_size=128,
                             collate_fn=collate_fn, num_workers=0)
    warm_batch = ev._to_device(next(iter(warm_loader)), device)
    forward_batch(model, stage2, warm_batch, args.conn_threshold)
    synchronize(device)

    rows = []
    for sigma_text, sdf_text in args.sdf:
        sigma = float(sigma_text)
        path = (ROOT / sdf_text).resolve()
        dataset, sdf_load_s = load_sdf_dataset(path, args)
        if len(dataset) == 0:
            raise ValueError(f'No molecules available for runtime benchmark: {path}')
        strata = size_strata(dataset)
        model_rows = [benchmark_model(
            dataset, list(range(len(dataset))), model, stage2, device, args, 'all'
        )]
        model_rows.extend(
            benchmark_model(dataset, indices, model, stage2, device, args, label)
            for label, indices in strata.items() if indices
        )
        for row in model_rows:
            row.update({'sigma': sigma, 'sdf_load_and_sanitize_s': sdf_load_s})
            if row['stratum'] == 'all':
                total = sdf_load_s + row['measured_inference_s']
                row['end_to_end_s'] = total
                row['end_to_end_throughput_mol_s'] = row['n_molecules'] / total
            rows.append(row)
        for method in ('RDKit', 'OpenBabel'):
            for row in benchmark_rule(method, dataset.mols):
                row.update({'sigma': sigma, 'sdf_load_and_sanitize_s': sdf_load_s})
                if row['stratum'] == 'all':
                    total = sdf_load_s + row['prediction_s']
                    row['end_to_end_s'] = total
                    row['end_to_end_throughput_mol_s'] = row['n_molecules'] / total
                rows.append(row)
        print(f'[BENCH] sigma={sigma:g} complete ({len(dataset)} molecules)', flush=True)

    output = (ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with (output / 'runtime.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    environment = {
        'platform': platform.platform(), 'python': platform.python_version(),
        'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
        'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'rdkit': rdBase.rdkitVersion,
        'openbabel': getattr(sys.modules.get('openbabel'), '__version__', 'unknown'),
        'openbabel_release': ob.OBReleaseVersion(),
        'babel_datadir': os.environ.get('BABEL_DATADIR'),
        'batch_size': args.batch_size, 'num_workers': args.num_workers,
        'checkpoint': args.checkpoint,
        'stage1': args.stage1, 'stage2': args.stage2,
        'sdfs': args.sdf,
    }
    (output / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    (output / 'runtime.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(f'[DONE] wrote {output}', flush=True)


if __name__ == '__main__':
    main()
