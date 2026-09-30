"""
Train Stage 2 bond-type classifier on bonded topology graphs.

Stage 1 predicts connectivity. Stage 2 receives the resulting bond topology and
classifies only heavy-heavy bonds. H-X bonds remain in the message-passing
topology when explicit hydrogens are present, but they are not supervised.
"""

import argparse
import logging
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from bondnet.data.dataset import (
    CachedBondNetDataset,
    ShardedBatchSampler,
    ShardedCachedBondNetDataset,
    collate_stage2_fn,
    collate_fn,
)
from bondnet.data.noise_augment import GaussianNoiseAugment, apply_dynamic_candidate_mask
from bondnet.loss.focal_loss import FocalLoss
from bondnet.model.bondnet import BondNet, _symmetrize_logits
from bondnet.model.bond_type_gnn import BondTypeGNN, compute_bonded_geometry
from bondnet.utils.metrics import BondNetMetrics


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _attach_run_log(output_dir: Path) -> None:
    path = output_dir / 'train.log'
    handler = logging.FileHandler(path, mode='a', encoding='utf-8')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logging.getLogger().addHandler(handler)


def _quantile(values, q):
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    i = int(round((len(xs) - 1) * q))
    i = max(0, min(i, len(xs) - 1))
    return xs[i]


def collate_stage2_train_fast(samples):
    """
    Lean collate for Stage2 teacher-forced train/val.

    Only keeps tensors used by _teacher_forced_edges and Stage2 forward, which
    avoids concatenating large unused fields on the CPU hot path.
    """
    valid = [s for s in samples if not s.get('_invalid', False)]
    if not valid:
        return {}

    atom_offset = 0
    elems_list = []
    coord_list = []
    edge_index_list = []
    bond_mask_list = []
    bond_type_full_list = []
    num_atoms_list = []
    sample_id_list = []

    for s in valid:
        n_atoms = int(s['num_atoms'])
        elems_list.append(s['elems'])
        coord_list.append(s['coord'])
        edge_index_list.append(s['edge_index'] + atom_offset)
        bond_mask_list.append(s['bond_mask'])
        bond_type_full_list.append(s['bond_type_full'])
        num_atoms_list.append(n_atoms)
        sample_id_list.append(
            s.get('geom_mol_idx', s.get('mol_idx', s.get('sample_id', -1)))
        )
        atom_offset += n_atoms

    return {
        'elems': torch.cat(elems_list),
        'coord': torch.cat(coord_list),
        'edge_index': torch.cat(edge_index_list),
        'bond_mask': torch.cat(bond_mask_list),
        'bond_type_full': torch.cat(bond_type_full_list),
        'num_atoms': sum(num_atoms_list),
        'num_mols': len(valid),
        'num_atoms_per_mol': torch.tensor(num_atoms_list, dtype=torch.long),
        'sample_ids': torch.tensor(sample_id_list, dtype=torch.long),
    }


def parse_args():
    p = argparse.ArgumentParser(description='Train Stage 2 BondTypeGNN')
    is_win = (sys.platform == 'win32')
    p.add_argument('--stage1_ckpt', required=True)
    p.add_argument('--init_checkpoint', default=None,
                   help='Optional Stage2 checkpoint used to initialize weights only. '
                        'Useful for predicted-topology fine-tuning after teacher-forced training.')
    p.add_argument('--cache_path', required=True)
    p.add_argument('--cache_max_samples', type=int, default=None,
                   help='Use only the first N samples from a cache.')
    p.add_argument('--shard_cache_size', type=int, default=2,
                   help='Number of shard files to keep loaded per DataLoader worker.')
    p.add_argument('--output_dir', default='./checkpoints/stage2')

    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--edge_embedding_size', type=int, default=32)
    p.add_argument('--bond_cutoff', type=float, default=3.0)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--random_node_feat_dim', type=int, default=0)
    p.add_argument('--random_node_feat_std', type=float, default=1.0)
    p.add_argument('--train_topology', choices=['teacher', 'predicted'], default='teacher',
                   help='Topology supplied to Stage2 during training. Predicted mode keeps '
                        'Stage1 false positives as message-passing context and supervises '
                        'detected true heavy-heavy edges.')
    p.add_argument('--conn_threshold', type=float, default=0.5,
                   help='Frozen Stage1 threshold used by --train_topology predicted.')

    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--noise_min', type=float, default=0.0)
    p.add_argument('--noise_max', type=float, default=0.02)
    p.add_argument('--noise_clean_prob', type=float, default=0.0)
    p.add_argument('--focal_gamma', type=float, default=2.0)
    p.add_argument('--cls_weights', type=float, nargs=4, default=None)
    p.add_argument('--auto_cls_weights', action='store_true',
                   help='Estimate 4-class bond-type weights from the training split.')
    p.add_argument('--auto_cls_weight_power', type=float, default=0.5,
                   help='Inverse-frequency power for --auto_cls_weights. 0.5 = inverse sqrt.')
    p.add_argument('--max_auto_weight', type=float, default=5.0,
                   help='Maximum per-class weight for --auto_cls_weights.')
    p.add_argument('--val_split', type=float, default=0.1)
    p.add_argument('--split_key', type=str, default=None)
    p.add_argument('--train_groups', type=int, default=None)
    p.add_argument('--legacy_resplit', action='store_true',
                   help=('Ignore fixed train/val/test labels and reproduce the legacy '
                         'grouped 90/10 re-split. Do not use for revision experiments.'))
    p.add_argument('--val_every', type=int, default=1)
    p.add_argument('--e2e_val_every', type=int, default=5)
    p.add_argument('--e2e_val_max_mols', type=int, default=2048)
    p.add_argument('--selection_noise_levels', type=float, nargs='+', default=[0.0, 0.1],
                   help='Validation noise levels averaged for best_e2e.pt selection.')
    p.add_argument('--selection_noise_seed', type=int, default=20260921,
                   help='Fixed seed for validation perturbations used in checkpoint selection.')
    p.add_argument('--save_every', type=int, default=10)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--num_workers', type=int, default=0 if is_win else 4)
    p.add_argument('--prefetch_factor', type=int, default=2 if is_win else 4)
    p.add_argument('--no_pin_memory', action='store_true')
    p.add_argument('--persistent_workers', action='store_true',
                   help='Keep DataLoader workers alive across epochs (can be unstable on Windows).')
    p.add_argument('--dataloader_timeout', type=float, default=120.0 if is_win else 0.0,
                   help='Seconds to wait for a DataLoader batch when num_workers>0. '
                        'Use >0 on Windows to fail fast instead of hanging forever.')
    p.add_argument('--mp_context', type=str, default='spawn' if is_win else None,
                   help='DataLoader multiprocessing context when num_workers>0, e.g., spawn/fork/forkserver.')
    p.add_argument('--profile_batches', type=int, default=0,
                   help='Profile first N train batches: data wait vs compute time.')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def _to_device(batch, device):
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    } if batch else batch


def _first_nonfinite_tensor(named_tensors):
    for name, tensor in named_tensors:
        if torch.is_tensor(tensor) and tensor.is_floating_point() and not torch.isfinite(tensor).all():
            return name
    return None


def build_datasets(args):
    if ShardedCachedBondNetDataset.is_sharded_cache(args.cache_path):
        return ShardedCachedBondNetDataset.split(
            args.cache_path,
            val_frac=args.val_split,
            noise_augment=None,
            seed=args.seed,
            split_key=args.split_key,
            train_groups=args.train_groups,
            max_samples=args.cache_max_samples,
            max_open_shards=args.shard_cache_size,
            respect_fixed_split=not args.legacy_resplit,
        )

    train_ds, val_ds = CachedBondNetDataset.split(
        args.cache_path,
        val_frac=args.val_split,
        noise_augment=None,
        seed=args.seed,
        split_key=args.split_key,
        train_groups=args.train_groups,
        respect_fixed_split=not args.legacy_resplit,
    )
    if args.cache_max_samples is not None:
        from torch.utils.data import Subset
        n_train = min(len(train_ds), max(1, int(args.cache_max_samples * (1.0 - args.val_split))))
        n_val = min(len(val_ds), max(1, int(args.cache_max_samples * args.val_split)))
        train_ds = Subset(train_ds, range(n_train))
        val_ds = Subset(val_ds, range(n_val))
    return train_ds, val_ds


def build_loaders(train_ds, val_ds, args):
    nw = args.num_workers
    persistent = bool(args.persistent_workers and nw > 0)
    timeout = float(args.dataloader_timeout) if nw > 0 else 0.0
    mp_context = args.mp_context if (nw > 0 and args.mp_context) else None

    train_collate = collate_stage2_fn if args.train_topology == 'predicted' else collate_stage2_train_fast
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    kw = dict(
        collate_fn=train_collate,
        num_workers=nw,
        pin_memory=not args.no_pin_memory,
        persistent_workers=persistent,
        prefetch_factor=(args.prefetch_factor if nw > 0 else None),
        timeout=timeout,
        multiprocessing_context=mp_context,
    )
    if isinstance(train_ds, ShardedCachedBondNetDataset):
        batch_sampler = ShardedBatchSampler(
            train_ds,
            batch_size=args.batch_size,
            drop_last=True,
            shuffle=True,
            seed=args.seed,
        )
        train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, **kw)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
            generator=train_generator, **kw
        )
    val_kw = dict(kw)
    val_kw['collate_fn'] = collate_stage2_train_fast
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **val_kw)
    log.info(
        'DataLoader config | num_workers=%d pin_memory=%s persistent_workers=%s prefetch_factor=%s timeout=%.1fs mp_context=%s',
        nw,
        str(not args.no_pin_memory),
        str(persistent),
        str(args.prefetch_factor if nw > 0 else None),
        timeout,
        str(mp_context),
    )
    return train_loader, val_loader


def load_stage1(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    saved = ck.get('args', {})
    model = BondNet(
        num_interactions=saved.get('num_interactions', 4),
        hidden_size=saved.get('hidden_size', 128),
        backbone=saved.get('backbone', 'painn'),
        cutoff=saved.get('cutoff', 5.0),
        edge_embedding_size=saved.get('edge_embedding_size', 20),
        vector_norm_limit=saved.get('vector_norm_limit',3.0),
        clof_coords_weight=saved.get('clof_coords_weight', 0.1),
        dropout=0.0,
    )
    model.load_state_dict(ck['model_state_dict'], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def _teacher_forced_edges(batch):
    bond_ei = batch['edge_index'][batch['bond_mask']]
    diff, dist = compute_bonded_geometry(batch['coord'], bond_ei)
    elems = batch['elems']
    readout_mask = (elems[bond_ei[:, 0]] != 1) & (elems[bond_ei[:, 1]] != 1)
    readout_ei = bond_ei[readout_mask]
    readout_diff, readout_dist = compute_bonded_geometry(batch['coord'], readout_ei)
    target = batch['bond_type_full'][readout_mask]
    return bond_ei, diff, dist, readout_ei, readout_diff, readout_dist, target


@torch.no_grad()
def _predicted_topology_edges(stage1, batch, conn_threshold):
    """Build Stage2 topology from frozen Stage1 predictions.

    False-positive edges remain in the message-passing topology, matching
    deployment exposure. The four-class Stage2 head has no ``no bond`` label,
    so its supervised readout is restricted to predicted true heavy-heavy
    edges; connectivity false negatives remain unrecoverable by Stage2.
    """
    batch = apply_dynamic_candidate_mask(
        batch,
        heavy_cutoff=float(stage1.cutoff),
        hydrogen_cutoff=float(stage1.cutoff),
    )
    _, _, edge_feat = stage1.backbone(
        batch['elems'], batch['coord'], batch['edge_index'],
        batch['edge_diff'], batch['edge_dist'],
        num_atoms_per_mol=batch.get('num_atoms_per_mol'),
    )
    pred_mask = stage1.stage1.predict(edge_feat, threshold=float(conn_threshold))
    if batch.get('train_edge_mask') is not None:
        pred_mask = pred_mask & batch['train_edge_mask']

    pred_pos = torch.where(pred_mask)[0]
    bond_ei = batch['edge_index'][pred_pos]
    diff, dist = compute_bonded_geometry(batch['coord'], bond_ei)
    if pred_pos.numel() == 0:
        empty_ei = batch['edge_index'].new_zeros((0, 2))
        empty_vec = batch['coord'].new_zeros((0, 3))
        empty_dist = batch['coord'].new_zeros((0,))
        empty_target = batch['bond_type_full'].new_zeros((0,), dtype=torch.long)
        return bond_ei, diff, dist, empty_ei, empty_vec, empty_dist, empty_target

    elems = batch['elems']
    is_hh = (elems[bond_ei[:, 0]] != 1) & (elems[bond_ei[:, 1]] != 1)
    supervised = is_hh & batch['bond_mask'][pred_pos]
    readout_pos = pred_pos[supervised]
    readout_ei = batch['edge_index'][readout_pos]
    readout_diff, readout_dist = compute_bonded_geometry(batch['coord'], readout_ei)

    target_by_edge = torch.full(
        (batch['edge_index'].shape[0],), -1, dtype=torch.long, device=batch['edge_index'].device
    )
    true_pos = torch.where(batch['bond_mask'])[0]
    target_by_edge[true_pos] = batch['bond_type_full']
    target = target_by_edge[readout_pos]
    if target.numel() and bool((target < 0).any()):
        raise RuntimeError('Predicted-topology target alignment produced an invalid label')
    return bond_ei, diff, dist, readout_ei, readout_diff, readout_dist, target


def _augment_batch_noise(batch, args):
    if args.noise_max <= 0.0:
        return batch
    aug = GaussianNoiseAugment(
        sigma=args.noise_max,
        sigma_min=args.noise_min,
        sigma_max=args.noise_max,
        cutoff=args.bond_cutoff,
    )
    return aug.augment_batch(
        batch,
        per_molecule=True,
        clean_prob=args.noise_clean_prob,
        base_seed=args.seed,
        epoch=getattr(args, '_current_epoch', 0),
    )


def train_epoch(stage2, stage1, loader, criterion, optimizer, device, args):
    stage2.train()
    totals, n = {}, 0
    data_total, compute_total, prof_n = 0.0, 0.0, 0
    data_hist, compute_hist = [], []
    last_t = time.time()
    for batch in loader:
        data_t = time.time() - last_t
        if not batch:
            last_t = time.time()
            continue
        compute_t0 = time.time()
        batch = _to_device(batch, device)
        batch = _augment_batch_noise(batch, args)
        if args.train_topology == 'predicted':
            bond_ei, diff, dist, readout_ei, readout_diff, readout_dist, target = (
                _predicted_topology_edges(stage1, batch, args.conn_threshold)
            )
        else:
            bond_ei, diff, dist, readout_ei, readout_diff, readout_dist, target = _teacher_forced_edges(batch)
        if readout_ei.shape[0] == 0:
            last_t = time.time()
            continue
        logits = stage2(batch['elems'], bond_ei, diff, dist, readout_ei, readout_diff, readout_dist)
        if logits.shape[0] != target.shape[0]:
            raise ValueError(f'Stage2 logits/target mismatch: {logits.shape[0]} vs {target.shape[0]}')
        if not torch.isfinite(logits).all():
            raise RuntimeError('Non-finite Stage2 logits')
        loss = criterion(logits, target)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite Stage2 loss')

        optimizer.zero_grad()
        loss.backward()

        bad_grad = _first_nonfinite_tensor(
            (name, p.grad) for name, p in stage2.named_parameters() if p.grad is not None
        )
        if bad_grad is not None:
            optimizer.zero_grad()
            raise RuntimeError(f'Non-finite Stage2 gradient: {bad_grad}')

        if args.grad_clip > 0:
            grad_norm = nn.utils.clip_grad_norm_(stage2.parameters(), args.grad_clip)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad()
                raise RuntimeError('Non-finite Stage2 gradient norm after clipping')
        optimizer.step()

        bad_param = _first_nonfinite_tensor(stage2.named_parameters())
        if bad_param is not None:
            raise RuntimeError(f'Non-finite Stage2 parameter after optimizer step: {bad_param}')

        totals['cls'] = totals.get('cls', 0.0) + float(loss.detach())
        n += 1

        compute_t = time.time() - compute_t0
        if args.profile_batches > 0 and prof_n < args.profile_batches:
            data_total += data_t
            compute_total += compute_t
            data_hist.append(data_t)
            compute_hist.append(compute_t)
            prof_n += 1
            if prof_n == args.profile_batches:
                log.info(
                    'Profile first %d train batches: '
                    'data_avg=%.3fs p50=%.3fs p95=%.3fs max=%.3fs | '
                    'compute_avg=%.3fs p50=%.3fs p95=%.3fs max=%.3fs',
                    prof_n,
                    data_total / prof_n,
                    _quantile(data_hist, 0.50),
                    _quantile(data_hist, 0.95),
                    max(data_hist) if data_hist else 0.0,
                    compute_total / prof_n,
                    _quantile(compute_hist, 0.50),
                    _quantile(compute_hist, 0.95),
                    max(compute_hist) if compute_hist else 0.0,
                )
        last_t = time.time()
    return {k: v / n for k, v in totals.items()} if n else {}


@torch.no_grad()
def val_epoch(stage2, loader, criterion, device):
    stage2.eval()
    totals, metrics, n = {}, BondNetMetrics(), 0
    for batch in loader:
        if not batch:
            continue
        batch = _to_device(batch, device)
        bond_ei, diff, dist, readout_ei, readout_diff, readout_dist, target = _teacher_forced_edges(batch)
        if readout_ei.shape[0] == 0:
            continue
        logits = stage2(batch['elems'], bond_ei, diff, dist, readout_ei, readout_diff, readout_dist)
        logits = _symmetrize_logits(logits, readout_ei)
        loss = criterion(logits, target)
        totals['cls'] = totals.get('cls', 0.0) + float(loss.detach())
        metrics.update_bond_types(logits.argmax(-1), target)
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}, metrics.compute()


@torch.no_grad()
def val_e2e(stage2, stage1, dataset, device, batch_size=256, num_workers=0,
            heavy_cutoff=2.5, hydrogen_cutoff=2.5,
            noise_sigma=0.0, noise_seed=20260921):
    stage2.eval()
    stage1.eval()
    metrics = BondNetMetrics()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_stage2_fn, num_workers=num_workers,
    )
    for batch in loader:
        if not batch:
            continue
        if noise_sigma > 0.0:
            aug = GaussianNoiseAugment(
                sigma=noise_sigma,
                sigma_min=noise_sigma,
                sigma_max=noise_sigma,
                cutoff=heavy_cutoff,
            )
            batch = aug.augment_batch(
                batch,
                per_molecule=True,
                base_seed=int(noise_seed) + int(round(noise_sigma * 10000.0)),
            )
        batch = apply_dynamic_candidate_mask(batch, heavy_cutoff, hydrogen_cutoff)
        batch = _to_device(batch, device)
        _, _, edge_feat = stage1.backbone(
            batch['elems'], batch['coord'], batch['edge_index'],
            batch['edge_diff'], batch['edge_dist'],
            num_atoms_per_mol=batch.get('num_atoms_per_mol'),
        )
        pred_mask = stage1.stage1.predict(edge_feat, threshold=0.5)
        train_edge_mask = batch.get('train_edge_mask')
        if train_edge_mask is not None:
            pred_mask = pred_mask & train_edge_mask
        true_mask = batch['bond_mask']
        metrics.update_connectivity(pred_mask.long(), true_mask.long())

        pred_pos = torch.where(pred_mask)[0]
        full = torch.full((batch['edge_index'].shape[0],), -1, dtype=torch.long, device=device)
        if pred_pos.numel() > 0:
            pred_ei = batch['edge_index'][pred_pos]
            diff, dist = compute_bonded_geometry(batch['coord'], pred_ei)
            elems = batch['elems']
            readout_mask = (elems[pred_ei[:, 0]] != 1) & (elems[pred_ei[:, 1]] != 1)
            readout_pos = pred_pos[readout_mask]
            readout_ei = pred_ei[readout_mask]
            if readout_ei.numel() > 0:
                readout_diff, readout_dist = compute_bonded_geometry(batch['coord'], readout_ei)
                logits = stage2(batch['elems'], pred_ei, diff, dist, readout_ei, readout_diff, readout_dist)
                logits = _symmetrize_logits(logits, readout_ei)
                full[readout_pos] = logits.argmax(-1)

        true_ei = batch['edge_index'][true_mask]
        elems = batch['elems']
        true_hh_mask = (elems[true_ei[:, 0]] != 1) & (elems[true_ei[:, 1]] != 1)
        true_pos = torch.where(true_mask)[0][true_hh_mask]
        pred = full[true_pos]
        true = batch['bond_type_full'][true_hh_mask]
        missed = pred == -1
        if missed.any():
            pred[missed] = torch.where(
                true[missed] == 0,
                torch.ones_like(true[missed]),
                torch.zeros_like(true[missed]),
            )
        if pred.shape[0] != true.shape[0]:
            raise ValueError(
                f'End-to-end bond targets disagree: {pred.shape[0]} != {true.shape[0]}'
            )
        if pred.numel() > 0:
            metrics.update_bond_types(pred, true)

        true_bond_mol = batch['bond_mol_idx'][true_hh_mask]
        edge_mol = batch['edge_mol_idx']
        for mol_i in range(int(batch['num_mols'])):
            mol_edges = edge_mol == mol_i
            conn_ok = bool((pred_mask[mol_edges] == true_mask[mol_edges]).all().item())
            mol_true = true_bond_mol == mol_i
            type_ok = bool((pred[mol_true] == true[mol_true]).all().item())
            metrics.update_molecule_validity(conn_ok and type_ok)
    return metrics.compute()


def estimate_cls_weights(dataset, power=0.5, max_weight=5.0, eps=1e-6):
    """Estimate 4-class bond-type weights from HH bond-type labels."""
    counts = torch.zeros(4, dtype=torch.float64)
    samples = getattr(dataset, 'samples', None)
    if samples is not None:
        iterator = samples
    elif hasattr(dataset, 'indices') and hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'samples'):
        base = dataset.dataset.samples
        iterator = (base[i] for i in dataset.indices)
    else:
        iterator = (dataset[i] for i in range(len(dataset)))

    for sample in iterator:
        target = sample.get('bond_type_train', sample.get('bond_type_full'))
        if target is None or not torch.is_tensor(target) or target.numel() == 0:
            continue
        counts += torch.bincount(target.long().clamp(0, 3), minlength=4).double()[:4]

    if float(counts.sum()) <= 0:
        return None, counts
    freq = counts / counts.sum()
    weights = (freq.mean() / freq.clamp_min(eps)).pow(float(power))
    weights = weights / weights.mean().clamp_min(eps)
    return weights.clamp(max=float(max_weight)).float(), counts


def main():
    args = parse_args()
    _seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    log.info(f'Device: {device}')

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _attach_run_log(out)
    log.info('Command: %s', ' '.join(sys.argv))
    log.info('Environment: torch=%s cuda=%s seed=%d', torch.__version__, torch.version.cuda, args.seed)

    train_ds, val_ds = build_datasets(args)
    log.info(f'Train: {len(train_ds)} | Val: {len(val_ds)}')
    train_loader, val_loader = build_loaders(train_ds, val_ds, args)

    stage1 = load_stage1(args.stage1_ckpt, device)
    hidden_size = stage1.backbone.hidden_state_size
    stage2 = BondTypeGNN(
        hidden_size=hidden_size,
        num_layers=args.num_layers,
        edge_embedding_size=args.edge_embedding_size,
        bond_cutoff=args.bond_cutoff,
        dropout=args.dropout,
        vector_norm_limit=5,
        random_node_feat_dim=args.random_node_feat_dim,
        random_node_feat_std=args.random_node_feat_std,
    ).to(device)
    if args.init_checkpoint:
        init_ck = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
        stage2.load_state_dict(init_ck['stage2_state_dict'])
        log.info('Initialized Stage 2 weights from %s', args.init_checkpoint)
    log.info(f'Stage 2 parameters: {sum(p.numel() for p in stage2.parameters()):,}')

    if args.auto_cls_weights:
        cls_w, cls_counts = estimate_cls_weights(
            train_ds,
            power=args.auto_cls_weight_power,
            max_weight=args.max_auto_weight,
        )
        if cls_w is None:
            log.warning('Could not estimate Stage2 class weights; falling back to --cls_weights')
            cls_w = torch.tensor(args.cls_weights, dtype=torch.float32) if args.cls_weights else None
        else:
            log.info(
                'Auto Stage2 class counts [single,double,triple,aromatic]=%s -> weights=%s',
                [int(x) for x in cls_counts.tolist()],
                [round(float(x), 4) for x in cls_w.tolist()],
            )
    else:
        cls_w = torch.tensor(args.cls_weights, dtype=torch.float32) if args.cls_weights else None
    criterion = FocalLoss(gamma=args.focal_gamma, weight=cls_w).to(device)
    optimizer = torch.optim.AdamW(stage2.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup = min(5, args.epochs // 10)
    def lr_fn(ep):
        if ep < warmup:
            return (ep + 1) / max(warmup, 1)
        t = (ep - warmup) / max(args.epochs - warmup, 1)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    start_epoch, best_val, best_e2e = 0, float('inf'), float('-inf')
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        stage2.load_state_dict(ck['stage2_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        if 'scheduler_state_dict' in ck:
            scheduler.load_state_dict(ck['scheduler_state_dict'])
        start_epoch = ck.get('epoch', 0) + 1
        best_val = ck.get('best_val', best_val)
        best_e2e = ck.get('best_e2e', best_e2e)

    if args.e2e_val_max_mols and len(val_ds) > args.e2e_val_max_mols:
        from torch.utils.data import Subset
        e2e_ds = Subset(val_ds, range(args.e2e_val_max_mols))
    else:
        e2e_ds = val_ds

    last_vl, last_vm, last_e2e_by_sigma = {}, {}, {}
    for epoch in range(start_epoch, args.epochs):
        args._current_epoch = epoch
        if hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch)
        t0 = time.time()
        tl = train_epoch(stage2, stage1, train_loader, criterion, optimizer, device, args)
        do_val = (epoch % max(args.val_every, 1) == 0) or epoch == 0
        if do_val:
            last_vl, last_vm = val_epoch(stage2, val_loader, criterion, device)
        do_e2e = (epoch % max(args.e2e_val_every, 1) == 0) or epoch == 0
        if do_e2e:
            for selection_sigma in args.selection_noise_levels:
                last_e2e_by_sigma[float(selection_sigma)] = val_e2e(
                    stage2,
                    stage1,
                    e2e_ds,
                    device,
                    args.batch_size,
                    args.num_workers,
                    heavy_cutoff=float(stage1.cutoff),
                    hydrogen_cutoff=float(stage1.cutoff),
                    noise_sigma=float(selection_sigma),
                    noise_seed=args.selection_noise_seed,
                )
        scheduler.step()

        clean_key = 0.0 if 0.0 in last_e2e_by_sigma else float(args.selection_noise_levels[0])
        e2e = last_e2e_by_sigma[clean_key]
        robust_e2e = sum(
            result.get('f1_macro', 0.0)
            for result in last_e2e_by_sigma.values()
        ) / max(len(last_e2e_by_sigma), 1)
        vl, vm = last_vl, last_vm
        f1_tf = f"s={vm.get('f1_single',0):.3f} d={vm.get('f1_double',0):.3f} t={vm.get('f1_triple',0):.3f} a={vm.get('f1_aromatic',0):.3f}"
        f1_e2e = f"s={e2e.get('f1_single',0):.3f} d={e2e.get('f1_double',0):.3f} t={e2e.get('f1_triple',0):.3f} a={e2e.get('f1_aromatic',0):.3f}"
        log.info(
            f'Epoch {epoch+1:3d}/{args.epochs} | train={tl.get("cls",0):.4f} '
            f'| val={vl.get("cls",0):.4f} | f1_tf={vm.get("f1_macro",0):.3f} '
            f'| tf[{f1_tf}] | conn_f1_e2e={e2e.get("conn_f1",0):.3f} '
            f'| f1_e2e={e2e.get("f1_macro",0):.3f} | e2e[{f1_e2e}] '
            f'| robust_select={robust_e2e:.3f} '
            f'| {time.time()-t0:.1f}s'
        )

        def save(path):
            ck_args = vars(args).copy()
            ck_args['hidden_size'] = hidden_size
            torch.save({
                'epoch': epoch,
                'stage2_state_dict': stage2.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val': best_val,
                'best_e2e': best_e2e,
                'args': ck_args,
            }, path)

        val_loss = vl.get('cls', float('inf'))
        if do_val and val_loss < best_val:
            best_val = val_loss
            save(out / 'best.pt')
        if do_e2e and robust_e2e > best_e2e:
            best_e2e = robust_e2e
            save(out / 'best_e2e.pt')
        if (epoch + 1) % args.save_every == 0:
            save(out / f'epoch_{epoch+1:04d}.pt')

    log.info(f'Done. Best val loss: {best_val:.4f} | Best e2e f1: {best_e2e:.4f}')


if __name__ == '__main__':
    main()
