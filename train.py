"""
BondNet Training Script

Usage:
  # Train from SDF (cache auto-built on first run):
  python train.py --data_path molecules.sdf --output_dir checkpoints/run1

  # Train from pre-built cache:
  python train.py --cache_path data/qm9_cache.pt --output_dir checkpoints/run1

  # Resume:
  python train.py --cache_path data/qm9_cache.pt --output_dir checkpoints/run1 \
                  --resume checkpoints/run1/best.pt --resume_weights_only
"""

import os, sys, math, time, argparse, logging, random
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from bondnet.model.bondnet import BondNet
from bondnet.data.dataset import (
    BondNetDataset,
    CachedBondNetDataset,
    ShardedBatchSampler,
    ShardedCachedBondNetDataset,
    collate_connectivity_fn,
    collate_fn,
)
from bondnet.data.featurizer import MoleculeFeaturizer
from bondnet.data.noise_augment import GaussianNoiseAugment, apply_dynamic_candidate_mask
from bondnet.loss.combined_loss import BondNetLoss
from bondnet.utils.metrics import BondNetMetrics

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
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


# ------------------------------------------------------------------ #
# Args                                                                  #
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description='Train BondNet')

    g = p.add_argument_group('Data')
    g.add_argument('--data_path',  type=str, default='data\qm9\raw\gdb9.sdf', help='.sdf file')
    g.add_argument('--data_dir',   type=str, default=None, help='directory of .sdf files')
    g.add_argument('--cache_path', type=str, default=None,
                   help='Pre-built cache (.pt) or sharded cache directory.')
    g.add_argument('--cache_shard_size', type=int, default=0,
                   help='If > 0, build/load a directory sharded cache with this many molecules per shard.')
    g.add_argument('--cache_max_samples', type=int, default=None,
                   help='Use only the first N samples from a cache. Useful for quick runs on large sharded caches.')
    g.add_argument('--shard_cache_size', type=int, default=2,
                   help='Number of shard files to keep loaded per DataLoader worker.')
    g.add_argument('--wiberg_npz', type=str, default=None)
    g.add_argument('--max_mols',   type=int, default=None)
    g.add_argument('--val_split',  type=float, default=0.1)
    g.add_argument('--split_key', type=str, default=None,
                   help='Optional sample key for grouped split, e.g. geom_mol_idx')
    g.add_argument('--train_groups', type=int, default=None,
                   help='Use the first N sorted split_key groups for training')
    g.add_argument('--legacy_resplit', action='store_true',
                   help=('Ignore fixed train/val/test labels and reproduce the legacy '
                         'grouped 90/10 re-split. Do not use for revision experiments.'))
    g.add_argument('--explicit_h', dest='explicit_h', action='store_true', default=True,
                   help='Keep explicit hydrogens as ordinary learnable nodes/edges')
    g.add_argument('--no_explicit_h', dest='explicit_h', action='store_false',
                   help='Remove hydrogens and train on heavy atoms only')

    g = p.add_argument_group('Model')
    g.add_argument('--num_interactions',   type=int,   default=4)
    g.add_argument('--hidden_size',        type=int,   default=128)
    g.add_argument('--backbone', choices=['painn', 'clof'], default='painn')
    g.add_argument('--cutoff',             type=float, default=2.5,
                   help='Radius cutoff for heavy-heavy candidate edges (A)')
    g.add_argument('--h_cutoff',           type=float, default=2.5,
                   help='Radius cutoff for H-X candidate edges (A)')
    g.add_argument('--edge_embedding_size',type=int,   default=20)
    g.add_argument('--vector_norm_limit', type=float, default=3.0,
                   help='Clip each PaiNN vector channel to this norm. 0 disables vector norm clipping.')
    g.add_argument('--clof_coords_weight', type=float, default=0.1,
                   help='Coordinate update scale for the Clof backbone.')
    g.add_argument('--dropout',            type=float, default=0.1)

    g = p.add_argument_group('Training')
    g.add_argument('--epochs',     type=int,   default=100)
    g.add_argument('--batch_size', type=int,   default=256)
    g.add_argument('--lr',         type=float, default=1e-3)
    g.add_argument('--lr_warmup_epochs', type=int, default=None,
                   help='LR warmup epochs. Default uses min(5, epochs//10). Use 0 for no warmup.')
    g.add_argument('--min_lr_factor', type=float, default=0.01,
                   help='Final cosine LR as a fraction of initial LR.')
    g.add_argument('--weight_decay',type=float,default=1e-5)
    g.add_argument('--grad_clip',  type=float, default=1.0)
    g.add_argument('--noise_min',  type=float, default=0.0,
                   help='Min noise sigma (A). Uniform in [noise_min, noise_max].')
    g.add_argument('--noise_max',  type=float, default=0.0,
                   help='Max noise sigma (A). 0 = no noise.')
    g.add_argument('--noise_warmup_epochs', type=int, default=0,
                   help='Linearly ramp training noise_max from 0 to the requested value over N epochs.')
    g.add_argument('--noise_clean_prob', type=float, default=0.0,
                   help='Probability that a training sample stays clean when noise augmentation is enabled.')
    g.add_argument('--focal_gamma',type=float, default=2.0)
    g.add_argument('--cls_weights',type=float, nargs=4, default=[1,10,0.5,5],
                   metavar=('W_S','W_D','W_T','W_A'),
                   help='Per-class weights [single,double,triple,aromatic].')
    g.add_argument('--auto_cls_weights', action='store_true',
                   help='Estimate bond-type class weights from the training split.')
    g.add_argument('--auto_cls_weight_power', type=float, default=0.5,
                   help='Inverse-frequency power for --auto_cls_weights. 0.5 = inverse sqrt.')
    g.add_argument('--max_auto_weight', type=float, default=5.0,
                   help='Maximum per-class weight for --auto_cls_weights.')
    g.add_argument('--pos_weight', type=float, default=3.0)
    g.add_argument('--conn_focal_gamma', type=float, default=0.0,
                   help='Binary focal gamma for Stage 1 connectivity. 0 = BCE only')
    g.add_argument('--connectivity_only', action='store_true',
                   help='Train only Stage 1 connectivity, skip bond type classification')
    g.add_argument('--weighted_sampling', action='store_true')
    g.add_argument('--val_every',  type=int, default=1)
    g.add_argument('--e2e_val_every', type=int, default=5)
    g.add_argument('--e2e_val_max_mols', type=int, default=2048)
    g.add_argument('--selection_noise_levels', type=float, nargs='+', default=[0.0, 0.1],
                   help='Validation noise levels averaged for best_e2e.pt selection.')
    g.add_argument('--selection_noise_seed', type=int, default=20260921,
                   help='Fixed seed for validation perturbations used in checkpoint selection.')
    g.add_argument('--profile_batches', type=int, default=0,
                   help='Log average data/compute time for the first N training batches.')

    g = p.add_argument_group('Checkpointing')
    g.add_argument('--output_dir', type=str, default='./checkpoints')
    g.add_argument('--save_every', type=int, default=10)
    g.add_argument('--resume',     type=str, default=None)
    g.add_argument('--resume_weights_only', action='store_true')
    g.add_argument('--device',     type=str, default='auto')
    g.add_argument('--num_workers',type=int, default=0)
    g.add_argument('--prefetch_factor', type=int, default=4,
                   help='DataLoader prefetch factor when num_workers > 0.')
    g.add_argument('--no_pin_memory', action='store_true',
                   help='Disable DataLoader pinned memory.')
    g.add_argument('--seed',       type=int, default=42)

    return p.parse_args()


# ------------------------------------------------------------------ #
# Data                                                                  #
# ------------------------------------------------------------------ #

def _build_cache(args, cache_path: str) -> None:
    """Featurize raw dataset and save cache. Called automatically on first run."""
    log.info(f'Building feature cache → {cache_path} (one-time cost)...')
    from bondnet.data.dataset import build_cache, build_sharded_cache_from_sdf
    featurizer = MoleculeFeaturizer(cutoff=args.cutoff, h_cutoff=args.h_cutoff, explicit_h=args.explicit_h)

    if (args.cache_shard_size and args.cache_shard_size > 0) or not cache_path.lower().endswith('.pt'):
        shard_size = args.cache_shard_size if args.cache_shard_size and args.cache_shard_size > 0 else 50000
        build_sharded_cache_from_sdf(
            output_dir=cache_path,
            sdf_path=args.data_path,
            sdf_dir=args.data_dir,
            wiberg_npz=args.wiberg_npz,
            max_mols=args.max_mols,
            featurizer=featurizer,
            shard_size=shard_size,
            show_progress=True,
        )
        return

    if args.data_dir:
        ds = BondNetDataset.from_sdf_dir(
            sdf_dir=args.data_dir, wiberg_npz=args.wiberg_npz,
            max_mols=args.max_mols, featurizer=featurizer,
        )
    elif args.data_path:
        ds = BondNetDataset.from_sdf(
            sdf_path=args.data_path, wiberg_npz=args.wiberg_npz,
            max_mols=args.max_mols, featurizer=featurizer,
        )
    else:
        raise ValueError('Provide --data_path or --data_dir to auto-build cache.')

    build_cache(ds, cache_path, num_workers=args.num_workers, show_progress=True)


def _check_cache_explicit_h(cache_path: str, explicit_h: bool) -> None:
    metadata = None
    try:
        if ShardedCachedBondNetDataset.is_sharded_cache(cache_path):
            metadata = ShardedCachedBondNetDataset._load_manifest(cache_path).get('metadata', {})
        elif os.path.exists(cache_path):
            metadata = torch.load(cache_path, map_location='cpu', weights_only=False).get('metadata', {})
    except Exception as e:
        log.warning(f'Could not inspect cache metadata for {cache_path}: {e}')
        return
    cached_explicit_h = metadata.get('explicit_h') if metadata else None
    if cached_explicit_h is not None and bool(cached_explicit_h) != bool(explicit_h):
        mode = 'explicit-H' if cached_explicit_h else 'heavy-only'
        requested = 'explicit-H' if explicit_h else 'heavy-only'
        raise ValueError(
            f'Cache explicit_h mismatch: cache is {mode}, but arguments request {requested}. '
            f'Use the matching cache, or rebuild with {"--explicit_h" if explicit_h else "--no_explicit_h"}.'
        )


def build_datasets(args):
    # Resolve cache path: explicit or auto-derived in output_dir
    cache_path = args.cache_path
    if cache_path is None:
        os.makedirs(args.output_dir, exist_ok=True)
        h_tag = '_explicit_h' if args.explicit_h else ''
        suffix = '' if args.cache_shard_size and args.cache_shard_size > 0 else '.pt'
        cache_path = os.path.join(
            args.output_dir,
            f'feature_cache_c{args.cutoff:.1f}{h_tag}{suffix}',
        )
        log.info(f'Auto cache path: {cache_path}')
    cache_path = os.path.normpath(cache_path)

    # Build cache if missing
    is_sharded = ShardedCachedBondNetDataset.is_sharded_cache(cache_path)
    if os.path.exists(cache_path) and (is_sharded or not os.path.isdir(cache_path)):
        log.info(f'Using existing feature cache: {cache_path}')
    else:
        _build_cache(args, cache_path)
        is_sharded = ShardedCachedBondNetDataset.is_sharded_cache(cache_path)
    _check_cache_explicit_h(cache_path, args.explicit_h)

    if is_sharded:
        train_ds, val_ds = ShardedCachedBondNetDataset.split(
            cache_path,
            val_frac=args.val_split,
            noise_augment=None,
            seed=args.seed,
            split_key=args.split_key,
            train_groups=args.train_groups,
            max_samples=args.cache_max_samples,
            max_open_shards=args.shard_cache_size,
            respect_fixed_split=not args.legacy_resplit,
        )
    else:
        train_ds, val_ds = CachedBondNetDataset.split(
            cache_path,
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


def _format_float_tag(value: float) -> str:
    text = f'{value:g}'.replace('-', 'm').replace('.', 'p')
    return text


def _auto_output_dir(args) -> None:
    default_out = os.path.normpath('./checkpoints')
    requested = os.path.normpath(args.output_dir)
    if requested != default_out:
        return

    h_tag = 'explicit_h' if args.explicit_h else 'heavy'
    noise_tag = (
        f'n{_format_float_tag(args.noise_min)}-{_format_float_tag(args.noise_max)}'
        if args.noise_max > 0 else 'n0'
    )
    mode_tag = 'conn' if args.connectivity_only else 'joint'
    args.output_dir = os.path.join(
        args.output_dir,
        f'run_c{_format_float_tag(args.cutoff)}_{h_tag}_{noise_tag}_{mode_tag}',
    )
    log.info(f'Auto output_dir: {args.output_dir}')


def build_loaders(train_ds, val_ds, args):
    sampler = None
    batch_sampler = None
    if isinstance(train_ds, ShardedCachedBondNetDataset):
        if args.weighted_sampling:
            log.warning(
                '--weighted_sampling is disabled for sharded caches because it '
                'destroys shard-local IO locality.'
            )
        batch_sampler = ShardedBatchSampler(
            train_ds,
            batch_size=args.batch_size,
            drop_last=True,
            shuffle=True,
            seed=args.seed,
        )
    elif args.weighted_sampling and hasattr(train_ds, '_diversity_weights'):
        from torch.utils.data import WeightedRandomSampler
        base = getattr(train_ds, 'dataset', train_ds)
        w = base._diversity_weights
        if hasattr(train_ds, 'indices'):
            w = w[list(train_ds.indices)]
        sampler = WeightedRandomSampler(w, len(train_ds), replacement=True)

    nw = args.num_workers
    pf = args.prefetch_factor if nw > 0 else None
    pw = nw > 0
    pin = not args.no_pin_memory
    train_collate = collate_connectivity_fn if args.connectivity_only else collate_fn
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    if batch_sampler is not None:
        train_loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                                  collate_fn=train_collate, num_workers=nw,
                                  pin_memory=pin, persistent_workers=pw,
                                  prefetch_factor=pf)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=sampler, shuffle=(sampler is None),
                                  collate_fn=train_collate, num_workers=nw,
                                  pin_memory=pin, drop_last=True,
                                  persistent_workers=pw, prefetch_factor=pf,
                                  generator=train_generator)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=nw, pin_memory=pin,
                              persistent_workers=pw, prefetch_factor=pf)
    return train_loader, val_loader


def _augment_batch_noise(batch, args, sigma_min: float, sigma_max: float):
    if sigma_max > 0.0:
        aug = GaussianNoiseAugment(
            sigma=sigma_max,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            cutoff=args.cutoff,
        )
        batch = aug.augment_batch(
            batch,
            per_molecule=True,
            clean_prob=args.noise_clean_prob,
            base_seed=args.seed,
            epoch=getattr(args, '_current_epoch', 0),
        )
    return apply_dynamic_candidate_mask(batch, args.cutoff, args.h_cutoff)


# ------------------------------------------------------------------ #
# Train / val loops                                                     #
# ------------------------------------------------------------------ #

def _to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()} if batch else batch


def _first_nonfinite_tensor(named_tensors):
    for name, tensor in named_tensors:
        if torch.is_tensor(tensor) and tensor.is_floating_point() and not torch.isfinite(tensor).all():
            return name
    return None


def _nonfinite_summary(named_tensors):
    parts = []
    for name, tensor in named_tensors:
        if not (torch.is_tensor(tensor) and tensor.is_floating_point()):
            continue
        finite = torch.isfinite(tensor)
        if finite.all():
            continue
        n_bad = int((~finite).sum().item())
        n_total = tensor.numel()
        n_nan = int(torch.isnan(tensor).sum().item())
        n_inf = int(torch.isinf(tensor).sum().item())
        max_abs = float(tensor[finite].abs().max().item()) if finite.any() else float('nan')
        parts.append(
            f'{name}: bad={n_bad}/{n_total} nan={n_nan} inf={n_inf} max_abs_finite={max_abs:.3g}'
        )
    return '; '.join(parts) if parts else 'none'


def train_epoch(model, loader, criterion, optimizer, device, args):
    model.train()
    totals, n = {}, 0
    data_total, compute_total, prof_n = 0.0, 0.0, 0
    last_t = time.time()
    sigma_min = getattr(args, '_current_noise_min', args.noise_min)
    sigma_max = getattr(args, '_current_noise_max', args.noise_max)

    for batch in loader:
        data_t = time.time() - last_t
        if not batch:
            last_t = time.time()
            continue
        compute_t0 = time.time()
        batch = _augment_batch_noise(batch, args, sigma_min, sigma_max)
        batch = _to_device(batch, device)
        if args.connectivity_only:
            batch['_connectivity_only'] = True
        output = model(batch)
        bad_out = _first_nonfinite_tensor(output.items())
        if bad_out is not None:
            raise RuntimeError(
                'Non-finite model output | ' + _nonfinite_summary(output.items())
            )
        losses = criterion(output, batch)

        if not all(torch.isfinite(v) for v in losses.values()):
            log.warning('Non-finite loss, skipping batch'); continue

        optimizer.zero_grad()
        losses['total'].backward()

        bad_grad = _first_nonfinite_tensor(
            (name, p.grad) for name, p in model.named_parameters() if p.grad is not None
        )
        if bad_grad is not None:
            optimizer.zero_grad()
            log.warning(f'Non-finite gradient ({bad_grad}), skipping batch')
            continue

        if args.grad_clip > 0:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad()
                log.warning('Non-finite gradient norm after clipping, skipping batch')
                continue
        
        optimizer.step()

        bad_param = _first_nonfinite_tensor(model.named_parameters())
        if bad_param is not None:
            raise RuntimeError(f'Non-finite parameter after optimizer step: {bad_param}')

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        n += 1

        compute_t = time.time() - compute_t0
        if args.profile_batches > 0 and prof_n < args.profile_batches:
            data_total += data_t
            compute_total += compute_t
            prof_n += 1
            if prof_n == args.profile_batches:
                log.info(
                    f'Profile first {prof_n} train batches: '
                    f'data={data_total/prof_n:.3f}s/batch '
                    f'compute={compute_total/prof_n:.3f}s/batch'
                )
        last_t = time.time()

    return {k: v / n for k, v in totals.items()} if n else {}


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    totals, metrics, n = {}, BondNetMetrics(), 0

    for batch in loader:
        if not batch:
            continue
        batch = apply_dynamic_candidate_mask(batch, model.cutoff, model.cutoff)
        batch = _to_device(batch, device)
        if criterion.connectivity_only:
            batch['_connectivity_only'] = True
        output = model(batch)
        bad_out = _first_nonfinite_tensor(output.items())
        if bad_out is not None:
            raise RuntimeError(
                'Non-finite validation output | ' + _nonfinite_summary(output.items())
            )
        losses = criterion(output, batch)

        if not all(torch.isfinite(v) for v in losses.values()):
            log.warning(
                'Non-finite val loss, skipping batch | '
                + ', '.join(f'{k}={float(v.detach())}' for k, v in losses.items())
            )
            continue

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + float(v)

        mask = batch.get('train_edge_mask')
        conn = torch.sigmoid(output['conn_logits']) >= 0.5
        metrics.update_connectivity(
            (conn[mask] if mask is not None else conn).long(),
            (batch['bond_exists'][mask] if mask is not None else batch['bond_exists']).long()
        )

        target = batch.get(
            'bond_type_train_active',
            batch.get('bond_type_train', batch['bond_type_full']),
        )
        preds = output['cls_logits'].argmax(-1)
        if preds.shape[0] > 0 and target.shape[0] > 0:
            if preds.shape[0] != target.shape[0]:
                raise ValueError(
                    f'Validation logits/target mismatch: {preds.shape[0]} vs {target.shape[0]}'
                )
            metrics.update_bond_types(preds, target)
        n += 1

    log.info(f'val_epoch: n_batches={n}, sample_loss={totals.get("total", 0) / max(n,1):.4f}')
    return {k: v / max(n, 1) for k, v in totals.items()}, metrics.compute()


@torch.no_grad()
def val_e2e(model, dataset, device, batch_size=256, num_workers=0, h_cutoff=None,
            noise_sigma=0.0, noise_seed=20260921):
    model.eval()
    metrics = BondNetMetrics()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=num_workers)

    for batch in loader:
        if not batch:
            continue
        if noise_sigma > 0.0:
            aug = GaussianNoiseAugment(
                sigma=noise_sigma,
                sigma_min=noise_sigma,
                sigma_max=noise_sigma,
                cutoff=model.cutoff,
            )
            batch = aug.augment_batch(
                batch,
                per_molecule=True,
                base_seed=int(noise_seed) + int(round(noise_sigma * 10000.0)),
            )
        batch = apply_dynamic_candidate_mask(
            batch, model.cutoff, model.cutoff if h_cutoff is None else h_cutoff
        )
        batch = _to_device(batch, device)
        result = model.predict(batch)

        pred_mask = result['bond_mask']
        true_mask = batch['bond_mask']
        metrics.update_connectivity(pred_mask.long(), true_mask.long())

        E = batch['edge_index'].shape[0]
        all_true_pos = torch.where(true_mask)[0]
        true_pos = all_true_pos
        true_hh = torch.zeros(all_true_pos.shape[0], dtype=torch.bool, device=device)
        if all_true_pos.numel() > 0:
            true_ei = batch['edge_index'][all_true_pos]
            true_hh = (batch['elems'][true_ei[:, 0]] != 1) & (batch['elems'][true_ei[:, 1]] != 1)
            true_pos = all_true_pos[true_hh]
        pred_pos = torch.where(pred_mask)[0]

        full = torch.full((E,), -1, dtype=torch.long, device=device)
        if result['bond_types'].shape[0] > 0:
            if pred_pos.shape[0] != result['bond_types'].shape[0]:
                raise ValueError(
                    'Predicted bond positions/types disagree: '
                    f'{pred_pos.shape[0]} != {result["bond_types"].shape[0]}'
                )
            full[pred_pos] = result['bond_types']

        pred = full[true_pos]
        true = batch['bond_type_full']
        true = true[true_hh]
        missed = pred == -1
        if missed.any():
            pred[missed] = torch.where(true[missed] == 0,
                                       torch.ones_like(true[missed]),
                                       torch.zeros_like(true[missed]))
        if pred.shape[0] != true.shape[0]:
            raise ValueError(
                f'End-to-end bond targets disagree: {pred.shape[0]} != {true.shape[0]}'
            )
        if pred.numel() > 0:
            metrics.update_bond_types(pred, true)

        true_bond_mol = batch['bond_mol_idx'][true_hh]
        edge_mol = batch['edge_mol_idx']
        for mol_i in range(int(batch['num_mols'])):
            mol_edges = edge_mol == mol_i
            conn_ok = bool((pred_mask[mol_edges] == true_mask[mol_edges]).all().item())
            mol_true = true_bond_mol == mol_i
            type_ok = bool((pred[mol_true] == true[mol_true]).all().item())
            metrics.update_molecule_validity(conn_ok and type_ok)

    return metrics.compute()


def estimate_cls_weights(dataset, power=0.5, max_weight=5.0, eps=1e-6):
    """Estimate 4-class bond-type weights from bond_type_train labels."""
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
        binc = torch.bincount(target.long().clamp(0, 3), minlength=4).double()
        counts += binc[:4]

    if float(counts.sum()) <= 0:
        return None, counts
    freq = counts / counts.sum()
    weights = (freq.mean() / freq.clamp_min(eps)).pow(float(power))
    weights = weights / weights.mean().clamp_min(eps)
    weights = weights.clamp(max=float(max_weight))
    return weights.float(), counts


# ------------------------------------------------------------------ #
# Checkpoint helpers                                                    #
# ------------------------------------------------------------------ #

def save_ck(path, model, optimizer, scheduler, epoch, best_val, args,
            best_e2e=float('-inf')):
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_loss': best_val, 'best_e2e': best_e2e,
                'args': vars(args)}, path)


def load_ck(path, model, optimizer=None, scheduler=None, weights_only=False, args=None):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    saved_args = ck.get('args', {})
    current = {
        'num_interactions': getattr(model.backbone, 'num_interactions', None),
        'hidden_size': getattr(model, 'hidden_size', None),
        'cutoff': getattr(model, 'cutoff', None),
        'edge_embedding_size': getattr(model.backbone, 'edge_embedding_size', None),
    }
    mismatches = []
    for key, cur in current.items():
        old = saved_args.get(key)
        if old is not None and cur is not None and old != cur:
            mismatches.append(f'{key}: ckpt={old} current={cur}')
    if args is not None:
        # Data/featurization mismatches can silently hurt transfer quality.
        data_keys = ('explicit_h', 'cutoff')
        for key in data_keys:
            old = saved_args.get(key)
            cur = getattr(args, key, None)
            if old is not None and cur is not None and old != cur:
                mismatches.append(f'{key}: ckpt={old} current={cur}')
    if mismatches:
        log.warning('Checkpoint/model config mismatch: ' + '; '.join(mismatches))
        if weights_only and args is not None:
            old_h = saved_args.get('explicit_h')
            cur_h = getattr(args, 'explicit_h', None)
            if old_h is not None and cur_h is not None and old_h != cur_h:
                raise ValueError(
                    'Refusing --resume_weights_only across explicit_h mismatch. '
                    'Start from scratch, or resume from a checkpoint trained with '
                    'the same explicit_h setting.'
                )
    inc = model.load_state_dict(ck['model_state_dict'], strict=False)
    if inc.unexpected_keys:
        log.info(f'Ignored keys: {sorted(inc.unexpected_keys)}')
    if inc.missing_keys:
        log.info(f'Missing keys: {sorted(inc.missing_keys)}')
    if not weights_only:
        if optimizer and 'optimizer_state_dict' in ck:
            optimizer.load_state_dict(ck['optimizer_state_dict'])
        if scheduler and ck.get('scheduler_state_dict'):
            scheduler.load_state_dict(ck['scheduler_state_dict'])
        if 'best_e2e' not in ck:
            raise ValueError(
                'Cannot safely resume checkpoint selection: this checkpoint '
                'does not record best_e2e. Start a new run or use '
                '--resume_weights_only with an explicit new output directory.'
            )
        return (ck.get('epoch', 0),
                ck.get('best_val_loss', float('inf')),
                ck['best_e2e'])
    return 0, float('inf'), float('-inf')


# ------------------------------------------------------------------ #
# Main                                                                  #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()
    _auto_output_dir(args)
    _seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
             if args.device == 'auto' else torch.device(args.device)
    log.info(f'Device: {device}')

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _attach_run_log(out)
    log.info('Command: %s', ' '.join(sys.argv))
    log.info('Environment: torch=%s cuda=%s seed=%d', torch.__version__, torch.version.cuda, args.seed)

    # Data
    log.info('Loading datasets...')
    train_ds, val_ds = build_datasets(args)
    log.info(f'Train: {len(train_ds)} | Val: {len(val_ds)}')
    train_loader, val_loader = build_loaders(train_ds, val_ds, args)

    # Model
    model = BondNet(
        num_interactions=args.num_interactions,
        hidden_size=args.hidden_size,
        backbone=args.backbone,
        cutoff=args.cutoff,
        edge_embedding_size=args.edge_embedding_size,
        vector_norm_limit=args.vector_norm_limit,
        clof_coords_weight=args.clof_coords_weight,
        dropout=args.dropout,
    ).to(device)
    log.info(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Loss
    if args.auto_cls_weights:
        cls_w, cls_counts = estimate_cls_weights(
            train_ds,
            power=args.auto_cls_weight_power,
            max_weight=args.max_auto_weight,
        )
        if cls_w is None:
            log.warning('Could not estimate class weights; falling back to --cls_weights')
            cls_w = torch.tensor(args.cls_weights, dtype=torch.float32) if args.cls_weights else None
        else:
            log.info(
                'Auto class counts [single,double,triple,aromatic]=%s -> weights=%s',
                [int(x) for x in cls_counts.tolist()],
                [round(float(x), 4) for x in cls_w.tolist()],
            )
    else:
        cls_w = torch.tensor(args.cls_weights, dtype=torch.float32) \
                if args.cls_weights else None
    criterion = BondNetLoss(
        pos_weight=args.pos_weight,
        focal_gamma=args.focal_gamma,
        conn_focal_gamma=args.conn_focal_gamma,
        cls_weights=cls_w,
        connectivity_only=args.connectivity_only,
    ).to(device)

    # Optimizer + scheduler (warmup 5 epochs → cosine)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
    warmup = min(5, args.epochs // 10) if args.lr_warmup_epochs is None else max(0, args.lr_warmup_epochs)
    def lr_fn(ep):
        if warmup > 0 and ep < warmup:
            return (ep + 1) / warmup
        t = (ep - warmup) / max(args.epochs - warmup, 1)
        min_factor = args.min_lr_factor
        return min_factor + (1.0 - min_factor) * 0.5 * (1 + math.cos(math.pi * t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    start_epoch, best_val = 0, float('inf')
    best_e2e = float('-inf')

    if args.resume:
        log.info(f'Resuming: {args.resume} (weights_only={args.resume_weights_only})')
        start_epoch, best_val, best_e2e = load_ck(
            args.resume, model, optimizer, scheduler, args.resume_weights_only, args)
        if not args.resume_weights_only:
            start_epoch += 1

    # E2E eval subset
    if args.e2e_val_max_mols and len(val_ds) > args.e2e_val_max_mols:
        from torch.utils.data import Subset
        e2e_ds = Subset(val_ds, range(args.e2e_val_max_mols))
    else:
        e2e_ds = val_ds

    last_vl, last_vm, last_e2e_by_sigma = {}, {}, {}

    log.info('Training...')
    for epoch in range(start_epoch, args.epochs):
        args._current_epoch = epoch
        if hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch)
        if args.noise_max > 0 and args.noise_warmup_epochs > 0:
            scale = min(1.0, float(epoch + 1) / float(args.noise_warmup_epochs))
            cur_max = args.noise_max * scale
            cur_min = args.noise_min * scale
            args._current_noise_min = cur_min
            args._current_noise_max = cur_max
            log.info(f'Noise schedule: sigma in [{cur_min:.4f}, {cur_max:.4f}] A')
        else:
            args._current_noise_min = args.noise_min
            args._current_noise_max = args.noise_max
        t0 = time.time()

        tl = train_epoch(model, train_loader, criterion, optimizer,
                         device, args)

        do_val = (epoch % max(args.val_every, 1) == 0) or epoch == 0
        if do_val:
            last_vl, last_vm = val_epoch(model, val_loader, criterion, device)
        vl, vm = last_vl, last_vm

        do_e2e = (epoch % max(args.e2e_val_every, 1) == 0) or epoch == 0
        if do_e2e:
            for selection_sigma in args.selection_noise_levels:
                last_e2e_by_sigma[float(selection_sigma)] = val_e2e(
                    model,
                    e2e_ds,
                    device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    h_cutoff=args.h_cutoff,
                    noise_sigma=float(selection_sigma),
                    noise_seed=args.selection_noise_seed,
                )
        clean_key = 0.0 if 0.0 in last_e2e_by_sigma else float(args.selection_noise_levels[0])
        e2e = last_e2e_by_sigma[clean_key]
        selection_metric = 'conn_f1' if args.connectivity_only else 'f1_macro'
        robust_e2e = sum(
            result.get(selection_metric, 0.0)
            for result in last_e2e_by_sigma.values()
        ) / max(len(last_e2e_by_sigma), 1)

        scheduler.step()

        f1_tf  = f"s={vm.get('f1_single',0):.3f} d={vm.get('f1_double',0):.3f} " \
                 f"t={vm.get('f1_triple',0):.3f} a={vm.get('f1_aromatic',0):.3f}"
        f1_e2e = f"s={e2e.get('f1_single',0):.3f} d={e2e.get('f1_double',0):.3f} " \
                 f"t={e2e.get('f1_triple',0):.3f} a={e2e.get('f1_aromatic',0):.3f}"

        log.info(
            f'Epoch {epoch+1:3d}/{args.epochs} '
            f'| train={tl.get("total",0):.4f} '
            f'(c={tl.get("conn",0):.4f}, y={tl.get("cls",0):.4f}) '
            f'| val={vl.get("total",0):.4f} '
            f'(c={vl.get("conn",0):.4f}, y={vl.get("cls",0):.4f}) '
            f'| val({"fresh" if do_val else "cached"}) '
            f'| conn_f1_tf={vm.get("conn_f1",0):.3f} '
            f'| f1_tf={vm.get("f1_macro",0):.3f} '
            f'| tf[{f1_tf}] '
            f'| conn_f1_e2e={e2e.get("conn_f1",0):.3f} '
            f'| f1_e2e={e2e.get("f1_macro",0):.3f} '
            f'| robust_select={robust_e2e:.3f} '
            f'| e2e({"fresh" if do_e2e else "cached"})[{f1_e2e}] '
            f'| {time.time()-t0:.1f}s'
        )

        # Checkpoints
        val_total = vl.get('total', float('inf'))
        if do_val and val_total < best_val:
            best_val = val_total
            save_ck(out / 'best.pt', model, optimizer, scheduler,
                    epoch, best_val, args, best_e2e)

        if do_e2e and robust_e2e > best_e2e:
            best_e2e = robust_e2e
            save_ck(out / 'best_e2e.pt', model, optimizer, scheduler,
                    epoch, best_val, args, best_e2e)

        if (epoch + 1) % args.save_every == 0:
            save_ck(out / f'epoch_{epoch+1:04d}.pt', model, optimizer, scheduler,
                    epoch, best_val, args, best_e2e)

    log.info(f'Done. Best val loss: {best_val:.4f} | Best e2e f1: {best_e2e:.4f}')


if __name__ == '__main__':
    main()
