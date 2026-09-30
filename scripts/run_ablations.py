"""
BondNet Ablation Study Runner  (Proposal Section 6.5)

Trains and evaluates all ablation variants and the full BondNet model,
then produces a summary table suitable for a paper.

Ablations:
  full           — Full BondNet (all components)
  flat           — No hierarchy: flat 5-class classifier
  no_bo          — No continuous BO regression head
  no_ring        — Bond-level aromaticity (no Stage 2)
  distance_only  — Distance-only backbone (no atom type / angular context)
  no_focal       — Replace focal loss with standard cross-entropy
  no_valence     — No valence constraint loss term
  no_resonance   — No resonance-aware masking

Usage:
  python scripts/run_ablations.py \\
      --data_path qm9.sdf --wiberg_npz bo_qm9.npz \\
      --output_dir results/ablations/ \\
      --epochs 50 --batch_size 32

  # Run only specific ablations
  python scripts/run_ablations.py \\
      --data_path qm9.sdf --ablations full flat distance_only \\
      --output_dir results/ablations/
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bondnet.model.bondnet import BondNet
from bondnet.model.ablations import (
    FlatBondNet, FlatBondNetLoss,
    DistanceOnlyBondNet,
    NoBOBondNet,
    NoRingBondNet,
)
from bondnet.data.dataset import BondNetDataset, collate_fn
from bondnet.data.noise_augment import GaussianNoiseAugment
from bondnet.loss.combined_loss import BondNetLoss
from bondnet.loss.focal_loss import FocalLoss
from bondnet.utils.metrics import BondNetMetrics

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')

ALL_ABLATIONS = [
    'full', 'flat', 'no_bo', 'no_ring',
    'distance_only', 'no_focal', 'no_valence', 'no_resonance',
]

ABLATION_DESCRIPTIONS = {
    'full':          'Full BondNet (all components)',
    'flat':          'w/o hierarchy (flat 5-class)',
    'no_bo':         'w/o continuous BO regression',
    'no_ring':       'w/o ring-level aromaticity (bond-level)',
    'distance_only': 'w/o angular/multi-hop (distance only)',
    'no_focal':      'w/o focal loss (standard CE)',
    'no_valence':    'w/o valence constraint loss',
    'no_resonance':  'w/o resonance-aware masking',
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path',  type=str, required=True)
    p.add_argument('--wiberg_npz', type=str, default=None)
    p.add_argument('--output_dir', type=str, default='./results/ablations')
    p.add_argument('--ablations',  type=str, nargs='+', default=ALL_ABLATIONS,
                   choices=ALL_ABLATIONS + ['all'])
    p.add_argument('--epochs',     type=int, default=50)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--noise_sigma',type=float, default=0.05)
    p.add_argument('--hidden_size',type=int, default=128)
    p.add_argument('--num_interactions', type=int, default=4)
    p.add_argument('--device',     type=str, default='auto')
    p.add_argument('--num_workers',type=int, default=0)
    p.add_argument('--max_mols',   type=int, default=None)
    p.add_argument('--seed',       type=int, default=42)
    return p.parse_args()


# ------------------------------------------------------------------ #
# Model + loss factory                                                  #
# ------------------------------------------------------------------ #

def build_model_and_loss(ablation: str, args, device, cls_weights=None):
    """Create model + loss function for a given ablation."""
    hidden = args.hidden_size
    n_int = args.num_interactions

    if ablation == 'full':
        model = BondNet(num_interactions=n_int, hidden_size=hidden)
        criterion = BondNetLoss(
            cls_weights=cls_weights,
            lambda_bo=1.0,
            mu_valence=0.1,
            focal_gamma=2.0,
        )

    elif ablation == 'flat':
        model = FlatBondNet(num_interactions=n_int, hidden_size=hidden)
        criterion = FlatBondNetLoss(gamma=2.0)

    elif ablation == 'no_bo':
        model = NoBOBondNet(num_interactions=n_int, hidden_size=hidden)
        criterion = BondNetLoss(
            cls_weights=cls_weights,
            lambda_bo=0.0,      # ← disable BO loss
            mu_valence=0.1,
            focal_gamma=2.0,
        )

    elif ablation == 'no_ring':
        model = NoRingBondNet(num_interactions=n_int, hidden_size=hidden)
        criterion = BondNetLoss(
            cls_weights=cls_weights,
            lambda_bo=1.0,
            mu_valence=0.1,
            focal_gamma=2.0,
        )

    elif ablation == 'distance_only':
        model = DistanceOnlyBondNet(hidden_size=hidden)
        criterion = BondNetLoss(
            cls_weights=cls_weights,
            lambda_bo=1.0,
            mu_valence=0.0,
            focal_gamma=2.0,
        )

    elif ablation == 'no_focal':
        model = BondNet(num_interactions=n_int, hidden_size=hidden)
        criterion = BondNetLoss(
            cls_weights=None,   # no class weighting
            lambda_bo=1.0,
            mu_valence=0.1,
            focal_gamma=0.0,    # γ=0 → standard cross-entropy
        )

    elif ablation == 'no_valence':
        model = BondNet(num_interactions=n_int, hidden_size=hidden)
        criterion = BondNetLoss(
            cls_weights=cls_weights,
            lambda_bo=1.0,
            mu_valence=0.0,     # ← disable valence loss
            focal_gamma=2.0,
        )

    elif ablation == 'no_resonance':
        model = BondNet(num_interactions=n_int, hidden_size=hidden)
        criterion = _BondNetLossNoResonance(
            cls_weights=cls_weights,
            lambda_bo=1.0,
            mu_valence=0.1,
            focal_gamma=2.0,
        )

    else:
        raise ValueError(f'Unknown ablation: {ablation}')

    return model.to(device), criterion.to(device)


class _BondNetLossNoResonance(BondNetLoss):
    """BondNetLoss with resonance masking disabled (ablation: no_resonance)."""

    def forward(self, output, batch):
        # Remove resonance mask from batch before passing to parent
        batch_no_res = dict(batch)
        batch_no_res['is_resonance'] = torch.zeros_like(
            batch.get('is_resonance', torch.zeros(1))
        )
        return super().forward(output, batch_no_res)


# ------------------------------------------------------------------ #
# Train / eval loops                                                    #
# ------------------------------------------------------------------ #

def _to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        if not batch:
            continue
        batch = _to_device(batch, device)
        try:
            out = model(batch)
            losses = criterion(out, batch)
            optimizer.zero_grad()
            losses['total'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total += float(losses['total'])
            n += 1
        except Exception as e:
            log.debug(f'Train step error: {e}')
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, ablation: str) -> Dict:
    model.eval()
    metrics = BondNetMetrics()
    total_loss = 0.0
    n = 0

    for batch in loader:
        if not batch:
            continue
        batch = _to_device(batch, device)
        try:
            out = model(batch)
            losses = criterion(out, batch)
            total_loss += float(losses['total'])
            n += 1

            # Collect predictions for metrics
            _update_metrics(metrics, out, batch, ablation)
        except Exception as e:
            log.debug(f'Eval step error: {e}')

    result = metrics.compute()
    result['val_loss'] = total_loss / max(n, 1)
    return result


def _update_metrics(metrics, output, batch, ablation):
    if ablation == 'flat':
        # For flat model: all-edge prediction, remap to connectivity + bond type
        flat_pred = output['flat_logits'].argmax(dim=-1)   # (E,) in {0..4}
        conn_pred = (flat_pred > 0).long()
        metrics.update_connectivity(conn_pred, batch['bond_exists'].long())
        # Bond type: only on bonded edges, remap {1,2,3,4} → {0,1,2,3}
        bonded = flat_pred[batch['bond_mask']] - 1
        bonded = bonded.clamp(min=0)
        if bonded.shape[0] > 0 and batch['bond_type_full'].shape[0] > 0:
            n = min(bonded.shape[0], batch['bond_type_full'].shape[0])
            metrics.update_bond_types(bonded[:n], batch['bond_type_full'][:n])
    else:
        # Standard: Stage 1 + Stage 3
        conn_pred = (torch.sigmoid(output['conn_logits']) >= 0.5).long()
        metrics.update_connectivity(conn_pred, batch['bond_exists'].long())

        if output['cls_logits'].shape[0] > 0:
            pred_types = output['cls_logits'].argmax(dim=-1)
            if ablation == 'no_ring':
                # No-ring model predicts 4 classes including aromatic
                if batch.get('bond_type_full') is not None:
                    n = min(pred_types.shape[0], batch['bond_type_full'].shape[0])
                    metrics.update_bond_types(pred_types[:n], batch['bond_type_full'][:n])
            else:
                if batch['bond_type'].shape[0] > 0:
                    n = min(pred_types.shape[0], batch['bond_type'].shape[0])
                    metrics.update_bond_types(pred_types[:n], batch['bond_type'][:n])

        if output['bo_pred'].shape[0] > 0 and batch.get('wiberg_bo') is not None:
            if batch['wiberg_bo'].shape[0] > 0:
                n = min(output['bo_pred'].shape[0], batch['wiberg_bo'].shape[0])
                metrics.update_bond_orders(output['bo_pred'][:n], batch['wiberg_bo'][:n])


# ------------------------------------------------------------------ #
# Main                                                                  #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device == 'auto' else torch.device(args.device)

    ablations = args.ablations
    if 'all' in ablations:
        ablations = ALL_ABLATIONS

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ────────────────────────────────────────────────── #
    log.info('Loading dataset...')
    noise = GaussianNoiseAugment(sigma=args.noise_sigma) if args.noise_sigma > 0 else None
    full_ds = BondNetDataset.from_sdf(
        args.data_path,
        wiberg_npz=args.wiberg_npz,
        noise_augment=noise,
        max_mols=args.max_mols,
    )
    n = len(full_ds)
    n_val = max(1, int(n * 0.1))
    train_ds, val_ds = random_split(full_ds, [n - n_val, n_val])
    log.info(f'Train: {len(train_ds)}  Val: {len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=args.num_workers,
                               drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=args.num_workers)

    # Compute class weights from training set
    cls_weights = None
    try:
        base_ds = full_ds
        counts = torch.zeros(3)
        for i in range(min(len(base_ds), 5000)):
            s = base_ds[i]
            if s.get('_invalid'):
                continue
            for bt in s['bond_type'].tolist():
                if bt < 3:
                    counts[bt] += 1
        if counts.sum() > 0:
            cls_weights = FocalLoss.class_weights_from_counts(counts).to(device)
            log.info(f'Class weights: {cls_weights.tolist()}')
    except Exception:
        pass

    # ── Run each ablation ───────────────────────────────────────────── #
    all_results = {}

    for ablation in ablations:
        log.info(f'\n{"="*60}')
        log.info(f'Ablation: {ablation}  —  {ABLATION_DESCRIPTIONS[ablation]}')
        log.info('='*60)

        model, criterion = build_model_and_loss(ablation, args, device, cls_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )

        n_params = sum(p.numel() for p in model.parameters())
        log.info(f'Parameters: {n_params:,}')

        best_val_loss = float('inf')
        best_metrics = {}

        t_start = time.time()
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_result = evaluate(model, val_loader, criterion, device, ablation)
            scheduler.step()

            val_loss = val_result.get('val_loss', 0.0)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_result
                torch.save(model.state_dict(),
                           output_dir / f'{ablation}_best.pt')

            if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                log.info(
                    f'  Epoch {epoch+1:3d}/{args.epochs} '
                    f'train={train_loss:.4f} val={val_loss:.4f} '
                    f'f1_macro={val_result.get("f1_macro", 0):.3f}'
                )

        elapsed = time.time() - t_start
        best_metrics['ablation'] = ablation
        best_metrics['description'] = ABLATION_DESCRIPTIONS[ablation]
        best_metrics['train_time_s'] = elapsed
        all_results[ablation] = best_metrics

        log.info(f'Best val loss: {best_val_loss:.4f}')
        log.info(f'Best metrics: {best_metrics}')

    # ── Print summary table ──────────────────────────────────────────── #
    print('\n' + '='*100)
    print(f"{'Ablation':<25} {'F1-single':>10} {'F1-double':>10} {'F1-triple':>10} {'F1-arom':>10} {'F1-macro':>10} {'BO-MAE':>8}")
    print('-'*100)
    for abl, r in all_results.items():
        print(
            f"{ABLATION_DESCRIPTIONS[abl]:<25} "
            f"{r.get('f1_single', 0):>10.4f} "
            f"{r.get('f1_double', 0):>10.4f} "
            f"{r.get('f1_triple', 0):>10.4f} "
            f"{r.get('f1_aromatic', 0):>10.4f} "
            f"{r.get('f1_macro', 0):>10.4f} "
            f"{r.get('bo_mae', 0):>8.4f}"
        )

    # ── Save ────────────────────────────────────────────────────────── #
    out_path = output_dir / 'ablation_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log.info(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
