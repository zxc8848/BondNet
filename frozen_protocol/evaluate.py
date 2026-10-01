"""
BondNet Evaluation Script

Evaluates BondNet under multiple noise levels (robustness curve) and compares
against OpenBabel and RDKit xyz2mol baselines.

Usage:
  # Evaluate trained BondNet checkpoint
  python evaluate.py --checkpoint checkpoints/best.pt \
                     --data_path test.sdf \
                     --output_dir results/

  # QM9 benchmark (download via PyG)
  python evaluate.py --checkpoint best.pt \
                     --dataset qm9 --pyg_root ./data/qm9 \
                     --output_dir results/

  # Noise levels to sweep (脜)
  python evaluate.py --checkpoint best.pt --data_path test.sdf \
                     --noise_levels 0.0 0.01 0.05 0.1 0.2 0.3
"""

import os
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from bondnet.model.bondnet import BondNet
from bondnet.data.dataset import (
    BondNetDataset,
    CachedBondNetDataset,
    ShardedCachedBondNetDataset,
    collate_fn,
)
from bondnet.data.featurizer import MoleculeFeaturizer
from bondnet.data.noise_augment import GaussianNoiseAugment, apply_dynamic_candidate_mask
from bondnet.utils.metrics import BondNetMetrics

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


_BOND_NAMES = ['single', 'double', 'triple', 'aromatic']


def _resolve_existing_path(path_str: Optional[str], anchor: Optional[Path] = None) -> Optional[Path]:
    if not path_str:
        return None

    candidates = []
    raw = Path(path_str)
    candidates.append(raw)
    if anchor is not None:
        candidates.append(anchor / raw)

    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            continue
        if resolved.exists():
            return resolved
    return None


def _check_stage2_cache_provenance(stage2_ckpt_path: str, stage2_args: Dict, eval_cache_path: Optional[str]) -> None:
    train_cache = stage2_args.get('cache_path')
    if not train_cache or not eval_cache_path:
        return

    ckpt_dir = Path(stage2_ckpt_path).resolve().parent
    train_cache_resolved = _resolve_existing_path(train_cache, anchor=ckpt_dir)
    eval_cache_resolved = _resolve_existing_path(eval_cache_path, anchor=Path.cwd())

    same_cache = False
    if train_cache_resolved is not None and eval_cache_resolved is not None:
        same_cache = train_cache_resolved == eval_cache_resolved
    else:
        same_cache = os.path.normcase(str(train_cache)) == os.path.normcase(str(eval_cache_path))

    if same_cache:
        return

    raise ValueError(
        'Stage 2 checkpoint cache mismatch: the checkpoint was trained/validated '
        f'on {train_cache!r} but evaluation is using {eval_cache_path!r}. '
        'This can invalidate Stage 2 results because the topology cache semantics '
        'may differ. Re-run Stage 2 on the target cache or pass '
        '--allow_stage2_cache_mismatch to override deliberately.'
    )


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate BondNet')

    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--dataset', choices=['sdf', 'sdf_dir', 'qm9'], default='sdf')
    p.add_argument('--data_path', type=str, default=None)
    p.add_argument('--data_dir', type=str, default=None)
    p.add_argument('--pyg_root', type=str, default='./data/qm9')
    p.add_argument('--wiberg_npz', type=str, default=None)
    p.add_argument('--cache_path', type=str, default=None,
                   help='Optional precomputed feature cache (.pt or sharded directory). When set, '
                        'evaluation loads cached tensors instead of rebuilding '
                        'features from source molecules.')
    p.add_argument('--max_mols', type=int, default=None)
    p.add_argument('--shard_cache_size', type=int, default=2,
                   help='Number of shard files to keep loaded for sharded cache evaluation')
    p.add_argument('--split_key', type=str, default=None,
                   help='Optional cached sample key for grouped evaluation split')
    p.add_argument('--train_groups', type=int, default=None,
                   help='When split_key is set, evaluate groups after this train cutoff')
    p.add_argument('--eval_split', choices=['all', 'train', 'val', 'test'], default='all',
                   help='Which grouped split to evaluate for cached datasets')
    p.add_argument('--explicit_h', dest='explicit_h', action='store_true', default=True,
                   help='Keep explicit hydrogens when building uncached evaluation data')
    p.add_argument('--no_explicit_h', dest='explicit_h', action='store_false',
                   help='Remove hydrogens for uncached evaluation data')
    p.add_argument('--cutoff', type=float, default=2.5,
                   help='Radius cutoff for heavy-heavy candidate edges (A)')
    p.add_argument('--h_cutoff', type=float, default=2.5,
                   help='Radius cutoff for H-X candidate edges (A)')

    p.add_argument('--noise_levels', type=float, nargs='+',
                   default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.3],
                   help='Noise 蟽 values (脜) for robustness curve')
    p.add_argument('--eval_noise_seed', type=int, default=20260921,
                   help='Base seed for deterministic evaluation perturbations. Each sigma uses '
                        'base_seed + round(10000*sigma), so separate methods see identical noise.')
    p.add_argument('--conn_threshold', type=float, default=0.5,
                   help='Stage 1 connectivity probability threshold. Select on validation data; '
                        'keep fixed for held-out test evaluation.')
    p.add_argument('--hydrogen_drop_fraction', type=float, default=0.0,
                   help='Deterministically remove this fraction of explicit-H atoms per molecule '
                        'before inference (0 to 1). Heavy atoms and HH labels are unchanged.')
    p.add_argument('--hydrogen_noise_sigma', type=float, default=0.0,
                   help='Additional Gaussian coordinate noise applied only to explicit-H atoms (A).')
    p.add_argument('--hydrogen_corruption_seed', type=int, default=20260925,
                   help='Base seed for deterministic missing/misplaced-H perturbations.')

    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--output_dir', type=str, default='./results')

    p.add_argument('--run_baselines', action='store_true',
                   help='Also evaluate OpenBabel and RDKit xyz2mol baselines')
    p.add_argument('--constrained_discretization', action='store_true',
                   help='Apply constrained integer-program discretization at inference')
    p.add_argument('--bo_guided_decode', action='store_true',
                   help='Use BO-guided decoding for low-confidence non-aromatic bonds')
    p.add_argument('--bo_conf_threshold', type=float, default=0.75,
                   help='Classification confidence threshold for BO-guided decoding')
    p.add_argument('--bo_resonance_lo', type=float, default=1.25,
                   help='Lower BO bound of resonance band where classifier decision is kept')
    p.add_argument('--bo_resonance_hi', type=float, default=1.75,
                   help='Upper BO bound of resonance band where classifier decision is kept')
    p.add_argument('--stage2_ckpt', type=str, default=None,
                   help='Optional Stage 2 BondTypeGNN checkpoint for topology-based bond typing')
    p.add_argument('--allow_stage2_cache_mismatch', action='store_true',
                   help='Allow evaluating a Stage 2 checkpoint on a different cache than the one '
                        'recorded in the checkpoint args. Disabled by default because mismatched '
                        'cache semantics can silently invalidate Stage 2 results.')
    p.add_argument('--hh_only', action='store_true',
                   help='Evaluate bond-type metrics on heavy-heavy bonds only (excludes H-X bonds). '
                        'Useful for fair comparison when stage2 is absent.')
    p.add_argument('--apply_ring_rules', action='store_true',
                   help='Apply valence-constraint post-processing after bond type prediction')
    p.add_argument('--resonance_aware', action='store_true',
                   help='Resonance-aware scoring: count symmetric delocalized oxy-group bonds '
                        '(nitro, carboxyl/carboxylate) as correct when the prediction represents '
                        'the same delocalized system. Report ALONGSIDE the strict metric.')
    p.add_argument('--valence_decode', action='store_true',
                   help='[EXPERIMENTAL, default OFF] Valence-constrained decoding over Stage 2 '
                        'logits (bondnet.inference.valence_decode). Use with --valence_conf_gate '
                        'and --valence_prune to target the extra-bond / HH-graph-exact gap.')
    p.add_argument('--valence_conf_gate', type=float, default=1.0,
                   help='Only modify edges whose max-softmax confidence is below this value '
                        '(1.0 = every edge eligible). Set e.g. 0.9 to touch only uncertain bonds.')
    p.add_argument('--valence_prune', action='store_true',
                   help='Allow the valence decoder to PRUNE spurious single bonds (remove extra '
                        'heavy-heavy edges) to repair over-coordination under noise. Improves the '
                        'strict HH-graph exact-match rate. Requires --valence_decode.')
    p.add_argument('--save_error_analysis', action='store_true',
                   help='Save heuristic error-analysis JSON alongside aggregate metrics')
    p.add_argument('--error_analysis_max_records', type=int, default=2000,
                   help='Maximum number of per-bond error records to save per sigma')

    return p.parse_args()


def _bond_name(label: int) -> str:
    if 0 <= int(label) < len(_BOND_NAMES):
        return _BOND_NAMES[int(label)]
    return str(int(label))


def _bo_band(value: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return 'nan'
    if value < 1.20:
        return '<1.20'
    if value < 1.35:
        return '1.20-1.35'
    if value < 1.50:
        return '1.35-1.50'
    if value < 1.80:
        return '1.50-1.80'
    if value < 2.20:
        return '1.80-2.20'
    return '>=2.20'


def _infer_error_reason(
    true_type: int,
    pred_type: int,
    pred_bo: Optional[float],
    connectivity_missed: bool,
) -> str:
    if connectivity_missed:
        return 'connectivity_missed'

    true_name = _bond_name(true_type)
    pred_name = _bond_name(pred_type)
    pair = {true_name, pred_name}

    if pair <= {'single', 'double'}:
        return f'single_double_boundary@{_bo_band(pred_bo)}'
    if pair <= {'double', 'aromatic'}:
        return f'double_aromatic_confusion@{_bo_band(pred_bo)}'
    if pair <= {'single', 'aromatic'}:
        return f'single_aromatic_confusion@{_bo_band(pred_bo)}'
    if 'triple' in pair:
        return f'triple_confusion@{_bo_band(pred_bo)}'
    if 'aromatic' in pair:
        return f'aromatic_confusion@{_bo_band(pred_bo)}'
    return f'other_type_confusion@{_bo_band(pred_bo)}'


def _new_error_summary() -> Dict:
    """Create counters that summarize every evaluation error without storing it."""
    return {
        'n_errors': 0,
        'by_reason': Counter(),
        'by_transition': Counter(),
        'by_pair': Counter(),
        'by_pred_bo_band': Counter(),
        'by_true_bo_band': Counter(),
        'double_reasons': Counter(),
    }


def _update_error_summary(summary: Dict, record: Dict) -> None:
    """Add one misclassified bond to the full-dataset error counters."""
    summary['n_errors'] += 1
    summary['by_reason'][record['reason']] += 1
    summary['by_transition'][f"{record['true_type']}->{record['pred_type']}"] += 1
    summary['by_pair'][record['atom_pair']] += 1
    summary['by_pred_bo_band'][record['pred_bo_band']] += 1
    summary['by_true_bo_band'][record['true_bo_band']] += 1
    if record['true_type'] == 'double':
        summary['double_reasons'][record['reason']] += 1


def _build_error_analysis(records: List[Dict], total_true_bonds: int, summary: Dict) -> Dict:
    """Return exact aggregate counts plus a bounded, inspectable error sample."""

    return {
        'n_true_bonds_evaluated': int(total_true_bonds),
        'n_errors': int(summary['n_errors']),
        'error_rate': float(summary['n_errors'] / max(total_true_bonds, 1)),
        'n_error_samples': int(len(records)),
        'errors_by_reason': dict(summary['by_reason'].most_common()),
        'errors_by_transition': dict(summary['by_transition'].most_common()),
        'errors_by_atom_pair': dict(summary['by_pair'].most_common(20)),
        'errors_by_pred_bo_band': dict(summary['by_pred_bo_band'].most_common()),
        'errors_by_true_bo_band': dict(summary['by_true_bo_band'].most_common()),
        'double_errors_by_reason': dict(summary['double_reasons'].most_common()),
        'sample_records': records,
    }


# ------------------------------------------------------------------ #
# Dataset / model loading                                               #
# ------------------------------------------------------------------ #

_ATOM_TENSOR_KEYS = ('elems', 'coord', 'atom_features', 'local_geom', 'expected_valence')
_EDGE_TENSOR_KEYS = (
    'edge_diff', 'edge_dist', 'bond_exists', 'bond_mask', 'train_edge_mask',
    'train_bond_mask', 'h_candidate_edge_mask',
)
_FULL_BOND_TENSOR_KEYS = (
    'bond_type_full', 'bond_is_aromatic', 'wiberg_bo_all',
    'is_resonance_all', 'non_aromatic_mask',
)


def corrupt_explicit_hydrogens(
    sample: Dict,
    drop_fraction: float = 0.0,
    hydrogen_noise_sigma: float = 0.0,
    seed: int = 20260925,
) -> Dict:
    """Return a deterministic missing/misplaced-H version of one cached sample.

    Dropping hydrogens removes their nodes and incident candidate/true-bond edges,
    while preserving every heavy atom and heavy-heavy label. Hydrogen-only noise
    changes only H coordinates. Geometry tensors are recomputed after either
    operation. The input cache sample is never mutated.
    """
    if not 0.0 <= float(drop_fraction) <= 1.0:
        raise ValueError('hydrogen_drop_fraction must be in [0, 1]')
    if float(hydrogen_noise_sigma) < 0.0:
        raise ValueError('hydrogen_noise_sigma must be non-negative')

    out = dict(sample)
    elems = sample['elems']
    coord = sample['coord']
    h_pos = torch.where(elems == 1)[0]
    if h_pos.numel() == 0 or (drop_fraction == 0.0 and hydrogen_noise_sigma == 0.0):
        return out

    mol_id = int(sample.get('geom_mol_idx', sample.get('mol_idx', 0)))
    generator = torch.Generator(device='cpu')
    generator.manual_seed((int(seed) + 1000003 * mol_id) % (2**63 - 1))

    keep_atom = torch.ones(elems.shape[0], dtype=torch.bool)
    if drop_fraction > 0.0:
        if drop_fraction >= 1.0:
            keep_atom[h_pos] = False
        else:
            keep_h = torch.rand(h_pos.shape[0], generator=generator) >= float(drop_fraction)
            keep_atom[h_pos] = keep_h

    old_to_new = torch.full((elems.shape[0],), -1, dtype=torch.long)
    old_to_new[keep_atom] = torch.arange(int(keep_atom.sum().item()), dtype=torch.long)

    for key in _ATOM_TENSOR_KEYS:
        value = sample.get(key)
        if isinstance(value, torch.Tensor) and value.shape[0] == elems.shape[0]:
            out[key] = value[keep_atom].clone()

    new_coord = out['coord']
    if hydrogen_noise_sigma > 0.0:
        kept_h = torch.where(out['elems'] == 1)[0]
        if kept_h.numel() > 0:
            h_noise = torch.randn(
                (kept_h.numel(), 3), generator=generator, dtype=new_coord.dtype
            ) * float(hydrogen_noise_sigma)
            new_coord = new_coord.clone()
            new_coord[kept_h] += h_noise
            out['coord'] = new_coord

    old_edge_index = sample['edge_index']
    edge_keep = keep_atom[old_edge_index[:, 0]] & keep_atom[old_edge_index[:, 1]]
    new_edge_index = old_to_new[old_edge_index[edge_keep]]
    out['edge_index'] = new_edge_index
    for key in _EDGE_TENSOR_KEYS:
        value = sample.get(key)
        if isinstance(value, torch.Tensor) and value.shape[0] == old_edge_index.shape[0]:
            out[key] = value[edge_keep].clone()

    old_bond_index = sample.get('edge_index_bond')
    if isinstance(old_bond_index, torch.Tensor):
        bond_keep = keep_atom[old_bond_index[:, 0]] & keep_atom[old_bond_index[:, 1]]
        out['edge_index_bond'] = old_to_new[old_bond_index[bond_keep]]
        for key in _FULL_BOND_TENSOR_KEYS:
            value = sample.get(key)
            if isinstance(value, torch.Tensor) and value.shape[0] == old_bond_index.shape[0]:
                out[key] = value[bond_keep].clone()

    if new_edge_index.numel() > 0:
        new_edge_diff = new_coord[new_edge_index[:, 1]] - new_coord[new_edge_index[:, 0]]
        new_edge_dist = new_edge_diff.norm(dim=-1)
    else:
        new_edge_diff = torch.zeros((0, 3), dtype=new_coord.dtype)
        new_edge_dist = torch.zeros((0,), dtype=new_coord.dtype)
    out['edge_diff'] = new_edge_diff
    out['edge_dist'] = new_edge_dist
    out['num_atoms'] = int(keep_atom.sum().item())

    if 'local_geom' in sample:
        out['local_geom'] = compute_local_geometry_features(
            new_coord, new_edge_index, new_edge_dist, cutoff=3.0
        )

    # Ring atoms are heavy, but their indices change after H removal.
    if isinstance(sample.get('ring_atoms'), list):
        out['ring_atoms'] = [
            [int(old_to_new[int(i)].item()) for i in ring]
            for ring in sample['ring_atoms']
        ]
    if isinstance(sample.get('ring_edge_indices'), list):
        edge_old_to_new = torch.full((old_edge_index.shape[0],), -1, dtype=torch.long)
        edge_old_to_new[edge_keep] = torch.arange(int(edge_keep.sum().item()), dtype=torch.long)
        out['ring_edge_indices'] = [
            [int(edge_old_to_new[int(i)].item()) for i in ring]
            for ring in sample['ring_edge_indices']
        ]
    return out


class HydrogenCorruptionDataset(Dataset):
    def __init__(self, base, drop_fraction: float, hydrogen_noise_sigma: float, seed: int):
        self.base = base
        self.drop_fraction = float(drop_fraction)
        self.hydrogen_noise_sigma = float(hydrogen_noise_sigma)
        self.seed = int(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return corrupt_explicit_hydrogens(
            self.base[idx], self.drop_fraction, self.hydrogen_noise_sigma, self.seed
        )


def apply_requested_hydrogen_corruption(dataset, args):
    drop_fraction = float(getattr(args, 'hydrogen_drop_fraction', 0.0))
    h_noise = float(getattr(args, 'hydrogen_noise_sigma', 0.0))
    if drop_fraction == 0.0 and h_noise == 0.0:
        return dataset
    return HydrogenCorruptionDataset(
        dataset, drop_fraction, h_noise,
        int(getattr(args, 'hydrogen_corruption_seed', 20260925)),
    )

def load_test_dataset(args, sigma: float = 0.0) -> BondNetDataset:
    # Evaluation noise is applied batch-wise inside evaluate_bondnet() so cached
    # datasets stay clean and avoid per-sample Python augmentation overhead.
    noise = None

    if args.cache_path:
        from torch.utils.data import Subset

        def _check_cache_explicit_h() -> None:
            metadata = None
            try:
                if ShardedCachedBondNetDataset.is_sharded_cache(args.cache_path):
                    metadata = ShardedCachedBondNetDataset._load_manifest(args.cache_path).get('metadata', {})
                else:
                    metadata = torch.load(args.cache_path, map_location='cpu', weights_only=False).get('metadata', {})
            except Exception as e:
                log.warning(f'Could not inspect cache metadata for {args.cache_path}: {e}')
                return
            cached_explicit_h = metadata.get('explicit_h') if metadata else None
            if cached_explicit_h is not None and bool(cached_explicit_h) != bool(args.explicit_h):
                mode = 'explicit-H' if cached_explicit_h else 'heavy-only'
                requested = 'explicit-H' if args.explicit_h else 'heavy-only'
                raise ValueError(
                    f'Cache explicit_h mismatch: cache is {mode}, but evaluation arguments request {requested}. '
                    'Use the matching cache, or pass the matching --explicit_h/--no_explicit_h flag.'
                )

        _check_cache_explicit_h()

        def _subset_by_fixed_split(ds, split_name: str):
            if split_name == 'all':
                return ds
            labels = [s.get('split') for s in ds.samples] if hasattr(ds, 'samples') else None
            if not labels or not any(lbl is not None for lbl in labels):
                return None
            idx = [i for i, lbl in enumerate(labels) if lbl == split_name]
            if not idx and split_name == 'val':
                idx = [i for i, lbl in enumerate(labels) if lbl == 'test']
            if not idx:
                return None
            return Subset(ds, idx)

        if ShardedCachedBondNetDataset.is_sharded_cache(args.cache_path):
            if args.eval_split != 'all':
                if args.eval_split == 'test':
                    ds_all = ShardedCachedBondNetDataset(
                        args.cache_path,
                        noise_augment=noise,
                        max_open_shards=args.shard_cache_size,
                    )
                    labels = [meta.get('split') for meta in (ds_all._sample_meta or [])]
                    if labels and any(lbl is not None for lbl in labels):
                        idx = [i for i, lbl in enumerate(labels) if lbl == 'test']
                        if idx:
                            return ShardedCachedBondNetDataset(
                                args.cache_path,
                                indices=idx,
                                noise_augment=noise,
                                max_open_shards=args.shard_cache_size,
                            )
                train_ds, val_ds = ShardedCachedBondNetDataset.split(
                    args.cache_path,
                    val_frac=0.1,
                    noise_augment=noise,
                    split_key=args.split_key,
                    train_groups=args.train_groups,
                    max_samples=args.max_mols,
                    max_open_shards=args.shard_cache_size,
                )
                return train_ds if args.eval_split == 'train' else val_ds
            ds = ShardedCachedBondNetDataset(
                args.cache_path,
                noise_augment=noise,
                max_open_shards=args.shard_cache_size,
            )
            if args.max_mols is not None and len(ds) > args.max_mols:
                ds = Subset(ds, range(args.max_mols))
            return ds

        ds = CachedBondNetDataset(
            cache_path=args.cache_path,
            noise_augment=noise,
        )
        if args.eval_split in {'train', 'val', 'test'}:
            fixed = _subset_by_fixed_split(ds, args.eval_split)
            if fixed is not None:
                return fixed
        if args.eval_split != 'all':
            train_ds, val_ds = CachedBondNetDataset.split(
                args.cache_path,
                val_frac=0.1,
                noise_augment=noise,
                split_key=args.split_key,
                train_groups=args.train_groups,
            )
            return train_ds if args.eval_split == 'train' else val_ds
        if args.max_mols is not None and len(ds) > args.max_mols:
            ds = Subset(ds, range(args.max_mols))
        return ds

    featurizer = MoleculeFeaturizer(cutoff=args.cutoff, h_cutoff=args.h_cutoff, explicit_h=args.explicit_h)

    if args.dataset == 'qm9':
        return BondNetDataset.from_pyg_qm9(
            root=args.pyg_root, split='test',
            noise_augment=noise, max_mols=args.max_mols,
            featurizer=featurizer,
        )
    elif args.dataset == 'sdf_dir':
        return BondNetDataset.from_sdf_dir(
            sdf_dir=args.data_dir, wiberg_npz=args.wiberg_npz,
            noise_augment=noise, max_mols=args.max_mols,
            featurizer=featurizer,
        )
    else:
        ds = BondNetDataset.from_sdf(
            sdf_path=args.data_path, wiberg_npz=args.wiberg_npz,
            noise_augment=noise, max_mols=args.max_mols,
            featurizer=featurizer,
        )
        if getattr(args, 'eval_split', 'all') != 'all':
            from bondnet.data.dataset import _assign_split_label
            from torch.utils.data import Subset
            target = args.eval_split
            keep = []
            for i, mol in enumerate(ds.mols):
                sample = {}
                if hasattr(mol, 'HasProp') and mol.HasProp('geom_mol_idx'):
                    try:
                        sample['geom_mol_idx'] = int(mol.GetProp('geom_mol_idx'))
                    except Exception:
                        pass
                lbl = _assign_split_label(sample, default_idx=i, val_frac=0.1, test_frac=0.0)
                if lbl == target:
                    keep.append(i)
            log.info(f'Filtered raw-SDF dataset to {len(keep)} {target} molecules (from {len(ds.mols)} total).')
            ds = Subset(ds, keep)
        return ds


def load_model(checkpoint_path: str, device: torch.device) -> BondNet:
    ck = torch.load(checkpoint_path, map_location='cpu')
    saved_args = ck.get('args', {})
    sd = ck['model_state_dict']

    H = saved_args.get('hidden_size', 128)
    emb = saved_args.get('edge_embedding_size', 20)
    backbone = saved_args.get('backbone', 'painn')

    # Infer readout extras from the saved first edge MLP layer width.
    actual_edge_dim = sd['backbone.edge_mlp.0.weight'].shape[1] if backbone == 'painn' else None
    base_edge_dim = 2 * H + emb + 2
    use_raw_edge_distance = saved_args.get('use_raw_edge_distance')
    use_directional_projections = saved_args.get('use_directional_projections')

    if backbone != 'painn':
        use_raw_edge_distance = False
        use_directional_projections = False
    elif use_raw_edge_distance is None or use_directional_projections is None:
        if actual_edge_dim == base_edge_dim:
            use_raw_edge_distance = False
            use_directional_projections = False
        elif actual_edge_dim == base_edge_dim + 1:
            use_raw_edge_distance = True
            use_directional_projections = False
        elif actual_edge_dim == base_edge_dim + 2 * H:
            use_raw_edge_distance = False
            use_directional_projections = True
        elif actual_edge_dim == base_edge_dim + 1 + 2 * H:
            use_raw_edge_distance = True
            use_directional_projections = True
        else:
            raise ValueError(
                f'Unrecognized edge readout width in checkpoint: {actual_edge_dim} '
                f'(expected one of {base_edge_dim}, {base_edge_dim + 1}, '
                f'{base_edge_dim + 2 * H}, {base_edge_dim + 1 + 2 * H})'
            )

    log.info(
        'Checkpoint edge_dim=%s, raw_edge_distance=%s, directional_projections=%s',
        actual_edge_dim,
        use_raw_edge_distance,
        use_directional_projections,
    )

    model = BondNet(
        num_interactions=saved_args.get('num_interactions', 4),
        hidden_size=H,
        backbone=backbone,
        cutoff=saved_args.get('cutoff', 5.0),
        edge_embedding_size=emb,
        vector_norm_limit=saved_args.get('vector_norm_limit', 0.0),
        clof_coords_weight=saved_args.get('clof_coords_weight', 0.1),
        vector_rms_norm_scale=saved_args.get('vector_rms_norm_scale', 0.0),
        use_raw_edge_distance=use_raw_edge_distance,
        use_directional_projections=use_directional_projections,
        dropout=0.0,
    ).to(device)

    # Try strict load first; fall back to non-strict with a warning.
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        log.warning(f'Strict load failed ({e}). Trying strict=False 鈥?'
                    f'missing/mismatched keys will be randomly initialized.')
        model.load_state_dict(sd, strict=False)

    model.eval()
    return model


def _get_device(spec: str) -> torch.device:
    if spec == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(spec)


def _to_device(batch: Dict, device: torch.device) -> Dict:
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device, non_blocking=True)
        else:
            result[k] = v
    return result


def _bo_guided_decode(
    pred_types: torch.Tensor,
    cls_logits: torch.Tensor,
    pred_bo: torch.Tensor,
    conf_threshold: float,
    resonance_lo: float,
    resonance_hi: float,
) -> torch.Tensor:
    """
    BO-guided decoding for non-aromatic, low-confidence bonds.

    Rules:
      1) High-confidence predictions keep classifier output.
      2) In resonance BO band [lo, hi], keep classifier output.
      3) Aromatic predictions (class=3) always keep classifier output.
      4) Otherwise snap BO to nearest integer order among {1,2,3}
         and map to classes {0,1,2}.
    """
    if pred_types.numel() == 0:
        return pred_types

    probs = torch.softmax(cls_logits, dim=-1)
    conf, _ = probs.max(dim=-1)

    in_res_band = (pred_bo >= resonance_lo) & (pred_bo <= resonance_hi)
    is_aromatic = pred_types == 3
    low_conf = conf < conf_threshold

    snap_mask = low_conf & (~in_res_band) & (~is_aromatic)
    if not snap_mask.any():
        return pred_types

    # Snap BO to nearest of {1,2,3}, then convert to class {0,1,2}
    allowed_orders = torch.tensor([1.0, 2.0, 3.0], device=pred_bo.device)
    bo_sel = pred_bo[snap_mask].unsqueeze(-1)  # (K, 1)
    nearest_idx = torch.argmin(torch.abs(bo_sel - allowed_orders.unsqueeze(0)), dim=-1)

    out = pred_types.clone()
    out[snap_mask] = nearest_idx.long()
    return out


# ------------------------------------------------------------------ #
# BondNet evaluation 鈥?true sequential inference via predict()          #
# ------------------------------------------------------------------ #

@torch.no_grad()
def evaluate_bondnet(
    model: BondNet,
    dataset: BondNetDataset,
    device: torch.device,
    batch_size: int = 1,
    num_workers: int = 0,
    use_constrained_disc: bool = False,
    use_bo_guided_decode: bool = False,
    bo_conf_threshold: float = 0.75,
    bo_resonance_lo: float = 1.25,
    bo_resonance_hi: float = 1.75,
    apply_ring_rules: bool = False,
    use_valence_decode: bool = False,
    valence_conf_gate: float = 1.0,
    valence_prune: bool = False,
    use_resonance_aware: bool = False,
    collect_error_analysis: bool = False,
    error_analysis_max_records: int = 2000,
    stage2=None,
    noise_sigma: float = 0.0,
    noise_cutoff: float = 2.5,
    noise_h_cutoff: Optional[float] = None,
    noise_seed: Optional[int] = None,
    hh_only: bool = False,
    conn_threshold: float = 0.5,
) -> Dict:
    """
    Evaluate BondNet using sequential inference on predicted bonded edges.

    A single BondNet predicts connectivity first, then classifies the bond type
    for each predicted bonded edge without ground-truth leakage.

    Bond-type F1 is computed over ground-truth bonded edges:
      - Correctly detected bond 鈫?Stage 3 type prediction evaluated
      - Missed bond            鈫?counts as wrong type prediction
    Molecule-level validity: exact per-molecule (all bonds correct or not).
    """
    from bondnet.inference.constrained_opt import ConstrainedDiscretizer
    disc = ConstrainedDiscretizer() if use_constrained_disc else None

    metrics = BondNetMetrics()
    t0 = time.time()
    n_processed = 0
    total_true_bonds = 0
    pipeline_tp = torch.zeros(4, dtype=torch.long)
    pipeline_fp = torch.zeros(4, dtype=torch.long)
    pipeline_fn = torch.zeros(4, dtype=torch.long)
    exact_type_molecules = 0
    hh_exact_molecules = 0
    error_records: List[Dict] = []
    error_summary = _new_error_summary() if collect_error_analysis else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    for batch in loader:
        if not batch:
            continue
        if noise_sigma > 0.0:
            aug = GaussianNoiseAugment(
                sigma=noise_sigma,
                sigma_min=noise_sigma,
                sigma_max=noise_sigma,
                cutoff=noise_cutoff,
            )
            batch = aug.augment_batch(
                batch,
                per_molecule=True,
                base_seed=noise_seed,
                epoch=0,
            )
        batch = apply_dynamic_candidate_mask(
            batch,
            noise_cutoff,
            noise_cutoff if noise_h_cutoff is None else noise_h_cutoff,
        )
        batch = _to_device(batch, device)

        elems = batch['elems']
        coord = batch['coord']
        edge_index = batch['edge_index']
        edge_diff = batch['edge_diff']
        edge_dist = batch['edge_dist']

        # Stage 0: backbone edge features
        _, _, edge_feat = model.backbone(
            elems, coord, edge_index, edge_diff, edge_dist,
            num_atoms_per_mol=batch.get('num_atoms_per_mol'),
        )

        # Stage 1: connectivity
        conn_logits = model.stage1(edge_feat)
        pred_train_bond_mask = torch.sigmoid(conn_logits) >= conn_threshold
        train_edge_mask = batch.get('train_edge_mask')
        if train_edge_mask is not None:
            pred_train_bond_mask = pred_train_bond_mask & train_edge_mask
        pred_bond_mask = pred_train_bond_mask

        # Stage 3/bond-type inputs are predicted heavy-heavy bonded edges.
        # Predicted H-X bonds remain connectivity-only and are assigned
        # single in exported/full predictions.
        pred_bonded_all_pos = torch.where(pred_train_bond_mask)[0]
        pred_bonded_ei = edge_index[pred_bonded_all_pos]
        pred_bonded_is_h = (
            (elems[pred_bonded_ei[:, 0]] == 1) | (elems[pred_bonded_ei[:, 1]] == 1)
        ) if pred_bonded_all_pos.numel() > 0 else torch.zeros(0, dtype=torch.bool, device=device)
        pred_stage3_pos = pred_bonded_all_pos[~pred_bonded_is_h]
        pred_bonded_pos = torch.where(pred_bond_mask)[0]
        pred_type_pos = pred_stage3_pos
        pred_bo_pos = pred_stage3_pos
        pred_types_full = torch.full((edge_index.shape[0],), -1, dtype=torch.long, device=device)
        if pred_bonded_all_pos.numel() > 0:
            pred_types_full[pred_bonded_all_pos] = 0
        if pred_stage3_pos.numel() > 0:
            from bondnet.model.bondnet import _symmetrize_logits
            if stage2 is not None:
                from bondnet.model.bond_type_gnn import compute_bonded_geometry
                # Stage2 message passing still uses all predicted bonded
                # topology edges, including H-X context; readout is HH only.
                pred_ei = edge_index[pred_bonded_all_pos]
                b_diff, b_dist = compute_bonded_geometry(coord, pred_ei)
                readout_mask = (elems[pred_ei[:, 0]] != 1) & (elems[pred_ei[:, 1]] != 1)
                readout_pos = pred_bonded_all_pos[readout_mask]
                pred_type_pos = readout_pos
                pred_bo_pos = readout_pos
                readout_ei = pred_ei[readout_mask]
                if readout_ei.numel() > 0:
                    readout_diff, readout_dist = compute_bonded_geometry(coord, readout_ei)
                    cls_logits = stage2(elems, pred_ei, b_diff, b_dist, readout_ei, readout_diff, readout_dist)
                    cls_logits = _symmetrize_logits(cls_logits, readout_ei)
                    pred_types = cls_logits.argmax(dim=-1)
                    pred_types_full[readout_pos] = pred_types
                else:
                    cls_logits = torch.zeros(0, 4, device=device)
                    pred_types = torch.zeros(0, dtype=torch.long, device=device)
                pred_bo = torch.zeros(cls_logits.shape[0], device=device)
            else:
                bonded_edge_feat = edge_feat[pred_stage3_pos]
                cls_logits, pred_bo = model.stage3(bonded_edge_feat)
                # Symmetrize before argmax
                cls_logits = _symmetrize_logits(cls_logits, edge_index[pred_stage3_pos])
                pred_types = cls_logits.argmax(dim=-1)
                pred_types_full[pred_stage3_pos] = pred_types
        else:
            cls_logits = torch.zeros(0, 4, device=device)
            pred_bo = torch.zeros(0, device=device)
            pred_types = torch.zeros(0, dtype=torch.long, device=device)
        E = batch['edge_index'].shape[0]
        pred_bond_mask = pred_bond_mask   # (E,) bool
        true_bond_mask = batch['bond_mask']    # (E,) bool

        # Stage 1: connectivity
        metrics.update_connectivity(pred_bond_mask.long(), true_bond_mask.long())

        # Bond-type metrics are aligned to ground-truth bonds. When Stage 2 is
        # enabled, H-X bonds stay in the topology graph but are not typed.
        true_bonded_pos = torch.where(true_bond_mask)[0]
        true_types = batch['bond_type_full']                # (E_b,) in {0,1,2,3}
        true_hh_mask = None
        if true_bonded_pos.numel() > 0:
            true_ei = edge_index[true_bonded_pos]
            true_hh_mask = (elems[true_ei[:, 0]] != 1) & (elems[true_ei[:, 1]] != 1)
            true_bonded_pos = true_bonded_pos[true_hh_mask]
            true_types = true_types[true_hh_mask]

        if use_bo_guided_decode and cls_logits.numel() > 0 and pred_bo.numel() > 0:
            pred_types = _bo_guided_decode(
                pred_types=pred_types,
                cls_logits=cls_logits,
                pred_bo=pred_bo,
                conf_threshold=bo_conf_threshold,
                resonance_lo=bo_resonance_lo,
                resonance_hi=bo_resonance_hi,
            )

        if apply_ring_rules and pred_types.shape[0] > 0:
            from bondnet.inference.ring_rules import ChemicalRuleFilter
            rule_filter = ChemicalRuleFilter()
            pred_bonded_edge_index = edge_index[pred_type_pos]
            pred_types = rule_filter(
                pred_types, pred_bonded_edge_index, elems=elems, n_atoms=int(elems.shape[0])
            )
            pred_types_full[pred_type_pos] = pred_types

        if (use_valence_decode and cls_logits.numel() > 0
                and pred_types.shape[0] == cls_logits.shape[0]):
            from bondnet.inference.valence_decode import (
                ValenceConstrainedDecoder, attached_hydrogen_counts, PRUNED,
            )
            _vdec = ValenceConstrainedDecoder(
                conf_gate=valence_conf_gate, allow_prune=valence_prune,
            )
            _n_atoms = int(elems.shape[0])
            # Budget = MAX valence per element, WITHOUT subtracting attached H.
            # This is a provably safe ceiling: an atom's heavy-heavy bond-order
            # sum can never exceed its total valence, so correct molecules are
            # never flagged.  Only genuine heavy-heavy over-valence (created by
            # noise) is repaired.  (Subtracting predicted H tightens the budget
            # but misfires when H-X connectivity over-counts hydrogens.)
            _vd_ei = edge_index[pred_type_pos]
            pred_types = _vdec(
                pred_types, cls_logits, _vd_ei, elems, _n_atoms, h_counts=None,
            )
            pred_types_full[pred_type_pos] = pred_types
            # Pruned edges (label PRUNED == -1) are removed from the predicted bond
            # set so they no longer count as extra heavy-heavy bonds. We drop them
            # from pred_bond_mask; any exporter that reads pred_bond_mask /
            # pred_types_full to compute the extra-bond / HH-graph-exact rate then
            # sees the pruned graph. (The connectivity metric recorded earlier in
            # this loop is pre-pruning; score the exported predictions, or move that
            # update below this block, to reflect pruning in the connectivity F1.)
            if valence_prune:
                _pruned = (pred_types == PRUNED)
                if bool(_pruned.any()):
                    pred_bond_mask[pred_type_pos[_pruned]] = False

        pred_for_true = pred_types_full[true_bonded_pos]   # (E_b,)
        total_true_bonds += int(true_types.shape[0])

        pred_bo_for_true = None
        # Stage 2 is a categorical classifier. Its placeholder all-zero
        # ``pred_bo`` tensor is only shape compatibility plumbing, not a bond
        # order prediction, so error reports must record it as unavailable.
        if stage2 is not None:
            pred_bo_for_true = torch.full((true_bonded_pos.shape[0],), float('nan'), device=device)
        elif pred_bo.shape[0] > 0:
            pred_bo_full = torch.full((E,), float('nan'), device=device)
            pred_bo_full[pred_bo_pos[:pred_bo.shape[0]]] = pred_bo
            pred_bo_for_true = pred_bo_full[true_bonded_pos]
        else:
            pred_bo_for_true = torch.full((true_bonded_pos.shape[0],), float('nan'), device=device)

        # Missed bonds 鈫?assign a provably wrong type
        missed = pred_for_true == -1
        if missed.any():
            pred_for_true[missed] = torch.where(
                true_types[missed] == 0,
                torch.ones_like(true_types[missed]),   # true=single, guess=double
                torch.zeros_like(true_types[missed]),  # true=other,  guess=single
            )

        bo_true = batch.get('wiberg_bo_all')
        if bo_true is not None and true_hh_mask is not None:
            bo_true = bo_true[true_hh_mask]

        # Optional per-molecule constrained discretization
        if disc is not None and pred_bo.shape[0] > 0:
            try:
                disc_types = disc(
                    bond_orders=pred_bo,
                    edge_index_bond=batch['edge_index'][pred_bo_pos],
                    elems=batch['elems'],
                )
                pred_types_disc = torch.full((E,), -1, dtype=torch.long, device=device)
                n_d = min(pred_bo_pos.shape[0], disc_types.shape[0])
                pred_types_disc[pred_bo_pos[:n_d]] = (disc_types[:n_d] - 1).clamp(min=0, max=2)
                pred_for_true = pred_types_disc[true_bonded_pos]
                pred_for_true[pred_for_true == -1] = 0
            except Exception:
                pass

        # Optional resonance-aware scoring: count symmetric delocalized oxy-group
        # bonds (nitro, carboxyl/carboxylate) as correct when the prediction
        # represents the same delocalized system (arbitrary single/double label).
        if use_resonance_aware and pred_for_true.numel() > 0:
            from bondnet.inference.resonance import resonance_aware_predictions
            pred_for_true = resonance_aware_predictions(
                pred_for_true, true_types, edge_index[true_bonded_pos], elems,
            )

        # Update bond-type metrics after optional decoding/post-processing.
        if pred_for_true.shape[0] != true_types.shape[0]:
            raise ValueError(
                'Prediction/target length mismatch during evaluation: '
                f'{pred_for_true.shape[0]} != {true_types.shape[0]}'
            )
        n = len(true_types)
        if n > 0:
            metrics.update_bond_types(pred_for_true[:n], true_types[:n])

            if collect_error_analysis:
                true_types_cpu = true_types[:n].detach().cpu().tolist()
                pred_types_cpu = pred_for_true[:n].detach().cpu().tolist()
                missed_cpu = missed[:n].detach().cpu().tolist()
                pred_bo_cpu = pred_bo_for_true[:n].detach().cpu().tolist()
                true_bo_cpu = bo_true[:n].detach().cpu().tolist() if bo_true is not None and bo_true.shape[0] >= n else [float('nan')] * n
                bond_edge_pos = true_bonded_pos[:n]
                bond_dist_cpu = batch['edge_dist'][bond_edge_pos].detach().cpu().tolist()
                bond_edges_cpu = batch['edge_index'][bond_edge_pos].detach().cpu().tolist()
                elems_cpu = batch['elems'].detach().cpu().tolist()
                bond_mol_idx = batch.get('bond_mol_idx')
                if true_hh_mask is not None and bond_mol_idx is not None:
                    bond_mol_idx = bond_mol_idx[true_hh_mask]
                bond_mol_idx_cpu = bond_mol_idx[:n].detach().cpu().tolist() if bond_mol_idx is not None else []
                mol_ids = batch.get('mol_idx')
                mol_ids_cpu = mol_ids.detach().cpu().tolist() if mol_ids is not None else []

                for idx, (t, p, was_missed) in enumerate(zip(true_types_cpu, pred_types_cpu, missed_cpu)):
                    if t == p:
                        continue
                    src, dst = bond_edges_cpu[idx]
                    z_src = int(elems_cpu[src])
                    z_dst = int(elems_cpu[dst])
                    pred_bo_value = float(pred_bo_cpu[idx]) if np.isfinite(pred_bo_cpu[idx]) else None
                    true_bo_value = float(true_bo_cpu[idx]) if np.isfinite(true_bo_cpu[idx]) else None
                    mol_pos = int(bond_mol_idx_cpu[idx]) if idx < len(bond_mol_idx_cpu) else -1
                    mol_id = int(mol_ids_cpu[mol_pos]) if 0 <= mol_pos < len(mol_ids_cpu) else -1
                    record = {
                        'mol_idx': mol_id,
                        'bond_index_in_mol': idx,
                        'atom_indices': [int(src), int(dst)],
                        'atom_pair': '-'.join(map(str, sorted((z_src, z_dst)))),
                        'distance': float(bond_dist_cpu[idx]),
                        'true_type': _bond_name(t),
                        'pred_type': _bond_name(p),
                        'connectivity_missed': bool(was_missed),
                        'pred_bo': pred_bo_value,
                        'true_bo': true_bo_value,
                        'pred_bo_band': _bo_band(pred_bo_value),
                        'true_bo_band': _bo_band(true_bo_value),
                        'reason': _infer_error_reason(t, p, pred_bo_value, bool(was_missed)),
                    }
                    _update_error_summary(error_summary, record)
                    if len(error_records) < error_analysis_max_records:
                        error_records.append(record)

        # Common-denominator pipeline scores include missed, mistyped, and
        # extra HH bonds. A wrong type contributes one FN to the true class and
        # one FP to the predicted class.
        hh_edge_mask = (elems[edge_index[:, 0]] != 1) & (elems[edge_index[:, 1]] != 1)
        true_type_full = torch.full((E,), -1, dtype=torch.long, device=device)
        true_type_full[true_bonded_pos] = true_types
        pred_hh_pos = torch.where(pred_bond_mask & hh_edge_mask)[0]
        for cls in range(4):
            true_cls = true_type_full == cls
            pred_cls = torch.zeros(E, dtype=torch.bool, device=device)
            if pred_hh_pos.numel() > 0:
                pred_cls[pred_hh_pos] = pred_types_full[pred_hh_pos] == cls
            pipeline_tp[cls] += int((true_cls & pred_cls).sum().item())
            pipeline_fp[cls] += int((~true_cls & pred_cls).sum().item())
            pipeline_fn[cls] += int((true_cls & ~pred_cls).sum().item())

        # Per-molecule exact validity, including molecules with no reference HH
        # bonds (which are exact unless an extra HH edge is predicted).
        if 'bond_mol_idx' not in batch or 'edge_mol_idx' not in batch:
            raise KeyError('Per-molecule evaluation requires bond_mol_idx and edge_mol_idx')
        true_bond_mol_idx = batch['bond_mol_idx']
        if true_hh_mask is not None:
            true_bond_mol_idx = true_bond_mol_idx[true_hh_mask]
        correct = pred_for_true == true_types
        for mol_i in range(int(batch.get('num_mols', 1))):
            mol_mask = true_bond_mol_idx == mol_i
            type_exact = bool(correct[mol_mask].all().item()) if mol_mask.any() else True
            mol_edges = batch['edge_mol_idx'] == mol_i
            extra_hh = bool(
                (pred_bond_mask[mol_edges] & hh_edge_mask[mol_edges]
                 & ~true_bond_mask[mol_edges]).any().item()
            )
            exact_type_molecules += int(type_exact)
            hh_exact_molecules += int(type_exact and not extra_hh)
            metrics.update_molecule_validity(type_exact)

        n_processed += int(batch.get('num_mols', 1))
        if n_processed % 1000 == 0:
            log.info(f'  {n_processed}/{len(dataset)} molecules evaluated...')

    elapsed = time.time() - t0
    out = metrics.compute()
    out['inference_time_s'] = elapsed
    out['n_molecules'] = n_processed
    pipeline_f1 = []
    for cls in range(4):
        tp = int(pipeline_tp[cls])
        fp = int(pipeline_fp[cls])
        fn = int(pipeline_fn[cls])
        denom = 2 * tp + fp + fn
        pipeline_f1.append((2 * tp / denom) if denom else 0.0)
    out['f1_pipeline_by_class'] = {
        name: value for name, value in zip(
            ('single', 'double', 'triple', 'aromatic'), pipeline_f1
        )
    }
    out['f1_macro_pipeline'] = float(sum(pipeline_f1) / len(pipeline_f1))
    out['true_bond_exact_type_rate'] = (
        exact_type_molecules / n_processed if n_processed else 0.0
    )
    out['full_graph_exact_match'] = (
        hh_exact_molecules / n_processed if n_processed else 0.0
    )
    out['conn_threshold'] = float(conn_threshold)
    if collect_error_analysis:
        out['error_analysis'] = _build_error_analysis(error_records, total_true_bonds, error_summary)
    return out




# ------------------------------------------------------------------ #
# Baseline evaluation                                                   #
# ------------------------------------------------------------------ #

def _get_mol_at(dataset, i: int):
    """Get the RDKit mol for sample i, supporting both BondNetDataset and Subset."""
    if hasattr(dataset, 'mols'):
        return dataset.mols[i]
    if hasattr(dataset, 'indices') and hasattr(dataset, 'dataset'):
        return dataset.dataset.mols[dataset.indices[i]]
    raise AttributeError('Cannot access mols from dataset type: ' + type(dataset).__name__)


def evaluate_openbabel(dataset: BondNetDataset) -> Dict:
    """Evaluate OpenBabel distance-lookup bond perception."""
    try:
        from openbabel import openbabel as ob  # type: ignore
    except ImportError:
        log.warning('OpenBabel not installed 鈥?skipping OB baseline')
        return {}

    metrics = BondNetMetrics()

    BOND_TYPE_OB = {1: 0, 2: 1, 3: 2, 5: 3}  # OB codes 鈫?{single/double/triple/aromatic}

    for i in range(len(dataset)):
        try:
            s = dataset[i]
            if s.get('_invalid'):
                continue
            mol = _get_mol_at(dataset, i)
            pos = mol.GetConformer().GetPositions()
            z = [a.GetAtomicNum() for a in mol.GetAtoms()]

            # Create OBMol from coordinates
            obmol = ob.OBMol()
            for zi, (x, y, zz) in zip(z, pos):
                a = obmol.NewAtom()
                a.SetAtomicNum(int(zi))
                a.SetVector(float(x), float(y), float(zz))
            obmol.ConnectTheDots()
            obmol.PerceiveBondOrders()

            # Compare with ground truth
            true_labels = s['bond_type_full'].tolist()
            pred_labels = []
            for bond in ob.OBMolBondIter(obmol):
                bo = bond.GetBondOrder()
                is_arom = bond.IsAromatic()
                if is_arom:
                    pred_labels.append(3)
                else:
                    pred_labels.append(BOND_TYPE_OB.get(bo, 0))

            n = min(len(true_labels), len(pred_labels))
            if n > 0:
                metrics.update_bond_types(
                    torch.tensor(pred_labels[:n]),
                    torch.tensor(true_labels[:n]),
                )
        except Exception:
            continue

    return metrics.compute()


def evaluate_rdkit_xyz2mol(dataset: BondNetDataset) -> Dict:
    """Evaluate RDKit xyz2mol bond perception."""
    try:
        from rdkit.Chem.rdDetermineBonds import DetermineBonds  # type: ignore
        from rdkit import Chem  # type: ignore
    except ImportError:
        log.warning('RDKit DetermineBonds not available 鈥?skipping RDKit baseline')
        return {}

    metrics = BondNetMetrics()
    BOND_MAP = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3,
    }

    for i in range(len(dataset)):
        try:
            s = dataset[i]
            if s.get('_invalid'):
                continue
            mol = _get_mol_at(dataset, i)

            # Create editable mol with only positions (no bonds)
            rw = Chem.RWMol()
            for atom in mol.GetAtoms():
                rw.AddAtom(Chem.Atom(atom.GetAtomicNum()))
            conf = Chem.Conformer(rw.GetNumAtoms())
            orig_conf = mol.GetConformer()
            for idx in range(rw.GetNumAtoms()):
                conf.SetAtomPosition(idx, orig_conf.GetAtomPosition(idx))
            rw.AddConformer(conf, assignId=True)
            raw_mol = rw.GetMol()

            DetermineBonds(raw_mol)

            true_labels = s['bond_type_full'].tolist()
            pred_labels = [
                BOND_MAP.get(b.GetBondType(), 0)
                for b in raw_mol.GetBonds()
            ]
            n = min(len(true_labels), len(pred_labels))
            if n > 0:
                metrics.update_bond_types(
                    torch.tensor(pred_labels[:n]),
                    torch.tensor(true_labels[:n]),
                )
        except Exception:
            continue

    return metrics.compute()


# ------------------------------------------------------------------ #
# Robustness curve                                                      #
# ------------------------------------------------------------------ #

def robustness_curve(
    model: BondNet,
    base_dataset_args,        # args for reloading dataset at each sigma
    noise_levels: List[float],
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 0,
) -> List[Dict]:
    """
    Evaluate BondNet at each noise level.

    Returns list of result dicts, one per noise level.
    """
    results = []
    for sigma in noise_levels:
        log.info(f'Evaluating at sigma={sigma:.3f} A ...')
        ds = load_test_dataset(base_dataset_args, sigma=sigma)
        r = evaluate_bondnet(
            model, ds, device, batch_size, num_workers,
            noise_sigma=sigma,
            noise_cutoff=getattr(base_dataset_args, 'cutoff', 2.5),
        )
        r['sigma'] = sigma
        results.append(r)
        log.info(
            f'  sigma={sigma:.3f}: f1_macro={r.get("f1_macro", 0):.4f} '
            f'mol_validity={r.get("mol_validity", 0)*100:.1f}%'
        )
    return results


# ------------------------------------------------------------------ #
# Reporting                                                             #
# ------------------------------------------------------------------ #

def print_results_table(results_by_method: Dict[str, List[Dict]]) -> None:
    header = f"{'Method':<15} {'sigma(A)':>8} {'F1-single':>10} {'F1-double':>10} {'F1-triple':>10} {'F1-arom':>10} {'F1-macro':>10} {'Mol-valid%':>11}"
    print('\n' + header)
    print('-' * len(header))

    for method, results in results_by_method.items():
        if isinstance(results, dict):
            results = [results]
        for r in results:
            sigma = r.get('sigma', 0.0)
            print(
                f"{method:<15} {sigma:>7.3f} "
                f"{r.get('f1_single', 0):>10.4f} "
                f"{r.get('f1_double', 0):>10.4f} "
                f"{r.get('f1_triple', 0):>10.4f} "
                f"{r.get('f1_aromatic', 0):>10.4f} "
                f"{r.get('f1_macro', 0):>10.4f} "
                f"{r.get('mol_validity', 0)*100:>10.1f}%"
            )


# ------------------------------------------------------------------ #
# Main                                                                  #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()
    device = _get_device(args.device)
    log.info(f'Device: {device}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Stage 1 model
    log.info(f'Loading checkpoint: {args.checkpoint}')
    model = load_model(args.checkpoint, device)
    log.info('Model loaded.')

    stage2 = None
    if args.stage2_ckpt:
        from bondnet.model.bond_type_gnn import BondTypeGNN
        ck2 = torch.load(args.stage2_ckpt, map_location='cpu', weights_only=False)
        s2_args = ck2.get('args', {})
        if not args.allow_stage2_cache_mismatch:
            _check_stage2_cache_provenance(args.stage2_ckpt, s2_args, args.cache_path)
        stage2 = BondTypeGNN(
            hidden_size=s2_args.get('hidden_size', model.backbone.hidden_state_size),
            num_layers=s2_args.get('num_layers', 4),
            edge_embedding_size=s2_args.get('edge_embedding_size', 16),
            bond_cutoff=s2_args.get('bond_cutoff', 3.0),
            random_node_feat_dim=s2_args.get('random_node_feat_dim', 0),
            random_node_feat_std=s2_args.get('random_node_feat_std', 1.0),
        ).to(device)
        stage2.load_state_dict(ck2['stage2_state_dict'])
        stage2.eval()
        log.info(f'Stage 2 loaded from {args.stage2_ckpt}')

    if args.constrained_discretization and args.bo_guided_decode:
        log.warning('Both --constrained_discretization and --bo_guided_decode enabled; constrained discretization is applied after BO-guided decode.')

    all_results: Dict[str, List[Dict]] = {}

    # 鈹€鈹€ BondNet robustness curve 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€ #
    log.info('Running BondNet robustness evaluation...')
    bondnet_results = []
    for sigma in args.noise_levels:
        log.info(f'  sigma={sigma:.3f} A...')
        sigma_seed = int(args.eval_noise_seed) + int(round(float(sigma) * 10000.0))
        torch.manual_seed(sigma_seed)
        np.random.seed(sigma_seed % (2**32 - 1))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sigma_seed)
        ds = load_test_dataset(args, sigma=sigma)
        ds = apply_requested_hydrogen_corruption(ds, args)
        r = evaluate_bondnet(
            model, ds, device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_constrained_disc=args.constrained_discretization,
            use_bo_guided_decode=args.bo_guided_decode,
            bo_conf_threshold=args.bo_conf_threshold,
            bo_resonance_lo=args.bo_resonance_lo,
            bo_resonance_hi=args.bo_resonance_hi,
            apply_ring_rules=args.apply_ring_rules,
            use_valence_decode=args.valence_decode,
            valence_conf_gate=args.valence_conf_gate,
            valence_prune=args.valence_prune,
            use_resonance_aware=args.resonance_aware,
            collect_error_analysis=args.save_error_analysis,
            error_analysis_max_records=args.error_analysis_max_records,
            stage2=stage2,
            noise_sigma=sigma,
            noise_cutoff=args.cutoff,
            noise_h_cutoff=args.h_cutoff,
            noise_seed=sigma_seed,
            hh_only=getattr(args, 'hh_only', False),
            conn_threshold=args.conn_threshold,
        )
        r['sigma'] = sigma
        r['eval_noise_seed'] = sigma_seed
        r['hydrogen_drop_fraction'] = float(args.hydrogen_drop_fraction)
        r['hydrogen_noise_sigma'] = float(args.hydrogen_noise_sigma)
        r['hydrogen_corruption_seed'] = int(args.hydrogen_corruption_seed)
        if args.save_error_analysis and 'error_analysis' in r:
            error_analysis = r.pop('error_analysis')
            error_path = output_dir / f'error_analysis_sigma_{sigma:.3f}.json'
            with open(error_path, 'w') as f:
                json.dump(error_analysis, f, indent=2)
            r['error_analysis_path'] = error_path.name
            r['error_summary'] = {
                'n_errors': error_analysis.get('n_errors', 0),
                'error_rate': error_analysis.get('error_rate', 0.0),
                'top_reasons': dict(list(error_analysis.get('errors_by_reason', {}).items())[:5]),
            }
        bondnet_results.append(r)
        log.info(
            f'    reference_pair_f1={r.get("f1_macro", 0):.4f} '
            f'pipeline_f1={r.get("f1_macro_pipeline", 0):.4f} '
            f'hh_exact={r.get("full_graph_exact_match", 0):.4f} '
            f'conn_f1={r.get("conn_f1", 0):.4f}'
        )

    all_results['BondNet'] = bondnet_results

    # 鈹€鈹€ Baselines 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€ #
    if args.run_baselines:
        if args.cache_path:
            log.warning('Skipping baselines for --cache_path evaluation; baselines require source molecules.')
        else:
            base_ds = load_test_dataset(args, sigma=0.0)

            log.info('Running OpenBabel baseline...')
            ob_result = evaluate_openbabel(base_ds)
            if ob_result:
                ob_result['sigma'] = 0.0
                all_results['OpenBabel'] = [ob_result]

            log.info('Running RDKit xyz2mol baseline...')
            rdkit_result = evaluate_rdkit_xyz2mol(base_ds)
            if rdkit_result:
                rdkit_result['sigma'] = 0.0
                all_results['RDKit-xyz2mol'] = [rdkit_result]

    # 鈹€鈹€ Print table 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€ #
    print_results_table(all_results)

    # 鈹€鈹€ Save results 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€ #
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log.info(f'Results saved to {results_path}')

    # 鈹€鈹€ Save robustness CSV 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€ #
    csv_path = output_dir / 'robustness_curve.csv'
    with open(csv_path, 'w') as f:
        f.write(
            'method,sigma,f1_single,f1_double,f1_triple,f1_aromatic,'
            'reference_pair_macro_f1,pipeline_macro_f1,'
            'true_bond_exact_type_rate,hh_graph_exact_match,mol_validity\n'
        )
        for method, results in all_results.items():
            for r in results:
                f.write(
                    f"{method},"
                    f"{r.get('sigma', 0):.4f},"
                    f"{r.get('f1_single', 0):.6f},"
                    f"{r.get('f1_double', 0):.6f},"
                    f"{r.get('f1_triple', 0):.6f},"
                    f"{r.get('f1_aromatic', 0):.6f},"
                    f"{r.get('f1_macro', 0):.6f},"
                    f"{r.get('f1_macro_pipeline', 0):.6f},"
                    f"{r.get('true_bond_exact_type_rate', 0):.6f},"
                    f"{r.get('full_graph_exact_match', 0):.6f},"
                    f"{r.get('mol_validity', 0):.6f}\n"
                )
    log.info(f'Robustness CSV saved to {csv_path}')


if __name__ == '__main__':
    main()
