"""
Evaluate Stage 2 with oracle connectivity.

This answers: if the bonded topology is known perfectly, how robust is the
bond-type head itself under coordinate noise?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bondnet.data.dataset import (  # noqa: E402
    CachedBondNetDataset,
    ShardedCachedBondNetDataset,
)
from bondnet.data.noise_augment import GaussianNoiseAugment  # noqa: E402
from bondnet.model.bond_type_gnn import BondTypeGNN, compute_bonded_geometry  # noqa: E402
from bondnet.model.bondnet import _symmetrize_logits  # noqa: E402
from bondnet.utils.metrics import BondNetMetrics  # noqa: E402
from train_stage2 import collate_stage2_train_fast  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Oracle-connectivity Stage2 evaluation")
    p.add_argument("--stage2_ckpt", required=True)
    p.add_argument("--cache_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--noise_levels", nargs="+", type=float, default=[0.0, 0.01, 0.05, 0.1, 0.2])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--eval_split", choices=["all", "val", "test"], default="test")
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--split_key", type=str, default=None)
    p.add_argument("--train_groups", type=int, default=None)
    p.add_argument("--max_mols", type=int, default=None)
    p.add_argument("--shard_cache_size", type=int, default=2)
    p.add_argument("--noise_clean_prob", type=float, default=0.0)
    p.add_argument("--ece_bins", type=int, default=15)
    return p.parse_args()


def _device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _load_dataset(args: argparse.Namespace):
    if ShardedCachedBondNetDataset.is_sharded_cache(args.cache_path):
        if args.eval_split == "val":
            _, val = ShardedCachedBondNetDataset.split(
                args.cache_path,
                val_frac=args.val_split,
                split_key=args.split_key,
                train_groups=args.train_groups,
                max_samples=args.max_mols,
                max_open_shards=args.shard_cache_size,
            )
            return val
        ds = ShardedCachedBondNetDataset(
            args.cache_path,
            max_open_shards=args.shard_cache_size,
        )
        if args.eval_split == "test":
            labels = [m.get("split") for m in (ds._sample_meta or [])]
            indices = [i for i, label in enumerate(labels) if label == "test"]
            if not indices:
                raise ValueError("The sharded cache has no fixed test labels.")
            ds = ShardedCachedBondNetDataset(
                args.cache_path,
                indices=indices,
                max_open_shards=args.shard_cache_size,
            )
    else:
        if args.eval_split == "val":
            _, val = CachedBondNetDataset.split(
                args.cache_path,
                val_frac=args.val_split,
                split_key=args.split_key,
                train_groups=args.train_groups,
            )
            return val
        ds = CachedBondNetDataset(args.cache_path)
        if args.eval_split == "test":
            indices = [i for i, sample in enumerate(ds.samples) if sample.get("split") == "test"]
            if not indices:
                raise ValueError("The cache has no fixed test labels.")
            ds = Subset(ds, indices)

    if args.max_mols is not None and len(ds) > args.max_mols:
        return Subset(ds, range(args.max_mols))
    return ds


def _load_stage2(path: str, device: torch.device) -> BondTypeGNN:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    saved = ck.get("args", {})
    model = BondTypeGNN(
        hidden_size=saved.get("hidden_size", 128),
        num_layers=saved.get("num_layers", 4),
        edge_embedding_size=saved.get("edge_embedding_size", 32),
        bond_cutoff=saved.get("bond_cutoff", 3.0),
        dropout=0.0,
        random_node_feat_dim=saved.get("random_node_feat_dim", 0),
        random_node_feat_std=saved.get("random_node_feat_std", 1.0),
    ).to(device)
    model.load_state_dict(ck["stage2_state_dict"])
    model.eval()
    return model


def _to_device(batch: Dict, device: torch.device) -> Dict:
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    } if batch else batch


def _teacher_forced_edges(batch: Dict):
    bond_ei = batch["edge_index"][batch["bond_mask"]]
    diff, dist = compute_bonded_geometry(batch["coord"], bond_ei)
    elems = batch["elems"]
    readout_mask = (elems[bond_ei[:, 0]] != 1) & (elems[bond_ei[:, 1]] != 1)
    readout_ei = bond_ei[readout_mask]
    readout_diff, readout_dist = compute_bonded_geometry(batch["coord"], readout_ei)
    target = batch["bond_type_full"][readout_mask]
    return bond_ei, diff, dist, readout_ei, readout_diff, readout_dist, target


def _calibration_stats(logits: torch.Tensor, target: torch.Tensor, bins: int) -> Dict[str, float]:
    if logits.numel() == 0:
        return {"nll": 0.0, "ece": 0.0, "mean_confidence": 0.0}
    probs = torch.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)
    correct = pred.eq(target).float()
    nll = F.cross_entropy(logits, target, reduction="mean")

    ece = torch.zeros((), device=logits.device)
    edges = torch.linspace(0, 1, bins + 1, device=logits.device)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece = ece + mask.float().mean() * (conf[mask].mean() - correct[mask].mean()).abs()

    return {
        "nll": float(nll.detach().cpu()),
        "ece": float(ece.detach().cpu()),
        "mean_confidence": float(conf.mean().detach().cpu()),
    }


@torch.no_grad()
def evaluate_one_sigma(stage2: BondTypeGNN, dataset, sigma: float, args, device: torch.device) -> Dict:
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_stage2_train_fast,
        num_workers=args.num_workers,
    )
    aug = None
    if sigma > 0:
        aug = GaussianNoiseAugment(
            sigma=sigma,
            sigma_min=sigma,
            sigma_max=sigma,
            cutoff=getattr(stage2, "bond_cutoff", 3.0),
        )

    metrics = BondNetMetrics()
    logits_all: List[torch.Tensor] = []
    target_all: List[torch.Tensor] = []
    n_mols = 0

    for batch in loader:
        if not batch:
            continue
        if aug is not None:
            batch = aug.augment_batch(batch, per_molecule=True, clean_prob=args.noise_clean_prob)
        batch = _to_device(batch, device)
        bond_ei, diff, dist, readout_ei, readout_diff, readout_dist, target = _teacher_forced_edges(batch)
        if readout_ei.numel() == 0:
            continue
        logits = stage2(batch["elems"], bond_ei, diff, dist, readout_ei, readout_diff, readout_dist)
        logits = _symmetrize_logits(logits, readout_ei)
        pred = logits.argmax(dim=-1)
        metrics.update_bond_types(pred, target)
        logits_all.append(logits.detach())
        target_all.append(target.detach())
        n_mols += int(batch.get("num_mols", 1))

    out = metrics.compute()
    if logits_all:
        cal = _calibration_stats(torch.cat(logits_all), torch.cat(target_all), args.ece_bins)
        out.update(cal)
    out["sigma"] = sigma
    out["n_molecules"] = n_mols
    out["mode"] = "stage2_oracle_connectivity"
    return out


def main() -> int:
    args = parse_args()
    device = _device(args.device)
    dataset = _load_dataset(args)
    stage2 = _load_stage2(args.stage2_ckpt, device)

    results = []
    for sigma in args.noise_levels:
        print(f"[oracle] sigma={sigma:g}")
        results.append(evaluate_one_sigma(stage2, dataset, sigma, args, device))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"Stage2Oracle": results}
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(out_dir / "robustness_curve.csv", "w", encoding="utf-8") as f:
        f.write("method,sigma,f1_single,f1_double,f1_triple,f1_aromatic,f1_macro,nll,ece,mean_confidence\n")
        for r in results:
            f.write(
                "Stage2Oracle,"
                f"{r.get('sigma', 0):.6f},"
                f"{r.get('f1_single', 0):.6f},"
                f"{r.get('f1_double', 0):.6f},"
                f"{r.get('f1_triple', 0):.6f},"
                f"{r.get('f1_aromatic', 0):.6f},"
                f"{r.get('f1_macro', 0):.6f},"
                f"{r.get('nll', 0):.6f},"
                f"{r.get('ece', 0):.6f},"
                f"{r.get('mean_confidence', 0):.6f}\n"
            )
    print(f"[oracle] wrote {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
