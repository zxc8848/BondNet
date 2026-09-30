"""Connectivity threshold sweep on a fixed validation or test split.

The model is run once per noise level. Threshold metrics are then computed from
the retained probabilities, so a dense precision--recall curve does not repeat
neural inference. Molecule-level exact connectivity is reported alongside edge
precision/recall/F1. Use validation output to choose a threshold and never choose
from the held-out test output.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bondnet.data.dataset import (  # noqa: E402
    CachedBondNetDataset,
    ShardedCachedBondNetDataset,
    collate_fn,
)
from bondnet.data.noise_augment import (  # noqa: E402
    GaussianNoiseAugment,
    apply_dynamic_candidate_mask,
)
from evaluate import load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep BondNet connectivity thresholds")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--eval_split", choices=["val", "test"], default="val")
    p.add_argument("--noise_levels", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.15, 0.20])
    p.add_argument("--thresholds", nargs="+", type=float, default=None)
    p.add_argument("--selection_sigma", type=float, default=0.10)
    p.add_argument("--eval_noise_seed", type=int, default=20260921)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--cutoff", type=float, default=2.5)
    p.add_argument("--shard_cache_size", type=int, default=2)
    return p.parse_args()


def _device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _load_fixed_split(args: argparse.Namespace):
    if ShardedCachedBondNetDataset.is_sharded_cache(args.cache_path):
        all_ds = ShardedCachedBondNetDataset(
            args.cache_path, max_open_shards=args.shard_cache_size
        )
        labels = [m.get("split") for m in (all_ds._sample_meta or [])]
        indices = [i for i, label in enumerate(labels) if label == args.eval_split]
        if not indices:
            raise ValueError(f"No {args.eval_split!r} labels in sharded cache")
        return ShardedCachedBondNetDataset(
            args.cache_path,
            indices=indices,
            max_open_shards=args.shard_cache_size,
        )

    ds = CachedBondNetDataset(args.cache_path)
    indices = [i for i, sample in enumerate(ds.samples) if sample.get("split") == args.eval_split]
    if not indices:
        raise ValueError(f"No {args.eval_split!r} labels in cache")
    return Subset(ds, indices)


@torch.no_grad()
def _collect_probabilities(
    model,
    dataset,
    sigma: float,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    probabilities: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    molecule_ids: List[np.ndarray] = []
    molecule_offset = 0
    augment = None
    if sigma > 0:
        augment = GaussianNoiseAugment(
            sigma=sigma,
            sigma_min=sigma,
            sigma_max=sigma,
            cutoff=args.cutoff,
        )

    for batch in loader:
        if not batch:
            continue
        if augment is not None:
            noise_seed = int(args.eval_noise_seed) + int(round(float(sigma) * 10000.0))
            batch = augment.augment_batch(
                batch, per_molecule=True, base_seed=noise_seed, epoch=0
            )
        batch = apply_dynamic_candidate_mask(batch, args.cutoff, args.cutoff)
        batch = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        _, _, edge_features = model.backbone(
            batch["elems"],
            batch["coord"],
            batch["edge_index"],
            batch["edge_diff"],
            batch["edge_dist"],
            num_atoms_per_mol=batch.get("num_atoms_per_mol"),
        )
        probs = torch.sigmoid(model.stage1(edge_features))
        train_edge_mask = batch.get("train_edge_mask")
        if train_edge_mask is not None:
            probs = torch.where(train_edge_mask, probs, torch.zeros_like(probs))
        edge_mol = batch["edge_mol_idx"] + molecule_offset
        probabilities.append(probs.detach().cpu().numpy())
        labels.append(batch["bond_mask"].detach().cpu().numpy().astype(bool))
        molecule_ids.append(edge_mol.detach().cpu().numpy())
        molecule_offset += int(batch.get("num_mols", 1))

    return (
        np.concatenate(probabilities),
        np.concatenate(labels),
        np.concatenate(molecule_ids),
        molecule_offset,
    )


def _score_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    molecule_ids: np.ndarray,
    n_molecules: int,
    threshold: float,
) -> Dict[str, float]:
    pred = probabilities >= threshold
    tp = int(np.logical_and(pred, labels).sum())
    fp = int(np.logical_and(pred, ~labels).sum())
    fn = int(np.logical_and(~pred, labels).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    per_edge_correct = pred == labels
    exact = np.ones(n_molecules, dtype=bool)
    np.logical_and.at(exact, molecule_ids, per_edge_correct)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "molecule_connectivity_exact": float(exact.mean()) if n_molecules else 0.0,
        "tp_directed": tp,
        "fp_directed": fp,
        "fn_directed": fn,
        "n_candidates_directed": int(labels.size),
        "n_positive_directed": int(labels.sum()),
        "positive_prevalence": float(labels.mean()) if labels.size else 0.0,
    }


def main() -> int:
    args = parse_args()
    thresholds = args.thresholds
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101).tolist()
    if any(t < 0 or t > 1 for t in thresholds):
        raise ValueError("Thresholds must lie in [0, 1]")

    device = _device(args.device)
    dataset = _load_fixed_split(args)
    model = load_model(args.checkpoint, device)
    model.eval()
    rows: List[Dict[str, float]] = []
    for sigma in args.noise_levels:
        sigma_seed = int(args.eval_noise_seed) + int(round(float(sigma) * 10000.0))
        torch.manual_seed(sigma_seed)
        np.random.seed(sigma_seed % (2**32 - 1))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sigma_seed)
        probs, labels, molecule_ids, n_molecules = _collect_probabilities(
            model, dataset, sigma, args, device
        )
        for threshold in thresholds:
            row = _score_threshold(probs, labels, molecule_ids, n_molecules, threshold)
            row["sigma"] = float(sigma)
            row["n_molecules"] = int(n_molecules)
            rows.append(row)

    target = min(args.noise_levels, key=lambda x: abs(x - args.selection_sigma))
    target_rows = [row for row in rows if row["sigma"] == float(target)]
    selected = max(
        target_rows,
        key=lambda row: (
            row["f1"],
            row["molecule_connectivity_exact"],
            -abs(row["threshold"] - 0.5),
        ),
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "connectivity_threshold_sweep.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(output / "connectivity_threshold_sweep.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "eval_split": args.eval_split,
                "eval_noise_seed": args.eval_noise_seed,
                "selection_sigma": target,
                "selected_threshold": selected["threshold"] if args.eval_split == "val" else None,
                "selection_rule": "maximum edge F1, then molecule exactness, then proximity to 0.5",
                "rows": rows,
            },
            handle,
            indent=2,
        )
    if args.eval_split == "val":
        print(
            f"split=val molecules={len(dataset)}; "
            f"sigma={target:g} best threshold={selected['threshold']:.2f} "
            f"P={selected['precision']:.6f} R={selected['recall']:.6f} "
            f"F1={selected['f1']:.6f} exact={selected['molecule_connectivity_exact']:.6f}"
        )
    else:
        print(
            f"split=test molecules={len(dataset)}; threshold sweep saved without "
            "test-based selection"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
