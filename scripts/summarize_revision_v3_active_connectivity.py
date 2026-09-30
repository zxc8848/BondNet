"""Summarize connectivity on the active 2.5-A graph, not the 3.0-A cache.

The archived per-molecule audit counted every non-H--H cache record. Model
predictions outside the active graph were forced negative, so its TP and FP
counts are already restricted to active records. Only its FN count includes
reference bonds that moved outside the active radius. Reconstruct the exact
evaluation-time masks without rerunning any checkpoint, subtract those bonds,
and combine the corrected counts with the archived seed-level audit files.
"""

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

from bondnet.data.dataset import CachedBondNetDataset, collate_fn
from bondnet.data.noise_augment import GaussianNoiseAugment, apply_dynamic_candidate_mask


def active_counts(dataset, sigma: float, noise_seed: int, batch_size: int,
                  cutoff: float) -> dict[str, int]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    aug = GaussianNoiseAugment(sigma=sigma, sigma_min=sigma,
                               sigma_max=sigma, cutoff=cutoff)
    counts = {"cache_candidates": 0, "active_candidates": 0,
              "cache_bonds": 0, "active_bonds": 0}
    for batch in loader:
        if sigma > 0:
            batch = aug.augment_batch(
                batch, per_molecule=True,
                base_seed=noise_seed + int(round(sigma * 10000.0)), epoch=0,
            )
        batch = apply_dynamic_candidate_mask(batch, cutoff, cutoff)
        edge_index = batch["edge_index"]
        elems = batch["elems"]
        src, dst = edge_index[:, 0], edge_index[:, 1]
        valid = ~((elems[src] == 1) & (elems[dst] == 1))
        active = batch["train_edge_mask"] & valid
        bonded = batch["bond_mask"].bool()
        counts["cache_candidates"] += int(valid.sum())
        counts["active_candidates"] += int(active.sum())
        counts["cache_bonds"] += int((bonded & valid).sum())
        counts["active_bonds"] += int((bonded & active).sum())
    counts["excluded_reference_bonds"] = (
        counts["cache_bonds"] - counts["active_bonds"]
    )
    return counts


def corrected_metrics(tp: int, fp: int, fn_cache: int,
                      excluded_reference_bonds: int) -> dict[str, float | int]:
    fn_active = fn_cache - excluded_reference_bonds
    if min(tp, fp, fn_active) < 0:
        raise ValueError("Negative confusion count after active-mask correction")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn_active) if tp + fn_active else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn_active) if 2 * tp + fp + fn_active else 0.0
    return {"tp": tp, "fp": fp, "fn": fn_active,
            "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="data/fixed_split_caches/geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed_test.pt")
    parser.add_argument("--audit-dir", default="results/revision_v3_undirected_audit/joint")
    parser.add_argument("--output", default="results/revision_v3_active_connectivity/summary.json")
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0.0, 0.1, 0.2])
    parser.add_argument("--eval-noise-seed", type=int, default=20260921)
    parser.add_argument("--cutoff", type=float, default=2.5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    dataset = CachedBondNetDataset(str(ROOT / args.cache))
    rows = []
    for sigma in args.noise_levels:
        graph = active_counts(dataset, sigma, args.eval_noise_seed,
                              args.batch_size, args.cutoff)
        seed_rows = []
        for seed in (42, 43, 44):
            path = ROOT / args.audit_dir / f"seed{seed}" / f"sigma_{round(100 * sigma):03d}.npz"
            with np.load(path) as data:
                if len(data["mol_idx"]) != len(dataset):
                    raise ValueError(f"Audit molecule count differs from cache: {path}")
                tp = int(data["conn_tp"].sum())
                fp = int(data["conn_fp"].sum())
                fn_cache = int(data["conn_fn"].sum())
                if tp + fn_cache != graph["cache_bonds"]:
                    raise ValueError(f"Audit reference-bond count differs from cache: {path}")
                metrics = corrected_metrics(tp, fp, fn_cache,
                                            graph["excluded_reference_bonds"])
                if metrics["tp"] + metrics["fn"] != graph["active_bonds"]:
                    raise ValueError(f"Active reference-bond count mismatch: {path}")
                aggregate_path = ROOT / args.audit_dir / f"seed{seed}" / "aggregate.json"
                aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
                matching = [row for row in aggregate if abs(row["sigma"] - sigma) < 1e-12]
                if len(matching) != 1:
                    raise ValueError(f"Missing or duplicate sigma row: {aggregate_path}")
                metrics["undirected_pipeline_f1_per_class"] = matching[0][
                    "undirected_pipeline_f1_per_class"
                ]
                metrics["seed"] = seed
                seed_rows.append(metrics)
        summary = {}
        for name in ("precision", "recall", "f1"):
            values = np.asarray([row[name] for row in seed_rows], dtype=float)
            summary[name] = {"mean": float(values.mean()),
                             "sample_sd": float(values.std(ddof=1))}
        class_values = np.asarray(
            [row["undirected_pipeline_f1_per_class"] for row in seed_rows],
            dtype=float,
        )
        summary["undirected_pipeline_f1_per_class"] = {
            name: {"mean": float(class_values[:, col].mean()),
                   "sample_sd": float(class_values[:, col].std(ddof=1))}
            for col, name in enumerate(("single", "double", "triple", "aromatic"))
        }
        row = {"sigma": sigma, "graph": graph,
               "seeds": seed_rows, "three_seed": summary}
        rows.append(row)
        print(f"sigma={sigma:.2f} active={graph['active_candidates']} "
              f"excluded_bonds={graph['excluded_reference_bonds']} "
              f"P={summary['precision']['mean']:.6f} "
              f"R={summary['recall']['mean']:.6f} "
              f"F1={summary['f1']['mean']:.6f}", flush=True)
    payload = {"protocol": {"cache": args.cache, "audit_dir": args.audit_dir,
                             "eval_noise_seed": args.eval_noise_seed,
                             "cutoff": args.cutoff, "batch_size": args.batch_size,
                             "unit": "directed non-H--H active candidate record"},
               "rows": rows}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
