"""Paired molecule bootstrap for revision-v2 explicit-H vs heavy-only models."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def macro_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0).mean(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats_root", type=Path,
                        default=Path("results/revision_v2_bootstrap/stats"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/revision_v2_bootstrap/hydrogen_bootstrap.csv"))
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for sigma in (0.0, 0.1, 0.2):
        tag = f"{round(100 * sigma):03d}"
        data = {}
        for training_seed in (42, 43, 44):
            data[training_seed] = {}
            for method in ("two_stage", "heavy_two_stage"):
                path = ROOT / args.stats_root / f"seed{training_seed}" / method / f"sigma_{tag}.npz"
                with np.load(path) as archive:
                    data[training_seed][method] = {key: archive[key] for key in archive.files}
        ids = data[42]["two_stage"]["mol_idx"]
        for seed_data in data.values():
            for stats in seed_data.values():
                if not np.array_equal(stats["mol_idx"], ids):
                    raise ValueError("Explicit-H and heavy-only molecule order differs")
        n = len(ids)
        points = []
        for seed_data in data.values():
            exp = seed_data["two_stage"]
            heavy = seed_data["heavy_two_stage"]
            points.append(
                macro_f1(exp["pipe_tp"].sum(0), exp["pipe_fp"].sum(0), exp["pipe_fn"].sum(0))
                - macro_f1(heavy["pipe_tp"].sum(0), heavy["pipe_fp"].sum(0), heavy["pipe_fn"].sum(0))
            )
        values = []
        remaining = args.replicates
        while remaining:
            chunk = min(20, remaining)
            indices = rng.integers(0, n, size=(chunk, n), dtype=np.int32)
            diffs = []
            for seed_data in data.values():
                scores = []
                for method in ("two_stage", "heavy_two_stage"):
                    stats = seed_data[method]
                    scores.append(macro_f1(
                        stats["pipe_tp"][indices].sum(1),
                        stats["pipe_fp"][indices].sum(1),
                        stats["pipe_fn"][indices].sum(1),
                    ))
                diffs.append(scores[0] - scores[1])
            values.extend(np.mean(diffs, axis=0).tolist())
            remaining -= chunk
        low, high = np.quantile(values, (0.025, 0.975))
        rows.append({"sigma": sigma, "n_molecules": n, "replicates": args.replicates,
                     "comparison": "explicit_minus_heavy_pipeline_macro_f1",
                     "point": float(np.mean(points)), "molecule_ci_low": float(low),
                     "molecule_ci_high": float(high),
                     "training_seed_sd": float(np.std(points, ddof=1))})
        print(f"sigma={sigma:g} point={rows[-1]['point']:.6f} CI=[{low:.6f},{high:.6f}]", flush=True)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
