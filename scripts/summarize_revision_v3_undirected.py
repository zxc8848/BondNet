"""Aggregate the fair undirected HH-pair audit over three training seeds."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path("results/revision_v3_undirected_audit")
MODELS = ("joint", "heavy", "staged")
SEEDS = (42, 43, 44)
SIGMAS = (0.0, 0.1, 0.2)
FIELDS = (
    "undirected_pipeline_macro_f1", "undirected_true_pair_exact_rate",
    "undirected_hh_graph_exact_rate", "pipeline_macro_f1", "hh_graph_exact_rate",
)
T95_DF2 = 4.302652729911275


def main() -> None:
    records = {}
    for model in MODELS:
        for seed in SEEDS:
            path = ROOT / model / f"seed{seed}" / "aggregate.json"
            rows = json.loads(path.read_text(encoding="utf-8"))
            if len(rows) != len(SIGMAS) or any(row["n_molecules"] != 27240 for row in rows):
                raise ValueError(f"Unexpected cohort or noise count: {path}")
            records[model, seed] = {float(row["sigma"]): row for row in rows}
            if set(records[model, seed]) != set(SIGMAS):
                raise ValueError(f"Unexpected noise levels: {path}")
    summary = []
    for sigma in SIGMAS:
        for model in MODELS:
            item = {"sigma": sigma, "model": model, "n_seeds": len(SEEDS), "n_molecules": 27240}
            for field in FIELDS:
                vals = [float(records[model, seed][sigma][field]) for seed in SEEDS]
                item[field + "_mean"] = statistics.mean(vals)
                item[field + "_sample_sd"] = statistics.stdev(vals)
                item[field + "_by_seed"] = dict(zip(map(str, SEEDS), vals))
            summary.append(item)
            print(
                f"sigma={sigma:.2f} {model:<6} "
                f"F1={item['undirected_pipeline_macro_f1_mean']:.6f} "
                f"HH={100 * item['undirected_hh_graph_exact_rate_mean']:.3f}%"
            )
    paired = []
    for sigma in SIGMAS:
        item = {"sigma": sigma, "comparison": "joint_minus_heavy"}
        for field in ("undirected_pipeline_macro_f1", "undirected_hh_graph_exact_rate"):
            differences = [
                records["joint", seed][sigma][field] - records["heavy", seed][sigma][field]
                for seed in SEEDS
            ]
            mean = statistics.mean(differences)
            sd = statistics.stdev(differences)
            half_width = T95_DF2 * sd / math.sqrt(len(SEEDS))
            item[field + "_mean_difference"] = mean
            item[field + "_sample_sd_difference"] = sd
            item[field + "_seed_t95_interval"] = [mean - half_width, mean + half_width]
        paired.append(item)
    output = {"summary": summary, "paired_joint_minus_heavy": paired,
              "note": "Training-seed sample SD and descriptive t interval; not a pristine-test confidence bound."}
    destination = ROOT / "three_seed_summary.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
