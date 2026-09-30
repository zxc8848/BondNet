#!/usr/bin/env python3
"""Aggregate joint-model element-pair strata over training seeds 42-44.

Reads results/revision_v3_joint_pair_strata/seed*/analysis_sigma_*.json
(written by scripts/analyze_pair_strata.py with --pair_policy undirected_or)
and writes three_seed_summary.json next to them.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "revision_v3_joint_pair_strata"
AUDIT = ROOT / "results" / "revision_v3_undirected_audit" / "joint"
SEEDS = (42, 43, 44)
SIGMAS = (0.0, 0.1, 0.2)
SUPPORT = ("unseen", "rare_1_99", "medium_100_999", "common_ge_1000")
SIZES = ("heavy_le_20", "heavy_21_30", "heavy_ge_31")
BONDS = ("single", "double", "triple", "aromatic")


def tag(sigma):
    return f"{round(100 * sigma):03d}"


def ms(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "by_seed": values}


def main():
    data = {}
    for seed in SEEDS:
        audit = {float(r["sigma"]): r for r in
                 json.loads((AUDIT / f"seed{seed}" / "aggregate.json").read_text(encoding="utf-8"))}
        for sigma in SIGMAS:
            path = BASE / f"seed{seed}" / f"analysis_sigma_{tag(sigma)}.json"
            d = json.loads(path.read_text(encoding="utf-8"))
            if d.get("pair_policy") != "undirected_or":
                raise ValueError(f"{path}: expected undirected_or pair policy")
            ref = audit[sigma]["undirected_pipeline_macro_f1"]
            if abs(d["bond_class_pipeline_macro_f1"] - ref) > 1e-6:
                raise ValueError(f"{path}: macro-F1 {d['bond_class_pipeline_macro_f1']} does not "
                                 f"reproduce the main-table value {ref}")
            data[seed, sigma] = d

    out = {"seeds": list(SEEDS), "pair_policy": "undirected_or", "sigmas": []}
    for sigma in SIGMAS:
        rows = [data[s, sigma] for s in SEEDS]
        entry = {"sigma": sigma, "n_molecules": rows[0]["n_molecules"],
                 "bond_class_pipeline_macro_f1": ms([r["bond_class_pipeline_macro_f1"] for r in rows])}
        entry["support_strata"] = []
        for name in SUPPORT:
            per = [next(x for x in r["support_strata"] if x["support_stratum"] == name) for r in rows]
            entry["support_strata"].append({
                "support_stratum": name, "n_true_bonds": per[0]["n_true_bonds"],
                "pipeline_bond_accuracy": ms([p["pipeline_bond_accuracy"] for p in per]),
                "missed_connectivity_rate": ms([p["missed_connectivity_rate"] for p in per]),
            })
        entry["size_strata"] = []
        for name in SIZES:
            per = [next(x for x in r["molecule_size_strata"] if x["size_stratum"] == name) for r in rows]
            entry["size_strata"].append({
                "size_stratum": name, "n_true_bonds": per[0]["true_bond_metrics"]["n_true_bonds"],
                "pipeline_bond_accuracy": ms([p["true_bond_metrics"]["pipeline_bond_accuracy"] for p in per]),
            })
        # reference-bond confusion (rows: true single..aromatic; cols: miss, single..aromatic)
        mats = [r["reference_bond_confusion_true_rows_pred_columns"] for r in rows]
        entry["reference_confusion_by_seed"] = dict(zip(map(str, SEEDS), mats))
        entry["reference_confusion_mean"] = [
            [statistics.mean(m[i][j] for m in mats) for j in range(5)] for i in range(4)]
        # extra heavy-heavy bonds: true no-bond row of the 5-class candidate matrix
        cms = [r["pipeline_candidate_metrics"]["confusion_matrix_true_rows_pred_columns"] for r in rows]
        entry["extra_bonds_by_predicted_class_mean"] = {
            BONDS[j - 1]: statistics.mean(c[0][j] for c in cms) for j in range(1, 5)}
        entry["reference_bond_accuracy"] = ms([r["reference_bond_metrics"]["pipeline_bond_accuracy"] for r in rows])
        out["sigmas"].append(entry)

    # weakest well-populated element-pair/type combinations at sigma = 0 (seed-pooled)
    import csv
    pooled = {}
    for seed in SEEDS:
        with (BASE / f"seed{seed}" / "pair_type_performance_sigma_000.csv").open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = (row["element_pair"], row["bond_type"])
                p = pooled.setdefault(key, {"n": 0, "correct": 0, "train_count": int(row["train_count"])})
                p["n"] += int(row["n"])
                p["correct"] += int(row["correct"])
    weakest = sorted(
        ({"element_pair": k[0], "bond_type": k[1], "test_bonds_per_seed": v["n"] // len(SEEDS),
          "train_count": v["train_count"], "accuracy_pooled_over_seeds": v["correct"] / v["n"]}
         for k, v in pooled.items() if v["n"] // len(SEEDS) >= 20),
        key=lambda r: r["accuracy_pooled_over_seeds"])[:10]
    out["weakest_combinations_sigma_0_min20_bonds"] = weakest

    (BASE / "three_seed_summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for e in out["sigmas"]:
        print(f"sigma={e['sigma']:.2f} macroF1={e['bond_class_pipeline_macro_f1']['mean']:.6f} "
              + " ".join(f"{s['support_stratum']}:{s['n_true_bonds']}/"
                         f"{(s['pipeline_bond_accuracy'] or {}).get('mean', float('nan')):.4f}"
                         for s in e["support_strata"]))
    print(BASE / "three_seed_summary.json")


if __name__ == "__main__":
    main()
