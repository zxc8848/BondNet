#!/usr/bin/env python3
"""Summarize the one-shot evaluation on the frozen v3 external cohort.

Reads only files written by scripts/run_revision_v3_external.ps1 and checks,
for every learned-model result, that the evaluated molecule sequence equals the
frozen CID list (ordered-id SHA-256 in data/external_v3/manifest.json).

Outputs (results/revision_v3_external/):
  summary.json   every number, per seed and aggregated
  summary.csv    one row per (sigma, method)
  summary.md     table in the layout of Table 5
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "external_v3"
RESULTS = REPO / "results" / "revision_v3_external"
MODELS = {"joint": "Joint explicit-H", "heavy": "Joint heavy-only", "staged": "Hard two-stage"}
SEEDS = (42, 43, 44)
SIGMAS = (0.0, 0.1, 0.2)
T95_DF2 = 4.302652729911275
LEARNED_FIELDS = (
    ("undirected_pipeline_macro_f1", "pipeline_f1"),
    ("pipeline_macro_f1", "pipeline_f1_directed"),
    ("reference_pair_macro_f1", "conditional_f1"),
    ("undirected_true_pair_exact_rate", "exact_type"),
    ("true_pair_exact_rate", "exact_type_directed"),
    ("undirected_hh_graph_exact_rate", "hh_exact"),
    ("hh_graph_exact_rate", "hh_exact_directed"),
    ("conn_exact_rate", "connectivity_exact"),
)
RULE_FIELDS = (
    ("f1_macro_pipeline", "pipeline_f1"),
    ("f1_macro_true_pair", "conditional_f1"),
    ("true_bond_exact_type_rate", "exact_type"),
    ("full_graph_exact_match", "hh_exact"),
    ("success_rate", "success_rate"),
)


def tag(sigma: float) -> str:
    return f"{round(100 * sigma):03d}"


def ids_digest(ids) -> str:
    return hashlib.sha256(",".join(str(int(x)) for x in ids).encode()).hexdigest()


def main() -> int:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    n_expected = manifest["selection"]["cohort"]["n_molecules"]
    id_hash = manifest["selection"]["cohort"]["ordered_id_sha256"]
    # START.json is written by Windows PowerShell, which may prepend a BOM.
    start = json.loads((RESULTS / "START.json").read_text(encoding="utf-8-sig"))
    if start["manifest_sha256"] != (DATA / "manifest.sha256").read_text().split()[0]:
        raise ValueError("results were produced against a different manifest")

    rows, per_seed = [], {}
    for model in MODELS:
        for seed in SEEDS:
            out = RESULTS / model / f"seed{seed}"
            agg = json.loads((out / "aggregate.json").read_text(encoding="utf-8"))
            by_sigma = {float(r["sigma"]): r for r in agg}
            if set(by_sigma) != set(SIGMAS):
                raise ValueError(f"{out}: noise levels {sorted(by_sigma)}")
            for sigma in SIGMAS:
                if by_sigma[sigma]["n_molecules"] != n_expected:
                    raise ValueError(f"{out}: {by_sigma[sigma]['n_molecules']} molecules")
                ids = np.load(out / f"sigma_{tag(sigma)}.npz")["mol_idx"]
                if ids_digest(ids) != id_hash:
                    raise ValueError(f"{out} sigma={sigma}: molecule list differs from manifest")
            per_seed[model, seed] = by_sigma

    for sigma in SIGMAS:
        for model, label in MODELS.items():
            row = {"sigma": sigma, "method": label, "n_molecules": n_expected, "n_seeds": 3}
            for src, dst in LEARNED_FIELDS:
                vals = [float(per_seed[model, s][sigma][src]) for s in SEEDS]
                row[dst] = statistics.mean(vals)
                row[dst + "_sd"] = statistics.stdev(vals)
                row[dst + "_by_seed"] = dict(zip(map(str, SEEDS), vals))
            rows.append(row)
        for method, label in (("rdkit", "RDKit"), ("openbabel", "OpenBabel")):
            path = RESULTS / "rule_baselines" / "scores" / f"{method}_sigma_{tag(sigma)}.json"
            rep = json.loads(path.read_text(encoding="utf-8"))
            if rep["n_molecules"] != n_expected:
                raise ValueError(f"{path}: {rep['n_molecules']} molecules")
            row = {"sigma": sigma, "method": label, "n_molecules": n_expected, "n_seeds": 1}
            for src, dst in RULE_FIELDS:
                row[dst] = float(rep[src])
            row["pipeline_f1_conditional_success"] = float(
                rep.get("f1_macro_pipeline_conditional_success", rep["f1_macro_pipeline"]))
            rows.append(row)

    validity = {}
    for seed in SEEDS:
        path = RESULTS / "joint_validity" / f"seed{seed}" / "summary.json"
        if path.exists():
            validity[str(seed)] = json.loads(path.read_text(encoding="utf-8"))

    paired = []
    for sigma in SIGMAS:
        for field in ("undirected_pipeline_macro_f1", "undirected_hh_graph_exact_rate"):
            for other in ("heavy", "staged"):
                d = [per_seed["joint", s][sigma][field] - per_seed[other, s][sigma][field]
                     for s in SEEDS]
                m, sd = statistics.mean(d), statistics.stdev(d)
                hw = T95_DF2 * sd / math.sqrt(len(SEEDS))
                paired.append({"sigma": sigma, "field": field, "comparison": f"joint_minus_{other}",
                               "mean": m, "sd": sd, "seed_t95": [m - hw, m + hw]})

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps({
        "manifest_sha256": start["manifest_sha256"], "n_molecules": n_expected,
        "rows": rows, "paired_seed_differences": paired, "joint_validity_by_seed": validity,
        "note": "Learned models: mean and sample SD over training seeds 42/43/44. "
                "Rule tools are deterministic. Pipeline F1 and HH exact use undirected "
                "pair scoring as in Table 5; directed values are reported alongside.",
    }, indent=2) + "\n", encoding="utf-8")

    keys = ["sigma", "method", "n_molecules", "n_seeds", "pipeline_f1", "pipeline_f1_sd",
            "conditional_f1", "exact_type", "exact_type_sd", "hh_exact", "hh_exact_sd",
            "pipeline_f1_directed", "hh_exact_directed", "success_rate"]
    with open(RESULTS / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def fmt(row, key, pct=False, digits=5):
        if key not in row:
            return "--"
        v = row[key] * (100 if pct else 1)
        s = f"{v:.2f}" if pct else f"{v:.{digits}f}"
        if key + "_sd" in row:
            sd = row[key + "_sd"] * (100 if pct else 1)
            s += f" ± {sd:.2f}" if pct else f" ± {sd:.{digits}f}"
        return s

    lines = [f"External PubChem3D cohort, n = {n_expected:,} (manifest {start['manifest_sha256'][:12]})",
             "", "| σ (Å) | Method | Pipeline F1 | Exact-type (%) | HH exact (%) | Success (%) |",
             "|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['sigma']:.2f} | {row['method']} | {fmt(row, 'pipeline_f1')} | "
                     f"{fmt(row, 'exact_type', True)} | {fmt(row, 'hh_exact', True)} | "
                     f"{fmt(row, 'success_rate', True) if 'success_rate' in row else '100.00'} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
