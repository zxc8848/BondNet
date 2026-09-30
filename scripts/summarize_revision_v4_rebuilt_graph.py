#!/usr/bin/env python3
"""Compare rebuilt-from-noisy-coordinates graphs with the primary cached-graph protocol.

Primary results:
  GEOM      results/revision_v3_undirected_audit/<model>/seed<S>/aggregate.json
  external  results/revision_v3_external/<model>/seed<S>/aggregate.json
Rebuilt:    results/revision_v4_rebuilt_graph/<cohort>/<model>/seed<S>/sigma_XXX/
Writes results/revision_v4_rebuilt_graph/summary.json and summary.md.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "revision_v4_rebuilt_graph"
PRIMARY = {"geom": ROOT / "results" / "revision_v3_undirected_audit",
           "external": ROOT / "results" / "revision_v3_external"}
MODELS = {"joint": "Joint, explicit H", "heavy": "Joint, heavy only", "staged": "Hard two-stage"}
SEEDS = (42, 43, 44)
SIGMAS = (0.0, 0.1, 0.2)
FIELDS = (("undirected_pipeline_macro_f1", "pipeline_f1"),
          ("undirected_hh_graph_exact_rate", "hh_exact"),
          ("undirected_true_pair_exact_rate", "exact_type"))
T95_DF2 = 4.302652729911275


def tag(s):
    return f"{round(100 * s):03d}"


def main():
    rows = []
    for cohort, pdir in PRIMARY.items():
        if not (BASE / cohort).exists():
            continue
        for model, label in MODELS.items():
            for sigma in SIGMAS:
                row = {"cohort": cohort, "model": label, "sigma": sigma}
                for src, dst in FIELDS:
                    prim, reb = [], []
                    for seed in SEEDS:
                        p = {float(r["sigma"]): r for r in json.loads(
                            (pdir / model / f"seed{seed}" / "aggregate.json").read_text(encoding="utf-8"))}[sigma]
                        rdir = BASE / cohort / model / f"seed{seed}" / f"sigma_{tag(sigma)}"
                        r = json.loads((rdir / "aggregate.json").read_text(encoding="utf-8"))[0]
                        ids_r = np.load(rdir / "sigma_000.npz")["mol_idx"]
                        ids_p = np.load(pdir / model / f"seed{seed}" / f"sigma_{tag(sigma)}.npz")["mol_idx"]
                        if not np.array_equal(ids_r, ids_p):
                            raise ValueError(f"{rdir}: molecule order differs from primary")
                        prim.append(float(p[src])); reb.append(float(r[src]))
                    d = [b - a for a, b in zip(prim, reb)]
                    md, sd = statistics.mean(d), statistics.stdev(d)
                    hw = T95_DF2 * sd / math.sqrt(3)
                    row[dst] = {"primary_mean": statistics.mean(prim), "primary_sd": statistics.stdev(prim),
                                "rebuilt_mean": statistics.mean(reb), "rebuilt_sd": statistics.stdev(reb),
                                "paired_diff_mean": md, "paired_diff_sd": sd,
                                "paired_diff_t95": [md - hw, md + hw],
                                "primary_by_seed": prim, "rebuilt_by_seed": reb}
                rows.append(row)
    out = {"rows": rows, "note": (
        "Rebuilt: 3.0 A envelope, message-passing edges and 2.5 A candidates built from the "
        "noisy coordinates. Heavy-only rebuilt inputs are RemoveHs of the noisy explicit-H "
        "molecules, so they share the explicit-H heavy-atom displacements; the primary "
        "heavy-only evaluation used independent draws.")}
    (BASE / "summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    lines = ["| Cohort | Model | σ | Pipeline F1 primary → rebuilt (Δ) | HH exact % primary → rebuilt (Δ) |",
             "|---|---|---|---|---|"]
    for r in rows:
        f, h = r["pipeline_f1"], r["hh_exact"]
        lines.append(f"| {r['cohort']} | {r['model']} | {r['sigma']:.2f} | "
                     f"{f['primary_mean']:.5f} → {f['rebuilt_mean']:.5f} ({f['paired_diff_mean']:+.5f}) | "
                     f"{100*h['primary_mean']:.2f} → {100*h['rebuilt_mean']:.2f} ({100*h['paired_diff_mean']:+.2f}) |")
    (BASE / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
