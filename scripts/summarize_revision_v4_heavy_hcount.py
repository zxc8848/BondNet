#!/usr/bin/env python3
"""Explicit-H vs heavy-only vs heavy-only + oracle H counts (three seeds each).

Reads the undirected-pair aggregates of the three input representations on the
GEOM fixed test and the external cohort and writes
results/revision_v4_heavy_hcount/summary.{json,md}.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision_v4_heavy_hcount"
SOURCES = {
    "geom": {"explicit_h": ROOT / "results/revision_v3_undirected_audit/joint",
             "heavy_only": ROOT / "results/revision_v3_undirected_audit/heavy",
             "heavy_hcount_oracle": OUT / "geom"},
    "external": {"explicit_h": ROOT / "results/revision_v3_external/joint",
                 "heavy_only": ROOT / "results/revision_v3_external/heavy",
                 "heavy_hcount_oracle": OUT / "external"},
}
SEEDS = (42, 43, 44)
FIELDS = ("undirected_pipeline_macro_f1", "undirected_hh_graph_exact_rate")


def main():
    rows = []
    for cohort, reps in SOURCES.items():
        data = {}
        for rep, base in reps.items():
            data[rep] = {s: {float(r["sigma"]): r for r in json.loads(
                (base / f"seed{s}" / "aggregate.json").read_text(encoding="utf-8"))} for s in SEEDS}
        for sigma in (0.0, 0.1, 0.2):
            row = {"cohort": cohort, "sigma": sigma}
            for rep in reps:
                for f in FIELDS:
                    v = [data[rep][s][sigma][f] for s in SEEDS]
                    row[f"{rep}|{f}"] = {"mean": statistics.mean(v), "sd": statistics.stdev(v), "by_seed": v}
            rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps({
        "rows": rows,
        "note": "heavy_hcount_oracle receives per-heavy-atom H counts from the reference graph "
                "(oracle diagnostic, not a deployable input). Same recipe as heavy_only. "
                "Noise: independent molecule-keyed draws per representation, as in the primary tables."},
        indent=2) + "\n", encoding="utf-8")
    lines = ["| Cohort | σ | Explicit H | Heavy only | Heavy + oracle H count |", "|---|---|---|---|---|"]
    for r in rows:
        cells = []
        for rep in ("explicit_h", "heavy_only", "heavy_hcount_oracle"):
            f = r[f"{rep}|undirected_pipeline_macro_f1"]; h = r[f"{rep}|undirected_hh_graph_exact_rate"]
            cells.append(f"F1 {f['mean']:.5f}; HH {100*h['mean']:.2f} ± {100*h['sd']:.2f}%")
        lines.append(f"| {r['cohort']} | {r['sigma']:.2f} | " + " | ".join(cells) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
