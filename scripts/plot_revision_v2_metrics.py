"""Make separate fixed-cohort figures for pipeline F1 and HH-graph exactness."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    ("explicit_two_stage", "BondNet two-stage", "#1f77b4", "o"),
    ("one_stage", "BondNet one-stage", "#ff7f0e", "s"),
    ("heavy_two_stage", "BondNet heavy-only", "#2ca02c", "^"),
    ("RDKit", "RDKit", "#d62728", "D"),
    ("OpenBabel", "OpenBabel", "#9467bd", "v"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path,
        default=Path("results/revision_v2_final_analysis/same_metric_summary.csv"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("paper_revision"))
    args = parser.parse_args()
    with (ROOT / args.summary).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_key = {(row["model"], float(row["sigma"])): row for row in rows}
    levels = (0.0, 0.1, 0.2)
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    specs = (
        ("pipeline_macro_f1", "Pipeline macro-F1", "revision_v2_pipeline_f1.png"),
        ("hh_graph_exact_match", "HH-graph exact match", "revision_v2_hh_graph_exact.png"),
    )
    for field, ylabel, filename in specs:
        fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
        for method, label, color, marker in METHODS:
            data = [by_key[(method, sigma)] for sigma in levels]
            values = [float(row[field + "_mean"]) for row in data]
            sds = [float(row[field + "_sd"]) if row[field + "_sd"] else 0.0 for row in data]
            ax.errorbar(levels, values, yerr=sds, label=label, color=color,
                        marker=marker, capsize=3, linewidth=1.8)
        ax.set(xlabel="Coordinate noise σ (Å)", ylabel=ylabel, xlim=(-0.012, 0.212),
               ylim=(-0.02, 1.04))
        ax.set_xticks(levels)
        ax.grid(alpha=0.22)
        ax.legend(loc="lower left", fontsize=8, frameon=True)
        fig.savefig(output / filename, dpi=220)
        plt.close(fig)
        print(output / filename)


if __name__ == "__main__":
    main()
