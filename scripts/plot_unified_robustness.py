#!/usr/bin/env python3
"""Plot the unified robustness summary used by paper/main.tex."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "results" / "robustness_26940_unified" / "summary.csv"
OUTPUT = REPO / "results" / "robustness_curve_v2.png"


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    methods = ["BondNet-aug", "BondNet-clean", "RDKit", "OpenBabel"]
    styles = {
        "BondNet-aug": dict(color="#0072B2", marker="o", linewidth=2.4),
        "BondNet-clean": dict(color="#56B4E9", marker="s", linewidth=2.0),
        "RDKit": dict(color="#D55E00", marker="^", linewidth=2.0),
        "OpenBabel": dict(color="#009E73", marker="D", linewidth=2.0),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharex=True)
    fields = [
        ("f1_macro_pipeline", "Pipeline macro-F1"),
        ("true_bond_exact_type_rate", "True-bond exact-type rate"),
    ]
    for ax, (field, ylabel) in zip(axes, fields):
        for method in methods:
            selected = sorted(
                (r for r in rows if r["method"] == method),
                key=lambda r: float(r["sigma_angstrom"]),
            )
            ax.plot(
                [float(r["sigma_angstrom"]) for r in selected],
                [float(r[field]) for r in selected],
                label=method,
                **styles[method],
            )
        ax.set_xlabel(r"Coordinate noise $\sigma$ ($\AA$)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([0.00, 0.05, 0.10, 0.15, 0.20])
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25, linewidth=0.7)
    axes[0].text(0.01, 0.04, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(0.01, 0.04, "(b)", transform=axes[1].transAxes, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
