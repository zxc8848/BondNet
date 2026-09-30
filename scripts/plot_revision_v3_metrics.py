#!/usr/bin/env python3
"""Figures 1-2 of the revision: pipeline macro-F1 and HH-graph exact match
versus coordinate noise, on the GEOM fixed test and the frozen PubChem3D cohort.

Every plotted number is read from a result file:
  GEOM learned    results/revision_v3_undirected_audit/three_seed_summary.json
  GEOM rules      results/revision_v2_rule_baselines/summary.csv
  External        results/revision_v3_external/summary.json
  RDKit Hückel    results/revision_v4_rdkit_configs_f1/*.json (post-hoc configuration)
  OpenBabel       results/revision_v4_openbabel_fixed/summary.json (re-run with BABEL_DATADIR set;
                  replaces the primary Open Babel values, which lacked bondtyp.txt)
Outputs paper_revision/revision_v3_{pipeline_f1,hh_graph_exact}.{pdf,png} and a
CSV of the plotted values.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_revision"
SIGMAS = (0.0, 0.1, 0.2)
# Fixed categorical order (never cycled): identity follows the method.
METHODS = [
    ("joint", "Joint, explicit H", "#2a78d6", "o"),
    ("heavy", "Joint, heavy only", "#eb6834", "s"),
    ("staged", "Hard two-stage", "#1baf7a", "^"),
    ("rdkit", "RDKit", "#eda100", "D"),
    ("openbabel", "OpenBabel", "#e87ba4", "X"),
    ("rdkit_hueckel", "RDKit, H\u00fcckel (post-hoc)", "#eda100", "D"),
]
HOLLOW = {"rdkit_hueckel"}
EXT_LABEL = {"joint": "Joint explicit-H", "heavy": "Joint heavy-only", "staged": "Hard two-stage",
             "rdkit": "RDKit", "openbabel": "OpenBabel"}


def load():
    vals = {}  # (cohort, metric, method, sigma) -> (mean, sd or None)
    geom = json.loads((ROOT / "results/revision_v3_undirected_audit/three_seed_summary.json").read_text())
    for r in geom["summary"]:
        for metric, key in (("f1", "undirected_pipeline_macro_f1"), ("hh", "undirected_hh_graph_exact_rate")):
            vals["geom", metric, r["model"], float(r["sigma"])] = (r[key + "_mean"], r[key + "_sample_sd"])
    with (ROOT / "results/revision_v2_rule_baselines/summary.csv").open() as fh:
        for r in csv.DictReader(fh):
            m = "rdkit" if r["method"] == "RDKit" else "openbabel"
            s = float(r["sigma_angstrom"])
            vals["geom", "f1", m, s] = (float(r["f1_macro_pipeline"]), None)
            vals["geom", "hh", m, s] = (float(r["heavy_graph_exact_match"]), None)
    ext = json.loads((ROOT / "results/revision_v3_external/summary.json").read_text(encoding="utf-8"))
    for r in ext["rows"]:
        m = next(k for k, v in EXT_LABEL.items() if v == r["method"])
        for metric, key in (("f1", "pipeline_f1"), ("hh", "hh_exact")):
            vals["ext", metric, m, float(r["sigma"])] = (r[key], r.get(key + "_sd"))
    # Post-hoc RDKit useHueckel=True configuration (not a frozen endpoint).
    import glob
    for cohort, prefix in (("geom", "geom_test"), ("ext", "external")):
        for path in sorted(glob.glob(str(ROOT / "results/revision_v4_rdkit_configs_f1" / f"{prefix}_sigma_*.json"))):
            d = json.loads(Path(path).read_text(encoding="utf-8"))
            s = round(int(Path(path).stem.split("_sigma_")[1][:3]) / 100.0, 2)
            for row in d["rows"]:
                if row["configuration"] == "use_hueckel":
                    vals[cohort, "f1", "rdkit_hueckel", s] = (row["pipeline_macro_f1"], None)
                    vals[cohort, "hh", "rdkit_hueckel", s] = (row["hh_exact_rate"], None)
    # Open Babel re-run with its data directory (Deviation 002); overrides the primary values.
    fixed = json.loads((ROOT / "results/revision_v4_openbabel_fixed/summary.json").read_text(encoding="utf-8"))
    for cohort, key in (("geom", "geom"), ("ext", "external")):
        for r in fixed["cohorts"][key]:
            c = r["corrected"]
            vals[cohort, "f1", "openbabel", float(r["sigma"])] = (c["f1_macro_pipeline"], None)
            vals[cohort, "hh", "openbabel", float(r["sigma"])] = (c["full_graph_exact_match"], None)
    return vals


def plot(vals, metric, ylabel, scale, ylim, stem, panel_titles):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6,
                         "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b"})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    offsets = [-0.0125, -0.0075, -0.0025, 0.0025, 0.0075, 0.0125]
    for ax, cohort, title in zip(axes, ("geom", "ext"), panel_titles):
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
        ax.set_axisbelow(True)
        for (key, label, color, marker), dx in zip(METHODS, offsets):
            xs = [s + dx for s in SIGMAS]
            ys = [vals[cohort, metric, key, s][0] * scale for s in SIGMAS]
            sds = [vals[cohort, metric, key, s][1] for s in SIGMAS]
            # Dashed connectors only guide the eye between the three tested levels.
            ax.plot(xs, ys, color=color, linewidth=1.0,
                    linestyle=(0, (1, 1.5)) if key in HOLLOW else (0, (3, 2)), zorder=2)
            if all(sd is not None for sd in sds):
                ax.errorbar(xs, ys, yerr=[sd * scale for sd in sds], fmt="none", ecolor=color,
                            elinewidth=1.0, capsize=2.5, zorder=3)
            if key in HOLLOW:
                ax.scatter(xs, ys, s=30, marker=marker, facecolors="#fcfcfb", edgecolors=color,
                           linewidths=1.2, zorder=4, label=label)
            else:
                ax.scatter(xs, ys, s=30, marker=marker, color=color, edgecolors="#fcfcfb",
                           linewidths=0.8, zorder=4, label=label)
        ax.set_xticks(SIGMAS, ["0", "0.10", "0.20"])
        ax.set_xlim(-0.03, 0.23)
        ax.set_xlabel("Gaussian noise $\\sigma$ (Å)")
        ax.set_title(title, fontsize=8.5, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(ylabel)
    axes[0].set_ylim(*ylim)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300)
    plt.close(fig)


def main():
    vals = load()
    titles = ("(a) GEOM-DRUGS fixed test (n = 27,240)", "(b) PubChem3D external cohort (n = 10,000)")
    plot(vals, "f1", "Pipeline macro-F1", 1.0, (0.0, 1.03), "revision_v3_pipeline_f1", titles)
    plot(vals, "hh", "HH-graph exact match (%)", 100.0, (0, 103), "revision_v3_hh_graph_exact", titles)
    with (OUT / "revision_v3_figure_values.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cohort", "metric", "method", "sigma", "mean", "seed_sd"])
        for (cohort, metric, method, sigma), (m, sd) in sorted(vals.items()):
            w.writerow([cohort, metric, method, sigma, m, "" if sd is None else sd])
    print("wrote", OUT / "revision_v3_pipeline_f1.pdf", OUT / "revision_v3_hh_graph_exact.pdf")


if __name__ == "__main__":
    main()
