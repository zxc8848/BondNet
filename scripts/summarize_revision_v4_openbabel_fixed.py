#!/usr/bin/env python3
"""Summarize the Open Babel baselines re-run with the data directory set.

Reads results/revision_v4_openbabel_fixed/ (written by
scripts/run_revision_v4_openbabel_datadir.ps1) and the primary Open Babel
scores that were produced without bondtyp.txt, and writes
results/revision_v4_openbabel_fixed/summary.{json,md}:

  * GEOM fixed test and frozen PubChem3D cohort, sigma = 0/0.10/0.20:
    corrected vs primary pipeline F1, conditional F1, exact-type and HH exact
  * PubChem3D Deviation-001 subset (9,895 molecules), recomputed from the
    corrected prediction files with the same scorer port used for the primary
    subset analysis (full-cohort values are re-derived first and must equal the
    stored scores)
  * 478-molecule shared-SDF subset (clean and sigma = 0.10)
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXED = REPO / "results" / "revision_v4_openbabel_fixed"
SIGMAS = (0.0, 0.1, 0.2)
KEYS = ("success_rate", "f1_macro_pipeline", "f1_macro_pipeline_conditional_success",
        "true_bond_exact_type_rate", "full_graph_exact_match", "extra_bonds_per_successful_molecule")


def tag(s):
    return f"{round(100 * s):03d}"


def load_subset_module():
    spec = importlib.util.spec_from_file_location("ext_subset", REPO / "scripts" / "summarize_revision_v3_external_subset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    out = {"note": "Open Babel ConnectTheDots + PerceiveBondOrders with BABEL_DATADIR set (bondtyp.txt available). "
                   "Primary values were produced with BABEL_DATADIR unset.", "cohorts": {}}
    primary_geom = {float(r["sigma_angstrom"]): r for r in csv.DictReader(open(REPO / "results/revision_v2_rule_baselines/summary.csv"))
                    if r["method"] == "OpenBabel"}
    for cohort, prim_dir in (("geom", None), ("external", REPO / "results/revision_v3_external/rule_baselines/scores")):
        rows = []
        for s in SIGMAS:
            new = json.loads((FIXED / cohort / "scores" / f"openbabel_sigma_{tag(s)}.json").read_text(encoding="utf-8"))
            if cohort == "geom":
                p = primary_geom[s]
                old = {"success_rate": float(p["success_rate"]), "f1_macro_pipeline": float(p["f1_macro_pipeline"]),
                       "f1_macro_pipeline_conditional_success": float(p["f1_macro_pipeline_conditional_success"]),
                       "true_bond_exact_type_rate": float(p["true_bond_exact_type_rate"]),
                       "full_graph_exact_match": float(p["heavy_graph_exact_match"]),
                       "extra_bonds_per_successful_molecule": float(p["extra_bonds_per_successful_molecule"])}
            else:
                pth = prim_dir / f"openbabel_sigma_{tag(s)}.json"
                old = json.loads(pth.read_text(encoding="utf-8")) if pth.exists() else None
            rows.append({"sigma": s, "corrected": {k: new[k] for k in KEYS},
                         "primary_no_datadir": ({k: old[k] for k in KEYS if k in old} if old else None)})
        out["cohorts"][cohort] = rows

    # Deviation-001 subset of the external cohort from the corrected predictions.
    sub = load_subset_module()
    p0c_spec = importlib.util.spec_from_file_location("p0c", REPO / "scripts" / "p0c_yuelbond_compare.py")
    p0c = importlib.util.module_from_spec(p0c_spec)
    p0c_spec.loader.exec_module(p0c)
    data = REPO / "data" / "external_v3"
    cids = [int(x) for x in (data / "cids.txt").read_text().split()]
    excluded = {int(x) for x in (data / "deviation_001_historical50k_cids.txt").read_text().split()}
    subset_rows = []
    for s in SIGMAS:
        mols = p0c._read_valid_molecules(str(data / f"sigma_{tag(s)}.sdf"))
        if [int(m.GetProp("geom_mol_idx")) for m in mols] != cids:
            raise ValueError("SDF order differs from cids.txt")
        preds = json.loads((FIXED / "external" / "predictions" / f"openbabel_sigma_{tag(s)}_preds.json").read_text(encoding="utf-8"))
        preds.pop("__meta__", None)
        stored = json.loads((FIXED / "external" / "scores" / f"openbabel_sigma_{tag(s)}.json").read_text(encoding="utf-8"))
        full = sub.rule_metrics(p0c, mols, preds, set(range(len(mols))))
        sub.check_equal(f"openbabel sigma={s}", full, stored, [k for k in full if k != "n_molecules"])
        keep = {i for i, c in enumerate(cids) if c not in excluded}
        part = sub.rule_metrics(p0c, mols, preds, keep)
        subset_rows.append({"sigma": s, "full": full, "subset_excl_historical50k": part,
                            "max_abs_diff": {k: abs(part[k] - full[k]) for k in full if k != "n_molecules"}})
    out["external_deviation001_subset"] = subset_rows

    out["shared_478"] = {c: {k: v for k, v in json.loads((FIXED / "shared" / f"openbabel_{c}_fixedtest478_score.json")
                                                             .read_text(encoding="utf-8")).items() if k in KEYS + ("n_molecules",)}
                         for c in ("clean", "noisy")}
    (FIXED / "summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    lines = ["| Cohort | σ | F1 corrected | F1 primary | HH exact corrected (%) | HH exact primary (%) |", "|---|---|---|---|---|---|"]
    for cohort, rows in out["cohorts"].items():
        for r in rows:
            c, p = r["corrected"], r["primary_no_datadir"] or {}
            lines.append(f"| {cohort} | {r['sigma']:.2f} | {c['f1_macro_pipeline']:.5f} | {p.get('f1_macro_pipeline', float('nan')):.5f} | "
                         f"{100*c['full_graph_exact_match']:.2f} | {100*p.get('full_graph_exact_match', float('nan')):.2f} |")
    lines.append("")
    lines.append("External Deviation-001 subset (9,895): max |Δ| F1 = "
                 f"{max(r['max_abs_diff']['f1_macro_pipeline'] for r in subset_rows):.5f}, max |Δ| HH exact = "
                 f"{100*max(r['max_abs_diff']['full_graph_exact_match'] for r in subset_rows):.3f} pp")
    for c, v in out["shared_478"].items():
        lines.append(f"Shared 478 {c}: F1 {v['f1_macro_pipeline']:.4f}, HH exact {100*v['full_graph_exact_match']:.2f}%")
    (FIXED / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
