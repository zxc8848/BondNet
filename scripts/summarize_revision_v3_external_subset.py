#!/usr/bin/env python3
"""Sensitivity analysis for Deviation 001 of the frozen v3 external cohort.

Recomputes every external metric on the molecules that were NOT part of the
historical 50,000-record PubChem3D transfer cohort (source records 0-49,999),
using only outputs that already exist:

  learned models   per-molecule sufficient statistics (sigma_*.npz)
  RDKit/OpenBabel  normalized prediction JSONs, re-scored with the p0c scorer logic
  joint validity   per_molecule.csv

Nothing is re-run.  For each source the script first recomputes the full
10,000-molecule metric and requires it to equal the stored result, so the
subset numbers come from a verified reimplementation.

Output: results/revision_v3_external/sensitivity_excl_historical50k/
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "external_v3"
RESULTS = REPO / "results" / "revision_v3_external"
OUT = RESULTS / "sensitivity_excl_historical50k"
EXCLUDE_FILE = DATA / "deviation_001_historical50k_cids.txt"
MODELS = {"joint": "Joint explicit-H", "heavy": "Joint heavy-only", "staged": "Hard two-stage"}
RULES = {"rdkit": "RDKit", "openbabel": "OpenBabel"}
SEEDS = (42, 43, 44)
SIGMAS = (0.0, 0.1, 0.2)
TOL = 1e-9


def tag(sigma: float) -> str:
    return f"{round(100 * sigma):03d}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ learned --
def f1_macro(tp, fp, fn):
    tp, fp, fn = tp.sum(axis=0), fp.sum(axis=0), fn.sum(axis=0)
    denom = 2 * tp + fp + fn
    per_class = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    return float(per_class.mean())


def learned_metrics(stats, mask):
    s = {k: np.asarray(v)[mask] for k, v in stats.items()}
    return {
        "n_molecules": int(mask.sum()),
        "undirected_pipeline_macro_f1": f1_macro(s["u_pipe_tp"], s["u_pipe_fp"], s["u_pipe_fn"]),
        "pipeline_macro_f1": f1_macro(s["pipe_tp"], s["pipe_fp"], s["pipe_fn"]),
        "reference_pair_macro_f1": f1_macro(s["ref_tp"], s["ref_fp"], s["ref_fn"]),
        "undirected_true_pair_exact_rate": float(s["u_true_pair_exact"].mean()),
        "true_pair_exact_rate": float(s["true_pair_exact"].mean()),
        "undirected_hh_graph_exact_rate": float(s["u_hh_graph_exact"].mean()),
        "hh_graph_exact_rate": float(s["hh_graph_exact"].mean()),
        "conn_exact_rate": float(s["conn_exact"].mean()),
    }


# -------------------------------------------------------------------- rules --
def rule_metrics(p0c, mols, preds, keep):
    """Line-for-line port of p0c.cmd_score restricted to indices in ``keep``."""
    tp = Counter(); fp_matched = Counter(); fp_pipeline = Counter(); fn = Counter()
    fn_success = Counter()
    n_mol = n_success = n_exact = n_full = extra_total = 0
    for idx, mol in enumerate(mols):
        if idx not in keep:
            continue
        n_mol += 1
        true = p0c._true_bonds(mol)
        key = str(idx)
        if key not in preds:
            for lab in true.values():
                fn[lab] += 1
            continue
        n_success += 1
        pred = {}
        for i, j, o in preds[key]:
            pred[(min(int(i), int(j)), max(int(i), int(j)))] = p0c._pred_label(o)
        ok = True
        for e, tl in true.items():
            pl = pred.get(e)
            if pl is None:
                fn[tl] += 1; fn_success[tl] += 1; ok = False
            elif pl == tl:
                tp[tl] += 1
            else:
                fp_matched[pl] += 1; fp_pipeline[pl] += 1
                fn[tl] += 1; fn_success[tl] += 1; ok = False
        extra = [e for e in pred if e not in true]
        for e in extra:
            fp_pipeline[pred[e]] += 1
        extra_total += len(extra)
        if ok:
            n_exact += 1
            if not extra:
                n_full += 1

    def macro(fp, fneg):
        vals = []
        for c in range(4):
            if tp[c] + fp[c] + fneg[c] == 0:
                continue
            p = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
            r = tp[c] / (tp[c] + fneg[c]) if tp[c] + fneg[c] else 0.0
            vals.append(2 * p * r / (p + r) if p + r else 0.0)
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "n_molecules": n_mol,
        "success_rate": n_success / max(n_mol, 1),
        "f1_macro_true_pair": macro(fp_matched, fn),
        "f1_macro_pipeline": macro(fp_pipeline, fn),
        "f1_macro_pipeline_conditional_success": macro(fp_pipeline, fn_success),
        "true_bond_exact_type_rate": n_exact / max(n_mol, 1),
        "full_graph_exact_match": n_full / max(n_mol, 1),
    }


# ------------------------------------------------------------------ helpers --
def check_equal(label, recomputed, stored, keys):
    for k in keys:
        if abs(float(recomputed[k]) - float(stored[k])) > TOL:
            raise ValueError(f"{label}: recomputed {k}={recomputed[k]!r} != stored {stored[k]!r}")


def seed_summary(values):
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values),
            "by_seed": dict(zip(map(str, SEEDS), values))}


def main() -> int:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    cids = [int(x) for x in (DATA / "cids.txt").read_text().split()]
    if hashlib.sha256(",".join(map(str, cids)).encode()).hexdigest() != \
            manifest["selection"]["cohort"]["ordered_id_sha256"]:
        raise ValueError("cids.txt does not match the frozen manifest")
    deviation = (DATA / "DEVIATION_001.md").read_text(encoding="utf-8")
    excl_sha = sha256(EXCLUDE_FILE)
    if excl_sha not in deviation:
        raise ValueError("exclusion list differs from the one recorded in DEVIATION_001.md")
    excluded = {int(x) for x in EXCLUDE_FILE.read_text().split()}
    if not excluded <= set(cids):
        raise ValueError("exclusion list contains CIDs outside the cohort")
    keep_mask = np.array([c not in excluded for c in cids])
    full_mask = np.ones(len(cids), dtype=bool)
    print(f"[subset] {int(keep_mask.sum())} of {len(cids)} molecules kept "
          f"({len(excluded)} historical-50k molecules removed)")

    learned = {}
    for model in MODELS:
        for seed in SEEDS:
            out = RESULTS / model / f"seed{seed}"
            stored = {float(r["sigma"]): r for r in
                      json.loads((out / "aggregate.json").read_text(encoding="utf-8"))}
            for sigma in SIGMAS:
                stats = dict(np.load(out / f"sigma_{tag(sigma)}.npz"))
                if [int(x) for x in stats["mol_idx"]] != cids:
                    raise ValueError(f"{out} sigma={sigma}: molecule order differs from cids.txt")
                full = learned_metrics(stats, full_mask)
                check_equal(f"{model} seed{seed} sigma={sigma}", full, stored[sigma],
                            [k for k in full if k != "n_molecules"])
                learned[model, seed, sigma] = {"full": full, "subset": learned_metrics(stats, keep_mask)}

    spec = importlib.util.spec_from_file_location("p0c", REPO / "scripts" / "p0c_yuelbond_compare.py")
    p0c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p0c)
    rules = {}
    for sigma in SIGMAS:
        mols = p0c._read_valid_molecules(str(DATA / f"sigma_{tag(sigma)}.sdf"))
        if [int(m.GetProp("geom_mol_idx")) for m in mols] != cids:
            raise ValueError(f"sigma={sigma}: SDF order differs from cids.txt")
        keep_all = set(range(len(mols)))
        keep_sub = {i for i, c in enumerate(cids) if c not in excluded}
        for method in RULES:
            pred_path = RESULTS / "rule_baselines" / "predictions" / f"{method}_sigma_{tag(sigma)}_preds.json"
            preds = json.loads(pred_path.read_text(encoding="utf-8"))
            preds.pop("__meta__", None)
            stored = json.loads((RESULTS / "rule_baselines" / "scores" /
                                 f"{method}_sigma_{tag(sigma)}.json").read_text(encoding="utf-8"))
            full = rule_metrics(p0c, mols, preds, keep_all)
            check_equal(f"{method} sigma={sigma}", full, stored, [k for k in full if k != "n_molecules"])
            rules[method, sigma] = {"full": full, "subset": rule_metrics(p0c, mols, preds, keep_sub)}

    validity = {}
    for seed in SEEDS:
        path = RESULTS / "joint_validity" / f"seed{seed}" / "per_molecule.csv"
        if not path.exists():
            continue
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        for sigma in SIGMAS:
            sel = [r for r in rows if abs(float(r["sigma"]) - sigma) < 1e-9]
            if [int(r["mol_idx"]) for r in sel] != cids:
                raise ValueError(f"{path}: molecule order differs at sigma={sigma}")
            for which, rs in (("full", sel), ("subset", [r for r in sel if int(r["mol_idx"]) not in excluded])):
                validity[seed, sigma, which] = {
                    "n_molecules": len(rs),
                    "sanitize_ok_rate": sum(int(r["reference_charge_sanitize_ok"]) for r in rs) / len(rs),
                    "zero_charge_sanitize_ok_rate": sum(int(r["zero_charge_sanitize_ok"]) for r in rs) / len(rs),
                }
        stored = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
        for row in stored:
            v = validity[seed, float(row["sigma"]), "full"]
            if abs(v["sanitize_ok_rate"] - row["reference_charge_sanitize_ok_rate"]) > TOL:
                raise ValueError(f"validity seed{seed} sigma={row['sigma']}: recomputation mismatch")

    rows = []
    for sigma in SIGMAS:
        for model, label in MODELS.items():
            row = {"sigma": sigma, "method": label}
            for which in ("full", "subset"):
                for key, short in (("undirected_pipeline_macro_f1", "pipeline_f1"),
                                   ("undirected_true_pair_exact_rate", "exact_type"),
                                   ("undirected_hh_graph_exact_rate", "hh_exact"),
                                   ("pipeline_macro_f1", "pipeline_f1_directed"),
                                   ("hh_graph_exact_rate", "hh_exact_directed")):
                    row[f"{which}_{short}"] = seed_summary(
                        [learned[model, s, sigma][which][key] for s in SEEDS])
                row[f"{which}_n"] = learned[model, SEEDS[0], sigma][which]["n_molecules"]
            if model == "joint" and all((s, sigma, "full") in validity for s in SEEDS):
                for which in ("full", "subset"):
                    row[f"{which}_sanitizable"] = seed_summary(
                        [validity[s, sigma, which]["sanitize_ok_rate"] for s in SEEDS])
            rows.append(row)
        for method, label in RULES.items():
            row = {"sigma": sigma, "method": label}
            for which in ("full", "subset"):
                r = rules[method, sigma][which]
                row[f"{which}_pipeline_f1"] = r["f1_macro_pipeline"]
                row[f"{which}_exact_type"] = r["true_bond_exact_type_rate"]
                row[f"{which}_hh_exact"] = r["full_graph_exact_match"]
                row[f"{which}_success_rate"] = r["success_rate"]
                row[f"{which}_n"] = r["n_molecules"]
            rows.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps({
        "deviation": "DEVIATION_001.md", "excluded_cids_sha256": excl_sha,
        "n_full": len(cids), "n_subset": int(keep_mask.sum()), "rows": rows,
        "verification": "full-cohort values recomputed from per-molecule outputs and prediction "
                        "files match the stored results to 1e-9 before subsetting",
    }, indent=2) + "\n", encoding="utf-8")

    def cell(row, which, key, pct):
        v = row.get(f"{which}_{key}")
        if v is None:
            return "--"
        if isinstance(v, dict):
            m, sd = v["mean"], v["sd"]
            return f"{100*m:.2f} ± {100*sd:.2f}" if pct else f"{m:.5f} ± {sd:.5f}"
        return f"{100*v:.2f}" if pct else f"{v:.5f}"

    lines = [f"Sensitivity analysis (Deviation 001): {int(keep_mask.sum()):,} of {len(cids):,} external "
             f"molecules, excluding the {len(excluded)} in the historical 50k PubChem3D cohort", "",
             "| σ (Å) | Method | Pipeline F1 (all) | Pipeline F1 (subset) | HH exact % (all) | HH exact % (subset) |",
             "|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['sigma']:.2f} | {row['method']} | {cell(row,'full','pipeline_f1',False)} | "
                     f"{cell(row,'subset','pipeline_f1',False)} | {cell(row,'full','hh_exact',True)} | "
                     f"{cell(row,'subset','hh_exact',True)} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
