#!/usr/bin/env python3
r"""
P0(a) -- Unified resonance-aware re-run (fixes the Table 7 / Table 8 contradiction, reviewer #7).

Problem it solves
-----------------
Table 8 (resonance) currently reports STRICT numbers (QM9 99.4%, PubChem3D 93.0%) that were
computed on an *earlier* held-out sample, while Table 7 (diversity) reports the Diverse-80k model
on the *unified* leakage-free held-out sets (QM9 98.7%, PubChem3D 93.4%). Both tables claim the
same model + "leakage-free held-out", so the mismatched strict numbers look contradictory.

This script recomputes STRICT and RESONANCE-AWARE scores for the Diverse-80k model on the SAME
clean held-out SDFs used by Table 7 (data/heldout_qm9_clean.sdf, data/heldout_pubchem_clean.sdf).
After running, replace the Table 8 rows so that the "Strict" column exactly matches Table 7 and the
"Resonance-aware" column is the within-cohort improvement. Then delete the "earlier diagnostic
cohort" caveat from the caption/text.

Usage
-----
    python scripts/p0a_resonance_unified.py
    # or override paths:
    python scripts/p0a_resonance_unified.py --stage1 <s1.pt> --stage2 <s2.pt>

It runs `evaluate.py` twice per held-out set (once strict, once --resonance_aware, sigma=0 only),
parses the resulting JSON, prints a comparison table, sanity-checks the strict exact-type against
Table 7, and emits ready-to-paste LaTeX rows.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- CONFIG: real paths in this repo (override on the CLI if needed) --------------------------
DEFAULTS = dict(
    stage1=os.path.join(REPO, "checkpoints", "diverse80k_stage1", "best.pt"),
    stage2=os.path.join(REPO, "checkpoints", "diverse80k_stage2", "best_e2e.pt"),
    heldouts={
        "QM9":       os.path.join(REPO, "data", "heldout_qm9_clean.sdf"),
        "PubChem3D": os.path.join(REPO, "data", "heldout_pubchem_clean.sdf"),
    },
)
# Table 7 (diversity) strict exact-type rates -- used only as a sanity check.
TABLE7_EXACT = {"QM9": 98.7, "PubChem3D": 93.4}
EXACT_KEYS = ("mol_validity", "mol_exact_rate", "exact_type_rate", "exact_match_rate",
              "true_bond_exact_type_rate", "mol_exact", "exact")


def newest_json(d):
    js = sorted(glob.glob(os.path.join(d, "*.json")), key=os.path.getmtime)
    js = [j for j in js if "error_analysis" not in os.path.basename(j)]
    if not js:
        raise FileNotFoundError(f"No results JSON written to {d}")
    return js[-1]


def load_sigma0(path):
    """Return the sigma=0 result dict from an evaluate.py results JSON."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "BondNet" in data:
        results = data["BondNet"]
    elif isinstance(data, dict) and "results" in data:
        results = data["results"]
    else:
        results = data
    if isinstance(results, dict):
        results = [results]
    if not isinstance(results, list) or not results or not all(isinstance(r, dict) for r in results):
        raise ValueError(f"Unsupported results JSON schema in {path}")
    best = min(results, key=lambda r: abs(float(r.get("sigma", 0.0))))
    return best


def get_exact(r):
    for k in EXACT_KEYS:
        if k in r:
            v = float(r[k])
            return v * 100.0 if v <= 1.0 else v
    return float("nan")


def run_eval(stage1, stage2, sdf, out_dir, resonance):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, os.path.join(REPO, "evaluate.py"),
        "--checkpoint", stage1,
        "--stage2_ckpt", stage2,
        "--allow_stage2_cache_mismatch",
        "--dataset", "sdf", "--data_path", sdf,
        "--eval_split", "all",
        "--noise_levels", "0.0",
        "--output_dir", out_dir,
    ]
    if resonance:
        cmd.append("--resonance_aware")
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO)
    result_path = os.path.join(out_dir, "results.json")
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"evaluate.py completed but did not write {result_path}")
    return load_sigma0(result_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=DEFAULTS["stage1"])
    ap.add_argument("--stage2", default=DEFAULTS["stage2"])
    ap.add_argument("--out_root", default=os.path.join(REPO, "results", "p0a_resonance_unified"))
    args = ap.parse_args()

    for p in (args.stage1, args.stage2, *DEFAULTS["heldouts"].values()):
        if not os.path.exists(p):
            sys.exit(f"[missing] {p}\nEdit the CONFIG/flags to point at real files.")

    rows = {}
    for name, sdf in DEFAULTS["heldouts"].items():
        strict = run_eval(args.stage1, args.stage2, sdf,
                          os.path.join(args.out_root, name, "strict"), resonance=False)
        reson = run_eval(args.stage1, args.stage2, sdf,
                         os.path.join(args.out_root, name, "resonance"), resonance=True)
        rows[name] = dict(strict=strict, reson=reson)

    print("\n" + "=" * 78)
    print(f"{'Held-out':<12}{'Scoring':<16}{'F1-double':>11}{'F1-macro':>11}{'Exact-type':>12}")
    print("-" * 78)
    for name, r in rows.items():
        for label, res in (("Strict", r["strict"]), ("Resonance-aware", r["reson"])):
            print(f"{name:<12}{label:<16}"
                  f"{res.get('f1_double', float('nan')):>11.4f}"
                  f"{res.get('f1_macro', float('nan')):>11.4f}"
                  f"{get_exact(res):>11.1f}%")
        # sanity check vs Table 7
        se = get_exact(r["strict"])
        want = TABLE7_EXACT[name]
        flag = "OK" if abs(se - want) <= 0.11 else "MISMATCH -- investigate cohort/checkpoint"
        print(f"    [check] strict exact-type {se:.1f}% vs Table 7 {want:.1f}%  -> {flag}")
    print("=" * 78)

    print("\n% ---- paste into Table 8 (tab:resonance); strict must equal Table 7 ----")
    for name, r in rows.items():
        s, rz = r["strict"], r["reson"]
        print(f"\\multirow{{2}}{{*}}{{{name}}} & Strict          & "
              f"{s.get('f1_double',0):.4f} & {s.get('f1_macro',0):.4f} & {get_exact(s):.1f}\\% \\\\")
        print(f"                           & Resonance-aware & "
              f"{rz.get('f1_double',0):.4f} & {rz.get('f1_macro',0):.4f} & {get_exact(rz):.1f}\\% \\\\")
    print("\nDone. If strict now equals Table 7, remove the 'earlier diagnostic cohort' caveat.")


if __name__ == "__main__":
    main()
