#!/usr/bin/env python3
r"""
P0(b) -- Truly independent GEOM test set + one-shot evaluation (reviewer #1, and #3 as a bonus).

Problem it solves
-----------------
The GEOM random1/re10 *validation* split is used for validation loss, periodic end-to-end
validation, model selection, AND the headline results (macro-F1 0.9989 / exact-type 99.1%).
That is not an untouched test set, so the headline numbers may be optimistic.

Because Stage-2 training consumed every non-validation molecule, there is no unused partition
inside the training source. This script therefore builds a *fresh* leakage-free GEOM test set from
a different GEOM dump (data/geom_drugs_200k.sdf), excluding by canonical (heavy-atom, non-isomeric)
SMILES every molecule that appears in the training source or in any held-out/diverse set. Whatever
remains is provably disjoint from the model's training data. The main two-stage model is then
evaluated exactly ONCE on it, with rule-based baselines on the identical molecules (which also gives
the same-cohort, same-molecule baseline comparison the reviewer asked for in #3).

Report the resulting numbers as the headline; keep the validation split for model selection only.
Expect a small drop from 99.1% -- that is the honest number.

Usage
-----
    python scripts/p0b_independent_test.py                 # build + evaluate
    python scripts/p0b_independent_test.py --skip_build    # reuse an existing test SDF
"""
import argparse
import glob
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = lambda *p: os.path.join(REPO, *p)

# ---- CONFIG: real paths in this repo --------------------------------------------------------
# Main two-stage model (the one that produced 0.9989 / 99.1% on the GEOM val split):
STAGE1 = D("checkpoints", "geom_drugs_all_random1_re10_stage1_c25", "best.pt")
STAGE2 = D("checkpoints", "geom_drugs_all_random1_re10_stage2_c25_officialcache", "best_e2e.pt")
# Source to draw NEW test molecules from (a different GEOM dump than the training source):
TEST_SOURCE = D("data", "geom_drugs_200k.sdf")
# Everything the model may have seen -- excluded by canonical SMILES so the test set is disjoint:
EXCLUDE = [
    D("data", "geom_drugs_all_random1_rel10.sdf"),   # main GEOM training source
    D("data", "diverse_subset_80k.sdf"),             # (safety) diversity training set
    D("data", "heldout_qm9_clean.sdf"),              # (safety) other held-outs
    D("data", "heldout_pubchem_clean.sdf"),
]
TEST_SDF = D("data", "geom_TEST_leakagefree.sdf")
TAKE = 20000
NOISE_LEVELS = ["0.0", "0.05", "0.1", "0.15", "0.2"]
REPORTED_VAL = dict(conn=0.9998, macro=0.9989, exact=99.1)   # for side-by-side printing


def count_sdf_records(path):
    count = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip() == b"$$$$":
                count += 1
    return count


def build_test_set(source, test_sdf, take, allow_short=False):
    cmd = [sys.executable, D("scripts", "extract_heldout.py"),
           "--source", source, "--take", str(take), "--out", test_sdf]
    for e in EXCLUDE:
        if os.path.exists(e):
            cmd += ["--exclude", e]
        else:
            print(f"[warn] exclude file missing, skipping: {e}")
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO)
    written = count_sdf_records(test_sdf)
    if written != take and not allow_short:
        raise RuntimeError(
            f"Requested {take} leakage-free molecules but only {written} remained. "
            "This source overlaps the exclusion sets too heavily; choose a genuinely new source. "
            "Use --allow_short only for debugging, never for paper results."
        )
    return written


def evaluate(stage1, stage2, test_sdf):
    out_dir = D("results", "geom_TEST")
    os.makedirs(out_dir, exist_ok=True)
    # BondNet only. Do NOT use evaluate.py --run_baselines: its inline RDKit/OpenBabel baseline
    # uses charge=0 + positional (index-order) bond matching and is broken (0% validity even on
    # clean molecules). We run the CORRECTED baseline (scripts/evaluate_rule_baselines.py, with
    # per-molecule formal charge + atom-pair matching) on the SAME test SDF below.
    cmd = [sys.executable, D("evaluate.py"),
           "--checkpoint", stage1,
           "--stage2_ckpt", stage2,
           "--allow_stage2_cache_mismatch",
           "--dataset", "sdf", "--data_path", test_sdf,
           "--eval_split", "all",
           "--noise_levels", *NOISE_LEVELS,
           "--output_dir", out_dir]
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO)
    # Corrected rule-based baselines on the identical test molecules (#3), same noise levels.
    base_cmd = [sys.executable, D("scripts", "evaluate_rule_baselines.py"),
                "--data_path", test_sdf,
                "--methods", "rdkit", "openbabel",
                "--noise_levels", *NOISE_LEVELS,
                "--output_dir", D("results", "geom_TEST_baselines")]
    print("\n$ " + " ".join(base_cmd))
    try:
        subprocess.run(base_cmd, check=True, cwd=REPO)
    except subprocess.CalledProcessError as e:
        print(f"[warn] corrected baseline run failed ({e}); BondNet results are still valid.")
    js = sorted(glob.glob(os.path.join(out_dir, "*.json")), key=os.path.getmtime)
    js = [j for j in js if "error_analysis" not in os.path.basename(j)]
    with open(js[-1]) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_build", action="store_true",
                    help="Reuse an existing test SDF instead of rebuilding it.")
    ap.add_argument("--source", default=TEST_SOURCE)
    ap.add_argument("--test_sdf", default=TEST_SDF)
    ap.add_argument("--take", type=int, default=TAKE)
    ap.add_argument("--stage1", default=STAGE1)
    ap.add_argument("--stage2", default=STAGE2)
    ap.add_argument("--allow_short", action="store_true",
                    help="Allow fewer than --take molecules (debugging only).")
    args = ap.parse_args()

    source = os.path.abspath(args.source)
    test_sdf = os.path.abspath(args.test_sdf)
    stage1 = os.path.abspath(args.stage1)
    stage2 = os.path.abspath(args.stage2)
    for p in (stage1, stage2, source):
        if not os.path.exists(p):
            sys.exit(f"[missing] {p}\nEdit CONFIG to point at the real file.")

    if not args.skip_build or not os.path.exists(test_sdf):
        build_test_set(source, test_sdf, args.take, args.allow_short)
    if not os.path.exists(test_sdf) or os.path.getsize(test_sdf) == 0:
        sys.exit("[error] test SDF was not built (0 molecules left after exclusion?).")
    n_test = count_sdf_records(test_sdf)
    if n_test != args.take and not args.allow_short:
        sys.exit(f"[error] {test_sdf} contains {n_test}, expected {args.take}. Rebuild from a new source.")

    results = evaluate(stage1, stage2, test_sdf)
    # evaluate.py writes {method: [row_per_sigma, ...]}. Pull BondNet's rows.
    if isinstance(results, dict):
        rows = results.get("BondNet") or next(iter(results.values()))
    else:
        rows = results if isinstance(results, list) else [results]
    clean = min(rows, key=lambda r: abs(float(r.get("sigma", 0.0))))

    def exact(r):
        for k in ("mol_exact_rate", "exact_type_rate", "true_bond_exact_type_rate",
                  "mol_exact", "mol_validity"):
            if k in r:
                v = float(r[k]); return v * 100 if v <= 1 else v
        return float("nan")

    print("\n" + "=" * 62)
    print("INDEPENDENT TEST vs. reported VALIDATION (clean, sigma=0)")
    print("-" * 62)
    print(f"{'metric':<14}{'VAL (reported)':>18}{'TEST (this run)':>18}")
    print(f"{'Conn F1':<14}{REPORTED_VAL['conn']:>18.4f}{clean.get('conn_f1', float('nan')):>18.4f}")
    print(f"{'macro-F1':<14}{REPORTED_VAL['macro']:>18.4f}{clean.get('f1_macro', float('nan')):>18.4f}")
    print(f"{'Exact-type':<14}{REPORTED_VAL['exact']:>17.1f}%{exact(clean):>17.1f}%")
    print("=" * 62)
    print(f"\nTest set: {test_sdf} ({n_test} molecules)")
    print("Report the TEST column as the headline; keep VAL for model selection only.")
    print("Baselines (RDKit/OpenBabel) were saved separately under results/geom_TEST_baselines "
          "but use the identical test molecules (#3).")


if __name__ == "__main__":
    main()
