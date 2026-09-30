"""Partition clean rule-baseline HH graph errors by connectivity and label type.

This reads existing predictions; it does not rerun or retune the baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import p0c_yuelbond_compare as scorer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", default="data/robustness_27240_keyed/sigma_000.sdf")
    parser.add_argument("--predictions", default="results/revision_v2_rule_baselines/predictions/rdkit_sigma_000_preds.json")
    parser.add_argument("--output", default="results/revision_v4_rule_label_audit/geom_rdkit_clean.json")
    args = parser.parse_args()
    molecules = scorer._read_valid_molecules(str(ROOT / args.sdf))
    predictions = json.loads((ROOT / args.predictions).read_text(encoding="utf-8"))
    meta = predictions.pop("__meta__", None)
    if meta is None or meta.get("n_molecules") != len(molecules):
        raise ValueError("Prediction file lacks a matching cohort fingerprint")
    digest = hashlib.sha256((ROOT / args.sdf).read_bytes()).hexdigest()
    if meta.get("sdf_sha256") != digest:
        raise ValueError("Prediction and reference SDF hashes differ")
    counts = Counter()
    mismatched_pairs = Counter()
    for idx, mol in enumerate(molecules):
        true = scorer._true_bonds(mol)
        raw = predictions.get(str(idx))
        if raw is None:
            counts["tool_failure"] += 1
            continue
        pred = {(min(int(i), int(j)), max(int(i), int(j))): scorer._pred_label(label)
                for i, j, label in raw}
        if true == pred:
            counts["exact"] += 1
            continue
        if set(true) != set(pred):
            counts["connectivity_error"] += 1
            continue
        counts["same_connectivity_label_error"] += 1
        errors = [(true[pair], pred[pair]) for pair in true if true[pair] != pred[pair]]
        for old, new in errors:
            mismatched_pairs[f"{old}->{new}"] += 1
        if any(3 in (old, new) for old, new in errors):
            counts["any_aromatic_label_error"] += 1
        if all(3 in (old, new) and {old, new} <= {0, 1, 3}
               for old, new in errors):
            counts["aromatic_vs_kekule_only"] += 1
        elif all({old, new} <= {0, 1} for old, new in errors):
            counts["single_double_only"] += 1
        else:
            counts["other_label_error"] += 1
    primary = ("tool_failure", "exact", "connectivity_error",
               "same_connectivity_label_error")
    if sum(counts[key] for key in primary) != len(molecules):
        raise ValueError("Error partition does not cover the cohort")
    result = {"sdf": args.sdf, "predictions": args.predictions,
              "n_molecules": len(molecules), "counts": dict(counts),
              "mismatched_bond_label_pairs": dict(mismatched_pairs),
              "exact_fraction": counts["exact"] / len(molecules),
              "aromatic_only_fraction_of_all_errors": (
                  counts["aromatic_vs_kekule_only"]
                  / (len(molecules) - counts["exact"])
              ),
              "interpretation": "Aromatic-vs-Kekule category is an upper bound on representation-only errors; chemical equivalence was not proven."}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["counts"], indent=2))
    print(f"exact={result['exact_fraction']:.6%}; aromatic-only share of errors="
          f"{result['aromatic_only_fraction_of_all_errors']:.2%}")


if __name__ == "__main__":
    main()
