"""Summarize corrected joint structured-distortion and validity audits."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SEEDS = (42, 43, 44)
DISTORTION_FIELDS = (
    "mean_abs_reference_bond_length_change_A",
    "pipeline_macro_f1", "hh_graph_exact_rate",
)
VALIDITY_FIELDS = (
    "reference_charge_sanitize_ok_rate",
    "reference_charge_smiles_roundtrip_ok_rate",
    "reference_charge_canonical_smiles_match_rate",
    "reference_full_graph_exact_rate",
)


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(root: Path, category: str, key: str, fields: tuple[str, ...]) -> list[dict]:
    by_seed = {}
    for seed in SEEDS:
        filename = "results.json" if category == "structured_directed" else "summary.json"
        path = root / f"revision_v3_joint_{category}" / f"seed{seed}" / filename
        records = load(path)
        if any(row.get("n_molecules") != 27240 for row in records):
            raise ValueError(f"Unexpected cohort in {path}")
        by_seed[seed] = {str(row[key]): row for row in records}
        if len(by_seed[seed]) != len(records):
            raise ValueError(f"Duplicate {key} in {path}")
    keys = list(by_seed[SEEDS[0]])
    if any(set(by_seed[seed]) != set(keys) for seed in SEEDS):
        raise ValueError(f"Inconsistent {key} values for {category}")
    output = []
    for value in keys:
        summary = {key: value, "n_seeds": len(SEEDS), "n_molecules": 27240}
        for field in fields:
            vals = [float(by_seed[seed][value][field]) for seed in SEEDS]
            summary[field + "_mean"] = statistics.mean(vals)
            summary[field + "_sample_sd"] = statistics.stdev(vals)
            summary[field + "_by_seed"] = dict(zip(map(str, SEEDS), vals))
        output.append(summary)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--structured-only", action="store_true")
    args = parser.parse_args()
    structured = aggregate(args.root, "structured_directed", "distortion", DISTORTION_FIELDS)
    validity = [] if args.structured_only else aggregate(args.root, "validity", "sigma", VALIDITY_FIELDS)
    for seed in SEEDS:
        clean = next(row for row in load(args.root / "revision_v3_joint_structured_directed" / f"seed{seed}" / "results.json") if row["distortion"] == "clean")
        main_clean = load(args.root / "revision_v2_direction_fixed_test" / f"seed{seed}" / "one_stage" / "results.json")["BondNet"][0]
        for actual, expected in ((clean["pipeline_macro_f1"], main_clean["f1_macro_pipeline"]),
                                 (clean["hh_graph_exact_rate"], main_clean["full_graph_exact_match"])):
            if abs(actual - expected) > 1e-12:
                raise ValueError(f"Structured clean control differs from main scorer for seed {seed}")
    output = args.root / "revision_v3_joint_secondary_summary"
    output.mkdir(parents=True, exist_ok=True)
    (output / "structured_three_seed.json").write_text(json.dumps(structured, indent=2) + "\n", encoding="utf-8")
    if not args.structured_only:
        (output / "validity_three_seed.json").write_text(json.dumps(validity, indent=2) + "\n", encoding="utf-8")
    print("Structured clean controls reproduce all three main scores exactly.")
    for row in structured:
        print(f"{row['distortion']:<17} F1={row['pipeline_macro_f1_mean']:.6f} "
              f"HH={100 * row['hh_graph_exact_rate_mean']:.2f}%")
    for row in validity:
        print(f"sigma={row['sigma']} sanitize={100 * row['reference_charge_sanitize_ok_rate_mean']:.2f}% "
              f"indexed-exact={100 * row['reference_full_graph_exact_rate_mean']:.2f}%")
    print(output)


if __name__ == "__main__":
    main()
