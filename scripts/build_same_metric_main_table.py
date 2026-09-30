"""Build the fixed-test learned/rule comparison on identical metrics.

The learned rows use the three-seed molecule-statistics export. Rule-based rows
use the same 27,240 materialized SDF cohort. This keeps pipeline macro-F1,
true-pair exactness, and strict HH-graph exactness on a common denominator.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


LEARNED_METRICS = {
    "pipeline_macro_f1": "pipeline_macro_f1",
    "true_pair_exact_rate": "true_bond_exact_type_rate",
    "hh_graph_exact_rate": "hh_graph_exact_match",
}


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bootstrap",
        type=Path,
        default=Path("results/revision_bootstrap/bootstrap_summary.csv"),
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=Path("results/revision_fixed_test_rule_baselines/summary.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/revision_same_metric_main_table.csv"),
    )
    args = parser.parse_args()

    bootstrap = _read_csv(args.bootstrap)
    rules = _read_csv(args.rules)
    output = []

    for sigma in (0.0, 0.1, 0.2):
        for comparison, label in (("one_stage", "One-stage"), ("two_stage", "Two-stage")):
            row = {
                "sigma": f"{sigma:.1f}",
                "method": label,
                "n_molecules": "27240",
                "success_rate": "1.0",
            }
            seed_sds = []
            for metric, column in LEARNED_METRICS.items():
                matches = [
                    item for item in bootstrap
                    if float(item["sigma"]) == sigma
                    and item["metric"] == metric
                    and item["scope"] == "three_seed_mean"
                    and item["comparison"] == comparison
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Expected one bootstrap row for {sigma=} {metric=} {comparison=}; "
                        f"found {len(matches)}"
                    )
                item = matches[0]
                row[column] = item["point"]
                row[f"{column}_molecule_ci_low"] = item["ci_low"]
                row[f"{column}_molecule_ci_high"] = item["ci_high"]
                row[f"{column}_seed_sd"] = item["seed_sd"]
                seed_sds.append(item["seed_sd"])
            output.append(row)

        for method in ("RDKit", "OpenBabel"):
            matches = [
                item for item in rules
                if float(item["sigma_angstrom"]) == sigma and item["method"] == method
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one rule row for {sigma=} {method=}; found {len(matches)}")
            item = matches[0]
            output.append({
                "sigma": f"{sigma:.1f}",
                "method": method,
                "n_molecules": item["n_molecules"],
                "success_rate": item["success_rate"],
                "pipeline_macro_f1": item["f1_macro_pipeline"],
                "true_bond_exact_type_rate": item["true_bond_exact_type_rate"],
                "hh_graph_exact_match": item["heavy_graph_exact_match"],
            })

    fields = [
        "sigma", "method", "n_molecules", "success_rate",
        "pipeline_macro_f1", "pipeline_macro_f1_molecule_ci_low",
        "pipeline_macro_f1_molecule_ci_high", "pipeline_macro_f1_seed_sd",
        "true_bond_exact_type_rate", "true_bond_exact_type_rate_molecule_ci_low",
        "true_bond_exact_type_rate_molecule_ci_high", "true_bond_exact_type_rate_seed_sd",
        "hh_graph_exact_match", "hh_graph_exact_match_molecule_ci_low",
        "hh_graph_exact_match_molecule_ci_high", "hh_graph_exact_match_seed_sd",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)
    print(f"Wrote {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()
