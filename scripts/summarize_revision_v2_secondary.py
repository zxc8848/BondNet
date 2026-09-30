"""Aggregate v2 chemical-validity and distortion audits over training seeds.

Only reads the per-seed JSON files produced by the two evaluation scripts.
The sample SD describes training-seed variation, not molecule uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


VALIDITY_FIELDS = {
    "sanitizable": "reference_charge_sanitize_ok_rate",
    "round_trip": "reference_charge_smiles_roundtrip_ok_rate",
    "canonical_match": "reference_charge_canonical_smiles_match_rate",
    "indexed_exact": "reference_full_graph_exact_rate",
}
DISTORTION_FIELDS = {
    "bond_length_change_A": "mean_abs_reference_bond_length_change_A",
    "pipeline_macro_f1": "pipeline_macro_f1",
    "hh_graph_exact": "hh_graph_exact_rate",
}


def aggregate(root: Path, subdir: str, key: str, fields: dict[str, str], seeds: list[int]):
    by_seed = {}
    for seed in seeds:
        path = root / subdir / f"seed{seed}" / ("summary.json" if key == "sigma" else "results.json")
        with path.open(encoding="utf-8") as stream:
            records = json.load(stream)
        indexed = {item[key]: item for item in records}
        if len(indexed) != len(records):
            raise ValueError(f"Duplicate {key} value in {path}")
        by_seed[seed] = indexed
    keys = list(by_seed[seeds[0]])
    if any(set(rows) != set(keys) for rows in by_seed.values()):
        raise ValueError(f"Inconsistent {key} values for {subdir}")
    output = []
    for value in keys:
        row = {key: value, "n_seeds": len(seeds)}
        for label, field in fields.items():
            values = [float(by_seed[seed][value][field]) for seed in seeds]
            row[f"{label}_mean"] = statistics.mean(values)
            row[f"{label}_sample_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(row)
    return output


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    validity = aggregate(args.results_root, "revision_v2_chemical_validity", "sigma", VALIDITY_FIELDS, args.seeds)
    distortions = aggregate(args.results_root, "revision_v2_structured_distortions", "distortion", DISTORTION_FIELDS, args.seeds)
    output_dir = args.results_root / "revision_v2_secondary_summary"
    write_csv(output_dir / "chemical_validity_three_seed.csv", validity)
    write_csv(output_dir / "structured_distortions_three_seed.csv", distortions)
    print(output_dir)


if __name__ == "__main__":
    main()
