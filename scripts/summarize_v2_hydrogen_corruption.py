"""Summarize the v2 hydrogen-corruption audit across training seeds."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


FIELDS = [
    "reference_pair_f1",
    "pipeline_f1",
    "hh_graph_exact",
    "conn_f1",
    "reference_bond_exact",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/revision_v2_hydrogen_corruption"))
    parser.add_argument("--n-seeds", type=int, default=3)
    args = parser.parse_args()
    with (args.root / "summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    conditions = list(dict.fromkeys(row["condition"] for row in rows))
    output = []
    for condition in conditions:
        subset = [row for row in rows if row["condition"] == condition]
        seeds = {int(row["seed"]) for row in subset}
        if len(seeds) != args.n_seeds or len(subset) != args.n_seeds:
            raise ValueError(f"Expected {args.n_seeds} distinct seeds for {condition}, got {seeds}")
        result = {"condition": condition, "n_seeds": len(seeds)}
        for field in FIELDS:
            values = [float(row[field]) for row in subset]
            result[f"{field}_mean"] = statistics.mean(values)
            result[f"{field}_sample_sd"] = statistics.stdev(values)
        output.append(result)
    with (args.root / "three_seed_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(args.root / "three_seed_summary.csv")


if __name__ == "__main__":
    main()
