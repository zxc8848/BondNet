"""Audit and summarize corrected joint-model H-corruption evaluations."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


CONDITIONS = (
    "intact", "drop025", "drop050", "drop100",
    "hnoise010", "hnoise020", "hnoise030",
)
SEEDS = (42, 43, 44)
FIELDS = ("f1_macro_pipeline", "full_graph_exact_match", "conn_f1")


def result_row(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document["BondNet"]
    clean_rows = [row for row in rows if row["sigma"] == 0]
    if len(clean_rows) != 1 or clean_rows[0]["n_molecules"] != 27240:
        raise ValueError(f"Unexpected evaluation cohort or noise protocol: {path}")
    return clean_rows[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/revision_v3_joint_h_corruption"))
    parser.add_argument("--main-root", type=Path, default=Path("results/revision_v2_direction_fixed_test"))
    args = parser.parse_args()
    rows: list[dict] = []
    for condition in CONDITIONS:
        seed_values = []
        for seed in SEEDS:
            path = args.root / f"seed{seed}" / condition / "results.json"
            row = result_row(path)
            if condition == "intact":
                control = result_row(args.main_root / f"seed{seed}" / "one_stage" / "results.json")
                for field in FIELDS:
                    if abs(row[field] - control[field]) > 1e-12:
                        raise ValueError(f"Intact control differs from primary scorer: seed={seed}, {field}")
            seed_values.append(row)
        summary = {"condition": condition, "n_seeds": len(SEEDS), "n_molecules": 27240}
        for field in FIELDS:
            values = [float(row[field]) for row in seed_values]
            summary[field + "_mean"] = statistics.mean(values)
            summary[field + "_sample_sd"] = statistics.stdev(values)
            summary[field + "_by_seed"] = dict(zip(map(str, SEEDS), values))
        rows.append(summary)
    destination = args.root / "three_seed_summary.json"
    destination.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print("Intact controls reproduce all three primary results exactly.")
    print("Condition         Pipeline F1 mean (SD)      HH exact % mean (SD)")
    for row in rows:
        print(
            f"{row['condition']:<16} "
            f"{row['f1_macro_pipeline_mean']:.6f} ({row['f1_macro_pipeline_sample_sd']:.6f})       "
            f"{100 * row['full_graph_exact_match_mean']:.3f} "
            f"({100 * row['full_graph_exact_match_sample_sd']:.3f})"
        )
    print(destination)


if __name__ == "__main__":
    main()
