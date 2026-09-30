"""Paired three-seed audit of v2 teacher-forced and predicted-topology Stage 2."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


METRICS = {
    "pipeline_f1": "f1_macro_pipeline",
    "hh_graph_exact": "full_graph_exact_match",
    "reference_pair_f1": "f1_macro",
}
T_CRIT_DF2_975 = 4.302652729911275


def load(path: Path):
    with path.open(encoding="utf-8") as stream:
        return {float(row["sigma"]): row for row in json.load(stream)["BondNet"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    by_seed = {}
    for seed in args.seeds:
        baseline = load(args.results_root / f"revision_v2_seed{seed}" / "explicit_two_stage" / "results.json")
        finetuned = load(args.results_root / "revision_v2_predicted_topology" / f"seed{seed}" / "results.json")
        if set(baseline) != set(finetuned):
            raise ValueError(f"Noise levels differ for seed {seed}")
        for sigma in baseline:
            for field in ("n_molecules", "eval_noise_seed"):
                if baseline[sigma][field] != finetuned[sigma][field]:
                    raise ValueError(f"{field} differs for seed {seed}, sigma {sigma}")
            if int(baseline[sigma]["n_molecules"]) != 27240:
                raise ValueError(f"Unexpected molecule count for seed {seed}, sigma {sigma}")
        by_seed[seed] = (baseline, finetuned)
    rows = []
    for sigma in sorted(by_seed[args.seeds[0]][0]):
        for label, key in METRICS.items():
            base = [float(by_seed[seed][0][sigma][key]) for seed in args.seeds]
            tuned = [float(by_seed[seed][1][sigma][key]) for seed in args.seeds]
            delta = [b - a for a, b in zip(base, tuned)]
            n = len(delta)
            sd = statistics.stdev(delta) if n > 1 else 0.0
            halfwidth = (T_CRIT_DF2_975 * sd / math.sqrt(n)) if n == 3 else float("nan")
            rows.append({
                "sigma": sigma,
                "metric": label,
                "n_seeds": n,
                "teacher_mean": statistics.mean(base),
                "teacher_sample_sd": statistics.stdev(base) if n > 1 else 0.0,
                "fine_tuned_mean": statistics.mean(tuned),
                "fine_tuned_sample_sd": statistics.stdev(tuned) if n > 1 else 0.0,
                "paired_delta_mean": statistics.mean(delta),
                "paired_delta_sample_sd": sd,
                "paired_delta_t95_lo": statistics.mean(delta) - halfwidth,
                "paired_delta_t95_hi": statistics.mean(delta) + halfwidth,
                **{f"seed{seed}_delta": diff for seed, diff in zip(args.seeds, delta)},
            })
    output = args.results_root / "revision_v2_predicted_topology" / "three_seed_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
