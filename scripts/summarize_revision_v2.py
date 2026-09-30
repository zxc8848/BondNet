"""CPU-only audit and summary of the fixed-test revision-v2 evaluations.

The interrupted seed-44 one-stage evaluation is excluded unless a replacement
robustness_curve.csv is supplied explicitly. No model inference is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
SIGMAS = (0.0, 0.1, 0.2)
METRICS = (
    "pipeline_macro_f1",
    "true_bond_exact_type_rate",
    "hh_graph_exact_match",
)
EXPECTED_NOISE_SEED = 20260921
EXPECTED_N = 27240
T_95_DF2 = 4.302652729911275


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_learned(path: Path, seed: int, model: str) -> dict[float, dict]:
    curve = read_csv(path)
    results_path = path.with_name("results.json")
    if not results_path.exists():
        raise FileNotFoundError(results_path)
    detailed = json.loads(results_path.read_text(encoding="utf-8"))["BondNet"]
    if len(curve) != len(SIGMAS) or len(detailed) != len(SIGMAS):
        raise ValueError(f"Unexpected number of noise levels: {path}")
    parsed = {}
    for row, detail in zip(curve, detailed, strict=True):
        sigma = float(row["sigma"])
        if sigma != float(detail["sigma"]) or sigma not in SIGMAS:
            raise ValueError(f"Noise level mismatch: {path}")
        if int(detail["n_molecules"]) != EXPECTED_N:
            raise ValueError(f"Test denominator mismatch: {path} sigma={sigma}")
        expected_seed = EXPECTED_NOISE_SEED + round(10000 * sigma)
        if int(detail["eval_noise_seed"]) != expected_seed:
            raise ValueError(f"Evaluation noise seed mismatch: {path} sigma={sigma}")
        for csv_key, json_key in (
            ("pipeline_macro_f1", "f1_macro_pipeline"),
            ("true_bond_exact_type_rate", "true_bond_exact_type_rate"),
            ("hh_graph_exact_match", "full_graph_exact_match"),
        ):
            if not math.isclose(float(row[csv_key]), float(detail[json_key]), abs_tol=6e-7):
                raise ValueError(f"CSV/JSON score mismatch: {path} {csv_key}")
        parsed[sigma] = {metric: float(row[metric]) for metric in METRICS}
    return parsed


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def paired_interval(differences: list[float]) -> tuple[float, float]:
    mean, sd = mean_sd(differences)
    halfwidth = T_95_DF2 * sd / math.sqrt(len(differences))
    return mean - halfwidth, mean + halfwidth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed44-one-stage", type=Path,
        help="Replacement seed-44 one-stage robustness_curve.csv after valid rerun",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/revision_v2_cpu_analysis"),
    )
    args = parser.parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    learned: dict[tuple[int, str], dict[float, dict]] = {}
    inputs = {}
    for seed in SEEDS:
        for model in ("explicit_two_stage", "heavy_two_stage", "one_stage"):
            if model == "one_stage" and seed == 44 and args.seed44_one_stage is None:
                continue
            path = (
                args.seed44_one_stage if model == "one_stage" and seed == 44
                else Path(f"results/revision_v2_seed{seed}/{model}/robustness_curve.csv")
            )
            path = (ROOT / path).resolve()
            learned[seed, model] = check_learned(path, seed, model)
            inputs[f"{model}_seed{seed}"] = str(path)

    baseline_path = ROOT / "results/revision_v2_rule_baselines/summary.csv"
    keyed_manifest = json.loads(
        (ROOT / "data/robustness_27240_keyed/manifest.json").read_text(encoding="utf-8")
    )
    if keyed_manifest["noise"]["noise_protocol"] != "torch-keyed":
        raise ValueError("Rule-baseline coordinates do not use molecule-keyed noise")
    if keyed_manifest["cohort"]["n_molecules"] != EXPECTED_N:
        raise ValueError("Rule-baseline materialized cohort size mismatch")
    baselines = {}
    for row in read_csv(baseline_path):
        sigma = float(row["sigma_angstrom"])
        if int(row["n_molecules"]) != EXPECTED_N:
            raise ValueError(f"Rule baseline denominator mismatch: {row}")
        tag = f"{round(100 * sigma):03d}"
        score_path = (
            ROOT / "results/revision_v2_rule_baselines/scores"
            / f"{row['method'].lower()}_sigma_{tag}.json"
        )
        detail = json.loads(score_path.read_text(encoding="utf-8"))
        expected_hash = keyed_manifest["outputs"][f"{sigma:.2f}"]["sha256"]
        if detail["sdf"]["sdf_sha256"] != expected_hash:
            raise ValueError(f"Rule-baseline coordinate hash mismatch: {score_path}")
        for csv_key, json_key in (
            ("success_rate", "success_rate"),
            ("f1_macro_pipeline", "f1_macro_pipeline"),
            ("true_bond_exact_type_rate", "true_bond_exact_type_rate"),
            ("heavy_graph_exact_match", "full_graph_exact_match"),
        ):
            if not math.isclose(float(row[csv_key]), float(detail[json_key]), abs_tol=1e-10):
                raise ValueError(f"Rule-baseline CSV/JSON mismatch: {score_path}")
        baselines[row["method"], sigma] = {
            "pipeline_macro_f1": float(row["f1_macro_pipeline"]),
            "true_bond_exact_type_rate": float(row["true_bond_exact_type_rate"]),
            "hh_graph_exact_match": float(row["heavy_graph_exact_match"]),
            "success_rate": float(row["success_rate"]),
        }
    if len(baselines) != 2 * len(SIGMAS):
        raise ValueError("Missing rule-baseline noise level")

    seed_rows = []
    for (seed, model), scores in sorted(learned.items()):
        for sigma, values in sorted(scores.items()):
            seed_rows.append({"seed": seed, "model": model, "sigma": sigma, **values})

    summary_rows = []
    for sigma in SIGMAS:
        for model in ("explicit_two_stage", "heavy_two_stage", "one_stage"):
            available = [seed for seed in SEEDS if (seed, model) in learned]
            if model == "one_stage" and len(available) < 3:
                continue
            row = {"sigma": sigma, "model": model, "n_seeds": len(available),
                   "success_rate": 1.0}
            for metric in METRICS:
                row[f"{metric}_mean"], row[f"{metric}_sd"] = mean_sd(
                    [learned[seed, model][sigma][metric] for seed in available]
                )
            summary_rows.append(row)
        for method in ("RDKit", "OpenBabel"):
            score = baselines[method, sigma]
            row = {"sigma": sigma, "model": method, "n_seeds": 0,
                   "success_rate": score["success_rate"]}
            for metric in METRICS:
                row[f"{metric}_mean"] = score[metric]
                row[f"{metric}_sd"] = ""
            summary_rows.append(row)

    difference_rows = []
    comparisons = (
        ("explicit_minus_heavy", "explicit_two_stage", "heavy_two_stage"),
        ("explicit_minus_rdkit", "explicit_two_stage", "RDKit"),
    )
    if args.seed44_one_stage is not None:
        comparisons += (("explicit_minus_one_stage", "explicit_two_stage", "one_stage"),)
    for sigma in SIGMAS:
        for name, left, right in comparisons:
            for metric in METRICS:
                differences = [
                    learned[seed, left][sigma][metric]
                    - (baselines[right, sigma][metric] if right == "RDKit"
                       else learned[seed, right][sigma][metric])
                    for seed in SEEDS
                ]
                mean, sd = mean_sd(differences)
                low, high = paired_interval(differences)
                difference_rows.append({
                    "sigma": sigma, "comparison": name, "metric": metric,
                    "seed42": differences[0], "seed43": differences[1],
                    "seed44": differences[2], "mean": mean, "sample_sd": sd,
                    "seed_t95_low": low, "seed_t95_high": high,
                })

    write_csv(output_dir / "seed_scores.csv", seed_rows)
    write_csv(output_dir / "same_metric_summary.csv", summary_rows)
    write_csv(output_dir / "seed_paired_differences.csv", difference_rows)
    audit = {
        "status": "final" if args.seed44_one_stage else "provisional",
        "excluded": [] if args.seed44_one_stage else [
            "seed44 one_stage: interrupted resume overwrote original best_e2e.pt"
        ],
        "test_molecules_per_evaluation": EXPECTED_N,
        "sigmas": SIGMAS,
        "eval_noise_seed_base": EXPECTED_NOISE_SEED,
        "baseline_coordinates": str(ROOT / "data/robustness_27240_keyed"),
        "inputs": inputs,
        "interpretation": (
            "Seed t intervals describe variation across only three fitted training "
            "seeds. They do not measure molecule-sampling uncertainty, and the "
            "small sample makes them unstable."
        ),
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Status: {audit['status']}; audited {len(learned)} learned evaluations")
    for row in summary_rows:
        print(
            f"sigma={row['sigma']:.2f} {row['model']:<18} "
            f"F1={row['pipeline_macro_f1_mean']:.4f} "
            f"exact={100*row['true_bond_exact_type_rate_mean']:.2f}% "
            f"graph={100*row['hh_graph_exact_match_mean']:.2f}%"
        )
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
