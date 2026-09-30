"""Summarize the matched v3 joint one-stage hydrogen comparison.

Both arms use the same fixed GEOM test molecule order and keyed perturbations.
This summary reports training-seed variation only; it is not a molecule-level
bootstrap and does not make the repeatedly inspected test cohort pristine.
"""

import json
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
ARMS = {
    "explicit_h": ROOT / "results/revision_v2_direction_fixed_test",
    "heavy_only": ROOT / "results/revision_v3_one_stage_heavy_test",
}
METRICS = ("f1_macro_pipeline", "full_graph_exact_match", "true_bond_exact_type_rate")


def load(seed: int, arm: str) -> dict[float, dict]:
    base = ARMS[arm]
    path = (base / f"seed{seed}" / "one_stage" / "results.json") if arm == "explicit_h" else (base / f"seed{seed}" / "results.json")
    with path.open(encoding="utf-8") as stream:
        rows = json.load(stream)["BondNet"]
    by_sigma = {float(row["sigma"]): row for row in rows}
    assert set(by_sigma) == {0.0, 0.1, 0.2}, path
    assert all(row["n_molecules"] == 27240 for row in rows), path
    expected_noise_seeds = {0.0: 20260921, 0.1: 20261921, 0.2: 20262921}
    assert all(row["eval_noise_seed"] == expected_noise_seeds[float(row["sigma"])] for row in rows), path
    return by_sigma


def main() -> None:
    all_rows = {arm: {seed: load(seed, arm) for seed in SEEDS} for arm in ARMS}
    print("GEOM fixed test (repeatedly inspected; descriptive only), n=27,240")
    print("delta = explicit H minus heavy only; sample SD is across 3 training seeds")
    for sigma in (0.0, 0.1, 0.2):
        print(f"\nsigma={sigma:.2f} A")
        for metric in METRICS:
            explicit = [all_rows["explicit_h"][seed][sigma][metric] for seed in SEEDS]
            heavy = [all_rows["heavy_only"][seed][sigma][metric] for seed in SEEDS]
            delta = [a - b for a, b in zip(explicit, heavy)]
            print(
                f"  {metric}: explicit={mean(explicit):.6f} +/- {stdev(explicit):.6f}; "
                f"heavy={mean(heavy):.6f} +/- {stdev(heavy):.6f}; "
                f"paired_delta={mean(delta):+.6f} +/- {stdev(delta):.6f}; "
                f"seeds={','.join(f'{x:+.6f}' for x in delta)}"
            )


if __name__ == "__main__":
    main()
