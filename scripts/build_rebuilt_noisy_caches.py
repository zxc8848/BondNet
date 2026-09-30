#!/usr/bin/env python3
"""Build feature caches whose whole input graph comes from the noisy coordinates.

The primary evaluation perturbs the coordinates of a clean-coordinate 3.0 A
cache: its directed records (message-passing neighbours) were selected on clean
geometry, and only the 2.5 A candidate mask is recomputed after noise.  This
script instead featurizes the materialized noisy SDFs from scratch, so the 3.0 A
envelope, every message-passing edge and every candidate pair are determined by
the perturbed coordinates alone -- the input a deployed model would receive.

For each cohort and noise level it writes an explicit-H cache and a heavy-only
cache (RDKit RemoveHs of the same noisy molecules, so heavy-only inputs share
the explicit-H heavy-atom displacements) and checks that
  * molecule order and ids equal the primary cache,
  * explicit-H coordinates equal primary coordinates + the molecule-keyed
    evaluation noise (the exact coordinates of the primary evaluation).

Outputs: data/rebuilt_noisy_v4/<cohort>/{explicit,heavy}_sigma_XXX.pt and
data/rebuilt_noisy_v4/manifest.json.  Inference only; nothing is trained.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COHORTS = {
    "geom": {
        "primary_cache": "data/fixed_split_caches/geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed_test.pt",
        "sdf": "data/robustness_27240_keyed/sigma_{tag}.sdf",
    },
    "external": {
        "primary_cache": "data/external_v3/cache_explicit_c30_h30.pt",
        "sdf": "data/external_v3/sigma_{tag}.sdf",
    },
}
SIGMAS = (0.0, 0.1, 0.2)
NOISE_SEED = 20260921
OUT = ROOT / "data" / "rebuilt_noisy_v4"


def tag(sigma):
    return f"{round(100 * sigma):03d}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_samples(path: Path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)["samples"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohorts", nargs="+", default=list(COHORTS), choices=list(COHORTS))
    ap.add_argument("--sigmas", nargs="+", type=float, default=list(SIGMAS))
    ap.add_argument("--check_mols", type=int, default=512)
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("p0d", ROOT / "scripts" / "p0d_unified_robustness.py")
    p0d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p0d)

    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for cohort in args.cohorts:
        cfg = COHORTS[cohort]
        primary = load_samples(ROOT / cfg["primary_cache"])
        primary_ids = [int(s["geom_mol_idx"]) for s in primary]
        n_check = min(args.check_mols, len(primary))
        (OUT / cohort).mkdir(parents=True, exist_ok=True)
        for sigma in args.sigmas:
            sdf = ROOT / cfg["sdf"].format(tag=tag(sigma))
            entry = {"sdf": str(sdf.relative_to(ROOT)).replace("\\", "/"), "sdf_sha256": sha256(sdf)}
            for rep, flags in (("explicit", ["--explicit_h"]), ("heavy", [])):
                out = OUT / cohort / f"{rep}_sigma_{tag(sigma)}.pt"
                if not out.exists():
                    cmd = [sys.executable, str(ROOT / "scripts" / "precompute_features.py"),
                           "--data_path", str(sdf), "--output", str(out),
                           "--cutoff", "3.0", "--h_cutoff", "3.0", *flags]
                    print("[build]", " ".join(cmd), flush=True)
                    subprocess.run(cmd, cwd=ROOT, check=True)
                samples = load_samples(out)
                ids = [int(s["geom_mol_idx"]) for s in samples]
                if ids != primary_ids:
                    raise RuntimeError(f"{out}: molecule ids/order differ from the primary cache")
                if rep == "explicit":
                    noise = p0d._torch_keyed_noise(primary[:n_check], (sigma,), NOISE_SEED)[sigma]
                    worst = 0.0
                    for k in range(n_check):
                        expected = primary[k]["coord"].numpy().astype(np.float64)
                        if noise is not None:
                            expected = expected + noise[k]
                        got = samples[k]["coord"].numpy().astype(np.float64)
                        if got.shape != expected.shape:
                            raise RuntimeError(f"{out}: atom count differs for molecule {k}")
                        worst = max(worst, float(np.abs(got - expected).max()))
                    # SDF coordinates carry four decimals.
                    if worst > 1e-3:
                        raise RuntimeError(f"{out}: coordinates differ from primary evaluation ({worst:.2e} A)")
                    entry["explicit_max_coord_diff_vs_primary_eval"] = worst
                entry[f"{rep}_cache"] = str(out.relative_to(ROOT)).replace("\\", "/")
                entry[f"{rep}_n_molecules"] = len(samples)
                entry[f"{rep}_mean_directed_records"] = float(np.mean([s["edge_index"].shape[0] for s in samples]))
            manifest[f"{cohort}_sigma_{tag(sigma)}"] = entry
            print(f"[ok] {cohort} sigma={sigma}: {entry}", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("[done]", manifest_path)


if __name__ == "__main__":
    main()
