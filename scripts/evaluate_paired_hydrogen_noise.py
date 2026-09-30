"""Post-hoc paired-noise hydrogen sensitivity on GEOM or external PubChem3D.

Keep the frozen explicit-H prediction files unchanged. For each molecule, draw
noise at the explicit-H atom count using the original keyed evaluator, then
apply the corresponding atom displacements to the separately trained
heavy-only model. This removes stochastic noise-realization imbalance without
retraining or replacing the frozen external headline evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bondnet.data.dataset import CachedBondNetDataset
from bondnet.data.noise_augment import GaussianNoiseAugment
from evaluate import load_model
from scripts.export_per_molecule_stats import collect_sigma, f1_from_stats


CONFIG = {
    "geom": {
        "explicit_cache": "data/fixed_split_caches/geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed_test.pt",
        "heavy_cache": "data/fixed_split_caches/geom_drugs_all_random1_re10_c30_heavy_fixed_test.pt",
        "original": "results/revision_v3_undirected_audit",
    },
    "external": {
        "explicit_cache": "data/external_v3/cache_explicit_c30_h30.pt",
        "heavy_cache": "data/external_v3/cache_heavy_c30.pt",
        "original": "results/revision_v3_external",
    },
}


def atom_mapping(explicit: dict, heavy: dict) -> torch.Tensor:
    """Map heavy-only atoms onto explicit-H atoms, including retained stereo H."""
    xelem, yelem = explicit["elems"], heavy["elems"]
    n_heavy = int((xelem != 1).sum())
    if not bool((xelem[:n_heavy] != 1).all() and (xelem[n_heavy:] == 1).all()):
        raise ValueError("Explicit-H cache does not put all heavy atoms first")
    if not torch.equal(xelem[:n_heavy], yelem[:n_heavy]):
        raise ValueError("Heavy-atom identity/order differs between representations")
    if not torch.equal(explicit["coord"][:n_heavy], heavy["coord"][:n_heavy]):
        raise ValueError("Heavy-atom clean coordinates differ between representations")
    if not bool((yelem[n_heavy:] == 1).all()):
        raise ValueError("Heavy-only cache has an unexpected atom after its heavy prefix")
    mapping = list(range(n_heavy))
    used_h = set()
    for target in heavy["coord"][n_heavy:]:
        matches = torch.where(torch.all(explicit["coord"][n_heavy:] == target, dim=1))[0]
        matches = [n_heavy + int(i) for i in matches.tolist() if n_heavy + int(i) not in used_h]
        if len(matches) != 1:
            raise ValueError("Retained hydrogen has no unique coordinate match")
        mapping.append(matches[0])
        used_h.add(matches[0])
    return torch.tensor(mapping, dtype=torch.long)


def full_explicit_noise(sample: dict, sigma: float, eval_seed: int) -> torch.Tensor:
    """Use the frozen GaussianNoiseAugment implementation and keyed seed."""
    if sigma == 0.0:
        return torch.zeros_like(sample["coord"])
    n = len(sample["coord"])
    batch = {
        "coord": sample["coord"],
        "edge_index": torch.empty((0, 2), dtype=torch.long),
        "num_atoms_per_mol": torch.tensor([n], dtype=torch.long),
        "sample_ids": torch.tensor([int(sample["geom_mol_idx"])], dtype=torch.long),
    }
    augment = GaussianNoiseAugment(sigma=sigma, sigma_min=sigma,
                                   sigma_max=sigma, cutoff=2.5)
    perturbed = augment.augment_batch(
        batch, per_molecule=True,
        base_seed=eval_seed + int(round(sigma * 10000.0)), epoch=0,
    )["coord"]
    return perturbed - sample["coord"]


class PairedHeavyDataset(Dataset):
    def __init__(self, explicit: CachedBondNetDataset,
                 heavy: CachedBondNetDataset, sigma: float, eval_seed: int,
                 max_mols: int | None = None):
        if len(explicit) != len(heavy):
            raise ValueError("Representation caches differ in molecule count")
        self.heavy = heavy
        self.size = min(len(heavy), max_mols) if max_mols else len(heavy)
        self.mappings = []
        self.noises = []
        retained_h_mols = 0
        for idx in range(self.size):
            x = explicit[idx]
            y = heavy[idx]
            if int(x["geom_mol_idx"]) != int(y["geom_mol_idx"]):
                raise ValueError(f"Molecule id/order mismatch at row {idx}")
            mapping = atom_mapping(x, y)
            retained_h_mols += int(bool((y["elems"] == 1).any()))
            self.mappings.append(mapping)
            self.noises.append(full_explicit_noise(x, sigma, eval_seed)[mapping])
        self.retained_h_molecules = retained_h_mols

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        sample = self.heavy[idx]
        coord = sample["coord"] + self.noises[idx]
        edge_index = sample["edge_index"]
        diff = coord[edge_index[:, 1]] - coord[edge_index[:, 0]]
        updated = dict(sample)
        updated["coord"] = coord
        updated["edge_diff"] = diff
        updated["edge_dist"] = diff.norm(dim=-1)
        return updated


def score(stats: dict) -> dict:
    _, macro = f1_from_stats(stats["u_pipe_tp"], stats["u_pipe_fp"], stats["u_pipe_fn"])
    return {"undirected_pipeline_macro_f1": macro,
            "undirected_hh_graph_exact_rate": float(stats["u_hh_graph_exact"].mean())}


def archived_stats(root: Path, arm: str, seed: int, tag: str) -> dict:
    with np.load(root / arm / f"seed{seed}" / f"sigma_{tag}.npz") as archive:
        return {key: archive[key] for key in
                ("mol_idx", "u_pipe_tp", "u_pipe_fp", "u_pipe_fn", "u_hh_graph_exact")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=CONFIG, required=True)
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.1, 0.2])
    parser.add_argument("--eval-noise-seed", type=int, default=20260921)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-mols", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    cfg = CONFIG[args.cohort]
    explicit = CachedBondNetDataset(str(ROOT / cfg["explicit_cache"]))
    heavy = CachedBondNetDataset(str(ROOT / cfg["heavy_cache"]))
    original = ROOT / cfg["original"]
    output = ROOT / (args.output_dir or f"results/revision_v4_paired_hydrogen/{args.cohort}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict_args = SimpleNamespace(batch_size=args.batch_size, cutoff=2.5,
                                   h_cutoff=2.5, conn_threshold=0.5,
                                   eval_noise_seed=args.eval_noise_seed)
    rows = []
    for sigma in args.sigmas:
        tag = f"{round(100 * sigma):03d}"
        paired = PairedHeavyDataset(explicit, heavy, sigma, args.eval_noise_seed,
                                    args.max_mols)
        print(f"[{args.cohort}] sigma={sigma:.2f} mapped={len(paired)} "
              f"retained-H={paired.retained_h_molecules}", flush=True)
        for seed in (42, 43, 44):
            checkpoint = ROOT / f"checkpoints/revision_v3_one_stage_heavy_seed{seed}/best_e2e.pt"
            model = load_model(str(checkpoint), device)
            model.eval()
            stats = collect_sigma(model, None, paired, device, predict_args, 0.0)
            paired_score = score(stats)
            seed_dir = output / f"seed{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(seed_dir / f"sigma_{tag}.npz", **stats)
            row = {"sigma": sigma, "seed": seed,
                   "n_molecules": len(paired),
                   "retained_h_molecules": paired.retained_h_molecules,
                   "heavy_paired": paired_score}
            if args.max_mols is None:
                joint = archived_stats(original, "joint", seed, tag)
                heavy_original = archived_stats(original, "heavy", seed, tag)
                for name, archived in (("joint_original", joint),
                                       ("heavy_independent_original", heavy_original)):
                    if not np.array_equal(stats["mol_idx"], archived["mol_idx"]):
                        raise ValueError(f"Molecule identity/order mismatch: {name}")
                    row[name] = score(archived)
                row["paired_joint_minus_heavy_f1"] = (
                    row["joint_original"]["undirected_pipeline_macro_f1"]
                    - paired_score["undirected_pipeline_macro_f1"]
                )
                row["paired_joint_minus_heavy_hh_exact"] = (
                    row["joint_original"]["undirected_hh_graph_exact_rate"]
                    - paired_score["undirected_hh_graph_exact_rate"]
                )
            rows.append(row)
            print(f"[{args.cohort}] sigma={sigma:.2f} seed={seed} "
                  f"paired heavy F1={paired_score['undirected_pipeline_macro_f1']:.6f} "
                  f"HH={paired_score['undirected_hh_graph_exact_rate']:.4%}", flush=True)
    (output / "summary.json").write_text(json.dumps({
        "protocol": {"cohort": args.cohort, "explicit_cache": cfg["explicit_cache"],
                     "heavy_cache": cfg["heavy_cache"], "noise_seed": args.eval_noise_seed,
                     "sigmas": args.sigmas, "max_mols": args.max_mols,
                     "status": "post-hoc paired-noise sensitivity; frozen primary results unchanged"},
        "rows": rows,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
