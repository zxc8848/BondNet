"""CPU-only audit of noisy 2.5-A candidates absent from the clean 3.0-A cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]


def mols(path: Path):
    supplier = Chem.SDMolSupplier(
        str(path), sanitize=False, removeHs=False, strictParsing=False
    )
    for index, mol in enumerate(supplier):
        if mol is None:
            raise ValueError(f"Unreadable SDF record {index}: {path}")
        yield mol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sdf-dir", type=Path, default=Path("data/robustness_27240_keyed")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/revision_v2_cpu_analysis/candidate_envelope.json"),
    )
    parser.add_argument("--cache-cutoff", type=float, default=3.0)
    parser.add_argument("--active-cutoff", type=float, default=2.5)
    args = parser.parse_args()
    sdf_dir = (ROOT / args.sdf_dir).resolve()
    clean_path = sdf_dir / "sigma_000.sdf"
    noisy_paths = {0.1: sdf_dir / "sigma_010.sdf",
                   0.2: sdf_dir / "sigma_020.sdf"}
    counts = {
        sigma: {
            "n_molecules": 0,
            "molecules_with_missing_active_pair": 0,
            "missing_active_pairs": 0,
            "missing_true_bond_pairs": 0,
            "molecules_with_missing_true_bond": 0,
            "missing_heavy_heavy_pairs": 0,
            "missing_hydrogen_heavy_pairs": 0,
            "active_pairs": 0,
            "active_heavy_heavy_pairs": 0,
        }
        for sigma in noisy_paths
    }
    clean_counts = {
        "n_molecules": 0,
        "active_pairs": 0,
        "active_heavy_heavy_pairs": 0,
        "active_heavy_heavy_true_bond_pairs": 0,
        "active_true_bond_pairs": 0,
        "reference_bond_pairs": 0,
    }
    streams = {sigma: mols(path) for sigma, path in noisy_paths.items()}
    sentinel = object()
    for index, clean in enumerate(mols(clean_path)):
        n_atoms = clean.GetNumAtoms()
        atoms = [atom.GetAtomicNum() for atom in clean.GetAtoms()]
        xyz_clean = np.asarray(clean.GetConformer().GetPositions())
        i, j = np.triu_indices(n_atoms, k=1)
        non_hh = (np.asarray(atoms)[i] != 1) | (np.asarray(atoms)[j] != 1)
        clean_dist = np.linalg.norm(xyz_clean[i] - xyz_clean[j], axis=1)
        outside_cache = (clean_dist > args.cache_cutoff) & non_hh
        bond_set = {
            tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
            for bond in clean.GetBonds()
        }
        is_true_bond = np.fromiter(
            ((int(a), int(b)) in bond_set for a, b in zip(i, j)),
            dtype=bool, count=len(i),
        )
        clean_active = (clean_dist <= args.active_cutoff) & non_hh
        heavy_heavy = (np.asarray(atoms)[i] != 1) & (np.asarray(atoms)[j] != 1)
        clean_counts["n_molecules"] += 1
        clean_counts["active_pairs"] += int(clean_active.sum())
        clean_counts["active_heavy_heavy_pairs"] += int((clean_active & heavy_heavy).sum())
        clean_counts["active_heavy_heavy_true_bond_pairs"] += int(
            (clean_active & heavy_heavy & is_true_bond).sum()
        )
        clean_counts["active_true_bond_pairs"] += int(
            (clean_active & is_true_bond).sum()
        )
        clean_counts["reference_bond_pairs"] += int((is_true_bond & non_hh).sum())
        for sigma, stream in streams.items():
            noisy = next(stream, sentinel)
            if noisy is sentinel:
                raise ValueError(f"Noisy SDF ended before clean SDF at {index}")
            if noisy.GetNumAtoms() != n_atoms:
                raise ValueError(f"Atom count mismatch at molecule {index}")
            if [atom.GetAtomicNum() for atom in noisy.GetAtoms()] != atoms:
                raise ValueError(f"Atom order mismatch at molecule {index}")
            if (clean.HasProp("geom_mol_idx") and noisy.HasProp("geom_mol_idx")
                    and clean.GetProp("geom_mol_idx") != noisy.GetProp("geom_mol_idx")):
                raise ValueError(f"Molecule id mismatch at molecule {index}")
            xyz_noisy = np.asarray(noisy.GetConformer().GetPositions())
            noisy_dist = np.linalg.norm(xyz_noisy[i] - xyz_noisy[j], axis=1)
            active = (noisy_dist <= args.active_cutoff) & non_hh
            missing = active & outside_cache
            missing_true = missing & is_true_bond
            stats = counts[sigma]
            stats["n_molecules"] += 1
            stats["active_pairs"] += int(active.sum())
            stats["active_heavy_heavy_pairs"] += int((active & heavy_heavy).sum())
            stats["molecules_with_missing_active_pair"] += int(bool(missing.any()))
            stats["missing_active_pairs"] += int(missing.sum())
            stats["missing_true_bond_pairs"] += int(missing_true.sum())
            stats["molecules_with_missing_true_bond"] += int(bool(missing_true.any()))
            stats["missing_heavy_heavy_pairs"] += int(
                (missing & (np.asarray(atoms)[i] != 1)
                 & (np.asarray(atoms)[j] != 1)).sum()
            )
            stats["missing_hydrogen_heavy_pairs"] += int(
                (missing & ((np.asarray(atoms)[i] == 1)
                            ^ (np.asarray(atoms)[j] == 1))).sum()
            )
        if (index + 1) % 5000 == 0:
            print(f"Audited {index + 1:,} molecules", flush=True)
    for sigma, stream in streams.items():
        if next(stream, sentinel) is not sentinel:
            raise ValueError(f"Noisy SDF has extra molecules at sigma={sigma}")
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "clean_sdf": str(clean_path),
        "noisy_sdfs": {str(sigma): str(path) for sigma, path in noisy_paths.items()},
        "cache_cutoff_angstrom": args.cache_cutoff,
        "active_cutoff_angstrom": args.active_cutoff,
        "pair_definition": "unordered non-H--H atom pairs; SDF reference bonds",
        "clean_candidate_graph": clean_counts,
        "by_sigma": counts,
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for sigma, stats in counts.items():
        print(f"sigma={sigma}: {stats}")
    print(f"clean: {clean_counts}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
