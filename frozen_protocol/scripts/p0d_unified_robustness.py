#!/usr/bin/env python3
"""Build and run the paper's unified 26,940-molecule robustness benchmark.

The fixed validation cache stores a clean candidate graph.  Adding coordinate
noise to that cache does not rebuild candidate edges, so it is unsuitable for
an end-to-end robustness claim.  This harness instead materializes one SDF per
noise level and makes every method read those exact coordinates.  BondNet's SDF
loader therefore rebuilds its distance-based candidate graph after perturbation.

Examples
--------
  python scripts/p0d_unified_robustness.py build
  python scripts/p0d_unified_robustness.py run --methods bondnet-aug bondnet-clean rdkit openbabel
  python scripts/p0d_unified_robustness.py table
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = REPO / "data" / "geom_drugs_all_random1_re10_c25_explicit_h_fixed_val.pt"
DEFAULT_SOURCE = REPO / "data" / "geom_drugs_all_random1_rel10.sdf"
DEFAULT_DATA_DIR = REPO / "data" / "robustness_26940"
DEFAULT_RESULTS_DIR = REPO / "results" / "robustness_26940_unified"
DEFAULT_STAGE1 = REPO / "checkpoints" / "geom_drugs_all_random1_re10_stage1_c25" / "best.pt"
DEFAULT_STAGE2_AUG = REPO / "checkpoints" / "stage2_c25_officialcache_noise010" / "best.pt"
DEFAULT_STAGE2_CLEAN = (
    REPO / "checkpoints" / "geom_drugs_all_random1_re10_stage2_c25_officialcache" / "best.pt"
)
DEFAULT_SIGMAS = (0.0, 0.05, 0.10, 0.15, 0.20)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _sigma_tag(sigma: float) -> str:
    return f"{round(100 * sigma):03d}"


def _sdf_path(data_dir: Path, sigma: float) -> Path:
    return data_dir / f"sigma_{_sigma_tag(sigma)}.sdf"


def _load_p0c():
    path = REPO / "scripts" / "p0c_yuelbond_compare.py"
    spec = importlib.util.spec_from_file_location("p0c_yuelbond_compare", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_cache(path: Path, expected_split: str | None = None):
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "samples" not in payload:
        raise ValueError(f"unexpected cache payload: {path}")
    samples = payload["samples"]
    metadata = payload.get("metadata", {})
    actual_split = metadata.get("kept_split", metadata.get("selected_split"))
    if expected_split and actual_split != expected_split:
        raise ValueError(
            f"cache split is {actual_split!r}, expected {expected_split!r}"
        )
    return samples, metadata


def _torch_batched_noise(samples, sigmas, seed: int, batch_size: int, device_name: str):
    """Reproduce evaluate.py's fixed-sigma, batched coordinate perturbations.

    Random draws depend on batch boundaries in PyTorch, so the benchmark records
    and reuses the evaluation batch size instead of generating noise molecule by
    molecule.  Returned arrays are indexed in cache order.
    """
    import torch

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--noise-device cuda requested but CUDA is unavailable")
    output = {}
    for sigma in sigmas:
        sigma = float(sigma)
        if sigma == 0.0:
            output[sigma] = None
            continue
        sigma_seed = int(seed) + int(round(sigma * 10000.0))
        torch.manual_seed(sigma_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sigma_seed)
        per_sample = []
        for start in range(0, len(samples), int(batch_size)):
            chunk = samples[start:start + int(batch_size)]
            coords = [sample["coord"].detach().cpu() for sample in chunk]
            counts = [int(coord.shape[0]) for coord in coords]
            joined = torch.cat(coords, dim=0).to(device)
            noise = torch.randn_like(joined) * sigma
            offset = 0
            for count in counts:
                per_sample.append(noise[offset:offset + count].cpu().numpy())
                offset += count
        output[sigma] = per_sample
    return output


def _torch_keyed_noise(samples, sigmas, seed: int):
    """Match GaussianNoiseAugment's molecule-keyed fixed-sigma protocol."""
    import torch

    output = {}
    modulus = 2**63 - 1
    for sigma in sigmas:
        sigma = float(sigma)
        if sigma == 0.0:
            output[sigma] = None
            continue
        base_seed = int(seed) + int(round(sigma * 10000.0))
        per_sample = []
        for sample in samples:
            mol_id = int(sample.get("geom_mol_idx", sample.get("mol_idx", -1)))
            if mol_id < 0:
                raise ValueError("torch-keyed noise requires a persistent molecule id")
            mixed_seed = (
                base_seed * 1_000_003
                + mol_id * 9_176
                + 0x5DEECE66D
            ) % modulus
            generator = torch.Generator(device="cpu")
            generator.manual_seed(mixed_seed)
            # GaussianNoiseAugment samples u even when sigma_min == sigma_max;
            # consume the identical draw before generating Cartesian noise.
            torch.rand((), generator=generator)
            coord = sample["coord"].detach().cpu()
            noise = torch.randn(
                coord.shape, generator=generator, dtype=coord.dtype
            ) * sigma
            per_sample.append(noise.numpy())
        output[sigma] = per_sample
    return output


def _sample_signature(sample: dict) -> tuple[list[int], np.ndarray]:
    elems = sample["elems"]
    coord = sample["coord"]
    if hasattr(elems, "detach"):
        elems = elems.detach().cpu().numpy()
    if hasattr(coord, "detach"):
        coord = coord.detach().cpu().numpy()
    return [int(x) for x in np.asarray(elems).tolist()], np.asarray(coord, dtype=np.float64)


def _mol_signature(mol) -> tuple[list[int], np.ndarray]:
    elems = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    coord = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    return elems, coord


def _noise_seed(seed: int, geom_mol_idx: int, geom_conf_idx: str, sigma: float) -> np.random.SeedSequence:
    # SeedSequence accepts uint32 entropy. Hashing the conformer identifier also
    # supports non-integer GEOM identifiers without Python's randomized hash().
    conf_hash = int.from_bytes(hashlib.sha256(str(geom_conf_idx).encode()).digest()[:4], "little")
    return np.random.SeedSequence(
        [int(seed) & 0xFFFFFFFF, int(geom_mol_idx) & 0xFFFFFFFF,
         conf_hash, int(round(sigma * 1000000)) & 0xFFFFFFFF]
    )


def cmd_build(args: argparse.Namespace) -> None:
    from rdkit import Chem

    cache = Path(args.cache).resolve()
    source = Path(args.source).resolve()
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    for path in (cache, source):
        if not path.exists():
            raise FileNotFoundError(path)

    samples, metadata = _load_cache(cache, args.expected_split)
    by_id = {}
    ordered_ids = []
    for pos, sample in enumerate(samples):
        geom_id = int(sample["geom_mol_idx"])
        if geom_id in by_id:
            raise ValueError(f"duplicate geom_mol_idx={geom_id}; this harness expects random1")
        by_id[geom_id] = (pos, sample)
        ordered_ids.append(geom_id)
    if len(samples) != int(metadata.get("n_molecules", len(samples))):
        raise ValueError("cache metadata count disagrees with sample count")

    sigmas = tuple(float(s) for s in args.sigmas)
    torch_noise = None
    if args.noise_protocol == "torch-batched":
        torch_noise = _torch_batched_noise(
            samples, sigmas, args.seed, args.noise_batch_size, args.noise_device
        )
    elif args.noise_protocol == "torch-keyed":
        torch_noise = _torch_keyed_noise(samples, sigmas, args.seed)
    outputs = {sigma: _sdf_path(data_dir, sigma) for sigma in sigmas}
    for path in outputs.values():
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to rebuild")
    writers = {sigma: Chem.SDWriter(str(path)) for sigma, path in outputs.items()}

    found = {}
    found_order = []
    max_coord_error = 0.0
    n_records = 0
    try:
        supplier = Chem.ForwardSDMolSupplier(str(source), removeHs=False, sanitize=False)
        for mol in supplier:
            n_records += 1
            if mol is None or not mol.HasProp("geom_mol_idx"):
                continue
            try:
                geom_id = int(mol.GetProp("geom_mol_idx"))
            except ValueError:
                continue
            if geom_id not in by_id:
                continue
            if geom_id in found:
                raise ValueError(f"source SDF contains duplicate target geom_mol_idx={geom_id}")
            pos, sample = by_id[geom_id]
            cache_elems, cache_coord = _sample_signature(sample)
            mol_elems, mol_coord = _mol_signature(mol)
            if cache_elems != mol_elems:
                raise ValueError(
                    f"element/order mismatch for geom_mol_idx={geom_id}: "
                    f"cache={cache_elems}, sdf={mol_elems}"
                )
            if cache_coord.shape != mol_coord.shape:
                raise ValueError(f"coordinate shape mismatch for geom_mol_idx={geom_id}")
            coord_error = float(np.max(np.abs(cache_coord - mol_coord)))
            max_coord_error = max(max_coord_error, coord_error)
            if coord_error > args.coord_tolerance:
                raise ValueError(
                    f"coordinate mismatch for geom_mol_idx={geom_id}: max |delta|={coord_error:.6g} A"
                )

            conf_id = sample.get("geom_conf_idx", mol.GetProp("geom_conf_idx") if mol.HasProp("geom_conf_idx") else 0)
            found[geom_id] = pos
            found_order.append(geom_id)
            for sigma, writer in writers.items():
                out_mol = Chem.Mol(mol)
                if sigma > 0:
                    if args.noise_protocol in ("torch-batched", "torch-keyed"):
                        noisy = mol_coord + torch_noise[sigma][pos]
                    else:
                        rng = np.random.default_rng(_noise_seed(args.seed, geom_id, str(conf_id), sigma))
                        noisy = mol_coord + rng.normal(0.0, sigma, size=mol_coord.shape)
                    conf = out_mol.GetConformer()
                    for atom_idx, xyz in enumerate(noisy):
                        conf.SetAtomPosition(atom_idx, tuple(float(x) for x in xyz))
                out_mol.SetProp("robustness_sigma_angstrom", f"{sigma:.4f}")
                out_mol.SetIntProp("robustness_seed", int(args.seed))
                out_mol.SetIntProp("robustness_cache_position", int(pos))
                writer.write(out_mol)
            if len(found) % args.progress_every == 0:
                print(f"[build] matched {len(found):,}/{len(samples):,} (source records {n_records:,})", flush=True)
            if len(found) == len(samples):
                break
    finally:
        for writer in writers.values():
            writer.close()

    missing = [geom_id for geom_id in ordered_ids if geom_id not in found]
    if missing:
        raise RuntimeError(f"missing {len(missing)} cache molecules in source SDF; first IDs: {missing[:20]}")

    # SDF order follows source order.  It should equal cache order for random1;
    # make this an explicit invariant because prediction indices depend on it.
    if found_order != ordered_ids:
        raise RuntimeError("source/cache order mismatch; outputs cannot safely share positional predictions")

    p0c = _load_p0c()
    output_meta = {}
    for sigma, path in outputs.items():
        mols = p0c._read_valid_molecules(str(path))
        if len(mols) != len(samples):
            raise RuntimeError(f"{path} contains {len(mols)} scorable molecules, expected {len(samples)}")
        actual_ids = [int(m.GetProp("geom_mol_idx")) for m in mols]
        if actual_ids != ordered_ids:
            raise RuntimeError(f"molecule order changed while reading {path}")
        output_meta[f"{sigma:.2f}"] = {
            "path": os.path.relpath(path, REPO),
            "sha256": _sha256(path),
            "n_molecules": len(mols),
        }

    id_digest = hashlib.sha256(",".join(map(str, ordered_ids)).encode()).hexdigest()
    determinism = (
        f"torch.manual_seed(global_seed + round(10000*sigma)); torch.randn_like on "
        f"cache-order batches of {args.noise_batch_size} molecules on {args.noise_device}"
        if args.noise_protocol == "torch-batched" else
        "GaussianNoiseAugment-compatible torch.Generator keyed by "
        "(global_seed + round(10000*sigma), geom_mol_idx)"
        if args.noise_protocol == "torch-keyed" else
        "SeedSequence(global_seed, geom_mol_idx, sha256(geom_conf_idx), sigma_microangstrom)"
    )
    manifest = {
        "protocol": "materialized-coordinate-noise; candidate graph rebuilt independently by each method",
        "noise": {
            "distribution": "iid Gaussian per Cartesian coordinate",
            "applied_to": "all atoms including explicit hydrogen",
            "seed": int(args.seed),
            "sigmas_angstrom": list(sigmas),
            "determinism": determinism,
            "noise_protocol": args.noise_protocol,
            "noise_batch_size": args.noise_batch_size if args.noise_protocol == "torch-batched" else None,
            "noise_device": args.noise_device if args.noise_protocol == "torch-batched" else None,
        },
        "cache": {
            "path": os.path.relpath(cache, REPO),
            "sha256": _sha256(cache),
            "metadata": metadata,
        },
        "source": {"path": os.path.relpath(source, REPO), "sha256": _sha256(source)},
        "cohort": {
            "n_molecules": len(samples),
            "id_key": "geom_mol_idx",
            "ordered_id_sha256": id_digest,
            "max_cache_source_coordinate_error_angstrom": max_coord_error,
        },
        "outputs": output_meta,
    }
    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[build] wrote {len(outputs)} matched SDFs and {manifest_path}")


def _score(p0c, sdf: Path, pred: Path, name: str, output: Path) -> None:
    p0c.cmd_score(SimpleNamespace(
        sdf=str(sdf), pred=str(pred), name=name, output=str(output), allow_unfingerprinted=False
    ))


def cmd_run(args: argparse.Namespace) -> None:
    p0c = _load_p0c()
    data_dir = Path(args.data_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    pred_dir = results_dir / "predictions"
    score_dir = results_dir / "scores"
    pred_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path}; run the build subcommand first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    methods = args.methods
    model_paths = {
        "bondnet-aug": (Path(args.stage1).resolve(), Path(args.stage2_aug).resolve()),
        "bondnet-clean": (Path(args.stage1).resolve(), Path(args.stage2_clean).resolve()),
    }
    for method in methods:
        if method in model_paths:
            for checkpoint in model_paths[method]:
                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)

    for sigma in (float(s) for s in args.sigmas):
        sdf = _sdf_path(data_dir, sigma)
        key = f"{sigma:.2f}"
        if not sdf.exists() or _sha256(sdf) != manifest["outputs"][key]["sha256"]:
            raise RuntimeError(f"missing or modified benchmark SDF: {sdf}")
        tag = _sigma_tag(sigma)
        print(f"\n[run] sigma={sigma:.2f} SDF={sdf}", flush=True)

        for method in methods:
            score_path = score_dir / f"{method}_sigma_{tag}.json"
            pred_path = pred_dir / f"{method}_sigma_{tag}_preds.json"
            if score_path.exists() and not args.overwrite:
                print(f"[skip] completed score exists: {score_path}")
                continue
            if method in model_paths:
                stage1, stage2 = model_paths[method]
                p0c._export_bondnet_predictions(
                    str(stage1), str(stage2), str(sdf), str(pred_path),
                    batch_size=args.batch_size, device_name=args.device,
                )
                display = "BondNet-aug" if method == "bondnet-aug" else "BondNet-clean"
                _score(p0c, sdf, pred_path, display, score_path)
            elif method in ("rdkit", "openbabel"):
                p0c.cmd_rule(SimpleNamespace(sdf=str(sdf), methods=[method]))
                generated = Path(p0c._prediction_path(method, str(sdf)))
                if not generated.exists():
                    raise RuntimeError(f"{method} did not produce {generated}")
                generated.replace(pred_path)
                _score(p0c, sdf, pred_path, "RDKit" if method == "rdkit" else "OpenBabel", score_path)
            else:
                raise ValueError(f"unknown method: {method}")
    print(f"\n[run] completed requested methods; scores are under {score_dir}")


def cmd_table(args: argparse.Namespace) -> None:
    score_dir = Path(args.results_dir).resolve() / "scores"
    rows = []
    for path in sorted(score_dir.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        stem = path.stem
        if "_sigma_" not in stem:
            continue
        method, tag = stem.rsplit("_sigma_", 1)
        sigma = int(tag) / 100.0
        display_name = {"RDKIT": "RDKit"}.get(report["name"], report["name"])
        row = {
            "sigma_angstrom": sigma,
            "method": display_name,
            "n_molecules": report["n_molecules"],
            "success_rate": report["success_rate"],
            "f1_macro_true_pair": report["f1_macro_true_pair"],
            "f1_macro_pipeline": report["f1_macro_pipeline"],
            "f1_macro_pipeline_conditional_success": report.get(
                "f1_macro_pipeline_conditional_success", report["f1_macro_pipeline"]
            ),
            "true_bond_exact_type_rate": report["true_bond_exact_type_rate"],
            "heavy_graph_exact_match": report["full_graph_exact_match"],
            "true_bond_exact_type_rate_conditional_success": report.get(
                "true_bond_exact_type_rate_conditional_success",
                report["true_bond_exact_type_rate"],
            ),
            "heavy_graph_exact_match_conditional_success": report.get(
                "full_graph_exact_match_conditional_success",
                report["full_graph_exact_match"],
            ),
            "extra_bonds_per_successful_molecule": report["extra_bonds_per_successful_molecule"],
        }
        for bond_type in ("single", "double", "triple", "aromatic"):
            row[f"pipeline_f1_{bond_type}"] = report["f1_pipeline_by_class"].get(bond_type, 0.0)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no score files under {score_dir}")
    order = {"BondNet-aug": 0, "BondNet-clean": 1, "RDKit": 2, "OpenBabel": 3}
    rows.sort(key=lambda row: (row["sigma_angstrom"], order.get(row["method"], 99)))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"sigma={row['sigma_angstrom']:.2f} {row['method']:<13} "
            f"success={100*row['success_rate']:6.2f}% pipe-F1={row['f1_macro_pipeline']:.4f} "
            f"pair-exact={100*row['true_bond_exact_type_rate']:6.2f}% "
            f"HH-exact={100*row['heavy_graph_exact_match']:6.2f}% "
            f"extra/mol={row['extra_bonds_per_successful_molecule']:.4f}"
        )
    print(f"[table] wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="materialize the exact fixed-validation cohort")
    build.add_argument("--cache", default=str(DEFAULT_CACHE))
    build.add_argument("--source", default=str(DEFAULT_SOURCE))
    build.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    build.add_argument("--sigmas", nargs="+", type=float, default=list(DEFAULT_SIGMAS))
    build.add_argument("--seed", type=int, default=20260713)
    build.add_argument("--expected-split", choices=["train", "val", "test"], default="val")
    build.add_argument(
        "--noise-protocol", choices=["seedsequence-per-molecule", "torch-batched", "torch-keyed"],
        default="seedsequence-per-molecule",
    )
    build.add_argument("--noise-batch-size", type=int, default=128)
    build.add_argument("--noise-device", choices=["cpu", "cuda"], default="cpu")
    build.add_argument("--coord-tolerance", type=float, default=1e-3)
    build.add_argument("--progress-every", type=int, default=2500)
    build.add_argument("--overwrite", action="store_true")
    build.set_defaults(func=cmd_build)

    run = sub.add_parser("run", help="run methods and the shared scorer")
    run.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    run.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    run.add_argument("--sigmas", nargs="+", type=float, default=list(DEFAULT_SIGMAS))
    run.add_argument(
        "--methods", nargs="+",
        choices=["bondnet-aug", "bondnet-clean", "rdkit", "openbabel"],
        default=["bondnet-aug", "bondnet-clean", "rdkit", "openbabel"],
    )
    run.add_argument("--stage1", default=str(DEFAULT_STAGE1))
    run.add_argument("--stage2-aug", default=str(DEFAULT_STAGE2_AUG))
    run.add_argument("--stage2-clean", default=str(DEFAULT_STAGE2_CLEAN))
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--device", default="auto")
    run.add_argument("--overwrite", action="store_true")
    run.set_defaults(func=cmd_run)

    table = sub.add_parser("table", help="combine unified score JSONs")
    table.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    table.add_argument("--output", default=str(DEFAULT_RESULTS_DIR / "summary.csv"))
    table.set_defaults(func=cmd_table)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
