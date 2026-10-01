"""
Experiment runner based on the project experiment plan.

This script orchestrates train.py, train_stage2.py, and evaluate.py for:
- In-domain baselines (A1/A2/A3)
- GEOM noise robustness (B)
- Cross-dataset generalization (C1..C5)
- Joint-training evaluations (D1/D2) when merged caches are provided
- Ablation entrypoints (E1/E2/E3)
- Cross-dataset matrix summary (F1) from finished result files

Typical usage:
  python scripts/run_experiment_plan.py --dry-run --experiments A C
  python scripts/run_experiment_plan.py --experiments A2 C3 --device cuda --num_workers 2
  python scripts/run_experiment_plan.py --experiments F1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DatasetSpec:
    name: str
    cache_path: Optional[Path]
    split_key: Optional[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run experiment plan matrix")

    p.add_argument("--python", default=sys.executable, help="Python executable used to launch sub-jobs")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    p.add_argument("--stop-on-error", action="store_true", help="Stop immediately if any command fails")
    p.add_argument("--skip-existing", action="store_true", help="Skip training/evaluation if expected output exists")

    p.add_argument(
        "--experiments",
        nargs="+",
        default=["A", "B", "C", "E", "F1"],
        help=(
            "Experiment ids/groups to run. Supported ids: "
            "A1 A2 A3 B C1 C2 C3 C4 C5 D1 D2 E1 E2 E3 F1. "
            "Groups: A B C D E."
        ),
    )

    # Dataset cache paths (recommended)
    p.add_argument("--qm9-cache", default="checkpoints/qm9_one_stage/feature_cache_c2.5_explicit_h.pt")
    p.add_argument("--geom-cache", default="data/geom_drugs_all_random1_re10_c25_explicit_h_fixed.pt")
    p.add_argument("--pubchem-cache", default="data/pubchem3d_1M_c25_explicit_h_sharded")

    # Optional merged caches for joint training (D1/D2)
    p.add_argument("--geom-pubchem-cache", default=None)
    p.add_argument("--qm9-pubchem-cache", default=None)

    # Shared train settings
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--dataloader-timeout", type=float, default=120.0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs-stage1", type=int, default=10)
    p.add_argument("--epochs-stage2", type=int, default=100)
    p.add_argument("--cutoff", type=float, default=2.5)
    p.add_argument("--h-cutoff", type=float, default=2.5)
    p.add_argument("--edge-embedding-size", type=int, default=32)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--num-interactions", type=int, default=4)
    p.add_argument("--lr-stage1", type=float, default=3e-4)
    p.add_argument("--lr-stage2", type=float, default=5e-4)
    p.add_argument("--grad-clip-stage1", type=float, default=0.5)
    p.add_argument("--grad-clip-stage2", type=float, default=1.0)
    p.add_argument("--val-split", type=float, default=0.05)
    p.add_argument("--e2e-val-max-mols", type=int, default=8192)
    p.add_argument("--cache-max-samples", type=int, default=None)

    # Noise settings
    p.add_argument("--train-noise-max", type=float, default=0.15)
    p.add_argument("--train-noise-min", type=float, default=0.0)
    p.add_argument("--noise-levels", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.15, 0.20])

    # Output roots
    p.add_argument("--checkpoints-root", default="checkpoints/exp_plan")
    p.add_argument("--results-root", default="results/exp_plan")

    return p.parse_args()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _run_cmd(cmd: List[str], dry_run: bool) -> int:
    print("\n[CMD] " + " ".join(cmd))
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=ROOT)
    return int(proc.returncode)


def _expand_experiment_tokens(tokens: Iterable[str]) -> List[str]:
    groups = {
        "A": ["A1", "A2", "A3"],
        "B": ["B"],
        "C": ["C1", "C2", "C3", "C4", "C5"],
        "D": ["D1", "D2"],
        "E": ["E1", "E2", "E3"],
        "F": ["F1"],
    }
    expanded: List[str] = []
    for t in tokens:
        key = t.strip().upper()
        if key in groups:
            expanded.extend(groups[key])
        else:
            expanded.append(key)
    # Keep order and deduplicate
    seen = set()
    out = []
    for x in expanded:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _two_stage_paths(exp_id: str, train_name: str, test_name: str, ckpt_root: Path, res_root: Path) -> Dict[str, Path]:
    tag = f"{exp_id}_{train_name}_to_{test_name}" if train_name != test_name else f"{exp_id}_{train_name}"
    stage1_dir = ckpt_root / f"{tag}_stage1"
    stage2_dir = ckpt_root / f"{tag}_stage2"
    result_dir = res_root / tag
    return {
        "stage1_dir": stage1_dir,
        "stage2_dir": stage2_dir,
        "stage1_best": stage1_dir / "best_e2e.pt",
        "stage1_best_fallback": stage1_dir / "best.pt",
        "stage2_best": stage2_dir / "best_e2e.pt",
        "result_dir": result_dir,
        "result_json": result_dir / "results.json",
    }


def _pick_existing(*paths: Path) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _append_common_dataloader_flags(cmd: List[str], args: argparse.Namespace) -> None:
    cmd.extend([
        "--num_workers", str(args.num_workers),
        "--prefetch_factor", str(args.prefetch_factor),
        "--dataloader_timeout", str(args.dataloader_timeout),
    ])


def _train_stage1_cmd(args: argparse.Namespace, cache_path: Path, out_dir: Path, split_key: Optional[str], connectivity_only: bool) -> List[str]:
    cmd = [
        args.python,
        "train.py",
        "--cache_path", str(cache_path),
        "--output_dir", str(out_dir),
        "--explicit_h",
        "--cutoff", str(args.cutoff),
        "--h_cutoff", str(args.h_cutoff),
        "--val_split", str(args.val_split),
        "--num_interactions", str(args.num_interactions),
        "--hidden_size", str(args.hidden_size),
        "--edge_embedding_size", str(args.edge_embedding_size),
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs_stage1),
        "--lr", str(args.lr_stage1),
        "--grad_clip", str(args.grad_clip_stage1),
        "--device", args.device,
    ]
    if split_key:
        cmd.extend(["--split_key", split_key])
    if args.cache_max_samples is not None:
        cmd.extend(["--cache_max_samples", str(args.cache_max_samples)])
    if connectivity_only:
        cmd.append("--connectivity_only")
    _append_common_dataloader_flags(cmd, args)
    return cmd


def _train_stage2_cmd(args: argparse.Namespace, cache_path: Path, stage1_ckpt: Path, out_dir: Path, split_key: Optional[str], noise_max: float) -> List[str]:
    cmd = [
        args.python,
        "train_stage2.py",
        "--stage1_ckpt", str(stage1_ckpt),
        "--cache_path", str(cache_path),
        "--output_dir", str(out_dir),
        "--val_split", str(args.val_split),
        "--num_layers", "4",
        "--edge_embedding_size", str(args.edge_embedding_size),
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs_stage2),
        "--lr", str(args.lr_stage2),
        "--grad_clip", str(args.grad_clip_stage2),
        "--noise_min", str(args.train_noise_min),
        "--noise_max", str(noise_max),
        "--e2e_val_max_mols", str(args.e2e_val_max_mols),
        "--device", args.device,
    ]
    if split_key:
        cmd.extend(["--split_key", split_key])
    if args.cache_max_samples is not None:
        cmd.extend(["--cache_max_samples", str(args.cache_max_samples)])
    _append_common_dataloader_flags(cmd, args)
    return cmd


def _evaluate_cmd(args: argparse.Namespace, test_cache: Path, stage1_ckpt: Path, stage2_ckpt: Optional[Path], out_dir: Path, split_key: Optional[str], noise_levels: List[float], run_baselines: bool = False) -> List[str]:
    cmd = [
        args.python,
        "evaluate.py",
        "--checkpoint", str(stage1_ckpt),
        "--cache_path", str(test_cache),
        "--eval_split", "val",
        "--noise_levels",
    ]
    cmd.extend([str(x) for x in noise_levels])
    cmd.extend([
        "--output_dir", str(out_dir),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--device", args.device,
    ])
    if stage2_ckpt is not None:
        cmd.extend(["--stage2_ckpt", str(stage2_ckpt)])
    if split_key:
        cmd.extend(["--split_key", split_key])
    if run_baselines:
        cmd.append("--run_baselines")
    return cmd


def _rule_baseline_cmd(args: argparse.Namespace, data_path: Path, out_dir: Path, noise_levels: List[float]) -> List[str]:
    cmd = [
        args.python,
        "scripts/evaluate_rule_baselines.py",
        "--data_path", str(data_path),
        "--methods", "rdkit", "openbabel",
        "--noise_levels",
    ]
    cmd.extend([str(x) for x in noise_levels])
    cmd.extend([
        "--output_dir", str(out_dir),
    ])
    return cmd


def _load_f1_macro(result_json: Path) -> Optional[float]:
    if not result_json.exists():
        return None
    with open(result_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return None
    for method_name in ("BondNet", "bondnet", "BONDNET"):
        if method_name in payload and payload[method_name]:
            row = payload[method_name][0]
            if isinstance(row, dict) and "f1_macro" in row:
                return float(row["f1_macro"])
    return None


def _run_two_stage(
    exp_id: str,
    args: argparse.Namespace,
    train_spec: DatasetSpec,
    test_spec: DatasetSpec,
    noise_levels_eval: List[float],
    noise_max_train: float,
    run_baselines: bool,
) -> List[Tuple[str, int]]:
    statuses: List[Tuple[str, int]] = []
    ckpt_root = ROOT / args.checkpoints_root
    res_root = ROOT / args.results_root
    paths = _two_stage_paths(exp_id, train_spec.name, test_spec.name, ckpt_root, res_root)

    paths["stage1_dir"].mkdir(parents=True, exist_ok=True)
    paths["stage2_dir"].mkdir(parents=True, exist_ok=True)
    paths["result_dir"].mkdir(parents=True, exist_ok=True)

    # Stage 1 (connectivity only)
    if args.skip_existing and (_pick_existing(paths["stage1_best"], paths["stage1_best_fallback"]) is not None):
        statuses.append((f"{exp_id}:stage1:skip", 0))
    else:
        cmd = _train_stage1_cmd(args, train_spec.cache_path, paths["stage1_dir"], train_spec.split_key, connectivity_only=True)
        rc = _run_cmd(cmd, args.dry_run)
        statuses.append((f"{exp_id}:stage1", rc))
        if rc != 0 and args.stop_on_error:
            return statuses

    stage1_ckpt = _pick_existing(paths["stage1_best"], paths["stage1_best_fallback"])
    if stage1_ckpt is None:
        stage1_ckpt = paths["stage1_best"]

    # Stage 2
    if args.skip_existing and paths["stage2_best"].exists():
        statuses.append((f"{exp_id}:stage2:skip", 0))
    else:
        cmd = _train_stage2_cmd(args, train_spec.cache_path, stage1_ckpt, paths["stage2_dir"], train_spec.split_key, noise_max=noise_max_train)
        rc = _run_cmd(cmd, args.dry_run)
        statuses.append((f"{exp_id}:stage2", rc))
        if rc != 0 and args.stop_on_error:
            return statuses

    # Evaluate
    if args.skip_existing and paths["result_json"].exists():
        statuses.append((f"{exp_id}:eval:skip", 0))
    else:
        stage2_ckpt = paths["stage2_best"]
        cmd = _evaluate_cmd(
            args,
            test_spec.cache_path,
            stage1_ckpt,
            stage2_ckpt,
            paths["result_dir"],
            test_spec.split_key,
            noise_levels_eval,
            run_baselines=run_baselines,
        )
        rc = _run_cmd(cmd, args.dry_run)
        statuses.append((f"{exp_id}:eval", rc))
        if rc != 0 and args.stop_on_error:
            return statuses

    return statuses


def _run_one_stage(
    exp_id: str,
    args: argparse.Namespace,
    train_spec: DatasetSpec,
    test_spec: DatasetSpec,
    noise_levels_eval: List[float],
) -> List[Tuple[str, int]]:
    statuses: List[Tuple[str, int]] = []
    ckpt_root = ROOT / args.checkpoints_root
    res_root = ROOT / args.results_root
    tag = f"{exp_id}_{train_spec.name}_to_{test_spec.name}_one_stage"
    stage1_dir = ckpt_root / tag
    stage1_best = stage1_dir / "best_e2e.pt"
    stage1_best_fallback = stage1_dir / "best.pt"
    result_dir = res_root / tag
    result_json = result_dir / "results.json"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and (_pick_existing(stage1_best, stage1_best_fallback) is not None):
        statuses.append((f"{exp_id}:one_stage_train:skip", 0))
    else:
        cmd = _train_stage1_cmd(args, train_spec.cache_path, stage1_dir, train_spec.split_key, connectivity_only=False)
        rc = _run_cmd(cmd, args.dry_run)
        statuses.append((f"{exp_id}:one_stage_train", rc))
        if rc != 0 and args.stop_on_error:
            return statuses

    stage1_ckpt = _pick_existing(stage1_best, stage1_best_fallback)
    if stage1_ckpt is None:
        stage1_ckpt = stage1_best

    if args.skip_existing and result_json.exists():
        statuses.append((f"{exp_id}:one_stage_eval:skip", 0))
    else:
        cmd = _evaluate_cmd(
            args,
            test_spec.cache_path,
            stage1_ckpt,
            stage2_ckpt=None,
            out_dir=result_dir,
            split_key=test_spec.split_key,
            noise_levels=noise_levels_eval,
            run_baselines=False,
        )
        rc = _run_cmd(cmd, args.dry_run)
        statuses.append((f"{exp_id}:one_stage_eval", rc))
        if rc != 0 and args.stop_on_error:
            return statuses

    return statuses


def _build_specs(args: argparse.Namespace) -> Dict[str, DatasetSpec]:
    return {
        "QM9": DatasetSpec("qm9", Path(args.qm9_cache) if args.qm9_cache else None, None),
        "GEOM": DatasetSpec("geom", Path(args.geom_cache) if args.geom_cache else None, "geom_mol_idx"),
        "PUBCHEM": DatasetSpec("pubchem", Path(args.pubchem_cache) if args.pubchem_cache else None, None),
        "GEOM_PUBCHEM": DatasetSpec("geom_pubchem", Path(args.geom_pubchem_cache) if args.geom_pubchem_cache else None, None),
        "QM9_PUBCHEM": DatasetSpec("qm9_pubchem", Path(args.qm9_pubchem_cache) if args.qm9_pubchem_cache else None, None),
    }


def _validate_cache(spec: DatasetSpec) -> Optional[str]:
    if spec.cache_path is None:
        return f"Missing cache path for {spec.name}."
    if not spec.cache_path.exists():
        return f"Cache path does not exist for {spec.name}: {spec.cache_path}"
    return None


def _write_matrix_summary(args: argparse.Namespace) -> int:
    res_root = ROOT / args.results_root
    out_csv = res_root / "F1_cross_dataset_matrix.csv"
    out_json = res_root / "F1_cross_dataset_matrix.json"
    res_root.mkdir(parents=True, exist_ok=True)

    experiments = {
        "C1_qm9_to_geom": res_root / "C1_qm9_to_geom" / "results.json",
        "C2_geom_to_qm9": res_root / "C2_geom_to_qm9" / "results.json",
        "C3_pubchem_to_geom": res_root / "C3_pubchem_to_geom" / "results.json",
        "C4_geom_to_pubchem": res_root / "C4_geom_to_pubchem" / "results.json",
        "C5_qm9_to_pubchem": res_root / "C5_qm9_to_pubchem" / "results.json",
        "A1_qm9": res_root / "A1_qm9" / "results.json",
        "A2_geom": res_root / "A2_geom" / "results.json",
        "A3_pubchem": res_root / "A3_pubchem" / "results.json",
    }

    rows = []
    for name, path in experiments.items():
        rows.append({"experiment": name, "f1_macro": _load_f1_macro(path), "path": str(path)})

    matrix = {
        "rows": rows,
        "note": "f1_macro extracted from evaluate.py output results.json",
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2)

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("experiment,f1_macro,path\n")
        for r in rows:
            val = "" if r["f1_macro"] is None else f"{r['f1_macro']:.6f}"
            f.write(f"{r['experiment']},{val},{r['path']}\n")

    print(f"[INFO] Wrote matrix summary: {out_csv}")
    print(f"[INFO] Wrote matrix summary: {out_json}")
    return 0


def main() -> int:
    args = parse_args()
    exp_ids = _expand_experiment_tokens(args.experiments)
    specs = _build_specs(args)

    statuses: List[Tuple[str, int]] = []

    def need(spec_name: str) -> bool:
        err = _validate_cache(specs[spec_name])
        if err:
            print(f"[WARN] {err}")
            return False
        return True

    for exp_id in exp_ids:
        if exp_id == "A1":
            if need("QM9"):
                statuses += _run_two_stage("A1", args, specs["QM9"], specs["QM9"], [0.0], noise_max_train=0.0, run_baselines=False)
        elif exp_id == "A2":
            if need("GEOM"):
                statuses += _run_two_stage("A2", args, specs["GEOM"], specs["GEOM"], [0.0], noise_max_train=0.0, run_baselines=False)
        elif exp_id == "A3":
            if need("PUBCHEM"):
                statuses += _run_two_stage("A3", args, specs["PUBCHEM"], specs["PUBCHEM"], [0.0], noise_max_train=0.0, run_baselines=False)
        elif exp_id == "B":
            if need("GEOM"):
                # GEOM-only robustness + optional built-in baselines from evaluate.py.
                statuses += _run_two_stage("B", args, specs["GEOM"], specs["GEOM"], args.noise_levels, noise_max_train=args.train_noise_max, run_baselines=True)
        elif exp_id == "C1":
            if need("QM9") and need("GEOM"):
                statuses += _run_two_stage("C1", args, specs["QM9"], specs["GEOM"], [0.0], noise_max_train=0.0, run_baselines=False)
        elif exp_id == "C2":
            if need("GEOM") and need("QM9"):
                statuses += _run_two_stage("C2", args, specs["GEOM"], specs["QM9"], [0.0], noise_max_train=0.0, run_baselines=False)
        elif exp_id == "C3":
            if need("PUBCHEM") and need("GEOM"):
                statuses += _run_two_stage("C3", args, specs["PUBCHEM"], specs["GEOM"], [0.0], noise_max_train=0.0, run_baselines=False)
        elif exp_id == "C4":
            if need("GEOM") and need("PUBCHEM"):
                statuses += _run_two_stage("C4", args, specs["GEOM"], specs["PUBCHEM"], [0.0], noise_max_train=0.0, run_baselines=False)
        elif exp_id == "C5":
            if need("QM9") and need("PUBCHEM"):
                statuses += _run_two_stage("C5", args, specs["QM9"], specs["PUBCHEM"], [0.0], noise_max_train=0.0, run_baselines=False)
        elif exp_id == "D1":
            if need("GEOM_PUBCHEM") and need("GEOM"):
                statuses += _run_two_stage("D1", args, specs["GEOM_PUBCHEM"], specs["GEOM"], [0.0, 0.15], noise_max_train=args.train_noise_max, run_baselines=False)
            else:
                print("[WARN] Skip D1: provide --geom-pubchem-cache.")
        elif exp_id == "D2":
            if need("QM9_PUBCHEM") and need("GEOM"):
                statuses += _run_two_stage("D2", args, specs["QM9_PUBCHEM"], specs["GEOM"], [0.0, 0.15], noise_max_train=args.train_noise_max, run_baselines=False)
            else:
                print("[WARN] Skip D2: provide --qm9-pubchem-cache.")
        elif exp_id == "E1":
            # One-stage vs two-stage under sigma=0.15 on GEOM
            if need("GEOM"):
                statuses += _run_one_stage("E1", args, specs["GEOM"], specs["GEOM"], [0.15])
                statuses += _run_two_stage("E1", args, specs["GEOM"], specs["GEOM"], [0.15], noise_max_train=args.train_noise_max, run_baselines=False)
        elif exp_id == "E2":
            # Explicit-H ablation entrypoint.
            # Current train/eval scripts already default to explicit-H from cache.
            # For a no-H counterpart, provide a no-H cache and run as C-style comparison.
            print("[INFO] E2 requires paired explicit-H and no-H caches. Use C-style runs with the two caches.")
        elif exp_id == "E3":
            # Cutoff ablation entrypoint; run via train.py/train_stage2.py using cache paths built for each cutoff.
            print("[INFO] E3 requires per-cutoff caches (2.0 / 2.5 / 3.0). Re-run with corresponding --*-cache paths.")
        elif exp_id == "F1":
            rc = _write_matrix_summary(args)
            statuses.append(("F1:summary", rc))
        else:
            print(f"[WARN] Unknown experiment id: {exp_id}")

    print("\n===== Experiment Status =====")
    n_fail = 0
    for name, rc in statuses:
        s = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"{name:<36} {s}")
        if rc != 0:
            n_fail += 1

    if n_fail > 0:
        print(f"\n[ERROR] {n_fail} task(s) failed.")
        return 1

    print("\n[INFO] All requested tasks finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
