"""
Run a fair 2x2 comparison on the stitched diverse80k dataset.

Matrix:
  - backbones: PaiNN, Clof
  - training/evaluation: one-stage joint, hard two-stage

Defaults are intentionally narrow for the current question:
  - data/diverse_subset_80k.sdf
  - cutoff=2.5, h_cutoff=2.5
  - train noise max=0.1
  - eval noise level=0.1
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diverse80k PaiNN/Clof one-stage/two-stage study")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")

    p.add_argument("--data_path", default="data/diverse_subset_80k.sdf")
    p.add_argument("--cache_path", default="data/diverse_subset_80k_c25_h25_explicit_h_v14.pt")
    p.add_argument("--build_cache", action="store_true")
    p.add_argument("--cache_max_mols", type=int, default=None,
                   help="Optional cache-building molecule cap for smoke tests.")
    p.add_argument("--cache_max_samples", type=int, default=None,
                   help="Optional training/eval sample cap for smoke tests.")

    p.add_argument("--output_root", default="checkpoints/diverse_backbone_stage_study")
    p.add_argument("--results_root", default="results/diverse_backbone_stage_study")
    p.add_argument("--eval_split", choices=["all", "train", "val", "test"], default="test")
    p.add_argument("--split_key", default=None)
    p.add_argument("--train_groups", type=int, default=None)

    p.add_argument("--backbones", nargs="+", choices=["painn", "clof"], default=["painn", "clof"])
    p.add_argument("--one_stage_epochs", type=int, default=50)
    p.add_argument("--stage1_epochs", type=int, default=20)
    p.add_argument("--stage2_epochs", type=int, default=50)
    p.add_argument("--train_noise_max", type=float, default=0.1)
    p.add_argument("--train_noise_min", type=float, default=0.0)
    p.add_argument("--train_noise_clean_prob", type=float, default=0.0)
    p.add_argument("--noise_levels", nargs="+", type=float, default=[0.1])

    p.add_argument("--cutoff", type=float, default=2.5)
    p.add_argument("--h_cutoff", type=float, default=2.5)
    p.add_argument("--bond_cutoff", type=float, default=3.0)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--clof_coords_weight", type=float, default=0.1)
    p.add_argument("--num_interactions", type=int, default=4)
    p.add_argument("--edge_embedding_size", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr_stage1", type=float, default=3e-4)
    p.add_argument("--lr_stage2", type=float, default=5e-4)
    p.add_argument("--lr_one_stage", type=float, default=3e-4)
    p.add_argument("--grad_clip_stage1", type=float, default=0.5)
    p.add_argument("--grad_clip_stage2", type=float, default=1.0)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--auto_cls_weights", action="store_true", default=True)
    p.add_argument("--no_auto_cls_weights", dest="auto_cls_weights", action="store_false")
    p.add_argument("--auto_cls_weight_power", type=float, default=0.5)
    p.add_argument("--max_auto_weight", type=float, default=5.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--dataloader_timeout", type=float, default=120.0)
    p.add_argument("--e2e_val_max_mols", type=int, default=8192)
    p.add_argument("--e2e_val_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_error_analysis", action="store_true", default=True)
    p.add_argument("--no_save_error_analysis", dest="save_error_analysis", action="store_false")
    return p.parse_args()


def _resolve_python(python_arg: str) -> str:
    raw = str(python_arg or "").strip().strip('"').strip("'") or sys.executable
    candidate = Path(raw)
    if candidate.exists():
        return str(candidate.resolve())
    found = shutil.which(raw)
    if found:
        return found
    raise FileNotFoundError(
        f"Cannot find Python executable {python_arg!r}. "
        r"Pass it explicitly, e.g. --python D:\minconda\python.exe"
    )


def _run(cmd: Sequence[str], dry_run: bool) -> int:
    print("\n[CMD] " + " ".join(str(x) for x in cmd), flush=True)
    if dry_run:
        return 0
    try:
        return int(subprocess.run(list(cmd), cwd=ROOT).returncode)
    except FileNotFoundError as e:
        print(f"[ERROR] Failed to launch command: {e}", flush=True)
        return 127


def _maybe(cmd: Sequence[str], expected: Path, label: str, args: argparse.Namespace) -> Tuple[str, int]:
    if args.skip_existing and expected.exists():
        print(f"[SKIP] {label}: {expected}", flush=True)
        return label + ":skip", 0
    return label, _run(cmd, args.dry_run)


def _tag_float(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _append_split_flags(flags: List[str], args: argparse.Namespace) -> None:
    if args.split_key:
        flags.extend(["--split_key", args.split_key])
    if args.train_groups is not None:
        flags.extend(["--train_groups", str(args.train_groups)])
    if args.cache_max_samples is not None:
        flags.extend(["--cache_max_samples", str(args.cache_max_samples)])


def _train_common(args: argparse.Namespace, backbone: str) -> List[str]:
    flags = [
        "--cache_path", args.cache_path,
        "--explicit_h",
        "--cutoff", str(args.cutoff),
        "--h_cutoff", str(args.h_cutoff),
        "--num_interactions", str(args.num_interactions),
        "--hidden_size", str(args.hidden_size),
        "--backbone", backbone,
        "--clof_coords_weight", str(args.clof_coords_weight),
        "--edge_embedding_size", str(args.edge_embedding_size),
        "--batch_size", str(args.batch_size),
        "--device", args.device,
        "--num_workers", str(args.num_workers),
        "--prefetch_factor", str(args.prefetch_factor),
        "--e2e_val_max_mols", str(args.e2e_val_max_mols),
        "--e2e_val_every", str(args.e2e_val_every),
        "--seed", str(args.seed),
    ]
    _append_split_flags(flags, args)
    return flags


def _eval_common(args: argparse.Namespace, result_dir: Path) -> List[str]:
    flags = [
        "--cache_path", args.cache_path,
        "--explicit_h",
        "--cutoff", str(args.cutoff),
        "--h_cutoff", str(args.h_cutoff),
        "--eval_split", args.eval_split,
        "--noise_levels",
    ]
    flags.extend(str(x) for x in args.noise_levels)
    flags.extend([
        "--output_dir", str(result_dir),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--device", args.device,
    ])
    if args.split_key:
        flags.extend(["--split_key", args.split_key])
    if args.train_groups is not None:
        flags.extend(["--train_groups", str(args.train_groups)])
    if args.save_error_analysis:
        flags.append("--save_error_analysis")
    return flags


def _ckpt(out_dir: Path) -> Path:
    for name in ("best.pt", "best_e2e.pt"):
        path = out_dir / name
        if path.exists():
            return path
    return out_dir / "best.pt"


def build_commands(args: argparse.Namespace) -> List[Tuple[str, List[str], Path]]:
    out_root = ROOT / args.output_root
    res_root = ROOT / args.results_root
    commands: List[Tuple[str, List[str], Path]] = []

    cache_path = ROOT / args.cache_path
    if args.build_cache:
        build = [
            args.python, "scripts/precompute_features.py",
            "--data_path", args.data_path,
            "--output", args.cache_path,
            "--cutoff", str(args.cutoff),
            "--h_cutoff", str(args.h_cutoff),
            "--explicit_h",
            "--split_seed", str(args.seed),
        ]
        if args.cache_max_mols is not None:
            build.extend(["--max_mols", str(args.cache_max_mols)])
        commands.append(("build_diverse_cache", build, cache_path))

    noise_tag = _tag_float(args.train_noise_max)
    for backbone in args.backbones:
        base = f"{backbone}_noise{noise_tag}_c{_tag_float(args.cutoff)}_h{_tag_float(args.h_cutoff)}"
        one_dir = out_root / backbone / "one_stage"
        s1_dir = out_root / backbone / "two_stage_stage1"
        s2_dir = out_root / backbone / "two_stage_stage2"
        one_res = res_root / backbone / "one_stage"
        two_res = res_root / backbone / "two_stage"

        one_train = [
            args.python, "train.py",
            *_train_common(args, backbone),
            "--output_dir", str(one_dir),
            "--epochs", str(args.one_stage_epochs),
            "--lr", str(args.lr_one_stage),
            "--grad_clip", str(args.grad_clip_stage1),
            "--noise_min", str(args.train_noise_min),
            "--noise_max", str(args.train_noise_max),
            "--noise_clean_prob", str(args.train_noise_clean_prob),
            "--focal_gamma", str(args.focal_gamma),
        ]
        if args.auto_cls_weights:
            one_train.extend([
                "--auto_cls_weights",
                "--auto_cls_weight_power", str(args.auto_cls_weight_power),
                "--max_auto_weight", str(args.max_auto_weight),
            ])

        s1_train = [
            args.python, "train.py",
            *_train_common(args, backbone),
            "--output_dir", str(s1_dir),
            "--epochs", str(args.stage1_epochs),
            "--lr", str(args.lr_stage1),
            "--grad_clip", str(args.grad_clip_stage1),
            "--noise_min", str(args.train_noise_min),
            "--noise_max", str(args.train_noise_max),
            "--noise_clean_prob", str(args.train_noise_clean_prob),
            "--connectivity_only",
        ]
        s1_ckpt = _ckpt(s1_dir)

        s2_train = [
            args.python, "train_stage2.py",
            "--stage1_ckpt", str(s1_ckpt),
            "--cache_path", args.cache_path,
            "--output_dir", str(s2_dir),
            "--epochs", str(args.stage2_epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr_stage2),
            "--grad_clip", str(args.grad_clip_stage2),
            "--noise_min", str(args.train_noise_min),
            "--noise_max", str(args.train_noise_max),
            "--noise_clean_prob", str(args.train_noise_clean_prob),
            "--bond_cutoff", str(args.bond_cutoff),
            "--edge_embedding_size", str(args.edge_embedding_size),
            "--device", args.device,
            "--num_workers", str(args.num_workers),
            "--prefetch_factor", str(args.prefetch_factor),
            "--dataloader_timeout", str(args.dataloader_timeout),
            "--e2e_val_max_mols", str(args.e2e_val_max_mols),
            "--e2e_val_every", str(args.e2e_val_every),
            "--seed", str(args.seed),
        ]
        _append_split_flags(s2_train, args)
        if args.auto_cls_weights:
            s2_train.extend([
                "--auto_cls_weights",
                "--auto_cls_weight_power", str(args.auto_cls_weight_power),
                "--max_auto_weight", str(args.max_auto_weight),
            ])

        one_eval = [
            args.python, "evaluate.py",
            "--checkpoint", str(_ckpt(one_dir)),
            *_eval_common(args, one_res),
        ]
        two_eval = [
            args.python, "evaluate.py",
            "--checkpoint", str(s1_ckpt),
            "--stage2_ckpt", str(_ckpt(s2_dir)),
            *_eval_common(args, two_res),
        ]

        commands.extend([
            (f"train_{base}_one_stage", one_train, one_dir / "best.pt"),
            (f"train_{base}_stage1", s1_train, s1_dir / "best.pt"),
            (f"train_{base}_stage2", s2_train, s2_dir / "best.pt"),
            (f"eval_{base}_one_stage", one_eval, one_res / "results.json"),
            (f"eval_{base}_two_stage", two_eval, two_res / "results.json"),
        ])

    return commands


def _load_results(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    for value in payload.values():
        if isinstance(value, list):
            return value
    return []


def _iter_result_jsons(results_root: Path) -> Iterable[Path]:
    yield from sorted(results_root.glob("**/results.json"))


def write_summary(args: argparse.Namespace) -> None:
    res_root = ROOT / args.results_root
    rows = []
    for result_json in _iter_result_jsons(res_root):
        rel = result_json.parent.relative_to(res_root).parts
        backbone = rel[0] if len(rel) >= 1 else ""
        stage = rel[1] if len(rel) >= 2 else ""
        for r in _load_results(result_json):
            rows.append({
                "backbone": backbone,
                "stage": stage,
                "sigma": r.get("sigma", ""),
                "f1_macro": r.get("f1_macro", ""),
                "f1_single": r.get("f1_single", ""),
                "f1_double": r.get("f1_double", ""),
                "f1_triple": r.get("f1_triple", ""),
                "f1_aromatic": r.get("f1_aromatic", ""),
                "conn_f1": r.get("conn_f1", ""),
                "mol_validity": r.get("mol_validity", ""),
                "nll": r.get("nll", ""),
                "ece": r.get("ece", ""),
                "mean_confidence": r.get("mean_confidence", ""),
            })

    res_root.mkdir(parents=True, exist_ok=True)
    out_csv = res_root / "summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "backbone", "stage", "sigma", "f1_macro", "f1_single", "f1_double",
            "f1_triple", "f1_aromatic", "conn_f1", "mol_validity",
            "nll", "ece", "mean_confidence",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "data_path": args.data_path,
        "cache_path": args.cache_path,
        "backbones": args.backbones,
        "cutoff": args.cutoff,
        "h_cutoff": args.h_cutoff,
        "train_noise_max": args.train_noise_max,
        "noise_levels": args.noise_levels,
        "one_stage_epochs": args.one_stage_epochs,
        "stage1_epochs": args.stage1_epochs,
        "stage2_epochs": args.stage2_epochs,
        "auto_cls_weights": args.auto_cls_weights,
        "eval_split": args.eval_split,
        "seed": args.seed,
    }
    (res_root / "study_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[SUMMARY] wrote {out_csv}", flush=True)


def main() -> int:
    args = parse_args()
    try:
        args.python = _resolve_python(args.python)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", flush=True)
        return 127
    print(f"[INFO] Child Python: {args.python}", flush=True)

    cache = ROOT / args.cache_path
    data = ROOT / args.data_path
    if not data.exists():
        print(f"[ERROR] data_path does not exist: {data}", flush=True)
        return 2
    if not cache.exists() and not args.build_cache:
        print(f"[WARN] cache does not exist yet: {cache}", flush=True)
        print("[WARN] Re-run with --build_cache, or build it separately first.", flush=True)

    statuses: List[Tuple[str, int]] = []
    for label, cmd, expected in build_commands(args):
        name, rc = _maybe(cmd, expected if expected.is_absolute() else ROOT / expected, label, args)
        statuses.append((name, rc))
        if rc != 0 and args.stop_on_error:
            break

    if not args.dry_run:
        write_summary(args)

    print("\n===== Status =====", flush=True)
    failures = 0
    for name, rc in statuses:
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"{name:<48} {status}", flush=True)
        failures += int(rc != 0)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
