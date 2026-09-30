"""
Run the revised BondNet paper experiments with the joint one-stage model as
the main system.

This script keeps the paper story separated into three buckets:
1. Main one-stage robustness curves across training noise levels.
2. Small one-stage design ablations such as cutoff and explicit-H.
3. Optional hard two-stage diagnostic, delegated to run_hierarchy_robustness_study.py.

The hard two-stage run is intentionally not treated as the default model.  It is
kept as a matched diagnostic for the question: does hard separation itself help?
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-stage-first paper experiment runner")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")

    p.add_argument("--cache_path", default="data/geom_drugs_all_random1_re10_c25_explicit_h_fixed.pt")
    p.add_argument("--cache_map", nargs="*", default=[],
                   help=(
                       "Optional cutoff-specific explicit-H caches, e.g. "
                       "2.5=data/geom_c25.pt 3.0=data/geom_c30.pt. "
                       "Cached candidate edges are not rebuilt from --cutoff."
                   ))
    p.add_argument("--heavy_cache_path", default=None,
                   help="Optional heavy-only cache for explicit-H ablation. If omitted, no-H runs are skipped.")
    p.add_argument("--split_key", default="geom_mol_idx")
    p.add_argument("--train_groups", type=int, default=None)
    p.add_argument("--eval_split", choices=["val", "all", "test"], default="val")
    p.add_argument("--cache_max_samples", type=int, default=None,
                   help="Optional smoke-test sample cap; omit for paper runs.")

    p.add_argument("--output_root", default="checkpoints/one_stage_main_study")
    p.add_argument("--results_root", default="results/one_stage_main_study")

    p.add_argument("--one_stage_epochs", type=int, default=50)
    p.add_argument("--stage1_epochs", type=int, default=20)
    p.add_argument("--stage2_epochs", type=int, default=50)
    p.add_argument("--noise_train_maxes", nargs="+", type=float,
                   default=[0.0, 0.05, 0.10, 0.15, 0.20])
    p.add_argument("--main_train_noise_max", type=float, default=0.15,
                   help="Noise level used for cutoff/H ablations and optional two-stage diagnostic.")
    p.add_argument("--train_noise_min", type=float, default=0.0)
    p.add_argument("--train_noise_clean_prob", type=float, default=0.0)
    p.add_argument("--noise_levels", nargs="+", type=float,
                   default=[0.0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20])

    p.add_argument("--cutoffs", nargs="+", type=float, default=[2.5, 3.0],
                   help="Heavy-heavy cutoffs for one-stage cutoff ablation.")
    p.add_argument("--main_cutoff", type=float, default=2.5,
                   help="Primary cutoff for main one-stage noise-grid runs.")
    p.add_argument("--h_cutoff", type=float, default=2.5)
    p.add_argument("--bond_cutoff", type=float, default=3.0)

    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--backbone", choices=["painn", "clof"], default="painn")
    p.add_argument("--clof_coords_weight", type=float, default=0.1)
    p.add_argument("--num_interactions", type=int, default=4)
    p.add_argument("--edge_embedding_size", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--auto_cls_weights", action="store_true")
    p.add_argument("--auto_cls_weight_power", type=float, default=0.5)
    p.add_argument("--max_auto_weight", type=float, default=5.0)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--lr_stage1", type=float, default=3e-4)
    p.add_argument("--lr_stage2", type=float, default=5e-4)
    p.add_argument("--grad_clip", type=float, default=0.5)
    p.add_argument("--grad_clip_stage2", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--dataloader_timeout", type=float, default=120.0)
    p.add_argument("--e2e_val_max_mols", type=int, default=8192)
    p.add_argument("--e2e_val_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--run_noise_grid", action="store_true", default=True)
    p.add_argument("--no_run_noise_grid", dest="run_noise_grid", action="store_false")
    p.add_argument("--run_cutoff_ablation", action="store_true", default=True)
    p.add_argument("--no_run_cutoff_ablation", dest="run_cutoff_ablation", action="store_false")
    p.add_argument("--run_h_ablation", action="store_true", default=True)
    p.add_argument("--no_run_h_ablation", dest="run_h_ablation", action="store_false")
    p.add_argument("--run_two_stage_diagnostic", action="store_true",
                   help="Also run matched hard two-stage diagnostic via run_hierarchy_robustness_study.py.")
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
        r"Pass the environment explicitly, e.g. --python D:\minconda\python.exe"
    )


def _tag_float(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _parse_cache_map(entries: Sequence[str]) -> Dict[float, str]:
    cache_map: Dict[float, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid --cache_map entry {entry!r}; expected cutoff=path")
        key, value = entry.split("=", 1)
        cache_map[float(key)] = value
    return cache_map


def _cache_for_cutoff(args: argparse.Namespace, cutoff: float) -> str:
    cache_map = getattr(args, "_cache_map", {})
    for key, value in cache_map.items():
        if abs(float(key) - float(cutoff)) < 1e-6:
            return value
    return args.cache_path


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


def _append_if_present(flags: List[str], args: argparse.Namespace) -> None:
    if args.split_key:
        flags.extend(["--split_key", args.split_key])
    if args.train_groups is not None:
        flags.extend(["--train_groups", str(args.train_groups)])
    if args.cache_max_samples is not None:
        flags.extend(["--cache_max_samples", str(args.cache_max_samples)])


def _train_common(args: argparse.Namespace, cache_path: str, cutoff: float, h_cutoff: float,
                  explicit_h: bool) -> List[str]:
    flags = [
        "--cache_path", cache_path,
        "--cutoff", str(cutoff),
        "--h_cutoff", str(h_cutoff),
        "--num_interactions", str(args.num_interactions),
        "--hidden_size", str(args.hidden_size),
        "--backbone", args.backbone,
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
    flags.append("--explicit_h" if explicit_h else "--no_explicit_h")
    _append_if_present(flags, args)
    return flags


def _eval_common(args: argparse.Namespace, cache_path: str, result_dir: Path, cutoff: float,
                 h_cutoff: float, explicit_h: bool) -> List[str]:
    flags = [
        "--cache_path", cache_path,
        "--cutoff", str(cutoff),
        "--h_cutoff", str(h_cutoff),
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
    flags.append("--explicit_h" if explicit_h else "--no_explicit_h")
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


def _one_stage_pair(
    args: argparse.Namespace,
    *,
    label: str,
    cache_path: str,
    output_dir: Path,
    result_dir: Path,
    train_noise_max: float,
    cutoff: float,
    h_cutoff: float,
    explicit_h: bool,
) -> List[Tuple[str, List[str], Path]]:
    train = [
        args.python, "train.py",
        *_train_common(args, cache_path, cutoff, h_cutoff, explicit_h),
        "--output_dir", str(output_dir),
        "--epochs", str(args.one_stage_epochs),
        "--lr", str(args.lr),
        "--grad_clip", str(args.grad_clip),
        "--noise_min", str(args.train_noise_min),
        "--noise_max", str(train_noise_max),
        "--noise_clean_prob", str(args.train_noise_clean_prob),
        "--focal_gamma", str(args.focal_gamma),
    ]
    if args.auto_cls_weights:
        train.extend([
            "--auto_cls_weights",
            "--auto_cls_weight_power", str(args.auto_cls_weight_power),
            "--max_auto_weight", str(args.max_auto_weight),
        ])
    eval_cmd = [
        args.python, "evaluate.py",
        "--checkpoint", str(_ckpt(output_dir)),
        *_eval_common(args, cache_path, result_dir, cutoff, h_cutoff, explicit_h),
    ]
    return [
        (f"train_{label}", train, output_dir / "best.pt"),
        (f"eval_{label}", eval_cmd, result_dir / "results.json"),
    ]


def build_commands(args: argparse.Namespace) -> List[Tuple[str, List[str], Path]]:
    commands: List[Tuple[str, List[str], Path]] = []
    out_root = ROOT / args.output_root
    res_root = ROOT / args.results_root

    if args.run_noise_grid:
        cache_path = _cache_for_cutoff(args, args.main_cutoff)
        for noise_max in args.noise_train_maxes:
            tag = f"noise{_tag_float(noise_max)}_c{_tag_float(args.main_cutoff)}_explicitH"
            commands.extend(_one_stage_pair(
                args,
                label=f"one_stage_{tag}",
                cache_path=cache_path,
                output_dir=out_root / "one_stage_noise_grid" / tag,
                result_dir=res_root / "one_stage_noise_grid" / tag,
                train_noise_max=noise_max,
                cutoff=args.main_cutoff,
                h_cutoff=args.h_cutoff,
                explicit_h=True,
            ))

    if args.run_cutoff_ablation:
        for cutoff in args.cutoffs:
            cache_path = _cache_for_cutoff(args, cutoff)
            tag = f"c{_tag_float(cutoff)}_noise{_tag_float(args.main_train_noise_max)}_explicitH"
            commands.extend(_one_stage_pair(
                args,
                label=f"cutoff_{tag}",
                cache_path=cache_path,
                output_dir=out_root / "cutoff_ablation" / tag,
                result_dir=res_root / "cutoff_ablation" / tag,
                train_noise_max=args.main_train_noise_max,
                cutoff=cutoff,
                h_cutoff=args.h_cutoff,
                explicit_h=True,
            ))

    if args.run_h_ablation:
        cache_path = _cache_for_cutoff(args, args.main_cutoff)
        tag = f"explicitH_c{_tag_float(args.main_cutoff)}_noise{_tag_float(args.main_train_noise_max)}"
        commands.extend(_one_stage_pair(
            args,
            label=f"h_ablation_{tag}",
            cache_path=cache_path,
            output_dir=out_root / "h_ablation" / tag,
            result_dir=res_root / "h_ablation" / tag,
            train_noise_max=args.main_train_noise_max,
            cutoff=args.main_cutoff,
            h_cutoff=args.h_cutoff,
            explicit_h=True,
        ))
        if args.heavy_cache_path:
            tag = f"heavyOnly_c{_tag_float(args.main_cutoff)}_noise{_tag_float(args.main_train_noise_max)}"
            commands.extend(_one_stage_pair(
                args,
                label=f"h_ablation_{tag}",
                cache_path=args.heavy_cache_path,
                output_dir=out_root / "h_ablation" / tag,
                result_dir=res_root / "h_ablation" / tag,
                train_noise_max=args.main_train_noise_max,
                cutoff=args.main_cutoff,
                h_cutoff=args.h_cutoff,
                explicit_h=False,
            ))
        else:
            marker = res_root / "h_ablation" / "heavy_only_SKIPPED.txt"
            commands.append((
                "h_ablation_heavy_only_skipped",
                [args.python, "scripts/run_one_stage_main_study.py", "_write_skip_marker", str(marker),
                 "No --heavy_cache_path was provided; heavy-only explicit-H ablation was skipped."],
                marker,
            ))

    if args.run_two_stage_diagnostic:
        cache_path = _cache_for_cutoff(args, args.main_cutoff)
        diag_res = res_root / "two_stage_diagnostic" / "summary.csv"
        diag = [
            args.python, "scripts/run_hierarchy_robustness_study.py",
            "--python", args.python,
            "--cache_path", cache_path,
            "--split_key", args.split_key,
            "--output_root", str(Path(args.output_root) / "two_stage_diagnostic"),
            "--results_root", str(Path(args.results_root) / "two_stage_diagnostic"),
            "--stage1_epochs", str(args.stage1_epochs),
            "--stage2_epochs", str(args.stage2_epochs),
            "--one_stage_epochs", str(args.one_stage_epochs),
            "--stage2_noise_max", str(args.main_train_noise_max),
            "--train_noise_min", str(args.train_noise_min),
            "--train_noise_clean_prob", str(args.train_noise_clean_prob),
            "--cutoff", str(args.main_cutoff),
            "--h_cutoff", str(args.h_cutoff),
            "--bond_cutoff", str(args.bond_cutoff),
            "--hidden_size", str(args.hidden_size),
            "--backbone", args.backbone,
            "--clof_coords_weight", str(args.clof_coords_weight),
            "--num_interactions", str(args.num_interactions),
            "--edge_embedding_size", str(args.edge_embedding_size),
            "--batch_size", str(args.batch_size),
            "--lr_stage1", str(args.lr_stage1),
            "--lr_stage2", str(args.lr_stage2),
            "--lr_one_stage", str(args.lr),
            "--grad_clip_stage1", str(args.grad_clip),
            "--grad_clip_stage2", str(args.grad_clip_stage2),
            "--device", args.device,
            "--num_workers", str(args.num_workers),
            "--prefetch_factor", str(args.prefetch_factor),
            "--dataloader_timeout", str(args.dataloader_timeout),
            "--e2e_val_max_mols", str(args.e2e_val_max_mols),
            "--e2e_val_every", str(args.e2e_val_every),
            "--seed", str(args.seed),
            "--noise_levels",
        ]
        diag.extend(str(x) for x in args.noise_levels)
        if args.train_groups is not None:
            diag.extend(["--train_groups", str(args.train_groups)])
        if args.cache_max_samples is not None:
            diag.extend(["--cache_max_samples", str(args.cache_max_samples)])
        if args.skip_existing:
            diag.append("--skip-existing")
        if args.stop_on_error:
            diag.append("--stop-on-error")
        if not args.save_error_analysis:
            diag.append("--no_save_error_analysis")
        commands.append(("run_two_stage_diagnostic", diag, diag_res))

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
        rel_parent = result_json.parent.relative_to(res_root).as_posix()
        for r in _load_results(result_json):
            rows.append({
                "experiment": rel_parent,
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
            "experiment", "sigma", "f1_macro", "f1_single", "f1_double",
            "f1_triple", "f1_aromatic", "conn_f1", "mol_validity",
            "nll", "ece", "mean_confidence",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SUMMARY] wrote {out_csv}", flush=True)

    manifest = {
        "paper_story": "joint one-stage is the main BondNet model; hard two-stage is diagnostic only",
        "main_train_noise_max": args.main_train_noise_max,
        "main_cutoff": args.main_cutoff,
        "h_cutoff": args.h_cutoff,
        "noise_train_maxes": args.noise_train_maxes,
        "noise_levels": args.noise_levels,
        "heavy_only_ablation": "enabled" if args.heavy_cache_path else "skipped: no --heavy_cache_path",
    }
    (res_root / "study_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_skip_marker(argv: Sequence[str]) -> int:
    if len(argv) < 2:
        return 2
    path = Path(argv[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(argv[1] + "\n", encoding="utf-8")
    print(f"[SKIP] {argv[1]}", flush=True)
    return 0


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "_write_skip_marker":
        return _write_skip_marker(sys.argv[2:])

    args = parse_args()
    args._cache_map = _parse_cache_map(args.cache_map)
    try:
        args.python = _resolve_python(args.python)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", flush=True)
        return 127
    print(f"[INFO] Child Python: {args.python}", flush=True)

    cache = ROOT / args.cache_path
    if not cache.exists():
        print(f"[WARN] explicit-H cache path does not exist yet: {cache}", flush=True)
    if args.heavy_cache_path and not (ROOT / args.heavy_cache_path).exists():
        print(f"[WARN] heavy-only cache path does not exist yet: {ROOT / args.heavy_cache_path}", flush=True)

    statuses: List[Tuple[str, int]] = []
    for label, cmd, expected in build_commands(args):
        name, rc = _maybe(cmd, ROOT / expected if not expected.is_absolute() else expected, label, args)
        statuses.append((name, rc))
        if rc != 0 and args.stop_on_error:
            break

    if not args.dry_run:
        write_summary(args)

    print("\n===== Status =====", flush=True)
    n_fail = 0
    for name, rc in statuses:
        ok = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"{name:<48} {ok}", flush=True)
        n_fail += int(rc != 0)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
