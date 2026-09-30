"""
Run the one-stage vs two-stage robustness study.

The study covers:
1. Robustness curves for joint one-stage and separate two-stage models.
2. Error decomposition via evaluate.py --save_error_analysis.
3. Stage2 oracle-connectivity evaluation.
4. Calibration of the oracle Stage2 head (NLL/ECE).
5. A compact summary table across sigmas.

Default training schedule follows the current working hypothesis:
- Stage 1 connectivity: 20 epochs.
- Stage 2 bond typing: 50 epochs.
- Joint one-stage baseline: 50 epochs.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-stage vs two-stage robustness study")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")

    p.add_argument(
        "--cache_path",
        default="data/geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed.pt",
        help=("Version-14 cache built with a 3.0 A edge envelope. The active "
              "candidate graph is rebuilt inside that envelope using --cutoff/--h_cutoff."),
    )
    p.add_argument(
        "--eval_cache_path",
        default=None,
        help=("Optional fixed-split subset cache used only for evaluation. Its samples "
              "must be exact copies from --cache_path."),
    )
    p.add_argument("--split_key", default="geom_mol_idx")
    p.add_argument("--train_groups", type=int, default=None)
    p.add_argument("--eval_split", choices=["val", "test", "all"], default="test")
    p.add_argument("--cache_max_samples", type=int, default=None)

    p.add_argument("--output_root", default="checkpoints/hierarchy_study")
    p.add_argument("--results_root", default="results/hierarchy_study")

    p.add_argument("--stage1_epochs", type=int, default=20)
    p.add_argument("--stage2_epochs", type=int, default=50)
    p.add_argument("--one_stage_epochs", type=int, default=50)
    p.add_argument("--stage1_noise_max", type=float, default=None,
                   help="Stage 1 training noise. Default: same as --stage2_noise_max for fair robustness comparison.")
    p.add_argument("--stage2_noise_max", type=float, default=0.15)
    p.add_argument("--train_noise_min", type=float, default=0.0)
    p.add_argument("--train_noise_clean_prob", type=float, default=0.0)

    p.add_argument("--noise_levels", nargs="+", type=float, default=[0.0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20])
    p.add_argument("--cutoff", type=float, default=2.5)
    p.add_argument("--h_cutoff", type=float, default=2.5)
    p.add_argument("--bond_cutoff", type=float, default=3.0)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--one_stage_hidden_size", type=int, default=None,
                   help=("Optional one-stage width used for parameter-budget matching. "
                         "For two 256-wide staged models, 360 gives a comparable total parameter count."))
    p.add_argument("--num_interactions", type=int, default=4)
    p.add_argument("--edge_embedding_size", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr_stage1", type=float, default=3e-4)
    p.add_argument("--lr_stage2", type=float, default=5e-4)
    p.add_argument("--lr_one_stage", type=float, default=3e-4)
    p.add_argument("--grad_clip_stage1", type=float, default=0.5)
    p.add_argument("--grad_clip_stage2", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--dataloader_timeout", type=float, default=120.0)
    p.add_argument("--e2e_val_max_mols", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_noise_seed", type=int, default=20260921)
    p.add_argument("--save_error_analysis", action="store_true", default=True)
    p.add_argument("--no_save_error_analysis", dest="save_error_analysis", action="store_false")
    p.add_argument("--run_clean_stage1_diagnostic", action="store_true",
                   help="Also train/evaluate a clean-only Stage 1 diagnostic arm.")
    return p.parse_args()


def _resolve_python(python_arg: str) -> str:
    """Resolve the Python executable used for child train/eval jobs."""
    raw = str(python_arg or "").strip().strip('"').strip("'")
    if not raw:
        raw = sys.executable

    candidate = Path(raw)
    if candidate.exists():
        return str(candidate.resolve())

    found = shutil.which(raw)
    if found:
        return found

    raise FileNotFoundError(
        "Cannot find the Python executable for child jobs: "
        f"{python_arg!r}\n"
        "Pass the interpreter that has torch/RDKit installed, for example:\n"
        r"  --python D:\minconda\envs\bondnet\python.exe"
    )


def _run(cmd: List[str], dry_run: bool) -> int:
    print("\n[CMD] " + " ".join(cmd))
    if dry_run:
        return 0
    try:
        return int(subprocess.run(cmd, cwd=ROOT).returncode)
    except FileNotFoundError as e:
        print(f"[ERROR] Failed to launch command: {e}")
        print(f"[ERROR] Executable was: {cmd[0]!r}")
        return 127


def _maybe(cmd: List[str], expected: Path, label: str, args: argparse.Namespace) -> Tuple[str, int]:
    if args.skip_existing and expected.exists():
        print(f"[SKIP] {label}: {expected}")
        return label + ":skip", 0
    return label, _run(cmd, args.dry_run)


def _common_train_flags(args: argparse.Namespace, hidden_size: Optional[int] = None) -> List[str]:
    width = args.hidden_size if hidden_size is None else int(hidden_size)
    flags = [
        "--cache_path", args.cache_path,
        "--explicit_h",
        "--cutoff", str(args.cutoff),
        "--h_cutoff", str(args.h_cutoff),
        "--num_interactions", str(args.num_interactions),
        "--hidden_size", str(width),
        "--edge_embedding_size", str(args.edge_embedding_size),
        "--batch_size", str(args.batch_size),
        "--device", args.device,
        "--num_workers", str(args.num_workers),
        "--prefetch_factor", str(args.prefetch_factor),
        "--e2e_val_max_mols", str(args.e2e_val_max_mols),
        "--seed", str(args.seed),
    ]
    if args.split_key:
        flags.extend(["--split_key", args.split_key])
    if args.train_groups is not None:
        flags.extend(["--train_groups", str(args.train_groups)])
    if args.cache_max_samples is not None:
        flags.extend(["--cache_max_samples", str(args.cache_max_samples)])
    return flags


def _common_eval_flags(args: argparse.Namespace, result_dir: Path) -> List[str]:
    eval_cache_path = args.eval_cache_path or args.cache_path
    flags = [
        "--cache_path", eval_cache_path,
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
        "--eval_noise_seed", str(args.eval_noise_seed),
    ])
    if args.split_key:
        flags.extend(["--split_key", args.split_key])
    if args.train_groups is not None:
        flags.extend(["--train_groups", str(args.train_groups)])
    if args.save_error_analysis:
        flags.append("--save_error_analysis")
    return flags


def _pick_ckpt(out_dir: Path) -> Path:
    # Prefer best.pt (selected by validation loss). The end-to-end-selected
    # checkpoint (best_e2e.pt) was found to underperform best.pt for GEOM Stage 2,
    # so we default to best.pt to avoid biasing the one-stage vs two-stage compare.
    for name in ("best.pt", "best_e2e.pt"):
        p = out_dir / name
        if p.exists():
            return p
    return out_dir / "best.pt"


def _pick_stage1_ckpt(out_dir: Path) -> Path:
    # Connectivity-only Stage 1 should be selected by validation loss, not by
    # end-to-end bond-type F1, because Stage 3 is intentionally not trained.
    for name in ("best.pt", "best_e2e.pt"):
        p = out_dir / name
        if p.exists():
            return p
    return out_dir / "best.pt"


def build_commands(args: argparse.Namespace):
    out_root = ROOT / args.output_root
    res_root = ROOT / args.results_root
    one_dir = out_root / "one_stage_joint"
    s1_dir = out_root / "two_stage_stage1"
    s2_dir = out_root / "two_stage_stage2"
    one_res = res_root / "one_stage_joint"
    two_res = res_root / "two_stage_predicted"
    two_clean_res = res_root / "two_stage_predicted_cleanS1"
    oracle_res = res_root / "two_stage_oracle"
    s1c_dir = out_root / "two_stage_stage1_clean"
    stage1_noise_max = args.stage2_noise_max if args.stage1_noise_max is None else args.stage1_noise_max
    one_stage_hidden_size = args.hidden_size if args.one_stage_hidden_size is None else args.one_stage_hidden_size

    one_train = [
        args.python, "train.py",
        *_common_train_flags(args, hidden_size=one_stage_hidden_size),
        "--output_dir", str(one_dir),
        "--epochs", str(args.one_stage_epochs),
        "--lr", str(args.lr_one_stage),
        "--grad_clip", str(args.grad_clip_stage1),
        "--noise_min", str(args.train_noise_min),
        "--noise_max", str(args.stage2_noise_max),
        "--noise_clean_prob", str(args.train_noise_clean_prob),
    ]
    s1_train = [
        args.python, "train.py",
        *_common_train_flags(args, hidden_size=args.hidden_size),
        "--output_dir", str(s1_dir),
        "--epochs", str(args.stage1_epochs),
        "--lr", str(args.lr_stage1),
        "--grad_clip", str(args.grad_clip_stage1),
        "--noise_min", str(args.train_noise_min),
        "--noise_max", str(stage1_noise_max),
        "--noise_clean_prob", str(args.train_noise_clean_prob),
        "--connectivity_only",
    ]
    s1_ckpt = _pick_stage1_ckpt(s1_dir)
    # Contrast arm: a clean-trained (noise-free) Stage 1. Comparing this against the
    # default noise-trained Stage 1 shows how much of the two-stage robustness gap
    # under noise is due to Stage-1 connectivity training vs. the Stage-2 typing head.
    s1c_train = [
        args.python, "train.py",
        *_common_train_flags(args, hidden_size=args.hidden_size),
        "--output_dir", str(s1c_dir),
        "--epochs", str(args.stage1_epochs),
        "--lr", str(args.lr_stage1),
        "--grad_clip", str(args.grad_clip_stage1),
        "--noise_min", "0.0",
        "--noise_max", "0.0",
        "--connectivity_only",
    ]
    s1c_ckpt = _pick_stage1_ckpt(s1c_dir)
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
        "--noise_max", str(args.stage2_noise_max),
        "--noise_clean_prob", str(args.train_noise_clean_prob),
        "--bond_cutoff", str(args.bond_cutoff),
        "--edge_embedding_size", str(args.edge_embedding_size),
        "--device", args.device,
        "--num_workers", str(args.num_workers),
        "--prefetch_factor", str(args.prefetch_factor),
        "--dataloader_timeout", str(args.dataloader_timeout),
        "--e2e_val_max_mols", str(args.e2e_val_max_mols),
        "--seed", str(args.seed),
    ]
    if args.split_key:
        s2_train.extend(["--split_key", args.split_key])
    if args.train_groups is not None:
        s2_train.extend(["--train_groups", str(args.train_groups)])
    if args.cache_max_samples is not None:
        s2_train.extend(["--cache_max_samples", str(args.cache_max_samples)])

    one_eval = [
        args.python, "evaluate.py",
        "--checkpoint", str(_pick_ckpt(one_dir)),
        *_common_eval_flags(args, one_res),
    ]
    two_eval = [
        args.python, "evaluate.py",
        "--checkpoint", str(s1_ckpt),
        "--stage2_ckpt", str(_pick_ckpt(s2_dir)),
        *_common_eval_flags(args, two_res),
    ]
    two_clean_eval = [
        args.python, "evaluate.py",
        "--checkpoint", str(s1c_ckpt),
        "--stage2_ckpt", str(_pick_ckpt(s2_dir)),
        *_common_eval_flags(args, two_clean_res),
    ]
    if args.eval_cache_path and Path(args.eval_cache_path).resolve() != Path(args.cache_path).resolve():
        two_eval.append("--allow_stage2_cache_mismatch")
        two_clean_eval.append("--allow_stage2_cache_mismatch")
    oracle_eval = [
        args.python, "scripts/evaluate_stage2_oracle.py",
        "--stage2_ckpt", str(_pick_ckpt(s2_dir)),
        "--cache_path", args.eval_cache_path or args.cache_path,
        "--output_dir", str(oracle_res),
        "--eval_split", args.eval_split,
        "--val_split", "0.1",
        "--noise_levels",
    ]
    oracle_eval.extend(str(x) for x in args.noise_levels)
    oracle_eval.extend([
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--device", args.device,
        "--noise_clean_prob", str(args.train_noise_clean_prob),
    ])
    if args.split_key:
        oracle_eval.extend(["--split_key", args.split_key])
    if args.train_groups is not None:
        oracle_eval.extend(["--train_groups", str(args.train_groups)])
    if args.cache_max_samples is not None:
        oracle_eval.extend(["--max_mols", str(args.cache_max_samples)])

    commands = [
        ("train_one_stage_joint", one_train, one_dir / "best.pt"),
        ("train_stage1_connectivity", s1_train, s1_dir / "best.pt"),
        ("train_stage2_bond_type", s2_train, s2_dir / "best.pt"),
        ("eval_one_stage_joint", one_eval, one_res / "results.json"),
        ("eval_two_stage_predicted", two_eval, two_res / "results.json"),
        ("eval_stage2_oracle", oracle_eval, oracle_res / "results.json"),
    ]
    if args.run_clean_stage1_diagnostic:
        commands.insert(2, ("train_stage1_connectivity_clean", s1c_train, s1c_dir / "best.pt"))
        commands.insert(-1, (
            "eval_two_stage_predicted_cleanS1",
            two_clean_eval,
            two_clean_res / "results.json",
        ))
    return commands


def _load_curve(path: Path, key: Optional[str] = None) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if key is not None:
        return payload.get(key, [])
    for value in payload.values():
        if isinstance(value, list):
            return value
    return []


def write_summary(args: argparse.Namespace) -> None:
    res_root = ROOT / args.results_root
    rows = []
    sources = {
        "one_stage_joint": res_root / "one_stage_joint" / "results.json",
        "two_stage_predicted": res_root / "two_stage_predicted" / "results.json",
        "two_stage_predicted_cleanS1": res_root / "two_stage_predicted_cleanS1" / "results.json",
        "two_stage_oracle": res_root / "two_stage_oracle" / "results.json",
    }
    for name, path in sources.items():
        for r in _load_curve(path):
            rows.append({
                "method": name,
                "sigma": r.get("sigma", 0.0),
                "f1_macro": r.get("f1_macro", 0.0),
                "f1_single": r.get("f1_single", 0.0),
                "f1_double": r.get("f1_double", 0.0),
                "f1_triple": r.get("f1_triple", 0.0),
                "f1_aromatic": r.get("f1_aromatic", 0.0),
                "conn_f1": r.get("conn_f1", ""),
                "mol_validity": r.get("mol_validity", ""),
                "nll": r.get("nll", ""),
                "ece": r.get("ece", ""),
                "mean_confidence": r.get("mean_confidence", ""),
            })

    res_root.mkdir(parents=True, exist_ok=True)
    out_csv = res_root / "summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "method", "sigma", "f1_macro", "f1_single", "f1_double",
            "f1_triple", "f1_aromatic", "conn_f1", "mol_validity",
            "nll", "ece", "mean_confidence",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SUMMARY] wrote {out_csv}")

    # Lightweight crossing-point hint for the main robustness question.
    by_method: Dict[str, Dict[float, float]] = {}
    for r in rows:
        by_method.setdefault(str(r["method"]), {})[float(r["sigma"])] = float(r["f1_macro"])
    one = by_method.get("one_stage_joint", {})
    two = by_method.get("two_stage_predicted", {})
    two_c = by_method.get("two_stage_predicted_cleanS1", {})
    common = sorted(set(one) & set(two))
    for sigma in common:
        line = f"[DELTA] sigma={sigma:g} two_stage(noiseS1) - one_stage = {two[sigma] - one[sigma]:+.6f}"
        if sigma in two_c:
            line += f" | two_stage(cleanS1) - one_stage = {two_c[sigma] - one[sigma]:+.6f}"
        print(line)


def write_protocol_manifest(args: argparse.Namespace) -> None:
    """Record the matched-comparison budget before any result is interpreted."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from bondnet.model.bond_type_gnn import BondTypeGNN
    from bondnet.model.bondnet import BondNet

    one_width = args.hidden_size if args.one_stage_hidden_size is None else args.one_stage_hidden_size
    one = BondNet(
        num_interactions=args.num_interactions,
        hidden_size=one_width,
        cutoff=args.cutoff,
        edge_embedding_size=args.edge_embedding_size,
    )
    staged_connectivity = BondNet(
        num_interactions=args.num_interactions,
        hidden_size=args.hidden_size,
        cutoff=args.cutoff,
        edge_embedding_size=args.edge_embedding_size,
    )
    staged_typing = BondTypeGNN(
        hidden_size=args.hidden_size,
        num_layers=4,
        edge_embedding_size=args.edge_embedding_size,
        bond_cutoff=args.bond_cutoff,
    )
    count = lambda model: sum(p.numel() for p in model.parameters())
    one_params = count(one)
    stage1_params = count(staged_connectivity)
    stage2_params = count(staged_typing)
    payload = {
        "split_policy": "fixed cache labels: train/val/test; test excluded from training and selection",
        "seed": args.seed,
        "cache_path": args.cache_path,
        "eval_cache_path": args.eval_cache_path or args.cache_path,
        "eval_noise_seed": args.eval_noise_seed,
        "cache_envelope_note": (
            "The cache stores an edge superset; active candidate masks and local geometry "
            "are recomputed from current coordinates at the requested cutoffs."
        ),
        "eval_split": args.eval_split,
        "augmentation": {
            "noise_min": args.train_noise_min,
            "noise_max": args.stage2_noise_max,
            "clean_probability": args.train_noise_clean_prob,
        },
        "batch_size": args.batch_size,
        "one_stage": {
            "hidden_size": one_width,
            "epochs": args.one_stage_epochs,
            "parameters": one_params,
        },
        "two_stage": {
            "hidden_size": args.hidden_size,
            "stage1_epochs": args.stage1_epochs,
            "stage2_epochs": args.stage2_epochs,
            "stage1_parameters": stage1_params,
            "stage2_parameters": stage2_params,
            "total_parameters": stage1_params + stage2_params,
        },
        "relative_parameter_difference": (
            one_params - (stage1_params + stage2_params)
        ) / float(stage1_params + stage2_params),
    }
    print(
        "[BUDGET] one-stage parameters="
        f"{one_params:,}; two-stage total={stage1_params + stage2_params:,}; "
        f"relative difference={100 * payload['relative_parameter_difference']:+.2f}%"
    )
    if not args.dry_run:
        out = ROOT / args.results_root / "protocol_manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"[PROTOCOL] wrote {out}")


def main() -> int:
    args = parse_args()
    try:
        args.python = _resolve_python(args.python)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 127
    print(f"[INFO] Child Python: {args.python}")
    write_protocol_manifest(args)

    cache = ROOT / args.cache_path
    if not cache.exists():
        print(f"[WARN] cache path does not exist yet: {cache}")

    statuses = []
    for label, cmd, expected in build_commands(args):
        name, rc = _maybe(cmd, expected, label, args)
        statuses.append((name, rc))
        if rc != 0 and args.stop_on_error:
            break

    if not args.dry_run:
        write_summary(args)

    print("\n===== Status =====")
    n_fail = 0
    for name, rc in statuses:
        ok = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"{name:<32} {ok}")
        n_fail += int(rc != 0)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
