"""Compare selected and final joint checkpoints on the original validation rule.

This is a validation-only diagnostic. It does not choose a new checkpoint or
inspect the fixed test or external confirmation cohort.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bondnet.data.dataset import CachedBondNetDataset
from evaluate import load_model
from train import val_e2e


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="data/fixed_split_caches/geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed_val.pt")
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--output", default="results/revision_v3_epoch50_validation/summary.json")
    parser.add_argument("--max-mols", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise-seed", type=int, default=20260921)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    full = CachedBondNetDataset(str(ROOT / args.cache))
    dataset = Subset(full, range(min(args.max_mols, len(full))))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in (42, 43, 44):
        ckpt_dir = ROOT / args.checkpoint_root / f"revision_v2_direction_fixed_lr1e4_seed{seed}" / "one_stage_joint"
        seed_rows = {}
        for label, name in (("selected", "best_e2e.pt"), ("epoch_50", "epoch_0050.pt")):
            path = ckpt_dir / name
            raw = torch.load(path, map_location="cpu", weights_only=False)
            model = load_model(str(path), device)
            model.eval()
            metrics = {}
            for sigma in (0.0, 0.1):
                result = val_e2e(
                    model, dataset, device, batch_size=args.batch_size,
                    num_workers=0, h_cutoff=2.5, noise_sigma=sigma,
                    noise_seed=args.noise_seed,
                )
                metrics[str(sigma)] = {
                    "f1_macro": float(result["f1_macro"]),
                    "conn_f1": float(result["conn_f1"]),
                    "mol_validity": float(result["mol_validity"]),
                }
            score = (metrics["0.0"]["f1_macro"] + metrics["0.1"]["f1_macro"]) / 2
            seed_rows[label] = {"checkpoint": str(path.relative_to(ROOT)),
                                "epoch_one_based": int(raw["epoch"]) + 1,
                                "selection_score": score, "metrics": metrics}
            print(f"seed={seed} {label} epoch={int(raw['epoch'])+1} "
                  f"score={score:.8f}", flush=True)
        seed_rows["epoch50_minus_selected"] = (
            seed_rows["epoch_50"]["selection_score"]
            - seed_rows["selected"]["selection_score"]
        )
        rows.append({"seed": seed, **seed_rows})
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "protocol": {"cache": args.cache, "n_molecules": len(dataset),
                     "noise_levels": [0.0, 0.1], "noise_seed": args.noise_seed,
                     "selection_metric": "mean reference-bond e2e macro-F1",
                     "evaluation_only": True},
        "rows": rows,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
