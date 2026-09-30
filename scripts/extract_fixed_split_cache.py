"""Extract fixed-label subsets from a monolithic BondNet feature cache.

The tensor samples are copied without modification.  This utility only reduces
evaluation startup time by avoiding repeated deserialization of the full cache.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Monolithic .pt feature cache")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=["train", "val", "test"], default=["val", "test"]
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(source, map_location="cpu", weights_only=False)
    samples = payload["samples"]
    weights = payload["diversity_weights"]
    metadata = dict(payload["metadata"])
    if len(samples) != len(weights):
        raise ValueError(f"sample/weight mismatch: {len(samples)} != {len(weights)}")

    for split in args.splits:
        indices = [i for i, sample in enumerate(samples) if sample.get("split") == split]
        if not indices:
            raise ValueError(f"No samples carry fixed split label {split!r}")
        subset_metadata = dict(metadata)
        subset_metadata.update(
            {
                "n_molecules": len(indices),
                "source_cache": str(source),
                "selected_split": split,
            }
        )
        out_path = output_dir / f"{source.stem}_{split}.pt"
        torch.save(
            {
                "samples": [samples[i] for i in indices],
                "diversity_weights": weights[indices].clone(),
                "metadata": subset_metadata,
            },
            out_path,
        )
        print(f"{split}: {len(indices)} samples -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
