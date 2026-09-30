"""Validation-only sensitivity of RDKit DetermineBonds settings.

Reports graph return, strict heavy--heavy exact match, and failure reasons on
the same labeled SDF. This is diagnostic and does not replace the frozen
default-configuration test/external baselines.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import p0c_yuelbond_compare as scorer


SETTINGS = {
    "default_cap1000": {},
    "use_vdw": {"useVdw": True},
    "use_hueckel": {"useHueckel": True},
    "no_embed_chiral": {"embedChiral": False},
    "no_charged_fragments": {"allowChargedFragments": False},
    "default_cap2000": {"maxIterations": 2000},
}


@contextlib.contextmanager
def quiet_native_stdout():
    """Suppress verbose Hückel distance diagnostics during bulk evaluation."""
    saved = os.dup(1)
    try:
        with open(os.devnull, "w") as null:
            os.dup2(null.fileno(), 1)
            yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)


def predict(mol, overrides: dict) -> dict:
    from rdkit import Chem
    from rdkit.Chem.rdDetermineBonds import DetermineBonds
    raw = Chem.RWMol()
    for atom in mol.GetAtoms():
        raw.AddAtom(Chem.Atom(atom.GetAtomicNum()))
    conf = Chem.Conformer(raw.GetNumAtoms())
    original = mol.GetConformer()
    for idx in range(raw.GetNumAtoms()):
        conf.SetAtomPosition(idx, original.GetAtomPosition(idx))
    raw.AddConformer(conf, assignId=True)
    raw = raw.GetMol()
    kwargs = {"charge": int(sum(a.GetFormalCharge() for a in mol.GetAtoms())),
              "maxIterations": 1000}
    kwargs.update(overrides)
    DetermineBonds(raw, **kwargs)
    heavy_map = scorer._heavy_map(mol)
    labels = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1,
              Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}
    bonds = {}
    for bond in raw.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in heavy_map and j in heavy_map:
            pair = tuple(sorted((heavy_map[i], heavy_map[j])))
            bonds[pair] = labels[bond.GetBondType()]
    return bonds


def update_pipeline_counts(true: dict, pred: dict, tp: list[int],
                           fp: list[int], fn: list[int]) -> None:
    """Score one unordered HH pair once, including extra/missing/wrong bonds."""
    for pair, label in true.items():
        if pred.get(pair) == label:
            tp[label] += 1
        else:
            fn[label] += 1
    for pair, label in pred.items():
        if true.get(pair) != label:
            fp[label] += 1


def pipeline_f1(tp: list[int], fp: list[int], fn: list[int]) -> list[float]:
    return [2 * t / (2 * t + f + n) if 2 * t + f + n else 0.0
            for t, f, n in zip(tp, fp, fn)]


def main() -> None:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", default="data/robustness_26940/sigma_020.sdf")
    parser.add_argument("--configs", nargs="+", choices=SETTINGS,
                        default=list(SETTINGS))
    parser.add_argument("--max-mols", type=int)
    parser.add_argument("--output", default="results/revision_v4_rdkit_configs/val_sigma_020.json")
    args = parser.parse_args()
    molecules = scorer._read_valid_molecules(str(ROOT / args.sdf))
    if args.max_mols:
        molecules = molecules[:args.max_mols]
    rows = []
    for name in args.configs:
        start = time.perf_counter()
        counts = Counter()
        failure_types = Counter()
        failure_messages = Counter()
        tp, fp, fn = [0] * 4, [0] * 4, [0] * 4
        with quiet_native_stdout() if name == "use_hueckel" else contextlib.nullcontext():
            for mol in molecules:
                true = scorer._true_bonds(mol)
                try:
                    pred = predict(mol, SETTINGS[name])
                except Exception as exc:
                    counts["failed"] += 1
                    failure_types[type(exc).__name__] += 1
                    failure_messages[str(exc).splitlines()[0][:160]] += 1
                    update_pipeline_counts(true, {}, tp, fp, fn)
                    continue
                update_pipeline_counts(true, pred, tp, fp, fn)
                counts["returned"] += 1
                if pred == true:
                    counts["exact"] += 1
                elif set(pred) == set(true):
                    counts["same_connectivity_wrong_label"] += 1
                else:
                    counts["wrong_connectivity"] += 1
        f1 = pipeline_f1(tp, fp, fn)
        rows.append({"configuration": name, "settings": SETTINGS[name],
                     "n_molecules": len(molecules), "counts": dict(counts),
                     "success_rate": counts["returned"] / len(molecules),
                     "hh_exact_rate": counts["exact"] / len(molecules),
                     "pipeline_macro_f1": sum(f1) / 4,
                     "pipeline_class_f1": f1,
                     "pipeline_tp": tp, "pipeline_fp": fp, "pipeline_fn": fn,
                     "seconds": time.perf_counter() - start,
                     "failure_types": dict(failure_types),
                     "failure_messages": dict(failure_messages)})
        print(f"{name}: success={rows[-1]['success_rate']:.4%} "
              f"HH-exact={rows[-1]['hh_exact_rate']:.4%} "
              f"fail={counts['failed']} time={rows[-1]['seconds']:.1f}s",
              flush=True)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"sdf": args.sdf, "rows": rows,
                                  "status": "validation diagnostic; no test tuning"},
                                 indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
