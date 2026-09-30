#!/usr/bin/env python3
"""Heavy-only feature cache whose atom tokens also encode the attached-H count.

Oracle diagnostic for the explicit-H versus heavy-only comparison (reviewer
request): does the explicit-H advantage come mainly from knowing how many H
atoms each heavy atom carries (valence information), or from the explicit 3D
hydrogen environment?  The heavy-only representation is kept unchanged (same
atoms, coordinates, candidate edges and labels); only the node token changes
from "element" to "(element, number of attached H)".

The H count is taken from the *reference* explicit-H molecule, so a model
trained on these caches receives information that a deployed coordinate-only
pipeline would not have.  Results must be labelled as an oracle diagnostic.

Token map: every (heavy element in the GEOM training set, n_H in 0..4) pair is
assigned a fixed unused atomic-number slot (< 119) of the existing embedding
table, so no model code changes.  Atoms that RDKit RemoveHs retains as H keep
token 1.  The map is written into the cache metadata.

Usage
  python scripts/build_heavy_hcount_cache.py --cache <heavy cache.pt> \
      --sdf <source SDF with geom_mol_idx> --output <new cache.pt>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Elements of the GEOM random1/re10 training split (see external-cohort selection record).
HEAVY_ELEMENTS = (5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53, 80, 83)
USED = set(HEAVY_ELEMENTS) | {1}
MAX_H = 4


def token_map():
    free = [z for z in range(2, 119) if z not in USED]
    pairs = [(z, n) for z in HEAVY_ELEMENTS for n in range(MAX_H + 1)]
    if len(pairs) > len(free):
        raise RuntimeError("not enough free embedding slots")
    return {f"{z}:{n}": free[k] for k, (z, n) in enumerate(pairs)}


def h_counts_from_sdf(path: Path, wanted: set[int] | None):
    """geom_mol_idx -> [(Z, n_H) for each non-H atom, in SDF atom order].

    n_H counts every attached hydrogen, including H/D atoms that RemoveHs keeps
    in the heavy-only cache (e.g. isotope-labelled or stereo H)."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    out = {}
    for mol in Chem.ForwardSDMolSupplier(str(path), removeHs=False, sanitize=False):
        if mol is None or not mol.HasProp("geom_mol_idx"):
            continue
        key = int(mol.GetProp("geom_mol_idx"))
        if wanted is not None and key not in wanted:
            continue
        counts = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue
            counts.append((atom.GetAtomicNum(),
                           sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1)))
        out[key] = counts
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="existing heavy-only cache (.pt)")
    ap.add_argument("--sdf", required=True, help="explicit-H SDF carrying geom_mol_idx")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import torch
    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    samples = payload["samples"]
    meta = dict(payload.get("metadata", {}))
    if meta.get("explicit_h"):
        raise ValueError("expected a heavy-only cache")
    wanted = {int(s["geom_mol_idx"]) for s in samples}
    counts = h_counts_from_sdf(Path(args.sdf), wanted)
    missing = wanted - set(counts)
    if missing:
        raise RuntimeError(f"{len(missing)} cache molecules not found in the SDF")
    tmap = token_map()
    retained_h = 0
    mols_with_retained_h = 0
    for s in samples:
        elems = s["elems"]
        ref = counts[int(s["geom_mol_idx"])]
        # RemoveHs keeps the relative order of the remaining atoms, but retained
        # H/D atoms (isotope or stereo H) may sit anywhere in the list, so align
        # the non-H atoms of the cache with the non-H atoms of the SDF by order.
        heavy_idx = [k for k, z in enumerate(elems.tolist()) if z != 1]
        if [int(elems[k]) for k in heavy_idx] != [z for z, _ in ref]:
            raise RuntimeError(f"heavy-atom element sequence mismatch for geom_mol_idx={s['geom_mol_idx']}")
        new = elems.clone()
        for k, (z, n) in zip(heavy_idx, ref):
            if n > MAX_H or f"{z}:{n}" not in tmap:
                raise ValueError(f"unsupported (element, nH)=({z},{n})")
            new[k] = tmap[f"{z}:{n}"]
        n_ret = int((elems == 1).sum())
        retained_h += n_ret
        mols_with_retained_h += int(n_ret > 0)
        s["elems"] = new
        s["element_z"] = elems
    meta.update({
        "atom_token": "element_and_attached_h_count (ORACLE: counts from reference explicit-H graph)",
        "hcount_token_map": tmap,
        "hcount_source_sdf": str(args.sdf),
        "hcount_source_cache": str(args.cache),
        "retained_h_atoms_kept_as_token_1": retained_h,
        "molecules_with_retained_h": mols_with_retained_h,
        "hcount_includes_retained_h": True,
    })
    payload["metadata"] = meta
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({"output": args.output, "n_molecules": len(samples),
                      "retained_h_atoms": retained_h,
                      "molecules_with_retained_h": mols_with_retained_h}, indent=2))


if __name__ == "__main__":
    main()
