#!/usr/bin/env python3
"""Build, materialize and freeze the v3 external confirmation cohort.

The GEOM fixed test cohort was inspected repeatedly during revision, so the
revision reports a separate confirmation cohort drawn from PubChem3D.  This
script builds that cohort and freezes it *before* any model sees it.

Run the subcommands in order.  None of them loads a model or scores anything:

    python scripts/build_external_cohort_v3.py select        # choose molecules
    python scripts/build_external_cohort_v3.py materialize   # caches + noisy SDFs
    python scripts/build_external_cohort_v3.py freeze        # write manifest.json
    python scripts/build_external_cohort_v3.py verify        # re-check every hash

``freeze`` refuses to run if any external result already exists, and the
evaluation runner (scripts/run_revision_v3_external.ps1) calls ``verify``
before its first model call.  Keep ``manifest.json`` and the printed SHA-256
somewhere with a timestamp (commit, e-mail, Zenodo draft) before evaluating.

Selection rules, in the order they are applied to each candidate record:

 1. parse_failed          RDKit cannot read the record.
 2. no_3d_conformer       no conformer, non-finite coordinates, or 2D block.
 3. isotope_label         any atom carries an isotope label (e.g. deuterium).
 4. formal_charge         any atom has a nonzero formal charge (GEOM cohort has none).
 5. sanitize_failed       MDL-aromaticity sanitization (as in training) fails.
 6. radical               any atom has unpaired electrons after sanitization.
 7. missing_explicit_h    no H atoms, or any atom still carries implicit H.
 8. h_not_after_heavy     an H atom precedes a heavy atom (keyed noise assumes
                          the heavy-atom prefix is shared with heavy-only caches).
 9. multi_fragment        more than one covalent component.
10. element_outside_geom_train
11. heavy_atoms_outside_geom_train_range
12. no_heavy_heavy_bond
13. identity_failed       InChIKey or canonical SMILES cannot be computed.
14. overlap_geom          InChIKey connectivity block OR canonical non-isomeric
                          heavy-atom SMILES matches any of the GEOM random1/re10
                          molecules (train, validation and test).
15. overlap_previous_cohort  same identity match against every file passed with
                          --exclude_sdf, or a shared PubChem CID.
16. duplicate_within_cohort  identity already accepted into this cohort.
17. featurization_failed  the BondNet featurizer rejects the molecule in the
                          explicit-H (3.0/3.0 A) or heavy-only (3.0 A) setting.

Candidates are visited in a seeded random permutation of all source records;
the first ``--n`` that pass every rule form the cohort (equivalent to a
uniform random sample of the eligible pool).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import math
import mmap
import os
import platform
import subprocess
import sys
from collections import Counter, OrderedDict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_SOURCE = "data/pubchem3d_1M.sdf"
DEFAULT_GEOM = "data/geom_drugs_all_random1_rel10.sdf"
DEFAULT_OUT = "data/external_v3"
DEFAULT_RESULTS = "results/revision_v3_external"
# Every file that was ever used as a held-out/evaluation cohort, or as a
# PubChem-containing training subset, in any earlier version of the study.
DEFAULT_EXCLUDES = (
    "data/heldout_pubchem.sdf",
    "data/heldout_pubchem_clean.sdf",
    "data/heldout_qm9.sdf",
    "data/heldout_qm9_clean.sdf",
    "data/heldout_geom.sdf",
    "data/diverse_subset_80k.sdf",
    "data/shared_bench_clean.sdf",
    "data/geom_TEST_leakagefree.sdf",
)
DEFAULT_N = 10000
DEFAULT_SELECTION_SEED = 20260929
EVAL_NOISE_SEED = 20260921          # identical to the GEOM fixed-test protocol
SIGMAS = (0.0, 0.1, 0.2)
CACHE_CUTOFF = 3.0                  # candidate envelope, as the GEOM fixed caches
EVAL_CUTOFF = 2.5                   # dynamic candidate mask used at evaluation
CONN_THRESHOLD = 0.5
SEEDS = (42, 43, 44)
CHECKPOINTS = OrderedDict(
    (f"{model}_seed{seed}", paths)
    for seed in SEEDS
    for model, paths in (
        ("joint", {"checkpoint": f"checkpoints/revision_v2_direction_fixed_lr1e4_seed{seed}/one_stage_joint/best_e2e.pt",
                   "cache": "explicit"}),
        ("heavy", {"checkpoint": f"checkpoints/revision_v3_one_stage_heavy_seed{seed}/best_e2e.pt",
                   "cache": "heavy"}),
        ("staged", {"checkpoint": f"checkpoints/revision_v2_seed{seed}/explicit_stage1/best_e2e.pt",
                    "stage2_ckpt": f"checkpoints/revision_v2_seed{seed}/explicit_stage2/best_e2e.pt",
                    "cache": "explicit"}),
    )
)
# Code whose behaviour defines the frozen protocol.  ``verify`` fails if any
# of these files changes after freezing.
PROTOCOL_CODE = (
    "scripts/build_external_cohort_v3.py",
    "scripts/export_per_molecule_stats.py",
    "scripts/evaluate_chemical_validity.py",
    "scripts/evaluate_rule_baselines.py",
    "scripts/p0c_yuelbond_compare.py",
    "scripts/p0d_unified_robustness.py",
    "scripts/precompute_features.py",
    "scripts/summarize_revision_v3_external.py",
    "scripts/run_revision_v3_external.ps1",
    "evaluate.py",
    "bondnet/data/dataset.py",
    "bondnet/data/featurizer.py",
    "bondnet/data/noise_augment.py",
)
RULE_ORDER = (
    "parse_failed", "no_3d_conformer", "isotope_label", "formal_charge",
    "sanitize_failed", "radical", "missing_explicit_h", "h_not_after_heavy",
    "multi_fragment", "element_outside_geom_train",
    "heavy_atoms_outside_geom_train_range", "no_heavy_heavy_bond",
    "identity_failed", "overlap_geom", "overlap_previous_cohort",
    "duplicate_within_cohort", "featurization_failed",
)


# --------------------------------------------------------------------------- #
# Small helpers                                                                #
# --------------------------------------------------------------------------- #

def rel(path: Path) -> str:
    try:
        return Path(os.path.relpath(Path(path).resolve(), REPO)).as_posix()
    except ValueError:
        return str(path)


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (REPO / path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_ids(ids) -> str:
    return hashlib.sha256(",".join(str(int(x)) for x in ids).encode()).hexdigest()


def now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def sigma_tag(sigma: float) -> str:
    return f"{round(100 * sigma):03d}"


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def index_records(path: Path) -> np.ndarray:
    """Byte offsets delimiting SDF records ("$$$$" at the start of a line)."""
    bounds = [0]
    size = path.stat().st_size
    with open(path, "rb") as handle:
        mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            pos = 0
            while True:
                i = mm.find(b"$$$$", pos)
                if i < 0:
                    break
                if i > 0 and mm[i - 1:i] not in (b"\n",):
                    pos = i + 4
                    continue
                j = mm.find(b"\n", i)
                end = size if j < 0 else j + 1
                bounds.append(end)
                pos = end
            tail = mm[bounds[-1]:size] if bounds[-1] < size else b""
        finally:
            mm.close()
    if tail.strip():
        raise ValueError(f"{path}: trailing data after the last $$$$ record")
    return np.asarray(bounds, dtype=np.int64)


def read_record(handle, bounds: np.ndarray, k: int) -> bytes:
    handle.seek(int(bounds[k]))
    return handle.read(int(bounds[k + 1] - bounds[k]))


BATCH = 8192  # records handed to the worker pool at a time (bounded memory)


def batched(iterable, size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_records(path: Path, bounds: np.ndarray, order=None, start: int = 0):
    order = range(len(bounds) - 1) if order is None else order
    with open(path, "rb") as handle:
        for pos, k in enumerate(order):
            if pos < start:
                continue
            yield int(k), read_record(handle, bounds, int(k))


# --------------------------------------------------------------------------- #
# Per-record analysis (runs in worker processes)                               #
# --------------------------------------------------------------------------- #

_WORKER_RULES = None


def _init_worker(rules):
    global _WORKER_RULES
    _WORKER_RULES = rules
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def _mdl_sanitize(mol):
    from rdkit import Chem
    Chem.SanitizeMol(mol)
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
    Chem.SanitizeMol(mol)
    return mol


def _parse(block: bytes):
    from rdkit import Chem
    text = block.decode("utf-8", errors="replace")
    supplier = Chem.SDMolSupplier()
    supplier.SetData(text, sanitize=False, removeHs=False, strictParsing=False)
    mol = supplier[0] if len(supplier) else None
    return text, mol


def _identity(mol):
    """(InChIKey connectivity block, canonical non-isomeric heavy-atom SMILES)."""
    from rdkit import Chem
    key = Chem.MolToInchiKey(mol)
    key14 = key.split("-")[0] if key else ""
    try:
        smi = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=False)
    except Exception:
        smi = Chem.MolToSmiles(mol, isomericSmiles=False)
    return key, key14, smi


def _prop(mol, name):
    return mol.GetProp(name) if mol is not None and mol.HasProp(name) else None


def analyze_reference(item):
    """Identity and descriptors of a reference molecule (GEOM or exclude file)."""
    k, block = item
    from rdkit import Chem
    out = {"k": k, "key14": None, "smi": None}
    try:
        _, mol = _parse(block)
    except Exception:
        mol = None
    if mol is None:
        return out
    out["geom_mol_idx"] = _prop(mol, "geom_mol_idx")
    out["cid"] = _prop(mol, "PUBCHEM_COMPOUND_CID")
    out["charged"] = any(a.GetFormalCharge() != 0 for a in mol.GetAtoms())
    out["elements"] = sorted({a.GetAtomicNum() for a in mol.GetAtoms()})
    out["heavy"] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    try:
        san = _mdl_sanitize(Chem.Mol(mol))
        out["n_frag"] = len(Chem.GetMolFrags(san))
        out["radical"] = any(a.GetNumRadicalElectrons() > 0 for a in san.GetAtoms())
        _, out["key14"], out["smi"] = _identity(san)
    except Exception:
        # Fall back to the unsanitized graph so that an exclusion file entry
        # still contributes an identity whenever RDKit can produce one.
        try:
            mol.UpdatePropertyCache(strict=False)
            _, out["key14"], out["smi"] = _identity(mol)
        except Exception:
            pass
    return out


def analyze_candidate(item):
    """Apply rules 1-13 to one source record; return the first failing rule."""
    k, block = item
    from rdkit import Chem
    rules = _WORKER_RULES
    out = {"k": k, "reason": None}
    try:
        text, mol = _parse(block)
    except Exception:
        mol = None
    if mol is None:
        out["reason"] = "parse_failed"
        return out
    out["cid"] = _prop(mol, "PUBCHEM_COMPOUND_CID") or mol.GetProp("_Name").strip()
    if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
        out["reason"] = "no_3d_conformer"
        return out
    pos = mol.GetConformer().GetPositions()
    if not np.isfinite(pos).all():
        out["reason"] = "no_3d_conformer"
        return out
    atoms = list(mol.GetAtoms())
    if any(a.GetIsotope() != 0 for a in atoms):
        out["reason"] = "isotope_label"
        return out
    if any(a.GetFormalCharge() != 0 for a in atoms):
        out["reason"] = "formal_charge"
        return out
    try:
        san = _mdl_sanitize(Chem.Mol(mol))
    except Exception:
        out["reason"] = "sanitize_failed"
        return out
    if any(a.GetNumRadicalElectrons() > 0 for a in san.GetAtoms()):
        out["reason"] = "radical"
        return out
    zs = [a.GetAtomicNum() for a in san.GetAtoms()]
    if 1 not in zs or any(a.GetNumImplicitHs() > 0 for a in san.GetAtoms()):
        out["reason"] = "missing_explicit_h"
        return out
    first_h = zs.index(1)
    if any(z != 1 for z in zs[first_h:]):
        out["reason"] = "h_not_after_heavy"
        return out
    if len(Chem.GetMolFrags(san)) != 1:
        out["reason"] = "multi_fragment"
        return out
    if not set(zs) <= set(rules["elements"]):
        out["reason"] = "element_outside_geom_train"
        return out
    heavy = sum(1 for z in zs if z > 1)
    if heavy < rules["heavy_min"] or heavy > rules["heavy_max"]:
        out["reason"] = "heavy_atoms_outside_geom_train_range"
        return out
    if not any(b.GetBeginAtom().GetAtomicNum() > 1 and b.GetEndAtom().GetAtomicNum() > 1
               for b in san.GetBonds()):
        out["reason"] = "no_heavy_heavy_bond"
        return out
    try:
        key, key14, smi = _identity(san)
        if not key14 or not smi:
            raise ValueError("empty identity")
    except Exception:
        out["reason"] = "identity_failed"
        return out
    out.update({"key": key, "key14": key14, "smi": smi, "heavy": heavy,
                "n_atoms": len(zs), "elements": sorted(set(zs))})
    return out


# --------------------------------------------------------------------------- #
# select                                                                       #
# --------------------------------------------------------------------------- #

def _reference_scan(path: Path, workers: int, label: str):
    bounds = index_records(path)
    n = len(bounds) - 1
    print(f"[select] {label}: {n:,} records in {rel(path)}", flush=True)
    rows = []
    with Pool(workers, initializer=_init_worker, initargs=(None,)) as pool:
        for batch in batched(iter_records(path, bounds), BATCH):
            rows.extend(pool.map(analyze_reference, batch, chunksize=64))
            print(f"[select]   {label}: {len(rows):,}/{n:,}", flush=True)
    return rows


def cmd_select(args) -> None:
    from rdkit import rdBase
    from bondnet.data.dataset import _assign_split_label

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = out_dir / "cohort.sdf"
    if cohort_path.exists() and not args.overwrite:
        raise FileExistsError(f"{cohort_path} exists; pass --overwrite to rebuild")
    if (out_dir / "manifest.json").exists():
        raise RuntimeError("manifest.json exists: the cohort is frozen and must not be re-selected")

    source = resolve(args.source)
    geom = resolve(args.geom_sdf)
    excludes = [resolve(p) for p in args.exclude_sdf]
    for path in [source, geom, *excludes]:
        if not path.exists():
            raise FileNotFoundError(path)

    # ---- GEOM reference -------------------------------------------------- #
    geom_rows = _reference_scan(geom, args.workers, "GEOM")
    geom_key14, geom_smi = set(), set()
    train_elements, train_heavy = set(), []
    geom_stats = Counter()
    for row in geom_rows:
        geom_stats["records"] += 1
        if row["key14"]:
            geom_key14.add(row["key14"])
        else:
            geom_stats["no_inchikey"] += 1
        if row["smi"]:
            geom_smi.add(row["smi"])
        else:
            geom_stats["no_smiles"] += 1
        if row.get("geom_mol_idx") is None:
            geom_stats["no_geom_mol_idx"] += 1
            continue
        geom_stats["charged"] += int(bool(row.get("charged")))
        geom_stats["radical"] += int(bool(row.get("radical")))
        geom_stats["multi_fragment"] += int((row.get("n_frag") or 1) > 1)
        split = _assign_split_label({"geom_mol_idx": int(row["geom_mol_idx"])}, default_idx=0,
                                    val_frac=0.1, test_frac=0.1, seed=42)
        geom_stats[f"split_{split}"] += 1
        if split == "train":
            train_elements.update(row["elements"])
            train_heavy.append(row["heavy"])
    if not train_heavy:
        raise RuntimeError("no GEOM training molecules found; wrong --geom_sdf?")
    rules = {"elements": sorted(train_elements), "heavy_min": int(min(train_heavy)),
             "heavy_max": int(max(train_heavy))}
    print(f"[select] GEOM training elements {rules['elements']}, heavy atoms "
          f"{rules['heavy_min']}-{rules['heavy_max']}; stats {dict(geom_stats)}", flush=True)

    # ---- previously used cohorts ---------------------------------------- #
    prev_key14, prev_smi, prev_cid = set(), set(), set()
    exclude_meta = []
    for path in excludes:
        rows = _reference_scan(path, args.workers, path.name)
        n_key = sum(1 for r in rows if r["key14"])
        prev_key14.update(r["key14"] for r in rows if r["key14"])
        prev_smi.update(r["smi"] for r in rows if r["smi"])
        prev_cid.update(str(r["cid"]).strip() for r in rows if r.get("cid"))
        exclude_meta.append({"path": rel(path), "sha256": sha256_file(path), "records": len(rows),
                             "records_with_identity": n_key,
                             "records_with_pubchem_cid": sum(1 for r in rows if r.get("cid"))})

    # ---- candidates in seeded random order -------------------------------- #
    bounds = index_records(source)
    n_source = len(bounds) - 1
    order = np.random.default_rng(args.seed).permutation(n_source)
    print(f"[select] source: {n_source:,} records; visiting in permutation seed {args.seed}",
          flush=True)

    from bondnet.data.featurizer import MoleculeFeaturizer
    feat_explicit = MoleculeFeaturizer(cutoff=CACHE_CUTOFF, h_cutoff=CACHE_CUTOFF, explicit_h=True)
    feat_heavy = MoleculeFeaturizer(cutoff=CACHE_CUTOFF, h_cutoff=CACHE_CUTOFF, explicit_h=False)

    reasons = Counter()
    accepted = []
    seen_key14, seen_smi = set(), set()
    examined = 0
    with Pool(args.workers, initializer=_init_worker, initargs=(rules,)) as pool, \
            open(source, "rb") as handle:
        def stream():
            for batch in batched(iter_records(source, bounds, order), BATCH):
                yield from pool.map(analyze_candidate, batch, chunksize=32)

        for row in stream():
            examined += 1
            reason = row["reason"]
            if reason is None:
                if row["key14"] in geom_key14 or row["smi"] in geom_smi:
                    reason = "overlap_geom"
                elif (row["key14"] in prev_key14 or row["smi"] in prev_smi
                      or str(row["cid"]).strip() in prev_cid):
                    reason = "overlap_previous_cohort"
                elif row["key14"] in seen_key14 or row["smi"] in seen_smi:
                    reason = "duplicate_within_cohort"
            if reason is None:
                block = read_record(handle, bounds, row["k"])
                try:
                    _, mol = _parse(block)
                    _mdl_sanitize(mol)
                    feat_explicit.featurize(mol)
                    feat_heavy.featurize(mol)
                except Exception:
                    reason = "featurization_failed"
            if reason is not None:
                reasons[reason] += 1
            else:
                seen_key14.add(row["key14"])
                seen_smi.add(row["smi"])
                row["examined_rank"] = examined - 1
                accepted.append((row, block))
                if len(accepted) % 1000 == 0:
                    print(f"[select]   accepted {len(accepted):,}/{args.n:,} after "
                          f"{examined:,} candidates", flush=True)
                if len(accepted) >= args.n:
                    break
    if len(accepted) < args.n:
        raise RuntimeError(f"only {len(accepted)} eligible molecules in the whole source")

    # ---- write cohort ----------------------------------------------------- #
    cids = []
    with open(cohort_path, "w", encoding="utf-8", newline="\n") as fh:
        for ext_idx, (row, block) in enumerate(accepted):
            text = block.decode("utf-8", errors="replace").replace("\r\n", "\n")
            end = text.find("M  END")
            if end < 0:
                raise ValueError(f"record {row['k']} has no 'M  END' line")
            molblock = text[:end + len("M  END")]
            cid = int(str(row["cid"]).strip())
            cids.append(cid)
            props = OrderedDict([
                ("geom_mol_idx", cid),       # persistent id used for keyed noise
                ("pubchem_cid", cid),
                ("external_idx", ext_idx),
                ("source_record_index", row["k"]),
                ("selection_examined_rank", row["examined_rank"]),
                ("inchikey", row["key"]),
                ("canonical_smiles_nonisomeric", row["smi"]),
            ])
            fh.write(molblock + "\n")
            for name, value in props.items():
                fh.write(f">  <{name}>\n{value}\n\n")
            fh.write("$$$$\n")
    if len(set(cids)) != len(cids):
        raise RuntimeError("duplicate PubChem CID in cohort")
    (out_dir / "cids.txt").write_text("\n".join(map(str, cids)) + "\n", encoding="utf-8")

    heavy = [r["heavy"] for r, _ in accepted]
    element_counts = Counter(z for r, _ in accepted for z in r["elements"])
    selection = OrderedDict([
        ("created_utc", now_utc()),
        ("script", rel(Path(__file__))),
        ("rdkit_version", rdBase.rdkitVersion),
        ("source", {"path": rel(source), "sha256": sha256_file(source), "records": n_source,
                    "description": "PubChem3D conformers (MMFF94s-optimized, OMEGA-generated) "
                                   "downloaded by scripts/download_pubchem3d.py"}),
        ("geom_reference", {"path": rel(geom), "sha256": sha256_file(geom),
                            "stats": dict(geom_stats),
                            "unique_inchikey_blocks": len(geom_key14),
                            "unique_canonical_smiles": len(geom_smi)}),
        ("previous_cohorts", exclude_meta),
        ("identity", "InChIKey connectivity block (first 14 characters) OR canonical "
                     "non-isomeric SMILES of the heavy-atom graph (RDKit, MDL aromaticity)"),
        ("rules", {"order": list(RULE_ORDER), **rules,
                   "rule_derivation": "elements and heavy-atom range observed in GEOM "
                                      "random1/re10 training split (stable hash, seed 42, 80/10/10)"}),
        ("sampling", {"method": "first N eligible records in a seeded random permutation of all "
                                "source records (numpy default_rng(seed).permutation)",
                      "seed": int(args.seed), "n": int(args.n),
                      "candidates_examined": examined,
                      "exclusions_among_examined": {r: int(reasons.get(r, 0)) for r in RULE_ORDER}}),
        ("cohort", {"path": rel(cohort_path), "sha256": sha256_file(cohort_path),
                    "n_molecules": len(cids), "id_key": "geom_mol_idx (= PubChem CID)",
                    "ordered_id_sha256": sha256_ids(cids),
                    "cids_path": rel(out_dir / "cids.txt"),
                    "cids_sha256": sha256_file(out_dir / "cids.txt"),
                    "heavy_atoms": {"min": int(min(heavy)), "median": float(np.median(heavy)),
                                    "max": int(max(heavy))},
                    "molecules_containing_element": {str(z): int(c) for z, c in
                                                     sorted(element_counts.items())}}),
    ])
    write_json(out_dir / "selection.json", selection)
    print(f"[select] wrote {len(cids):,} molecules to {rel(cohort_path)}; "
          f"examined {examined:,}; exclusions {dict(reasons)}", flush=True)


# --------------------------------------------------------------------------- #
# materialize                                                                  #
# --------------------------------------------------------------------------- #

def _load_cache(path: Path):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["samples"], payload.get("metadata", {})


def cmd_materialize(args) -> None:
    import torch
    from rdkit import Chem
    from bondnet.data.dataset import CachedBondNetDataset, collate_fn
    from bondnet.data.noise_augment import GaussianNoiseAugment

    out_dir = resolve(args.out_dir)
    if (out_dir / "manifest.json").exists():
        raise RuntimeError("manifest.json exists: the cohort is already frozen")
    selection = json.loads((out_dir / "selection.json").read_text(encoding="utf-8"))
    cohort_path = out_dir / "cohort.sdf"
    if sha256_file(cohort_path) != selection["cohort"]["sha256"]:
        raise RuntimeError("cohort.sdf changed after selection")
    cids = [int(x) for x in (out_dir / "cids.txt").read_text().split()]

    caches = {
        "explicit": (out_dir / "cache_explicit_c30_h30.pt", ["--explicit_h"]),
        "heavy": (out_dir / "cache_heavy_c30.pt", []),
    }
    for name, (path, flags) in caches.items():
        if path.exists() and not args.overwrite:
            print(f"[materialize] reuse {rel(path)}")
            continue
        cmd = [sys.executable, str(REPO / "scripts" / "precompute_features.py"),
               "--data_path", str(cohort_path), "--output", str(path),
               "--cutoff", str(CACHE_CUTOFF), "--h_cutoff", str(CACHE_CUTOFF), *flags]
        print("[materialize] " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=REPO, check=True)

    # ---- structural checks ------------------------------------------------ #
    ref_mols = [m for m in Chem.SDMolSupplier(str(cohort_path), removeHs=False, sanitize=False)]
    if len(ref_mols) != len(cids) or any(m is None for m in ref_mols):
        raise RuntimeError("cohort.sdf could not be re-read completely")
    exp_samples, exp_meta = _load_cache(caches["explicit"][0])
    hvy_samples, hvy_meta = _load_cache(caches["heavy"][0])
    for label, samples in (("explicit", exp_samples), ("heavy", hvy_samples)):
        ids = [int(s["geom_mol_idx"]) for s in samples]
        if ids != cids:
            raise RuntimeError(f"{label} cache order/ids differ from cohort (n={len(ids)})")
    max_err = 0.0
    heavy_cache_retained_h = 0
    for mol, es, hs in zip(ref_mols, exp_samples, hvy_samples):
        zs = [a.GetAtomicNum() for a in mol.GetAtoms()]
        xyz = mol.GetConformer().GetPositions()
        if es["elems"].tolist() != zs:
            raise RuntimeError(f"explicit cache atom order differs for CID {es['geom_mol_idx']}")
        # The heavy-only featurizer calls Chem.RemoveHs, which keeps the rare
        # H that defines stereochemistry (e.g. an imine N-H); GEOM caches were
        # built the same way.  Require the heavy prefix and RemoveHs parity.
        heavy_n = sum(1 for z in zs if z > 1)
        stripped = Chem.RemoveHs(_mdl_sanitize(Chem.Mol(mol)))
        if (hs["elems"].tolist() != [a.GetAtomicNum() for a in stripped.GetAtoms()]
                or hs["elems"].tolist()[:heavy_n] != zs[:heavy_n]):
            raise RuntimeError(f"heavy cache atoms differ from RemoveHs for CID {hs['geom_mol_idx']}")
        heavy_cache_retained_h += int(len(hs["elems"]) > heavy_n)
        max_err = max(max_err, float(np.abs(es["coord"].numpy() - xyz).max()),
                      float(np.abs(hs["coord"].numpy()
                                    - stripped.GetConformer().GetPositions()).max()))
    if max_err > 1e-3:
        raise RuntimeError(f"cache/SDF coordinate mismatch {max_err:.3g} A")

    # ---- keyed noise, materialized for the rule baselines ------------------ #
    p0d = load_module("p0d_unified_robustness", REPO / "scripts" / "p0d_unified_robustness.py")
    noise = p0d._torch_keyed_noise(exp_samples, SIGMAS, EVAL_NOISE_SEED)
    outputs = OrderedDict()
    for sigma in SIGMAS:
        path = out_dir / f"sigma_{sigma_tag(sigma)}.sdf"
        writer = Chem.SDWriter(str(path))
        try:
            for pos, mol in enumerate(ref_mols):
                out = Chem.Mol(mol)
                if sigma > 0:
                    xyz = mol.GetConformer().GetPositions() + noise[sigma][pos]
                    conf = out.GetConformer()
                    for i, p in enumerate(xyz):
                        conf.SetAtomPosition(i, tuple(float(v) for v in p))
                out.SetProp("robustness_sigma_angstrom", f"{sigma:.4f}")
                out.SetIntProp("robustness_seed", EVAL_NOISE_SEED)
                out.SetIntProp("robustness_cache_position", pos)
                writer.write(out)
        finally:
            writer.close()

    # ---- parity: materialized noise == in-evaluation keyed noise ----------- #
    parity = {}
    n_check = min(args.parity_mols, len(cids))
    # Only the explicit-H cache is checked: its noise is what the rule tools
    # see.  The heavy-only cache receives the same per-molecule seed, but
    # torch's CPU normal sampler is not prefix-stable for different tensor
    # sizes, so heavy-only displacements are independent draws of the same
    # distribution, not the explicit-H heavy-atom displacements.
    for label, cache_path in (("explicit", caches["explicit"][0]),):
        ds = CachedBondNetDataset(str(cache_path))
        batch = collate_fn([ds[i] for i in range(n_check)])
        clean = batch["coord"].clone()
        counts = batch["num_atoms_per_mol"].tolist()
        worst = 0.0
        for sigma in SIGMAS[1:]:
            aug = GaussianNoiseAugment(sigma=sigma, sigma_min=sigma, sigma_max=sigma,
                                       cutoff=EVAL_CUTOFF)
            noisy = aug.augment_batch(
                {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()},
                per_molecule=True, base_seed=EVAL_NOISE_SEED + int(round(sigma * 10000.0)),
                epoch=0)["coord"]
            delta = (noisy - clean).numpy()
            off = 0
            for pos, cnt in enumerate(counts):
                expected = noise[sigma][pos][:cnt]
                worst = max(worst, float(np.abs(delta[off:off + cnt] - expected).max()))
                off += cnt
        parity[label] = {"molecules_checked": n_check, "max_abs_difference_angstrom": worst}
        if worst > 1e-5:
            raise RuntimeError(f"keyed-noise parity failed for {label}: {worst:.3g} A")

    # ---- the rule-baseline reader sees the same molecules in the same order - #
    p0c = load_module("p0c_yuelbond_compare", REPO / "scripts" / "p0c_yuelbond_compare.py")
    for sigma in SIGMAS:
        path = out_dir / f"sigma_{sigma_tag(sigma)}.sdf"
        mols = p0c._read_valid_molecules(str(path))
        ids = [int(m.GetProp("geom_mol_idx")) for m in mols]
        if ids != cids:
            raise RuntimeError(f"{path.name}: scorer reads {len(ids)} molecules / wrong order")
        outputs[f"{sigma:.2f}"] = {"path": rel(path), "sha256": sha256_file(path),
                                   "n_molecules": len(mols)}

    materialize = OrderedDict([
        ("created_utc", now_utc()),
        ("caches", {name: {"path": rel(path), "sha256": sha256_file(path),
                           "metadata": {k: v for k, v in meta.items()
                                        if isinstance(v, (int, float, str, bool, dict))}}
                    for (name, (path, _)), meta in zip(caches.items(), (exp_meta, hvy_meta))}),
        ("max_cache_sdf_coordinate_error_angstrom", max_err),
        ("heavy_cache_molecules_with_retained_stereo_h", heavy_cache_retained_h),
        ("noise", {"distribution": "iid Gaussian per Cartesian coordinate",
                   "seed": EVAL_NOISE_SEED, "sigmas_angstrom": list(SIGMAS),
                   "determinism": "GaussianNoiseAugment-compatible torch.Generator keyed by "
                                  "(seed + round(10000*sigma), geom_mol_idx = PubChem CID)",
                   "parity_with_evaluation_noise": parity,
                   "heavy_only_note": "heavy-only inputs use the same keyed seed but, because "
                                      "torch CPU randn is not prefix-stable across tensor sizes, "
                                      "receive independent draws of the same distribution"}),
        ("outputs", outputs),
    ])
    write_json(out_dir / "materialize.json", materialize)
    print(f"[materialize] caches, 3 noisy SDFs and parity checks OK "
          f"(max coord error {max_err:.2g} A; parity {parity})", flush=True)


# --------------------------------------------------------------------------- #
# freeze / verify                                                              #
# --------------------------------------------------------------------------- #

def _checkpoint_record(path: Path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    a = vars(a) if not isinstance(a, dict) else a
    info = {"path": rel(path), "sha256": sha256_file(path), "epoch": ck.get("epoch"),
            "training_cache": a.get("cache_path"), "training_data_dir": a.get("data_dir")}
    cache = str(info["training_cache"] or "").lower()
    if "pubchem" in cache or "geom_drugs_all_random1_re10" not in cache:
        raise RuntimeError(f"{path}: unexpected training cache {info['training_cache']!r}")
    if a.get("data_dir") and "pubchem" in str(a.get("data_dir")).lower():
        raise RuntimeError(f"{path}: training data_dir mentions PubChem")
    return info


def _results_exist(results_dir: Path) -> list[str]:
    if not results_dir.exists():
        return []
    return [rel(p) for p in results_dir.rglob("*") if p.is_file()]


def cmd_freeze(args) -> None:
    import torch
    from rdkit import rdBase

    out_dir = resolve(args.out_dir)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"{rel(manifest_path)} already exists; the cohort is frozen")
    existing = _results_exist(resolve(args.results_dir))
    if existing:
        raise RuntimeError("external results already exist, so the cohort cannot be frozen "
                           f"before evaluation: {existing[:5]}")
    selection = json.loads((out_dir / "selection.json").read_text(encoding="utf-8"))
    materialize = json.loads((out_dir / "materialize.json").read_text(encoding="utf-8"))

    files = OrderedDict()
    files[selection["cohort"]["path"]] = selection["cohort"]["sha256"]
    files[selection["cohort"]["cids_path"]] = selection["cohort"]["cids_sha256"]
    for item in materialize["caches"].values():
        files[item["path"]] = item["sha256"]
    for item in materialize["outputs"].values():
        files[item["path"]] = item["sha256"]
    files[rel(out_dir / "selection.json")] = sha256_file(out_dir / "selection.json")
    files[rel(out_dir / "materialize.json")] = sha256_file(out_dir / "materialize.json")
    for path, digest in files.items():
        if sha256_file(resolve(path)) != digest:
            raise RuntimeError(f"{path} changed since it was written")

    code = OrderedDict()
    for path in PROTOCOL_CODE:
        p = resolve(path)
        if not p.exists():
            raise FileNotFoundError(f"protocol file missing: {path}")
        code[path] = sha256_file(p)

    checkpoints = OrderedDict()
    for name, spec in CHECKPOINTS.items():
        entry = {"model_file": _checkpoint_record(resolve(spec["checkpoint"])),
                 "cache": materialize["caches"][spec["cache"]]["path"]}
        if "stage2_ckpt" in spec:
            entry["stage2_file"] = _checkpoint_record(resolve(spec["stage2_ckpt"]))
        checkpoints[name] = entry

    manifest = OrderedDict([
        ("title", "BondNet revision v3 external confirmation cohort (PubChem3D)"),
        ("frozen_utc", now_utc()),
        ("statement", "This manifest was written before any model or baseline was run on the "
                      "cohort. Each model and baseline is evaluated once with the settings below; "
                      "no parameter is tuned on this cohort, and every result is reported."),
        ("environment", {"python": platform.python_version(), "platform": platform.platform(),
                         "torch": torch.__version__, "rdkit": rdBase.rdkitVersion}),
        ("selection", selection),
        ("materialization", materialize),
        # p0d_unified_robustness.py `run` reads this key to validate the SDFs.
        ("outputs", materialize["outputs"]),
        ("frozen_files_sha256", files),
        ("protocol_code_sha256", code),
        ("evaluation_plan", OrderedDict([
            ("models", checkpoints),
            ("rule_baselines", ["RDKit rdDetermineBonds (charge 0; failures counted)",
                                "OpenBabel ConnectTheDots + PerceiveBondOrders"]),
            ("sigmas_angstrom", list(SIGMAS)),
            ("eval_noise_seed", EVAL_NOISE_SEED),
            ("learned_settings", {"cutoff": EVAL_CUTOFF, "h_cutoff": EVAL_CUTOFF,
                                  "conn_threshold": CONN_THRESHOLD, "batch_size": 128,
                                  "script": "scripts/export_per_molecule_stats.py"}),
            ("scoring", "Same as Table 5: heavy-heavy pairs scored as undirected pairs "
                        "(OR across directed records, labels must agree); directed strict "
                        "scores reported alongside (Table S1 convention)."),
            ("metrics", ["pipeline macro-F1 (undirected; primary)", "pipeline macro-F1 (directed)",
                         "reference-pair (conditional) macro-F1", "molecule exact-type rate",
                         "HH-graph exact rate", "connectivity exact rate",
                         "rule-tool success rate", "joint-model sanitizable-graph rate"]),
            ("summary", "mean and sample SD over training seeds 42/43/44; rule tools are "
                        "deterministic (single run)"),
            ("runner", "scripts/run_revision_v3_external.ps1"),
            ("deviations", "Any deviation from this plan is reported in the manuscript."),
        ])),
    ])
    write_json(manifest_path, manifest)
    digest = sha256_file(manifest_path)
    (out_dir / "manifest.sha256").write_text(f"{digest}  manifest.json\n", encoding="utf-8")
    print(f"[freeze] wrote {rel(manifest_path)}")
    print(f"[freeze] manifest SHA-256 {digest}")
    print("[freeze] record this hash with a timestamp before running any evaluation.")


def cmd_verify(args) -> None:
    out_dir = resolve(args.out_dir)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = (out_dir / "manifest.sha256").read_text().split()[0]
    problems = []
    if sha256_file(manifest_path) != recorded:
        problems.append("manifest.json does not match manifest.sha256")
    for path, digest in manifest["frozen_files_sha256"].items():
        if sha256_file(resolve(path)) != digest:
            problems.append(f"data changed: {path}")
    for path, digest in manifest["protocol_code_sha256"].items():
        if sha256_file(resolve(path)) != digest:
            problems.append(f"protocol code changed: {path}")
    for name, entry in manifest["evaluation_plan"]["models"].items():
        for key in ("model_file", "stage2_file"):
            if key in entry and sha256_file(resolve(entry[key]["path"])) != entry[key]["sha256"]:
                problems.append(f"checkpoint changed: {entry[key]['path']}")
    if problems:
        for p in problems:
            print(f"[verify] FAIL {p}")
        raise SystemExit(1)
    print(f"[verify] OK: manifest {recorded} (frozen {manifest['frozen_utc']}); "
          f"{len(manifest['frozen_files_sha256'])} data files, "
          f"{len(manifest['protocol_code_sha256'])} code files, "
          f"{len(manifest['evaluation_plan']['models'])} checkpoints unchanged")


# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("select")
    s.add_argument("--source", default=DEFAULT_SOURCE)
    s.add_argument("--geom_sdf", default=DEFAULT_GEOM)
    s.add_argument("--exclude_sdf", nargs="*", default=list(DEFAULT_EXCLUDES))
    s.add_argument("--n", type=int, default=DEFAULT_N)
    s.add_argument("--seed", type=int, default=DEFAULT_SELECTION_SEED)
    s.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    s.add_argument("--out_dir", default=DEFAULT_OUT)
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(func=cmd_select)

    m = sub.add_parser("materialize")
    m.add_argument("--out_dir", default=DEFAULT_OUT)
    m.add_argument("--parity_mols", type=int, default=256)
    m.add_argument("--overwrite", action="store_true")
    m.set_defaults(func=cmd_materialize)

    f = sub.add_parser("freeze")
    f.add_argument("--out_dir", default=DEFAULT_OUT)
    f.add_argument("--results_dir", default=DEFAULT_RESULTS)
    f.set_defaults(func=cmd_freeze)

    v = sub.add_parser("verify")
    v.add_argument("--out_dir", default=DEFAULT_OUT)
    v.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
