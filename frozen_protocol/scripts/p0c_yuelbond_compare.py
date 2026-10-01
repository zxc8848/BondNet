#!/usr/bin/env python3
# P0(c) harness -- BondNet vs YuelBond, unified scorer + corrected rule baselines.
r"""
P0(c) -- Head-to-head comparison harness: BondNet vs. YuelBond (reviewer #9).

Goal
----
Compare BondNet and YuelBond (Wang & Dokholyan, JCIM 2026, 66(2):1003-1012;
code: https://bitbucket.org/dokhlab/yuel_bond) on the SAME molecules with the SAME coordinates
and the SAME metric -- including a pipeline-level metric that penalizes extra/missing bonds, so the
comparison is apples-to-apples (not conditional-on-success F1 vs. a 100%-success model).

Subcommands
-----------
  build   : sample N molecules from a leakage-free set and write a clean SDF + a fixed-seed
            Gaussian-noise SDF at --sigma. Both tools must run on these exact files.
  bondnet : run BondNet and export normalized per-molecule predictions for the unified scorer.
  rule    : run the CORRECTED RDKit/OpenBabel baselines (per-molecule formal charge + atom-pair
              matching) and emit normalized heavy-index predictions for the unified scorer.
  adapt   : convert a YuelBond/raw prediction JSON into the normalized, fingerprinted format.
  score   : score ANY tool's predictions on the shared SDF with the unified metric
            (matched macro-F1, true-bond exact-type, full-graph exact match, extra-bond rate,
            success rate). Use it for BondNet, RDKit, OpenBabel, and YuelBond alike.

IMPORTANT: do NOT use `evaluate.py --run_baselines` for the rule tools -- its inline RDKit/OpenBabel
path uses charge=0 and positional (index-order) bond matching and is broken (0% validity on clean
molecules). The `rule` subcommand here uses the corrected predictor.

Also: do NOT rebuild the shared SDF between `rule` and `score`. `rule` fingerprints the molecule set;
`score` refuses to run if the predictions were made for a different SDF (prevents index-mismatch
garbage where every bond looks "extra").

Normalized prediction JSON format (what `score` expects)
--------------------------------------------------------
  { "<mol_index_in_sdf>": [[i, j, order], ...], ... }
  where i, j are 0-based HEAVY-atom indices (H removed), and order is one of:
  1 (single), 2 (double), 3 (triple), "ar"/1.5 (aromatic). Molecules a tool failed on are omitted
  -> counted against its success rate. (`rule` also adds a "__meta__" fingerprint key.)

Workflow
--------
  python scripts/p0c_yuelbond_compare.py build   --source data/heldout_geom.sdf --n 5000 --sigma 0.1
  python scripts/p0c_yuelbond_compare.py bondnet
  python scripts/p0c_yuelbond_compare.py rule --sdf data/shared_bench_clean.sdf
  python scripts/p0c_yuelbond_compare.py rule --sdf data/shared_bench_noisy.sdf
  python scripts/p0c_yuelbond_compare.py score --sdf data/shared_bench_clean.sdf \
         --pred data/bondnet_shared_bench_clean_preds.json --name BondNet
  python scripts/p0c_yuelbond_compare.py score --sdf data/shared_bench_clean.sdf \
         --pred data/rdkit_shared_bench_clean_preds.json --name RDKit

  # After running YuelBond on the exact same SDF, normalize and fingerprint its raw JSON:
  python scripts/p0c_yuelbond_compare.py adapt --sdf data/shared_bench_clean.sdf \
         --input yuelbond_raw_clean.json --output data/yuelbond_shared_bench_clean_preds.json \
         --atom-index-space full --zero-based-classes
  python scripts/p0c_yuelbond_compare.py score --sdf data/shared_bench_clean.sdf \
         --pred data/yuelbond_shared_bench_clean_preds.json --name YuelBond

Repeat `score` for the noisy SDF and for OpenBabel. Prediction JSONs carry a SHA-256 fingerprint;
the scorer refuses stale or mismatched files.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = lambda *p: os.path.join(REPO, *p)

STAGE1 = D("checkpoints", "geom_drugs_all_random1_re10_stage1_c25", "best.pt")
# `best.pt` is the validation-loss-selected Stage-2 checkpoint.  A 2x2 checkpoint
# ablation on the shared 5k cohort showed that it, rather than `best_e2e.pt`, is
# responsible for the large clean/noisy typing improvement.  Keep this choice
# explicit: the two files contain different weights.
STAGE2 = D("checkpoints", "stage2_c25_officialcache_noise010", "best.pt")
CLEAN = D("data", "shared_bench_clean.sdf")
NOISY = D("data", "shared_bench_noisy.sdf")

_LABEL = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
_NAMES = ["single", "double", "triple", "aromatic"]

def _resolve(path):
    """Make a path absolute; if relative, resolve it against the repo root (not the cwd)."""
    return path if os.path.isabs(path) else os.path.join(REPO, path)


def _open_supplier(path):
    """Open an SDF with a clear error if it is missing or empty (RDKit's OSError is opaque)."""
    from rdkit import Chem
    p = _resolve(path)
    if not os.path.exists(p):
        sys.exit(f"[missing] {p}\n"
                 f"  Run the previous step first (e.g. `build`), or pass an existing SDF.")
    if os.path.getsize(p) == 0:
        sys.exit(f"[empty] {p} has 0 bytes -- the previous step wrote nothing.\n"
                 f"  Check that its input SDF existed and contained molecules.")
    return Chem.SDMolSupplier(p, removeHs=False, sanitize=False)


def _read_valid_molecules(path):
    """Read the exact sanitized molecule sequence used by rule/adapt/score."""
    mols = []
    for mol in _open_supplier(path):
        if mol is None:
            continue
        try:
            _mdl(mol)
        except Exception:
            continue
        mols.append(mol)
    return mols


def _sha256(path):
    h = hashlib.sha256()
    with open(_resolve(path), "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _prediction_meta(path, mols):
    return {
        "sdf": os.path.basename(path),
        "sdf_sha256": _sha256(path),
        "n_molecules": len(mols),
        "heavy": [_heavy_count(m) for m in mols],
    }


def _checkpoint_meta(path):
    path = _resolve(path)
    return {
        "path": os.path.relpath(path, REPO),
        "sha256": _sha256(path),
    }


def _prediction_path(method, sdf):
    return os.path.join(
        os.path.dirname(_resolve(sdf)),
        f"{method}_{os.path.splitext(os.path.basename(sdf))[0]}_preds.json",
    )


def _sdf_row_key(batch_start, local_index):
    """The shared scorer keys predictions by SDF row, not geom_mol_idx."""
    return str(int(batch_start) + int(local_index))


# --------------------------------------------------------------------------------------------
def _mdl(mol):
    from rdkit import Chem
    Chem.SanitizeMol(mol)
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
    Chem.SanitizeMol(mol)
    return mol


def _true_bonds(mol):
    """Heavy-heavy bonds of a ground-truth molecule -> {(i,j): label} on the H-removed index space."""
    hm = _heavy_map(mol)
    out = {}
    for b in mol.GetBonds():
        full_i, full_j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if full_i not in hm or full_j not in hm:
            continue
        i, j = hm[full_i], hm[full_j]
        bt = "AROMATIC" if b.GetIsAromatic() else b.GetBondType().name
        out[(min(i, j), max(i, j))] = _LABEL.get(bt, 0)
    return out


def _pred_label(o):
    """Map a predicted bond order (1/2/3, or 'ar'/1.5 for aromatic) to a class label 0..3."""
    if isinstance(o, str):
        return 3 if o.lower().startswith("ar") else {1: 0, 2: 1, 3: 2}[int(float(o))]
    if abs(float(o) - 1.5) < 0.1:
        return 3  # aromatic
    return {1: 0, 2: 1, 3: 2}[int(round(float(o)))]


def _heavy_map(mol):
    """full-atom index -> heavy-atom index (matches Chem.RemoveHs re-indexing)."""
    hm = {}; h = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            hm[atom.GetIdx()] = h; h += 1
    return hm


def _heavy_count(mol):
    """Count atoms with Z>1; unlike Chem.RemoveHs this never retains special hydrogens."""
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


# --------------------------------------------------------------------------------------------
def _count_sdf_records(path):
    count = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip() == b"$$$$":
                count += 1
    return count


def _pick_source(user_source, min_records):
    src = _resolve(user_source)
    if not os.path.exists(src):
        sys.exit(f"[missing] --source {src}")
    if _count_sdf_records(src) < min_records:
        sys.exit(f"[too small] --source {src} contains fewer than {min_records} SDF records")
    return src


def cmd_build(a):
    from rdkit import Chem
    import numpy as np
    source = _pick_source(a.source, a.n)
    print(f"[build] source = {source}")
    if "geom" in os.path.basename(source).lower():
        print("[caution] YuelBond was trained on GEOM. Unless this cohort is proven disjoint from "
              "its split, the comparison may favor YuelBond. A GEOM-excluded PubChem held-out "
              "cohort is safer for a paper-level head-to-head comparison.")
    supp = _open_supplier(source)
    rng = np.random.default_rng(a.seed)
    # Deterministic reservoir sampling avoids the previous "first N molecules"
    # bias without holding the entire source in memory.
    reservoir = []
    n_usable = 0
    for mol in supp:
        if mol is None:
            continue
        try:
            _mdl(mol)
        except Exception:
            continue
        if mol.GetNumConformers() == 0:
            continue
        n_usable += 1
        copy = Chem.Mol(mol)
        if len(reservoir) < a.n:
            reservoir.append(copy)
        else:
            j = int(rng.integers(0, n_usable))
            if j < a.n:
                reservoir[j] = copy

    if len(reservoir) != a.n:
        sys.exit(
            f"[error] requested {a.n} molecules but source contained only "
            f"{n_usable} usable sanitized 3D molecules. No benchmark was written."
        )
    rng.shuffle(reservoir)

    clean_out = _resolve(a.clean_out)
    noisy_out = _resolve(a.noisy_out)
    os.makedirs(os.path.dirname(clean_out), exist_ok=True)
    os.makedirs(os.path.dirname(noisy_out), exist_ok=True)
    wc = Chem.SDWriter(clean_out); wc.SetKekulize(False)
    wn = Chem.SDWriter(noisy_out); wn.SetKekulize(False)
    for mol in reservoir:
        wc.write(mol)
        noisy = Chem.Mol(mol)
        conf = noisy.GetConformer()
        for k in range(noisy.GetNumAtoms()):
            p = conf.GetAtomPosition(k)
            d = rng.normal(0.0, a.sigma, 3)
            conf.SetAtomPosition(k, (p.x + d[0], p.y + d[1], p.z + d[2]))
        wn.write(noisy)
    wc.close(); wn.close()
    clean_mols = _read_valid_molecules(clean_out)
    noisy_mols = _read_valid_molecules(noisy_out)
    if len(clean_mols) != a.n or len(noisy_mols) != a.n:
        sys.exit(
            f"[error] post-write validation failed: clean={len(clean_mols)}, "
            f"noisy={len(noisy_mols)}, expected={a.n}."
        )
    manifest = {
        "source": os.path.abspath(source), "n": a.n, "sigma": a.sigma, "seed": a.seed,
        "clean": _prediction_meta(clean_out, clean_mols),
        "noisy": _prediction_meta(noisy_out, noisy_mols),
    }
    manifest_path = os.path.join(os.path.dirname(clean_out), "shared_bench_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {a.n} randomly sampled molecules:\n  clean -> {clean_out}"
          f"\n  noisy(sigma={a.sigma}) -> {noisy_out}\n  manifest -> {manifest_path}")
    print("Run BOTH tools on these exact files. Do NOT rebuild between `rule` and `score`.")


def _load_stage2(path, model, device):
    import torch
    from bondnet.model.bond_type_gnn import BondTypeGNN
    ck = torch.load(path, map_location="cpu", weights_only=False)
    args = ck.get("args", {})
    stage2 = BondTypeGNN(
        hidden_size=args.get("hidden_size", model.backbone.hidden_state_size),
        num_layers=args.get("num_layers", 4),
        edge_embedding_size=args.get("edge_embedding_size", 16),
        bond_cutoff=args.get("bond_cutoff", 3.0),
        random_node_feat_dim=args.get("random_node_feat_dim", 0),
        random_node_feat_std=args.get("random_node_feat_std", 1.0),
    ).to(device)
    stage2.load_state_dict(ck["stage2_state_dict"])
    stage2.eval()
    return stage2


def _export_bondnet_predictions(stage1_path, stage2_path, sdf, out_path, batch_size=64,
                                device_name="auto", h_cutoff=2.0):
    """Export the hard-staged model or joint model when stage2_path is None."""
    import torch
    from torch.utils.data import DataLoader
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    import evaluate as ev
    from bondnet.data.dataset import collate_fn
    from bondnet.data.noise_augment import apply_dynamic_candidate_mask
    from bondnet.model.bond_type_gnn import compute_bonded_geometry
    from bondnet.model.bondnet import _symmetrize_logits

    device = ev._get_device(device_name)
    model = ev.load_model(stage1_path, device)
    model.eval()
    stage2 = _load_stage2(stage2_path, model, device) if stage2_path else None
    ds_args = SimpleNamespace(
        cache_path=None, dataset="sdf", data_path=_resolve(sdf), data_dir=None,
        pyg_root=None, wiberg_npz=None, max_mols=None, shard_cache_size=2,
        split_key=None, train_groups=None, eval_split="all", explicit_h=True,
        cutoff=2.5, h_cutoff=h_cutoff,
    )
    dataset = ev.load_test_dataset(ds_args, sigma=0.0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    order = {0: 1, 1: 2, 2: 3, 3: "ar"}
    preds = {}
    exported = 0

    with torch.no_grad():
        for batch in loader:
            if not batch:
                continue
            batch = apply_dynamic_candidate_mask(batch, 2.5, h_cutoff)
            batch = ev._to_device(batch, device)
            elems, coord = batch["elems"], batch["coord"]
            edge_index = batch["edge_index"]
            _, _, edge_feat = model.backbone(
                elems, coord, edge_index, batch["edge_diff"], batch["edge_dist"],
                num_atoms_per_mol=batch.get("num_atoms_per_mol"),
            )
            conn = torch.sigmoid(model.stage1(edge_feat)) >= 0.5
            if batch.get("train_edge_mask") is not None:
                conn = conn & batch["train_edge_mask"]
            bonded_pos = torch.where(conn)[0]
            if bonded_pos.numel():
                bonded_ei = edge_index[bonded_pos]
                b_diff, b_dist = compute_bonded_geometry(coord, bonded_ei)
                hh = (elems[bonded_ei[:, 0]] != 1) & (elems[bonded_ei[:, 1]] != 1)
                readout_pos = bonded_pos[hh]
                readout_ei = bonded_ei[hh]
                if readout_ei.numel():
                    if stage2 is None:
                        logits, _ = model.stage3(edge_feat[readout_pos])
                    else:
                        r_diff, r_dist = compute_bonded_geometry(coord, readout_ei)
                        logits = stage2(elems, bonded_ei, b_diff, b_dist,
                                        readout_ei, r_diff, r_dist)
                    labels = _symmetrize_logits(logits, readout_ei).argmax(dim=-1)
                    edge_mols = batch["edge_mol_idx"][readout_pos]
                else:
                    labels = torch.zeros(0, dtype=torch.long, device=device)
                    edge_mols = torch.zeros(0, dtype=torch.long, device=device)
            else:
                readout_ei = torch.zeros((0, 2), dtype=torch.long, device=device)
                labels = torch.zeros(0, dtype=torch.long, device=device)
                edge_mols = torch.zeros(0, dtype=torch.long, device=device)

            sizes = batch["num_atoms_per_mol"].detach().cpu().tolist()
            mol_ids = batch["mol_idx"].detach().cpu().tolist()
            offsets = []
            total = 0
            for size in sizes:
                offsets.append(total); total += int(size)
            local_maps = []
            elems_cpu = elems.detach().cpu().tolist()
            for off, size in zip(offsets, sizes):
                hm = {}; heavy_i = 0
                for local_i, z in enumerate(elems_cpu[off:off + int(size)]):
                    if int(z) > 1:
                        hm[local_i] = heavy_i; heavy_i += 1
                local_maps.append(hm)
            per_batch = [dict() for _ in sizes]
            for edge, label, mol_i in zip(
                    readout_ei.detach().cpu().tolist(),
                    labels.detach().cpu().tolist(),
                    edge_mols.detach().cpu().tolist()):
                mi = int(mol_i); off = offsets[mi]
                li, lj = int(edge[0]) - off, int(edge[1]) - off
                hm = local_maps[mi]
                if li not in hm or lj not in hm:
                    continue
                pair = tuple(sorted((hm[li], hm[lj])))
                previous = per_batch[mi].get(pair)
                if previous is not None and previous != int(label):
                    raise RuntimeError(f"conflicting directed predictions for molecule {mol_ids[mi]} edge {pair}")
                per_batch[mi][pair] = int(label)
            for mi, mol_id in enumerate(mol_ids):
                # collate_fn exposes persistent geom_mol_idx in batch['mol_idx'];
                # the shared-cohort scorer instead keys records by SDF row.
                preds[_sdf_row_key(exported, mi)] = [
                    [i, j, order[label]] for (i, j), label in sorted(per_batch[mi].items())
                ]
            exported += len(mol_ids)

    mols = _read_valid_molecules(sdf)
    if len(preds) != len(mols) or set(preds) != {str(i) for i in range(len(mols))}:
        raise RuntimeError(
            f"BondNet exported {len(preds)} molecules, but SDF row keys 0..{len(mols)-1} "
            f"were not all present for {sdf}."
        )
    meta = _prediction_meta(sdf, mols)
    meta.update({
        "method": "BondNet-joint" if stage2_path is None else "BondNet",
        "stage1": _checkpoint_meta(stage1_path),
        "stage2": _checkpoint_meta(stage2_path) if stage2_path else None,
        "h_cutoff": h_cutoff,
    })
    payload = {"__meta__": meta, **preds}
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"[bondnet] exported {len(preds)} fingerprinted predictions -> {out_path}")


def cmd_bondnet(a):
    clean = _resolve(a.clean)
    noisy = _resolve(a.noisy)
    stage1 = _resolve(a.stage1)
    stage2 = _resolve(a.stage2)
    if not (os.path.exists(clean) and os.path.exists(noisy)):
        sys.exit(f"[missing] shared benchmark SDFs not found.\n  {CLEAN}\n  {NOISY}\n"
                   f"Run `python scripts/p0c_yuelbond_compare.py build` first.")
    for checkpoint in (stage1, stage2):
        if not os.path.exists(checkpoint):
            sys.exit(f"[missing] checkpoint {checkpoint}")
    print(f"[BondNet] Stage 1: {stage1}")
    print(f"[BondNet] Stage 2: {stage2}")
    pairs = {"clean": (clean, "clean"), "noisy": (noisy, "noisy")}
    selected = pairs.values() if a.which == "both" else [pairs[a.which]]
    for sdf, tag in selected:
        if os.path.getsize(sdf) == 0:
            print(f"[skip] {sdf} is empty."); continue
        bench_name = os.path.splitext(os.path.basename(sdf))[0]
        out_dir = D("results", "p0c", bench_name)
        os.makedirs(out_dir, exist_ok=True)
        # NOTE: do NOT use evaluate.py --run_baselines here -- its inline RDKit/OpenBabel baseline
        # uses charge=0 and positional (index-order) bond matching, which is broken. Score the rule
        # tools with the `rule` subcommand, which uses the corrected atom-pair-matched baseline.
        cmd = [sys.executable, D("evaluate.py"),
               "--checkpoint", stage1, "--stage2_ckpt", stage2,
               "--allow_stage2_cache_mismatch",
               "--dataset", "sdf", "--data_path", sdf, "--eval_split", "all",
               "--noise_levels", "0.0",
               "--output_dir", out_dir]
        print("\n$ " + " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=REPO)
        js = sorted(glob.glob(os.path.join(out_dir, "*.json")), key=os.path.getmtime)
        js = [j for j in js if "error_analysis" not in os.path.basename(j)]
        if js:
            with open(js[-1]) as f:
                r = json.load(f)
            row = None
            if isinstance(r, dict) and "BondNet" in r:
                row = r["BondNet"][0]
            elif isinstance(r, list):
                row = r[0]
            if row:
                print(f"[BondNet {tag}] macro-F1={row.get('f1_macro'):.4f} "
                      f"conn-F1={row.get('conn_f1'):.4f} mol-valid={row.get('mol_validity'):.3f}")
        pred_path = _prediction_path("bondnet", sdf)
        _export_bondnet_predictions(stage1, stage2, sdf, pred_path,
                                    batch_size=a.batch_size, device_name=a.device)
    print("\nBondNet predictions and aggregate metrics saved. Next run rule+score on:")
    print(f"  python scripts/p0c_yuelbond_compare.py rule --sdf {clean}")
    print(f"  python scripts/p0c_yuelbond_compare.py rule --sdf {noisy}")


def cmd_joint(a):
    """Score corrected joint checkpoints on the archived shared 5k SDFs."""
    checkpoint = _resolve(a.checkpoint)
    if not os.path.exists(checkpoint):
        sys.exit(f"[missing] checkpoint {checkpoint}")
    for tag, sdf in (("clean", a.clean), ("noisy", a.noisy)):
        sdf = _resolve(sdf)
        if not os.path.exists(sdf):
            sys.exit(f"[missing] shared benchmark SDF {sdf}")
        if a.which != "both" and tag != a.which:
            continue
        out_path = D("results", "p0c", "joint_v3", f"seed{a.seed}_{tag}_preds.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        _export_bondnet_predictions(checkpoint, None, sdf, out_path,
                                    batch_size=a.batch_size, device_name=a.device,
                                    h_cutoff=2.5)
        score_path = D("results", "p0c", "joint_v3", f"seed{a.seed}_{tag}_score.json")
        cmd_score(SimpleNamespace(sdf=sdf, pred=out_path,
                                  name=f"BondNet-joint-seed{a.seed}",
                                  output=score_path, allow_unfingerprinted=False,
                                  subset_split="test"))


def cmd_rule(a):
    """Run the CORRECTED RDKit/OpenBabel baselines and emit normalized heavy-index predictions,
    so the rule tools can be scored through the SAME `score` function as BondNet and YuelBond."""
    from rdkit import Chem
    sys.path.insert(0, D("scripts"))
    from evaluate_rule_baselines import _rdkit_predict, _openbabel_predict
    order = {0: 1, 1: 2, 2: 3, 3: "ar"}
    predictors = {"rdkit": _rdkit_predict}
    try:
        from openbabel import openbabel as _ob  # noqa: F401
        predictors["openbabel"] = _openbabel_predict
    except Exception:
        print("[warn] OpenBabel not importable -- skipping it.")

    methods = a.methods or list(predictors)
    mols = _read_valid_molecules(a.sdf)

    # Fingerprint the exact molecule set so `score` can detect a preds/SDF mismatch.
    for method in methods:
        if method not in predictors:
            print(f"[skip] {method} unavailable."); continue
        preds = {"__meta__": _prediction_meta(a.sdf, mols)}
        n_fail = 0
        for idx, mol in enumerate(mols):
            try:
                raw = predictors[method](mol)                     # {(full_i,full_j): label}
            except Exception:
                n_fail += 1; continue                             # tool failed -> omit
            hm = _heavy_map(mol)
            hh = []
            for (i, j), lab in raw.items():
                if i in hm and j in hm:                           # heavy-heavy only
                    hh.append([hm[i], hm[j], order[lab]])
            preds[str(idx)] = hh
        out = _prediction_path(method, a.sdf)
        with open(out, "w") as f:
            json.dump(preds, f)
        print(f"[{method}] wrote {len(preds) - 1} predictions ({n_fail} tool-failures) -> {out}")
        print(f"    score it:  python scripts/p0c_yuelbond_compare.py score "
              f"--sdf {a.sdf} --pred {out} --name {method}")


def _first_present(d, keys, default=None):
    for key in keys:
        if key in d:
            return d[key]
    return default


def _raw_bond(entry):
    if isinstance(entry, (list, tuple)) and len(entry) >= 3:
        return entry[0], entry[1], entry[2]
    if not isinstance(entry, dict):
        raise ValueError(f"unsupported bond record: {entry!r}")
    i = _first_present(entry, ("i", "atom_i", "begin", "source", "src", "a1"))
    j = _first_present(entry, ("j", "atom_j", "end", "target", "dst", "a2"))
    value = _first_present(entry, ("order", "bond_order", "type", "prediction", "pred"))
    if value is None:
        probs = _first_present(entry, ("probabilities", "probs", "logits"))
        if probs is not None:
            value = int(max(range(len(probs)), key=lambda k: probs[k]))
    if i is None or j is None or value is None:
        raise ValueError(f"bond record lacks atom indices or prediction: {entry!r}")
    return i, j, value


def _iter_raw_predictions(raw):
    """Yield (molecule index, bond records) from common JSON layouts."""
    if isinstance(raw, dict) and "predictions" in raw:
        raw = raw["predictions"]
    if isinstance(raw, dict):
        for key, bonds in raw.items():
            if str(key).startswith("__"):
                continue
            yield int(key), bonds.get("bonds", []) if isinstance(bonds, dict) else bonds
        return
    if isinstance(raw, list):
        grouped = {}
        for record in raw:
            if not isinstance(record, dict):
                raise ValueError("list prediction JSON must contain objects")
            mol_idx = _first_present(record, ("mol_idx", "molecule_index", "molecule", "index"))
            if mol_idx is None:
                raise ValueError(f"prediction record lacks molecule index: {record!r}")
            if "bonds" in record:
                grouped.setdefault(int(mol_idx), []).extend(record["bonds"])
            else:
                grouped.setdefault(int(mol_idx), []).append(record)
        yield from sorted(grouped.items())
        return
    raise ValueError("unsupported raw prediction JSON; expected dict or list")


def _adapt_order(value, zero_based_classes, class_order):
    if isinstance(value, str):
        low = value.lower()
        if low in {"none", "no_bond", "no-bond", "nonbond", "non-bond"}:
            return None
        if low.startswith("ar"):
            return "ar"
        names = {"single": 1, "double": 2, "triple": 3}
        if low in names:
            return names[low]
        value = float(value)
    value = float(value)
    if zero_based_classes and value.is_integer() and 0 <= int(value) < len(class_order):
        return _adapt_order(class_order[int(value)], False, class_order)
    if abs(value - 1.5) < 0.1:
        return "ar"
    iv = int(round(value))
    if iv not in (1, 2, 3):
        raise ValueError(f"unsupported bond order/class value {value!r}")
    return iv


def cmd_adapt(a):
    """Normalize an external/YuelBond JSON and bind it cryptographically to the target SDF."""
    from rdkit import Chem
    mols = _read_valid_molecules(a.sdf)
    out = {"__meta__": _prediction_meta(a.sdf, mols)}
    input_path = _resolve(a.input)

    # Official YuelBond writes a heavy-atom-only SDF. Molecule names are
    # ``mol_N`` where N is the 1-based position in the original input SDF;
    # use that identifier rather than output order so skipped failures cannot
    # shift every subsequent prediction.
    if input_path.lower().endswith(".sdf"):
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.warning")
        supplier = Chem.SDMolSupplier(
            input_path, removeHs=False, sanitize=False, strictParsing=False
        )
        seen = set()
        for output_pos, mol in enumerate(supplier):
            if mol is None:
                continue
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            match = re.fullmatch(r"mol_(\d+)", name.strip())
            if match is None:
                raise ValueError(
                    f"YuelBond SDF record {output_pos} has name {name!r}; expected mol_N "
                    "to preserve the original input index."
                )
            mol_idx = int(match.group(1)) - 1
            if not 0 <= mol_idx < len(mols):
                raise ValueError(f"YuelBond molecule name {name!r} is outside the reference SDF")
            if mol_idx in seen:
                raise ValueError(f"duplicate YuelBond molecule identifier {name!r}")
            seen.add(mol_idx)
            expected_heavy = _heavy_count(mols[mol_idx])
            if mol.GetNumAtoms() != expected_heavy:
                raise ValueError(
                    f"atom-count mismatch for {name}: YuelBond={mol.GetNumAtoms()}, "
                    f"reference-heavy={expected_heavy}"
                )
            bonds = []
            for bond in mol.GetBonds():
                i, j = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
                if bond.GetIsAromatic() or bond.GetBondType() == Chem.BondType.AROMATIC:
                    order = "ar"
                elif bond.GetBondType() == Chem.BondType.DOUBLE:
                    order = 2
                elif bond.GetBondType() == Chem.BondType.TRIPLE:
                    order = 3
                else:
                    order = 1
                bonds.append([i, j, order])
            out[str(mol_idx)] = sorted(bonds)
        output = _resolve(a.output)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w") as f:
            json.dump(out, f)
        print(f"[adapt] normalized {len(out)-1}/{len(mols)} YuelBond SDF predictions -> {output}")
        print("Missing mol_N records remain omitted and will receive zero credit in `score`.")
        return

    with open(input_path) as f:
        raw = json.load(f)
    class_order = a.class_order
    for mol_idx, bonds in _iter_raw_predictions(raw):
        if not 0 <= mol_idx < len(mols):
            raise ValueError(f"molecule index {mol_idx} outside 0..{len(mols)-1}")
        mol = mols[mol_idx]
        hm = _heavy_map(mol)
        n_heavy = _heavy_count(mol)
        normalized = {}
        for entry in bonds:
            i, j, value = _raw_bond(entry)
            i, j = int(i) - a.index_base, int(j) - a.index_base
            if a.atom_index_space == "full":
                if i not in hm or j not in hm:
                    continue  # discard H-X predictions; paper comparison is HH-only
                i, j = hm[i], hm[j]
            elif not (0 <= i < n_heavy and 0 <= j < n_heavy):
                raise ValueError(f"heavy atom index out of range for molecule {mol_idx}: {(i, j)}")
            if i == j:
                raise ValueError(f"self bond in molecule {mol_idx}: {(i, j)}")
            pair = tuple(sorted((i, j)))
            order = _adapt_order(value, a.zero_based_classes, class_order)
            if order is None:
                continue
            if pair in normalized and normalized[pair] != order:
                raise ValueError(f"conflicting predictions for molecule {mol_idx}, edge {pair}")
            normalized[pair] = order
        out[str(mol_idx)] = [[i, j, order] for (i, j), order in sorted(normalized.items())]
    output = _resolve(a.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(out, f)
    print(f"[adapt] normalized {len(out)-1}/{len(mols)} molecule predictions -> {output}")
    print("Missing molecules remain omitted and will receive zero credit in `score`.")


def cmd_score(a):
    """Unified scorer for ANY tool's normalized predictions on the shared SDF."""
    from collections import Counter
    from rdkit import Chem
    # Look for the predictions file in a few sensible places, since `rule` writes it next to the SDF.
    cands = [_resolve(a.pred),
             os.path.join(REPO, "data", os.path.basename(a.pred)),
             os.path.join(os.path.dirname(_resolve(a.sdf)), os.path.basename(a.pred))]
    pred_path = next((c for c in cands if os.path.exists(c)), None)
    if pred_path is None:
        sys.exit("[missing] could not find --pred. Tried:\n  " + "\n  ".join(cands))
    print(f"[score] predictions: {pred_path}")
    with open(pred_path) as f:
        preds = json.load(f)
    meta = preds.pop("__meta__", None)

    # Read + sanitize the SDF once.
    sdf_mols = _read_valid_molecules(a.sdf)
    subset_split = getattr(a, "subset_split", "all")
    selected_indices = None
    if subset_split != "all":
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        from bondnet.data.dataset import _assign_split_label
        selected_indices = set()
        for idx, mol in enumerate(sdf_mols):
            if not mol.HasProp("geom_mol_idx"):
                raise ValueError(f"Missing geom_mol_idx on shared SDF molecule {idx}")
            label = _assign_split_label({"geom_mol_idx": int(mol.GetProp("geom_mol_idx"))}, idx)
            if label == subset_split:
                selected_indices.add(idx)
        if not selected_indices:
            raise ValueError(f"No molecules with fixed split {subset_split!r}")
        print(f"[subset] fixed GEOM {subset_split}: {len(selected_indices)}/{len(sdf_mols)} molecules")

    current_meta = _prediction_meta(a.sdf, sdf_mols)
    if meta is None and not a.allow_unfingerprinted:
        sys.exit(
            "[unsafe] prediction JSON has no __meta__ SDF fingerprint. "
            "Run `adapt` for YuelBond/external output, or regenerate it with `rule`/`bondnet`."
        )
    if meta is not None:
        if (meta.get("sdf_sha256") != current_meta["sdf_sha256"]
                or meta.get("n_molecules") != current_meta["n_molecules"]
                or meta.get("heavy") != current_meta["heavy"]):
            sys.exit(
                "[mismatch] this predictions file was generated for a DIFFERENT molecule set "
                f"than {os.path.basename(a.sdf)}\n"
                f"  preds: sdf={meta.get('sdf')}, n={meta.get('n_molecules')}, "
                f"sha256={str(meta.get('sdf_sha256'))[:12]}...\n"
                f"  this:  sdf={current_meta['sdf']}, n={current_meta['n_molecules']}, "
                f"sha256={current_meta['sdf_sha256'][:12]}...\n"
                "  Regenerate: re-run `build`, then `rule --sdf <that same sdf>`, then `score`.\n"
                "  (Do not rebuild the SDF between `rule` and `score`.)")

    tp = Counter(); fp_matched = Counter(); fp_pipeline = Counter(); fn = Counter()
    # Conditional-on-success metrics exclude omitted/tool-failure molecules but
    # retain every missing, mistyped, and extra bond within successful outputs.
    fn_success = Counter()
    n_mol = 0; n_success = 0; n_exact_true = 0; n_full_graph = 0; extra_total = 0; true_total = 0
    for idx, mol in enumerate(sdf_mols):
        if selected_indices is not None and idx not in selected_indices:
            continue
        n_mol += 1
        true = _true_bonds(mol)
        true_total += len(true)
        key = str(idx)
        if key not in preds:            # tool failed / omitted this molecule
            for lab in true.values():
                fn[lab] += 1
            continue
        n_success += 1
        pred = {}
        for i, j, o in preds[key]:
            pred[(min(int(i), int(j)), max(int(i), int(j)))] = _pred_label(o)
        all_true_correct = True
        for e, tl in true.items():
            pl = pred.get(e)
            if pl is None:
                fn[tl] += 1; fn_success[tl] += 1; all_true_correct = False
            elif pl == tl:
                tp[tl] += 1
            else:
                fp_matched[pl] += 1; fp_pipeline[pl] += 1
                fn[tl] += 1; fn_success[tl] += 1; all_true_correct = False
        extra = [e for e in pred if e not in true]     # bonds predicted that are not real
        for e in extra:
            fp_pipeline[pred[e]] += 1
        extra_total += len(extra)
        if all_true_correct:
            n_exact_true += 1
            if not extra:
                n_full_graph += 1

    def compute_f1s(fp, false_negatives):
        out = {}
        for c in range(4):
            if (tp[c] + fp[c] + false_negatives[c]) == 0:
                continue
            precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
            recall = tp[c] / (tp[c] + false_negatives[c]) if (tp[c] + false_negatives[c]) else 0.0
            out[c] = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return out
    matched_f1s = compute_f1s(fp_matched, fn)
    pipeline_f1s = compute_f1s(fp_pipeline, fn)
    conditional_matched_f1s = compute_f1s(fp_matched, fn_success)
    conditional_pipeline_f1s = compute_f1s(fp_pipeline, fn_success)
    matched_macro = sum(matched_f1s.values()) / len(matched_f1s) if matched_f1s else 0.0
    pipeline_macro = sum(pipeline_f1s.values()) / len(pipeline_f1s) if pipeline_f1s else 0.0
    conditional_matched_macro = (
        sum(conditional_matched_f1s.values()) / len(conditional_matched_f1s)
        if conditional_matched_f1s else 0.0
    )
    conditional_pipeline_macro = (
        sum(conditional_pipeline_f1s.values()) / len(conditional_pipeline_f1s)
        if conditional_pipeline_f1s else 0.0
    )

    print("\n" + "=" * 60)
    print(f"UNIFIED SCORE -- {a.name}  on {_resolve(a.sdf)}")
    print("-" * 60)
    print(f"molecules scored            : {n_mol}")
    print(f"success rate                : {100*n_success/max(n_mol,1):.1f}%")
    print(f"macro-F1 (true-pair typing): {matched_macro:.4f}   "
          + "  ".join(f"{_NAMES[c]}={matched_f1s[c]:.3f}" for c in sorted(matched_f1s)))
    print(f"macro-F1 (pipeline; extras): {pipeline_macro:.4f}   "
          + "  ".join(f"{_NAMES[c]}={pipeline_f1s[c]:.3f}" for c in sorted(pipeline_f1s)))
    print(f"macro-F1 conditional success: {conditional_pipeline_macro:.4f}")
    print(f"true-bond exact-type rate   : {100*n_exact_true/max(n_mol,1):.1f}%")
    print(f"FULL-GRAPH exact match      : {100*n_full_graph/max(n_mol,1):.1f}%   "
          f"(all true bonds correct AND no extra bonds)")
    print(f"extra predicted bonds / mol : {extra_total/max(n_success,1):.3f}")
    print("=" * 60)
    report = {
        "name": a.name,
        "sdf": current_meta,
        "prediction_file": os.path.abspath(pred_path),
        "n_molecules": n_mol,
        "fixed_split_subset": subset_split,
        "success_rate": n_success / max(n_mol, 1),
        "f1_macro_true_pair": matched_macro,
        "f1_macro_pipeline": pipeline_macro,
        "f1_macro_true_pair_conditional_success": conditional_matched_macro,
        "f1_macro_pipeline_conditional_success": conditional_pipeline_macro,
        "f1_true_pair_by_class": {_NAMES[c]: matched_f1s[c] for c in matched_f1s},
        "f1_pipeline_by_class": {_NAMES[c]: pipeline_f1s[c] for c in pipeline_f1s},
        "f1_pipeline_by_class_conditional_success": {
            _NAMES[c]: conditional_pipeline_f1s[c] for c in conditional_pipeline_f1s
        },
        "true_bond_exact_type_rate": n_exact_true / max(n_mol, 1),
        "full_graph_exact_match": n_full_graph / max(n_mol, 1),
        "true_bond_exact_type_rate_conditional_success": n_exact_true / max(n_success, 1),
        "full_graph_exact_match_conditional_success": n_full_graph / max(n_success, 1),
        "extra_bonds_per_successful_molecule": extra_total / max(n_success, 1),
    }
    if meta is not None:
        report["prediction_provenance"] = {
            key: meta[key] for key in ("method", "stage1", "stage2") if key in meta
        }
    output = (_resolve(a.output) if a.output else
              D("results", "p0c", "scores",
                f"{a.name.lower().replace(' ', '_')}_{os.path.splitext(os.path.basename(a.sdf))[0]}.json"))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"score JSON                  : {output}")
    print("Score BondNet, RDKit, OpenBabel, and YuelBond through this SAME command for a fair table.")


def cmd_rebind(a):
    """Refresh only the SDF fingerprint after scorer metadata logic changes."""
    pred_path = _resolve(a.pred)
    with open(pred_path) as f:
        preds = json.load(f)
    mols = _read_valid_molecules(a.sdf)
    for key, bonds in preds.items():
        if key == "__meta__":
            continue
        idx = int(key)
        if not 0 <= idx < len(mols):
            raise ValueError(f"prediction molecule index out of range: {idx}")
        n_heavy = _heavy_count(mols[idx])
        for i, j, _ in bonds:
            if not (0 <= int(i) < n_heavy and 0 <= int(j) < n_heavy):
                raise ValueError(f"prediction atom index out of range: mol={idx}, edge={(i, j)}")
    old_meta = preds.get("__meta__") or {}
    new_meta = _prediction_meta(a.sdf, mols)
    for key in ("method", "stage1", "stage2"):
        if key in old_meta:
            new_meta[key] = old_meta[key]
    preds["__meta__"] = new_meta
    with open(pred_path, "w") as f:
        json.dump(preds, f)
    print(f"[rebind] validated and refreshed fingerprint -> {pred_path}")


def cmd_table(a):
    files = sorted(glob.glob(os.path.join(_resolve(a.scores), "*.json")))
    rows = []
    for path in files:
        with open(path) as f:
            r = json.load(f)
        if "f1_macro_pipeline" not in r:
            continue
        sdf_name = r.get("sdf", {}).get("sdf", "")
        condition = "noisy" if "noisy" in sdf_name.lower() else "clean"
        rows.append({
            "condition": condition, "method": r.get("name", "tool"),
            "success_rate": r["success_rate"],
            "f1_macro_true_pair": r["f1_macro_true_pair"],
            "f1_macro_pipeline": r["f1_macro_pipeline"],
            "true_bond_exact_type_rate": r["true_bond_exact_type_rate"],
            "full_graph_exact_match": r["full_graph_exact_match"],
            "extra_bonds_per_molecule": r["extra_bonds_per_successful_molecule"],
        })
    if not rows:
        sys.exit(f"[missing] no score JSON files found under {_resolve(a.scores)}")
    rank = {"BondNet": 0, "YuelBond": 1, "RDKit": 2, "OpenBabel": 3}
    rows.sort(key=lambda r: ({"clean": 0, "noisy": 1}[r["condition"]], rank.get(r["method"], 99)))
    output = _resolve(a.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{'Condition':<10}{'Method':<12}{'Success':>9}{'F1-pair':>10}{'F1-pipe':>10}"
          f"{'Exact':>9}{'FullGraph':>11}{'Extra/mol':>11}")
    for r in rows:
        print(f"{r['condition']:<10}{r['method']:<12}{100*r['success_rate']:>8.1f}%"
              f"{r['f1_macro_true_pair']:>10.4f}{r['f1_macro_pipeline']:>10.4f}"
              f"{100*r['true_bond_exact_type_rate']:>8.1f}%"
              f"{100*r['full_graph_exact_match']:>10.1f}%"
              f"{r['extra_bonds_per_molecule']:>11.3f}")
    print(f"\nWrote comparison CSV -> {output}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--n", type=int, default=5000)
    b.add_argument("--sigma", type=float, default=0.1); b.add_argument("--seed", type=int, default=0)
    b.add_argument("--source", required=True,
                   help="Explicit leakage-free source SDF. For YuelBond comparison, prefer a "
                        "GEOM-excluded PubChem cohort because YuelBond was trained on GEOM.")
    b.add_argument("--clean-out", default=CLEAN)
    b.add_argument("--noisy-out", default=NOISY)
    b.set_defaults(func=cmd_build)
    e = sub.add_parser("bondnet")
    e.add_argument("--clean", default=CLEAN); e.add_argument("--noisy", default=NOISY)
    e.add_argument("--stage1", default=STAGE1); e.add_argument("--stage2", default=STAGE2)
    e.add_argument("--batch-size", type=int, default=64); e.add_argument("--device", default="auto")
    e.add_argument("--which", choices=["both", "clean", "noisy"], default="both")
    e.set_defaults(func=cmd_bondnet)
    j = sub.add_parser("joint", help="corrected joint model on the archived shared SDFs")
    j.add_argument("--clean", default=CLEAN); j.add_argument("--noisy", default=NOISY)
    j.add_argument("--checkpoint", required=True); j.add_argument("--seed", type=int, required=True)
    j.add_argument("--batch-size", type=int, default=64); j.add_argument("--device", default="auto")
    j.add_argument("--which", choices=["both", "clean", "noisy"], default="both")
    j.set_defaults(func=cmd_joint)
    ru = sub.add_parser("rule"); ru.add_argument("--sdf", required=True)
    ru.add_argument("--methods", nargs="+", choices=["rdkit", "openbabel"], default=None)
    ru.set_defaults(func=cmd_rule)
    ad = sub.add_parser("adapt", help="normalize YuelBond/external prediction JSON")
    ad.add_argument("--sdf", required=True); ad.add_argument("--input", required=True)
    ad.add_argument("--output", required=True)
    ad.add_argument("--atom-index-space", choices=["full", "heavy"], default="full")
    ad.add_argument("--index-base", type=int, choices=[0, 1], default=0)
    ad.add_argument("--zero-based-classes", action="store_true",
                    help="Interpret integer 0..3 predictions through --class-order.")
    ad.add_argument("--class-order", nargs="+",
                    default=["single", "double", "aromatic", "triple"],
                    help="Class names for zero-based outputs; may include no_bond. Verify against the checkpoint.")
    ad.set_defaults(func=cmd_adapt)
    s = sub.add_parser("score"); s.add_argument("--sdf", required=True)
    s.add_argument("--pred", required=True); s.add_argument("--name", default="tool")
    s.add_argument("--output", default=None)
    s.add_argument("--subset-split", choices=["all", "train", "val", "test"], default="all")
    s.add_argument("--allow-unfingerprinted", action="store_true",
                   help="Unsafe debugging escape hatch; paper results must not use it.")
    s.set_defaults(func=cmd_score)
    rb = sub.add_parser("rebind", help="validate predictions and refresh their SDF fingerprint")
    rb.add_argument("--sdf", required=True); rb.add_argument("--pred", required=True)
    rb.set_defaults(func=cmd_rebind)
    t = sub.add_parser("table", help="combine score JSONs into a comparison CSV")
    t.add_argument("--scores", default=D("results", "p0c", "scores"))
    t.add_argument("--output", default=D("results", "p0c", "comparison.csv"))
    t.set_defaults(func=cmd_table)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
