#!/usr/bin/env python3
"""Full-molecule identity for BondNet and the rule-based tools under one protocol.

The headline HH metrics compare discrete bond labels with the reference
convention. This diagnostic instead asks whether each method reconstructs the
same *molecule*, for every method on the same coordinates:

  returned        the method produced a bonded graph
  sanitizable     the first RDKit SanitizeMol of the predicted full explicit-H molecule succeeds
  normalized      the subsequent identity normalization (Kekulize with aromatic flags cleared,
                  re-sanitize, RemoveHs, SMILES) also succeeds
  inchi_ok        a non-stereo standard InChI was generated
  smiles_match    canonical non-isomeric SMILES equals the reference after both molecules
                  are re-sanitized with RDKit's default aromaticity model, so Kekule and
                  aromatic depictions of the same pi system count as identical
  inchi_match     non-stereo standard InChI (/SNon) equals the reference; InChI additionally
                  normalizes mobile-H tautomers, so this is a broader equivalence reported separately

Methods
  joint           corrected joint explicit-H BondNet, seeds 42-44; predicted H-A bonds are
                  single and formal charges are zero (the model predicts neither)
  rdkit           RDKit DetermineBonds, default configuration of the main tables
  rdkit_hueckel   post-hoc useHueckel=True configuration
  openbabel       ConnectTheDots + PerceiveBondOrders
Rule tools keep the formal charges they assign.

Coordinates are the molecule-keyed noisy coordinates of the primary evaluation
(materialized SDFs for the rule tools; the same keyed noise applied to the cache
for BondNet). Inference only.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

COHORTS = {
    "geom": {"cache": "data/fixed_split_caches/geom_drugs_all_random1_re10_c30_h30_explicit_h_fixed_test.pt",
             "sdf": "data/robustness_27240_keyed/sigma_{tag}.sdf"},
    "external": {"cache": "data/external_v3/cache_explicit_c30_h30.pt",
                 "sdf": "data/external_v3/sigma_{tag}.sdf"},
}
JOINT = "checkpoints/revision_v2_direction_fixed_lr1e4_seed{seed}/one_stage_joint/best_e2e.pt"
SIGMAS = (0.0, 0.1, 0.2)
SEEDS = (42, 43, 44)
RULES = ("rdkit", "rdkit_hueckel", "openbabel")
FIELDS = ("returned", "sanitizable", "normalized", "inchi_ok", "smiles_match", "inchi_match")
TEXT_FIELDS = ("fail_stage", "fail_error")


def tag(s):
    return f"{round(100 * s):03d}"


# ----------------------------------------------------------------- identity --
def _normalize_staged(mol):
    """Stage-wise identity normalization.

    Returns (stage_ok, smiles, inchi, fail_stage, fail_error) where stage_ok
    records whether each stage succeeded:
      sanitizable  first Chem.SanitizeMol on the predicted molecule as returned
      normalized   Kekulize(clearAromaticFlags) + re-sanitize + RemoveHs + SMILES
      inchi_ok     non-stereo standard InChI generated (non-empty)
    """
    from rdkit import Chem
    ok = {"sanitizable": False, "normalized": False, "inchi_ok": False}
    m = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(m)
    except Exception as exc:
        return ok, None, None, "sanitize", f"{type(exc).__name__}: {str(exc).strip()[:160]}"
    ok["sanitizable"] = True
    try:
        # Drop every aromatic flag and re-perceive with RDKit's default model, so the
        # reference (MDL aromaticity), Kekule outputs and aromatic-labelled outputs
        # are compared in one representation.
        Chem.Kekulize(m, clearAromaticFlags=True)
        Chem.SanitizeMol(m)
        smi = Chem.MolToSmiles(Chem.RemoveHs(m), isomericSmiles=False)
    except Exception as exc:
        return ok, None, None, "normalize", f"{type(exc).__name__}: {str(exc).strip()[:160]}"
    ok["normalized"] = True
    # Identity without stereochemistry: drop 3D coordinates (from which InChI
    # would otherwise derive stereo layers) and request a non-stereo InChI.
    m.RemoveAllConformers()
    try:
        inchi = Chem.MolToInchi(m, options="/SNon") or ""
    except Exception as exc:
        return ok, smi, "", "inchi", f"{type(exc).__name__}: {str(exc).strip()[:160]}"
    ok["inchi_ok"] = bool(inchi)
    return ok, smi, inchi, ("" if inchi else "inchi"), ("" if inchi else "empty InChI")


def _normalized(mol):
    """(canonical non-isomeric SMILES, standard InChI) or raises."""
    ok, smi, inchi, stage, err = _normalize_staged(mol)
    if not ok["normalized"]:
        raise ValueError(f"{stage}: {err}")
    return smi, inchi


def reference_identity(ref_mol):
    from rdkit import Chem
    m = Chem.Mol(ref_mol)
    m.RemoveAllConformers()
    return _normalized(m)


def score_prediction(pred_mol, ref_smi, ref_inchi):
    out = {"returned": pred_mol is not None, "sanitizable": False, "normalized": False,
           "inchi_ok": False, "smiles_match": False, "inchi_match": False,
           "fail_stage": "" if pred_mol is not None else "no_graph", "fail_error": ""}
    if pred_mol is None:
        return out
    try:
        ok, smi, inchi, stage, err = _normalize_staged(pred_mol)
    except Exception as exc:  # defensive: unexpected failure outside the staged calls
        out.update(fail_stage="other", fail_error=f"{type(exc).__name__}: {str(exc)[:160]}")
        return out
    out.update(ok)
    out["fail_stage"], out["fail_error"] = stage, err
    out["smiles_match"] = smi is not None and smi == ref_smi
    out["inchi_match"] = bool(inchi) and inchi == ref_inchi
    return out


# --------------------------------------------------------------- rule tools --
def _bare_mol(mol):
    from rdkit import Chem
    raw = Chem.RWMol()
    for atom in mol.GetAtoms():
        raw.AddAtom(Chem.Atom(atom.GetAtomicNum()))
    conf = Chem.Conformer(raw.GetNumAtoms())
    src = mol.GetConformer()
    for idx in range(raw.GetNumAtoms()):
        conf.SetAtomPosition(idx, src.GetAtomPosition(idx))
    raw.AddConformer(conf, assignId=True)
    return raw.GetMol()


def run_rdkit(mol, hueckel):
    from rdkit.Chem.rdDetermineBonds import DetermineBonds
    raw = _bare_mol(mol)
    kwargs = {"charge": 0, "maxIterations": 1000}
    if hueckel:
        kwargs["useHueckel"] = True
    DetermineBonds(raw, **kwargs)
    return raw


def run_openbabel(mol):
    from openbabel import openbabel as ob
    from rdkit import Chem
    obmol = ob.OBMol()
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        a = obmol.NewAtom()
        a.SetAtomicNum(int(atom.GetAtomicNum()))
        a.SetVector(float(p.x), float(p.y), float(p.z))
    obmol.ConnectTheDots()
    obmol.PerceiveBondOrders()
    rw = Chem.RWMol()
    for i in range(1, obmol.NumAtoms() + 1):
        a = obmol.GetAtom(i)
        ra = Chem.Atom(int(a.GetAtomicNum()))
        ra.SetFormalCharge(int(a.GetFormalCharge()))
        ra.SetNoImplicit(True)
        rw.AddAtom(ra)
    for b in ob.OBMolBondIter(obmol):
        i, j = b.GetBeginAtomIdx() - 1, b.GetEndAtomIdx() - 1
        order = int(b.GetBondOrder())
        bt = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
              3: Chem.BondType.TRIPLE}.get(order, Chem.BondType.SINGLE)
        rw.AddBond(i, j, bt)
    return rw.GetMol()


def _quiet_worker():
    """Silence native stdout (RDKit's Hückel diagnostics) in pool workers."""
    import os
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)


def _rule_worker(item):
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    idx, molblock, method, ref_smi, ref_inchi = item
    from rdkit import Chem
    mol = Chem.MolFromMolBlock(molblock, sanitize=False, removeHs=False)
    try:
        if method == "openbabel":
            pred = run_openbabel(mol)
        else:
            pred = run_rdkit(mol, hueckel=(method == "rdkit_hueckel"))
    except Exception:
        pred = None
    return idx, score_prediction(pred, ref_smi, ref_inchi)


# ------------------------------------------------------------------ BondNet --
def run_joint(cohort, seed, sigma, refs, device, batch_size):
    import torch
    from torch.utils.data import DataLoader
    import evaluate as ev
    from bondnet.data.dataset import CachedBondNetDataset, collate_fn
    ecv = importlib.import_module("evaluate_chemical_validity")
    model = ev.load_model(str(ROOT / JOINT.format(seed=seed)), device)
    model.eval()
    ds = CachedBondNetDataset(str(ROOT / COHORTS[cohort]["cache"]))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    rows, k = [], 0
    for batch in loader:
        per_mol, _ = ecv.predict_batch(model, None, batch, device, 2.5, 2.5, 0.5, sigma, 20260921)
        for bonds in per_mol:
            ref_mol, ref_smi, ref_inchi = refs[k]
            try:
                pred = ecv.build_predicted_mol(ref_mol, bonds, use_reference_charge=False)
            except Exception:
                pred = None
            rows.append(score_prediction(pred, ref_smi, ref_inchi))
            k += 1
    if k != len(refs):
        raise RuntimeError(f"joint seed {seed}: {k} predictions for {len(refs)} molecules")
    return rows


def load_refs(cohort, sigma):
    from rdkit import Chem
    ecv = importlib.import_module("evaluate_chemical_validity")
    path = ROOT / COHORTS[cohort]["sdf"].format(tag=tag(sigma))
    refs, blocks = [], []
    for mol in ecv.iter_reference_mols(path):
        smi, inchi = reference_identity(mol)
        refs.append((mol, smi, inchi))
        blocks.append(Chem.MolToMolBlock(mol, kekulize=True))
    return refs, blocks


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["mol_pos", *FIELDS, *TEXT_FIELDS])
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"mol_pos": i, **{f: int(r[f]) for f in FIELDS},
                        **{f: r.get(f, "") for f in TEXT_FIELDS}})


def fail_breakdown(rows):
    """{"stage | exception type": count} over molecules that were not normalized."""
    from collections import Counter
    c = Counter()
    for r in rows:
        if r.get("fail_stage"):
            etype = (r.get("fail_error") or "").split(":")[0]
            c[f"{r['fail_stage']} | {etype}" if etype else r["fail_stage"]] += 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def rates(rows):
    n = len(rows)
    return {f: sum(int(r[f]) for r in rows) / n for f in FIELDS}


def read_rows(path):
    """Completed per-molecule CSV -> rows (resume support), or None."""
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not set(FIELDS) <= set(reader.fieldnames or ()):
            return None   # written by an older version without stage fields
        return [{**{f: bool(int(r[f])) for f in FIELDS}, **{f: r.get(f, "") for f in TEXT_FIELDS}}
                for r in reader]


FAIL = {"returned": False, "sanitizable": False, "normalized": False, "inchi_ok": False,
        "smiles_match": False, "inchi_match": False, "fail_stage": "no_graph", "fail_error": ""}
_WARMUP = "\n     RDKit          3D\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n"


def _worker_loop(conn, method):
    """Dedicated worker: warm up the tool once, report ready, then score items."""
    _quiet_worker()
    try:
        _rule_worker((-1, _WARMUP, method, "", ""))
    except Exception:
        pass
    conn.send(("ready", None))
    while True:
        item = conn.recv()
        if item is None:
            break
        try:
            conn.send(("ok", _rule_worker(item)))
        except Exception:
            conn.send(("ok", (item[0], {**FAIL, "fail_stage": "worker_error"})))


class _Worker:
    def __init__(self, ctx, method, start_timeout):
        self.parent, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker_loop, args=(child, method), daemon=True)
        self.proc.start()
        child.close()
        self.item = None
        self.t0 = None
        # Startup (spawn + imports + first tool call) is excluded from the
        # per-molecule clock: wait for "ready" before handing out work.
        try:
            ok = self.parent.poll(start_timeout)
            if ok:
                self.parent.recv()
        except (EOFError, OSError):
            ok = False
        if not ok:
            self.kill()
            raise RuntimeError("rule-tool worker did not start")

    def submit(self, item):
        import time
        self.item, self.t0 = item, time.time()
        self.parent.send(item)

    def kill(self):
        try:
            self.proc.kill()
            self.proc.join(5)
        except Exception:
            pass


def _start_worker(ctx, method, start_timeout, tries=3):
    for k in range(tries):
        try:
            return _Worker(ctx, method, start_timeout)
        except RuntimeError:
            print(f"[worker] start attempt {k + 1} timed out; retrying", flush=True)
    raise RuntimeError("could not start a rule-tool worker")


def _run_pass(items, method, workers, timeout, log, tag, done):
    """Process items with dedicated workers; returns the indices that timed out."""
    import time
    from multiprocessing import get_context
    ctx = get_context("spawn")
    queue = list(items)
    timed_out = []
    # Start workers one at a time so first tool calls never overlap.
    pool = [_start_worker(ctx, method, max(timeout, 120.0)) for _ in range(min(workers, max(len(queue), 1)))]
    last_report = time.time()
    try:
        while queue or any(w.item is not None for w in pool):
            for w in pool:
                if w.item is None and queue:
                    w.submit(queue.pop(0))
            progressed = False
            now = time.time()
            for n, w in enumerate(pool):
                if w.item is None:
                    continue
                # A worker killed by a native crash closes its pipe: on Windows
                # poll() then raises BrokenPipeError, on Linux recv() raises EOFError.
                state, idx, row = "busy", None, None
                try:
                    if w.parent.poll():
                        _, (idx, row) = w.parent.recv()
                        state = "done"
                except (EOFError, OSError):
                    state = "crash"
                if state == "busy" and (not w.proc.is_alive()):
                    state = "crash"
                if state == "busy" and now - w.t0 > timeout:
                    state = "timeout"
                if state == "done":
                    done[idx] = row
                    log.write(json.dumps({"i": idx, "row": row, "pass": tag}) + "\n")
                    w.item = None
                    progressed = True
                elif state in ("crash", "timeout"):
                    # Provisional failure; retried once in a fresh worker by the caller.
                    idx = w.item[0]
                    done[idx] = {**FAIL, "fail_stage": "worker_crash" if state == "crash" else "timeout"}
                    timed_out.append(idx)
                    log.write(json.dumps({"i": idx, "row": done[idx], "timeout": True, "pass": tag}) + "\n")
                    log.flush()
                    what = "crashed its worker" if state == "crash" else f"exceeded {timeout:.0f} s"
                    print(f"[{state}:{tag}] molecule {idx} {what}; replacing the worker", flush=True)
                    w.kill()
                    pool[n] = _start_worker(ctx, method, max(timeout, 120.0))
                    progressed = True
            if not progressed:
                time.sleep(0.002)
            if now - last_report > 60:
                log.flush()
                print(f"[progress:{tag}] {len(done)} molecules scored", flush=True)
                last_report = now
    finally:
        for w in pool:
            try:
                w.parent.send(None)
            except Exception:
                pass
            w.kill()
    log.flush()
    return timed_out


def run_rules_robust(items, workers, timeout, partial_path):
    """Score rule-tool predictions with a per-molecule wall-clock limit.

    Each worker is a dedicated process that is warmed up (imports and one tool
    call) before it receives molecules, so process start-up never counts
    against a molecule. A molecule that exceeds ``timeout`` seconds, or whose
    worker dies, is recorded provisionally; only that worker is replaced. All
    provisional timeouts are then retried once, one at a time, in a fresh
    worker; a molecule is counted as a timeout (tool failure) only if it also
    fails the retry. Results are appended to ``partial_path`` so an interrupted
    run resumes where it stopped.
    """
    done, retried, first_timeouts, final_timeouts = {}, set(), set(), set()
    if partial_path.exists():
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            i = rec["i"]
            done[i] = rec["row"]
            if rec.get("pass") == "retry":
                retried.add(i)
                final_timeouts.discard(i)
                if rec.get("timeout"):
                    final_timeouts.add(i)
            elif rec.get("timeout"):
                first_timeouts.add(i)
            else:
                first_timeouts.discard(i)
    method = items[0][2]
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    with partial_path.open("a", encoding="utf-8") as log:
        queue = [it for it in items if it[0] not in done]
        if queue:
            first_timeouts |= set(_run_pass(queue, method, workers, timeout, log, "main", done))
        pending = [items[i] for i in sorted(first_timeouts - retried)]
        if pending:
            print(f"[retry] {len(pending)} provisional timeouts, retrying one at a time", flush=True)
            final_timeouts |= set(_run_pass(pending, method, 1, timeout, log, "retry", done))
    return [done[i] for i in range(len(items))], sorted(final_timeouts)


def ensure_openbabel_datadir():
    """Make sure Open Babel can read its data files (bondtyp.txt).

    Without them PerceiveBondOrders silently falls back to built-in tables and
    clean-GEOM HH exact match drops from ~93% to ~64% (see
    results/revision_v4_openbabel_datadir_check). If BABEL_DATADIR is unset or
    lacks bondtyp.txt, look under the Python prefix and set it (inherited by
    spawned workers); abort if nothing is found. Returns the directory used.
    """
    import os
    cur = os.environ.get("BABEL_DATADIR")
    if cur and (Path(cur) / "bondtyp.txt").exists():
        return cur
    try:
        import openbabel
        roots = [Path(sys.prefix), Path(openbabel.__file__).resolve().parent]
    except Exception:
        roots = [Path(sys.prefix)]
    for root in roots:
        for cand in [root / "share" / "openbabel", root / "Library" / "share" / "openbabel"]:
            hits = sorted(cand.rglob("bondtyp.txt")) if cand.exists() else []
            if hits:
                os.environ["BABEL_DATADIR"] = str(hits[0].parent)
                print(f"[openbabel] BABEL_DATADIR set to {hits[0].parent}", flush=True)
                return str(hits[0].parent)
    raise RuntimeError("Open Babel data files (bondtyp.txt) not found; set BABEL_DATADIR")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohorts", nargs="+", default=list(COHORTS), choices=list(COHORTS))
    ap.add_argument("--methods", nargs="+", default=["joint", *RULES], choices=["joint", *RULES])
    ap.add_argument("--sigmas", nargs="+", type=float, default=list(SIGMAS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--output_root", default="results/revision_v4_identity_v2")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="per-molecule wall-clock limit for rule tools (seconds); exceeding it counts as failure")
    args = ap.parse_args()
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    out_root = ROOT / args.output_root
    summary_path = out_root / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if "openbabel" in args.methods:
        summary["openbabel_datadir"] = ensure_openbabel_datadir()

    for cohort in args.cohorts:
        for sigma in args.sigmas:
            refs, blocks = load_refs(cohort, sigma)
            for method in args.methods:
                key = f"{cohort}|{method}|{sigma:.2f}"
                if method == "joint":
                    import evaluate as ev
                    device = ev._get_device(args.device)
                    per_seed = {}
                    for seed in SEEDS:
                        path = out_root / cohort / "joint" / f"seed{seed}" / f"sigma_{tag(sigma)}.csv"
                        rows = read_rows(path)
                        if rows is None or len(rows) != len(refs):
                            rows = run_joint(cohort, seed, sigma, refs, device, args.batch_size)
                            write_rows(path, rows)
                        per_seed[str(seed)] = rates(rows)
                        summary[f"{key}|fail_stages|seed{seed}"] = fail_breakdown(rows)
                    summary[key] = {f: {"mean": statistics.mean(v[f] for v in per_seed.values()),
                                        "sd": statistics.stdev(v[f] for v in per_seed.values()),
                                        "by_seed": {s: v[f] for s, v in per_seed.items()}}
                                    for f in FIELDS}
                else:
                    path = out_root / cohort / method / f"sigma_{tag(sigma)}.csv"
                    rows = read_rows(path)
                    timeouts = summary.get(key + "|timeouts", [])
                    partial = out_root / cohort / method / f"sigma_{tag(sigma)}.partial.jsonl"
                    needs_retry = bool(timeouts) and partial.exists() and not any(
                        '"pass": "retry"' in ln for ln in partial.read_text(encoding="utf-8").splitlines())
                    if rows is None or len(rows) != len(refs) or needs_retry:
                        items = [(i, blocks[i], method, refs[i][1], refs[i][2]) for i in range(len(refs))]
                        rows, timeouts = run_rules_robust(
                            items, args.workers, args.timeout,
                            partial)
                        write_rows(path, rows)
                    summary[key] = rates(rows)
                    summary[key + "|fail_stages"] = fail_breakdown(rows)
                    summary[key + "|timeouts"] = timeouts
                summary[key + "|n"] = len(refs)
                print(f"[done] {key}: {summary[key]}", flush=True)
                out_root.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
