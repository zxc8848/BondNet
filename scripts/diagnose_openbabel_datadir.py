#!/usr/bin/env python3
"""Check whether Open Babel finds its data files (bondtyp.txt) and how that
changes bond perception on the GEOM clean test set.

Without bondtyp.txt, OBMol::PerceiveBondOrders falls back to built-in tables
and its heavy-heavy graph exact match on GEOM drops from ~93% to ~64% (checked
with openbabel-wheel by hiding that one file). This script reports:
  * Open Babel version, BABEL_DATADIR / BABEL_LIBDIR as seen by Python
  * every bondtyp.txt found under the Python prefix
  * HH-graph exact match on the first N clean GEOM molecules, first with the
    current environment and then with BABEL_DATADIR pointed at each data
    directory found (each in a fresh subprocess).

Usage: python scripts/diagnose_openbabel_datadir.py [--n 2000]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SDF = ROOT / "data/robustness_27240_keyed/sigma_000.sdf"


def measure(n: int) -> dict:
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from openbabel import openbabel as ob
    from scripts import p0c_yuelbond_compare as sc
    from evaluate_rule_baselines import _openbabel_predict
    mols = sc._read_valid_molecules(str(SDF))[:n]
    exact = 0
    for m in mols:
        true = sc._true_bonds(m)
        hm = sc._heavy_map(m)
        pred = {}
        for (i, j), lab in _openbabel_predict(m).items():
            if i in hm and j in hm:
                a, b = hm[i], hm[j]
                pred[(min(a, b), max(a, b))] = lab
        exact += pred == true
    return {"ob_version": ob.OBReleaseVersion(),
            "BABEL_DATADIR": os.environ.get("BABEL_DATADIR"),
            "BABEL_LIBDIR": os.environ.get("BABEL_LIBDIR"),
            "n": len(mols), "hh_exact": exact / len(mols)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        print(json.dumps(measure(args.n)))
        return
    print("python:", sys.executable)
    print("BABEL_DATADIR (shell):", os.environ.get("BABEL_DATADIR"))
    found = sorted({str(p.parent) for p in Path(sys.prefix).rglob("bondtyp.txt")})
    print("bondtyp.txt found in:", found or "NONE")
    runs = [("current environment", dict(os.environ))]
    for d in found:
        env = dict(os.environ)
        env["BABEL_DATADIR"] = d
        runs.append((f"BABEL_DATADIR={d}", env))
    for label, env in runs:
        out = subprocess.run([sys.executable, __file__, "--child", "--n", str(args.n)],
                             env=env, capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith("{")]
        res = json.loads(line[-1]) if line else {"error": out.stderr[-500:]}
        print(f"\n[{label}]\n  {res}")


if __name__ == "__main__":
    main()
