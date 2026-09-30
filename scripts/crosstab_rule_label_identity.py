#!/usr/bin/env python3
"""Cross-tabulate the clean-GEOM RDKit HH-error audit with full-molecule identity.

The HH audit (scripts/audit_rule_label_disagreements.py) partitions the saved
RDKit predictions into exact / connectivity error / same HH connectivity but a
different discrete label / tool failure. This script asks, molecule by
molecule, whether each audited molecule reproduces the reference molecule under
the identity protocol of scripts/evaluate_full_molecule_identity.py.

For every molecule it re-runs the same RDKit DetermineBonds call (default
configuration, charge 0, maxIterations 1000), checks that the heavy-heavy bond
labels equal the saved prediction (so the identity result refers to the audited
prediction), and scores identity with the stage-wise normalization.

Output: results/revision_v4_rule_label_audit/geom_rdkit_clean_identity_crosstab.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import evaluate_full_molecule_identity as ident  # noqa: E402
from scripts import p0c_yuelbond_compare as scorer  # noqa: E402


def hh_labels(pred_mol, ref_mol):
    """Heavy-heavy bond labels of a DetermineBonds output in the scorer's index space."""
    hm = scorer._heavy_map(ref_mol)
    lab = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
    out = {}
    for b in pred_mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in hm and j in hm:
            a, c = hm[i], hm[j]
            out[(min(a, c), max(a, c))] = lab.get(b.GetBondType().name, 0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sdf", default="data/robustness_27240_keyed/sigma_000.sdf")
    ap.add_argument("--predictions", default="results/revision_v2_rule_baselines/predictions/rdkit_sigma_000_preds.json")
    ap.add_argument("--output", default="results/revision_v4_rule_label_audit/geom_rdkit_clean_identity_crosstab.json")
    ap.add_argument("--all", action="store_true", help="also score the exact-HH molecules")
    args = ap.parse_args()
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    audit_mols = scorer._read_valid_molecules(str(ROOT / args.sdf))
    preds = json.loads((ROOT / args.predictions).read_text(encoding="utf-8"))
    preds.pop("__meta__", None)
    ref_mols = list(ident.importlib.import_module("evaluate_chemical_validity").iter_reference_mols(ROOT / args.sdf))
    if len(ref_mols) != len(audit_mols):
        raise RuntimeError("audit and identity readers disagree on cohort size")

    table = defaultdict(Counter)
    errors = defaultdict(Counter)
    mismatched_prediction = []
    examples = defaultdict(list)
    for idx, (amol, rmol) in enumerate(zip(audit_mols, ref_mols)):
        if amol.GetNumAtoms() != rmol.GetNumAtoms():
            raise RuntimeError(f"atom-count mismatch at {idx}")
        true = scorer._true_bonds(amol)
        raw = preds.get(str(idx))
        if raw is None:
            cat = "tool_failure"
        else:
            pred = {(min(int(i), int(j)), max(int(i), int(j))): scorer._pred_label(l) for i, j, l in raw}
            if pred == true:
                cat = "exact"
            elif set(pred) != set(true):
                cat = "connectivity_error"
            else:
                errs = [(true[p], pred[p]) for p in true if true[p] != pred[p]]
                cat = ("label_error_aromatic" if any(3 in e for e in errs) else "label_error_single_double")
        if cat == "exact" and not args.all:
            continue
        ref_smi, ref_inchi = ident.reference_identity(rmol)
        try:
            pmol = ident.run_rdkit(rmol, hueckel=False)
        except Exception:
            pmol = None
        if ((pmol is None) != (raw is None)) or (pmol is not None and hh_labels(pmol, rmol) != pred):
            # DetermineBonds returned a different assignment than the saved (audited)
            # prediction, e.g. across RDKit builds; identity would then not refer to
            # the audited prediction, so these molecules are reported separately.
            mismatched_prediction.append(idx)
            table[cat]["recomputed_prediction_differs"] += 1
            continue
        row = ident.score_prediction(pmol, ref_smi, ref_inchi)
        if not row["returned"]:
            outcome = "no_graph"
        elif not row["sanitizable"]:
            outcome = "not_sanitizable"
        elif not row["normalized"]:
            outcome = "normalization_failed"
        elif row["smiles_match"]:
            outcome = "smiles_match"
        elif row["inchi_match"]:
            outcome = "inchi_match_only"
        else:
            outcome = "different_molecule"
        table[cat][outcome] += 1
        if row["fail_stage"]:
            errors[cat][f"{row['fail_stage']} | {row['fail_error'].split(':')[0]}"] += 1
        if outcome in ("inchi_match_only", "different_molecule") and len(examples[f"{cat}|{outcome}"]) < 5:
            smi = Chem_smiles(pmol)
            examples[f"{cat}|{outcome}"].append({"mol_pos": idx, "reference": ref_smi, "rdkit": smi})

    result = {
        "sdf": args.sdf, "predictions": args.predictions,
        "rdkit_version": __import__("rdkit").__version__,
        "crosstab": {k: dict(v) for k, v in table.items()},
        "category_totals": {k: sum(v.values()) for k, v in table.items()},
        "failure_stages": {k: dict(v) for k, v in errors.items()},
        "prediction_mismatches_vs_saved": mismatched_prediction,
        "examples": dict(examples),
        "note": "outcome: smiles_match = aromaticity-normalized non-stereo canonical SMILES equal; "
                "inchi_match_only = SMILES differ but non-stereo InChI equal; different_molecule = neither.",
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("crosstab", "category_totals", "failure_stages")}, indent=2))
    print("prediction mismatches vs saved:", len(mismatched_prediction))


def Chem_smiles(mol):
    from rdkit import Chem
    try:
        m = Chem.Mol(mol)
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(Chem.RemoveHs(m), isomericSmiles=False)
    except Exception:
        return None


if __name__ == "__main__":
    main()
