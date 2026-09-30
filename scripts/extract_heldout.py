#!/usr/bin/env python3
"""Extract a canonical-SMILES leakage-free held-out test set from a source SDF.

Index skipping alone is NOT sufficient to avoid leakage: the same molecule can
appear in multiple source datasets (e.g. a small molecule present in both QM9 and
PubChem) and at multiple indices within one file. This tool therefore excludes any
candidate whose canonical (non-isomeric) SMILES appears in a training set passed via
``--exclude`` (repeatable). Canonicalization matches the training/selection pipeline
(MDL aromaticity + ``isomericSmiles=False``), so exclusion is exact at the level the
selection script deduplicated on. Held-out molecules are also deduplicated among
themselves.
"""
import argparse, math, sys
from pathlib import Path
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')


def mdl_sanitize(mol):
    Chem.SanitizeMol(mol)
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
    Chem.SanitizeMol(mol)


def canonical_smiles(mol):
    """Molecule-identity key: canonical, non-isomeric SMILES on the heavy-atom
    skeleton (explicit H removed). Using RemoveHs makes the key insensitive to
    explicit-hydrogen placement, so the exclusion set and the held-out candidates
    are compared on the same basis regardless of how each SDF stored hydrogens."""
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=False)
    except Exception:
        return Chem.MolToSmiles(mol, isomericSmiles=False)


def finite_3d(mol):
    if mol.GetNumConformers() == 0:
        return False
    c = mol.GetConformer()
    if not c.Is3D():
        return False
    return all(math.isfinite(v) for i in range(mol.GetNumAtoms())
               for v in (c.GetAtomPosition(i).x, c.GetAtomPosition(i).y, c.GetAtomPosition(i).z))


def explicit_h(mol):
    return all(a.GetNumImplicitHs() == 0 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def load_exclusion(paths):
    """Return the set of canonical SMILES appearing in any --exclude training SDF."""
    excl = set()
    for p in paths:
        n = 0
        supp = Chem.ForwardSDMolSupplier(p, removeHs=False, sanitize=False)
        for mol in supp:
            if mol is None:
                continue
            # Always recompute the key the same way as for held-out candidates.
            # (Stored BONDNET_CANONICAL_SMILES uses explicit-H SMILES and would not
            # match the RemoveHs key, silently missing overlaps.)
            try:
                mdl_sanitize(mol)
            except Exception:
                continue
            excl.add(canonical_smiles(mol)); n += 1
        print(f'[exclude] {n} canonical SMILES loaded from {p}', file=sys.stderr)
    return excl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--skip', type=int, default=0,
                    help='Skip the first N molecules of the source before sampling.')
    ap.add_argument('--take', type=int, default=20000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--exclude', action='append', default=[],
                    help='Training SDF whose canonical SMILES are excluded from the '
                         'held-out set (repeatable). This is what guarantees the '
                         'held-out is leakage-free.')
    ap.add_argument('--allow_implicit_h', action='store_true')
    a = ap.parse_args()

    excl = load_exclusion(a.exclude)
    print(f'[exclude] total unique training canonical SMILES = {len(excl)}', file=sys.stderr)

    supp = Chem.ForwardSDMolSupplier(a.source, removeHs=False, sanitize=False)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(a.out); w.SetKekulize(False)
    seen = 0; written = 0
    seen_out = set()
    stats = {'unparsed': 0, 'sanitize_fail': 0, 'invalid_3d': 0, 'implicit_h': 0,
             'excluded_overlap': 0, 'dup_heldout': 0}
    for mol in supp:
        if written >= a.take:
            break
        if mol is None:
            stats['unparsed'] += 1; seen += 1; continue
        seen += 1
        if seen <= a.skip:
            continue
        try:
            mdl_sanitize(mol)
        except Exception:
            stats['sanitize_fail'] += 1; continue
        if not finite_3d(mol):
            stats['invalid_3d'] += 1; continue
        if not a.allow_implicit_h and not explicit_h(mol):
            stats['implicit_h'] += 1; continue
        smi = canonical_smiles(mol)
        if smi in excl:
            stats['excluded_overlap'] += 1; continue     # leakage — drop
        if smi in seen_out:
            stats['dup_heldout'] += 1; continue
        seen_out.add(smi)
        w.write(mol); written += 1
    w.close()
    print(f'wrote {written} leakage-free molecules to {a.out} '
          f'(skipped first {a.skip}); drops={stats}')


if __name__ == '__main__':
    main()
