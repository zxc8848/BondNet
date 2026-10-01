"""
Evaluate rule-based bond perception baselines on an SDF file.

The metrics are aligned by unordered atom pairs, not by RDKit/OpenBabel bond
iteration order. Connectivity is measured on all bonds, while bond-type metrics
are measured on heavy-heavy true bonds to match the Stage 2 evaluation protocol.

Examples:
    python scripts/evaluate_rule_baselines.py ^
        --data_path data/geom_drugs_200k.sdf ^
        --methods rdkit ^
        --max_mols 5000
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from bondnet.utils.metrics import BondNetMetrics


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate rule-based bond baselines')
    p.add_argument('--data_path', required=True, help='Input SDF with 3D coordinates and ground-truth bonds')
    p.add_argument('--methods', nargs='+', default=['rdkit'], choices=['rdkit', 'openbabel'])
    p.add_argument('--max_mols', type=int, default=None)
    p.add_argument('--noise_levels', type=float, nargs='+', default=[0.0],
                   help='Gaussian noise sigma values (A) to evaluate at')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', type=str, default='results/rule_baselines')
    p.add_argument('--progress_every', type=int, default=10000)
    return p.parse_args()


def _bond_type_map():
    from rdkit import Chem

    return {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3,
    }


def _sanitize_mdl(mol):
    from rdkit import Chem

    Chem.SanitizeMol(mol)
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
    Chem.SanitizeMol(mol)
    return mol


def _true_bonds(mol) -> Dict[Tuple[int, int], int]:
    bmap = _bond_type_map()
    out = {}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        out[tuple(sorted((i, j)))] = bmap.get(bond.GetBondType(), 0)
    return out


def _add_noise(mol, sigma: float, rng):
    """Return a copy of mol with Gaussian noise added to atom positions."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    mol = Chem.RWMol(mol)
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    noise = rng.normal(0.0, sigma, size=(n, 3))
    for i in range(n):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(pos.x + noise[i, 0],
                                        pos.y + noise[i, 1],
                                        pos.z + noise[i, 2]))
    return mol.GetMol()


def _rdkit_predict(mol) -> Dict[Tuple[int, int], int]:
    from rdkit import Chem
    from rdkit.Chem.rdDetermineBonds import DetermineBonds

    bmap = _bond_type_map()
    rw = Chem.RWMol()
    for atom in mol.GetAtoms():
        rw.AddAtom(Chem.Atom(atom.GetAtomicNum()))
    conf = Chem.Conformer(rw.GetNumAtoms())
    orig = mol.GetConformer()
    for idx in range(rw.GetNumAtoms()):
        conf.SetAtomPosition(idx, orig.GetAtomPosition(idx))
    rw.AddConformer(conf, assignId=True)
    raw = rw.GetMol()

    charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    # Distorted geometries can make bond-order search combinatorial.  RDKit's
    # default maxIterations=0 is unbounded and can stall an entire benchmark on
    # one molecule.  Treat exceeding a generous fixed budget as tool failure;
    # the unified scorer includes such failures in the common denominator.
    DetermineBonds(raw, charge=int(charge), maxIterations=1000)

    out = {}
    for bond in raw.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        out[tuple(sorted((i, j)))] = bmap.get(bond.GetBondType(), 0)
    return out


def _openbabel_predict(mol) -> Dict[Tuple[int, int], int]:
    from openbabel import openbabel as ob  # type: ignore

    obmol = ob.OBMol()
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        ob_atom = obmol.NewAtom()
        ob_atom.SetAtomicNum(int(atom.GetAtomicNum()))
        ob_atom.SetVector(float(pos.x), float(pos.y), float(pos.z))
    obmol.ConnectTheDots()
    obmol.PerceiveBondOrders()

    code_to_type = {1: 0, 2: 1, 3: 2, 5: 3}
    out = {}
    for bond in ob.OBMolBondIter(obmol):
        i = int(bond.GetBeginAtomIdx()) - 1
        j = int(bond.GetEndAtomIdx()) - 1
        if bond.IsAromatic():
            label = 3
        else:
            label = code_to_type.get(int(bond.GetBondOrder()), 0)
        out[tuple(sorted((i, j)))] = label
    return out


def _wrong_type(true_label: int) -> int:
    return 1 if true_label == 0 else 0


def _update_metrics(metrics: BondNetMetrics, mol, true: Dict[Tuple[int, int], int], pred: Dict[Tuple[int, int], int]):
    all_pairs = sorted(set(true) | set(pred))
    if all_pairs:
        y_conn = torch.tensor([1 if p in true else 0 for p in all_pairs], dtype=torch.long)
        p_conn = torch.tensor([1 if p in pred else 0 for p in all_pairs], dtype=torch.long)
        metrics.update_connectivity(p_conn, y_conn)

    y_type = []
    p_type = []
    for pair, true_label in true.items():
        i, j = pair
        if mol.GetAtomWithIdx(i).GetAtomicNum() == 1 or mol.GetAtomWithIdx(j).GetAtomicNum() == 1:
            continue
        pred_label = pred.get(pair)
        y_type.append(true_label)
        p_type.append(_wrong_type(true_label) if pred_label is None else pred_label)

    if y_type:
        y = torch.tensor(y_type, dtype=torch.long)
        p = torch.tensor(p_type, dtype=torch.long)
        metrics.update_bond_types(p, y)
        metrics.update_molecule_validity(bool((p == y).all().item()))


def iter_mols(path: str) -> Iterable:
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(path, sanitize=False, removeHs=False)
    for mol in supplier:
        if mol is None:
            continue
        try:
            yield _sanitize_mdl(mol)
        except Exception:
            continue


def evaluate_method(method: str, data_path: str, sigma: float = 0.0,
                    max_mols: int = None, progress_every: int = 10000, rng=None):
    import numpy as np
    if rng is None:
        rng = np.random.default_rng(42)

    if method == 'rdkit':
        from rdkit.Chem.rdDetermineBonds import DetermineBonds  # noqa: F401
        predictor = _rdkit_predict
    elif method == 'openbabel':
        from openbabel import openbabel as ob  # noqa: F401
        predictor = _openbabel_predict
    else:
        raise ValueError(method)

    metrics = BondNetMetrics()
    n_ok = 0
    n_failed = 0
    t0 = time.time()

    for mol in iter_mols(data_path):
        if max_mols is not None and n_ok >= max_mols:
            break
        try:
            true = _true_bonds(mol)
            noisy_mol = _add_noise(mol, sigma, rng) if sigma > 0 else mol
            pred = predictor(noisy_mol)
            _update_metrics(metrics, mol, true, pred)
            n_ok += 1
        except Exception:
            n_failed += 1

        if progress_every > 0 and n_ok > 0 and n_ok % progress_every == 0:
            elapsed = time.time() - t0
            print(f'  {method} σ={sigma:.2f}: {n_ok} mols, {n_ok / max(elapsed, 1e-9):.1f} mol/s')

    out = metrics.compute()
    out['n_molecules'] = int(n_ok)
    out['n_failed'] = int(n_failed)
    out['sigma'] = sigma
    out['time_s'] = float(time.time() - t0)
    return out


def main():
    import numpy as np
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # results[method][sigma_str] = metrics dict
    all_results = {}
    for method in args.methods:
        all_results[method] = {}
        for sigma in args.noise_levels:
            print(f'Evaluating {method} at sigma={sigma:.3f} A...')
            try:
                row = evaluate_method(
                    method, args.data_path,
                    sigma=sigma,
                    max_mols=args.max_mols,
                    progress_every=args.progress_every,
                    rng=rng,
                )
                all_results[method][sigma] = row
                print(
                    f"  conn_f1={row.get('conn_f1', 0):.4f} "
                    f"f1_macro={row.get('f1_macro', 0):.4f} "
                    f"d={row.get('f1_double', 0):.4f} "
                    f"a={row.get('f1_aromatic', 0):.4f} "
                    f"mol_valid={100 * row.get('mol_validity', 0):.1f}%"
                )
            except ImportError as exc:
                print(f'  skipping {method}: {exc}')
                break

    json_path = out_dir / 'rule_baselines.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    csv_path = out_dir / 'rule_baselines.csv'
    fields = [
        'method', 'sigma', 'n_molecules', 'conn_f1',
        'f1_single', 'f1_double', 'f1_triple', 'f1_aromatic', 'f1_macro',
        'mol_validity', 'time_s',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method, sigma_dict in all_results.items():
            for sigma, row in sigma_dict.items():
                writer.writerow({'method': method, 'sigma': sigma,
                                 **{k: row.get(k, '') for k in fields if k not in ('method', 'sigma')}})

    print(f'\nWrote {json_path}')
    print(f'Wrote {csv_path}')
    print(f'\n{"Method":<12} {"sigma(A)":>8} {"F1-macro":>10} {"F1-double":>10} {"F1-arom":>10} {"Mol-valid%":>11}')
    print('-' * 55)
    for method, sigma_dict in all_results.items():
        for sigma, row in sigma_dict.items():
            print(f"{method:<12} {sigma:>8.3f} {row.get('f1_macro',0):>10.4f} "
                  f"{row.get('f1_double',0):>10.4f} {row.get('f1_aromatic',0):>10.4f} "
                  f"{100*row.get('mol_validity',0):>10.1f}%")


if __name__ == '__main__':
    main()
