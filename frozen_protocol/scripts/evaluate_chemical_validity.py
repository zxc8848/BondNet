"""Evaluate RDKit sanitization and round-trip validity of BondNet graphs.

BondNet predicts connectivity for all candidate pairs, but assigns bond types
only to heavy-heavy edges; predicted H-X edges are therefore single bonds.  It
does not predict formal charge.  We report two explicitly different diagnostics:

* reference-charge: copy each atom's reference formal charge before sanitizing;
* zero-charge: use only atomic numbers and the predicted graph (all charges 0).

The first isolates graph/typing validity.  The second exposes the limitation of
BondNet's current output space and must not be presented as a pure graph error.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluate as ev
from bondnet.data.dataset import CachedBondNetDataset, collate_fn
from bondnet.data.noise_augment import GaussianNoiseAugment, apply_dynamic_candidate_mask
from bondnet.model.bond_type_gnn import BondTypeGNN, compute_bonded_geometry
from bondnet.model.bondnet import _symmetrize_logits


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default=None,
                   help='Joint one-stage BondNet checkpoint.')
    p.add_argument('--stage1', default=None,
                   help='Stage 1 checkpoint for the hard two-stage comparator.')
    p.add_argument('--stage2', default=None,
                   help='Stage 2 checkpoint; required with --stage1.')
    p.add_argument('--cache', required=True)
    p.add_argument('--reference_sdf', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--noise_levels', type=float, nargs='+', default=[0.0, 0.1, 0.2])
    p.add_argument('--eval_noise_seed', type=int, default=20260921)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--cutoff', type=float, default=2.5)
    p.add_argument('--h_cutoff', type=float, default=2.5)
    p.add_argument('--conn_threshold', type=float, default=0.5)
    p.add_argument('--device', default='auto')
    return p.parse_args()


def load_stage2(path, model, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    a = ck.get('args', {})
    stage2 = BondTypeGNN(
        hidden_size=a.get('hidden_size', model.backbone.hidden_state_size),
        num_layers=a.get('num_layers', 4),
        edge_embedding_size=a.get('edge_embedding_size', 16),
        bond_cutoff=a.get('bond_cutoff', 3.0),
        random_node_feat_dim=a.get('random_node_feat_dim', 0),
        random_node_feat_std=a.get('random_node_feat_std', 1.0),
    ).to(device)
    stage2.load_state_dict(ck['stage2_state_dict'])
    stage2.eval()
    return stage2


def classify_failure(exc: Exception) -> str:
    text = str(exc).lower()
    if 'valence' in text:
        return 'valence'
    if 'kekul' in text or 'aromatic' in text:
        return 'aromatic_or_kekulize'
    if 'ring' in text:
        return 'ring'
    return 'other'


def bond_type_from_label(label: int):
    from rdkit import Chem
    return {
        0: Chem.BondType.SINGLE,
        1: Chem.BondType.DOUBLE,
        2: Chem.BondType.TRIPLE,
        3: Chem.BondType.AROMATIC,
    }[int(label)]


def graph_signature(mol):
    from rdkit import Chem
    bmap = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3,
    }
    bonds = {}
    for bond in mol.GetBonds():
        pair = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        bonds[pair] = bmap.get(bond.GetBondType(), -1)
    charges = tuple(atom.GetFormalCharge() for atom in mol.GetAtoms())
    elems = tuple(atom.GetAtomicNum() for atom in mol.GetAtoms())
    return elems, charges, bonds


def reference_bonds(mol):
    return graph_signature(mol)[2]


def build_predicted_mol(ref_mol, pred_bonds, use_reference_charge: bool):
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    rw = Chem.RWMol()
    for ref_atom in ref_mol.GetAtoms():
        atom = Chem.Atom(ref_atom.GetAtomicNum())
        atom.SetFormalCharge(ref_atom.GetFormalCharge() if use_reference_charge else 0)
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    aromatic_atoms = set()
    for (i, j), label in sorted(pred_bonds.items()):
        rw.AddBond(int(i), int(j), bond_type_from_label(label))
        if int(label) == 3:
            aromatic_atoms.update((int(i), int(j)))
    mol = rw.GetMol()
    for idx in aromatic_atoms:
        mol.GetAtomWithIdx(idx).SetIsAromatic(True)
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.AROMATIC:
            bond.SetIsAromatic(True)

    if ref_mol.GetNumConformers():
        src = ref_mol.GetConformer()
        conf = Chem.Conformer(ref_mol.GetNumAtoms())
        conf.Set3D(True)
        for idx in range(ref_mol.GetNumAtoms()):
            p = src.GetAtomPosition(idx)
            conf.SetAtomPosition(idx, Point3D(float(p.x), float(p.y), float(p.z)))
        mol.AddConformer(conf, assignId=True)
    return mol


def validate_graph(ref_mol, pred_bonds, use_reference_charge: bool):
    from rdkit import Chem

    try:
        mol = build_predicted_mol(ref_mol, pred_bonds, use_reference_charge)
        Chem.SanitizeMol(mol)
    except Exception as exc:
        return False, False, False, False, classify_failure(exc)

    molblock_ok = False
    smiles_ok = False
    canonical_match = False
    failure = ''
    try:
        block = Chem.MolToMolBlock(mol, kekulize=False)
        reread = Chem.MolFromMolBlock(
            block, sanitize=True, removeHs=False, strictParsing=True
        )
        molblock_ok = reread is not None and graph_signature(reread) == graph_signature(mol)
    except Exception as exc:
        failure = 'molblock_' + classify_failure(exc)
    try:
        smiles = Chem.MolToSmiles(mol, canonical=True, allHsExplicit=True)
        params = Chem.SmilesParserParams()
        params.removeHs = False
        params.sanitize = True
        reread = Chem.MolFromSmiles(smiles, params)
        if reread is not None:
            smiles2 = Chem.MolToSmiles(reread, canonical=True, allHsExplicit=True)
            smiles_ok = smiles2 == smiles
        pred_canonical = Chem.MolToSmiles(
            mol, canonical=True, allHsExplicit=True, isomericSmiles=False
        )
        ref_canonical = Chem.MolToSmiles(
            ref_mol, canonical=True, allHsExplicit=True, isomericSmiles=False
        )
        canonical_match = pred_canonical == ref_canonical
    except Exception as exc:
        if not failure:
            failure = 'smiles_' + classify_failure(exc)
    return True, molblock_ok, smiles_ok, canonical_match, failure


@torch.no_grad()
def predict_batch(model, stage2, batch, device, cutoff, h_cutoff, threshold, sigma, eval_noise_seed):
    if sigma > 0:
        aug = GaussianNoiseAugment(
            sigma=sigma, sigma_min=sigma, sigma_max=sigma, cutoff=cutoff
        )
        noise_seed = int(eval_noise_seed) + int(round(sigma * 10000.0))
        batch = aug.augment_batch(
            batch, per_molecule=True, base_seed=noise_seed, epoch=0
        )
    batch = apply_dynamic_candidate_mask(batch, cutoff, h_cutoff)
    batch = ev._to_device(batch, device)
    elems, coord = batch['elems'], batch['coord']
    edge_index = batch['edge_index']
    _, _, edge_feat = model.backbone(
        elems, coord, edge_index, batch['edge_diff'], batch['edge_dist'],
        num_atoms_per_mol=batch.get('num_atoms_per_mol'),
    )
    conn = (torch.sigmoid(model.stage1(edge_feat)) >= threshold) & batch['train_edge_mask']
    bonded_pos = torch.where(conn)[0]
    pred_type_full = torch.full((edge_index.shape[0],), -1, dtype=torch.long, device=device)
    pred_type_full[bonded_pos] = 0
    if bonded_pos.numel():
        bonded_ei = edge_index[bonded_pos]
        bdiff, bdist = compute_bonded_geometry(coord, bonded_ei)
        hh = (elems[bonded_ei[:, 0]] != 1) & (elems[bonded_ei[:, 1]] != 1)
        readout_pos = bonded_pos[hh]
        readout_ei = bonded_ei[hh]
        if readout_ei.numel():
            if stage2 is None:
                logits, _ = model.stage3(edge_feat[readout_pos])
            else:
                rdiff, rdist = compute_bonded_geometry(coord, readout_ei)
                logits = stage2(elems, bonded_ei, bdiff, bdist, readout_ei, rdiff, rdist)
            labels = _symmetrize_logits(logits, readout_ei).argmax(dim=-1)
            pred_type_full[readout_pos] = labels

    sizes = [int(x) for x in batch['num_atoms_per_mol'].detach().cpu().tolist()]
    offsets = np.cumsum([0] + sizes[:-1]).tolist()
    edge_mol = batch['edge_mol_idx'].detach().cpu()
    edge_cpu = edge_index.detach().cpu()
    type_cpu = pred_type_full.detach().cpu()
    per_mol = []
    for mol_i, offset in enumerate(offsets):
        positions = torch.where((edge_mol == mol_i) & (type_cpu >= 0))[0]
        bonds = {}
        for pos in positions.tolist():
            i = int(edge_cpu[pos, 0]) - int(offset)
            j = int(edge_cpu[pos, 1]) - int(offset)
            pair = tuple(sorted((i, j)))
            label = int(type_cpu[pos])
            old = bonds.get(pair)
            if old is None:
                bonds[pair] = label
            elif old != label:
                # Directed predictions should agree after symmetrization.  If
                # only one direction was typed differently, keep the higher
                # order deterministically and expose the graph to sanitization.
                bonds[pair] = max(old, label)
        per_mol.append(bonds)
    return per_mol, batch


def iter_reference_mols(path):
    from rdkit import Chem
    supplier = Chem.ForwardSDMolSupplier(str(path), removeHs=False, sanitize=False)
    for mol in supplier:
        if mol is None:
            raise RuntimeError('reference SDF contains an unreadable molecule')
        # SD serialization may kekulize aromatic bonds. Restore the same MDL
        # aromatic representation used to construct the feature cache before
        # computing graph exactness.
        Chem.SanitizeMol(mol)
        Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
        Chem.SanitizeMol(mol)
        yield mol


def summarize(rows, sigma):
    subset = [row for row in rows if float(row['sigma']) == float(sigma)]
    output = {'sigma': float(sigma), 'n_molecules': len(subset)}
    for mode in ('reference_charge', 'zero_charge'):
        for field in (
            'sanitize_ok', 'molblock_roundtrip_ok', 'smiles_roundtrip_ok',
            'canonical_smiles_match',
        ):
            count = sum(int(row[f'{mode}_{field}']) for row in subset)
            output[f'{mode}_{field}_count'] = count
            output[f'{mode}_{field}_rate'] = count / len(subset)
        output[f'{mode}_failure_reasons'] = dict(Counter(
            row[f'{mode}_failure_reason'] or 'none' for row in subset
            if not int(row[f'{mode}_sanitize_ok'])
        ))
    output['reference_full_graph_exact_rate'] = (
        sum(int(row['reference_full_graph_exact']) for row in subset) / len(subset)
    )
    output['charged_molecule_fraction'] = (
        sum(int(row['reference_has_charge']) for row in subset) / len(subset)
    )
    return output


def main():
    args = parse_args()
    if bool(args.checkpoint) == bool(args.stage1):
        raise ValueError('Specify either --checkpoint or --stage1, but not both.')
    if args.stage1 and not args.stage2:
        raise ValueError('--stage2 is required with --stage1.')
    if args.checkpoint and args.stage2:
        raise ValueError('--stage2 must not be used with a joint --checkpoint.')
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.error')
    RDLogger.DisableLog('rdApp.warning')
    device = ev._get_device(args.device)
    model_path = args.checkpoint if args.checkpoint else args.stage1
    model = ev.load_model(str((ROOT / model_path).resolve()), device)
    model.eval()
    stage2 = load_stage2(str((ROOT / args.stage2).resolve()), model, device) if args.stage2 else None
    dataset = CachedBondNetDataset(str((ROOT / args.cache).resolve()))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    reference_path = (ROOT / args.reference_sdf).resolve()
    rows = []

    for sigma in args.noise_levels:
        sigma = float(sigma)
        sigma_seed = int(args.eval_noise_seed) + int(round(sigma * 10000.0))
        torch.manual_seed(sigma_seed)
        np.random.seed(sigma_seed % (2**32 - 1))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sigma_seed)
        refs = iter_reference_mols(reference_path)
        n_done = 0
        for batch in loader:
            pred_per_mol, used_batch = predict_batch(
                model, stage2, batch, device, args.cutoff, args.h_cutoff,
                args.conn_threshold, sigma, args.eval_noise_seed,
            )
            mol_ids = used_batch['mol_idx'].detach().cpu().tolist()
            elems_cpu = used_batch['elems'].detach().cpu().tolist()
            sizes = [int(x) for x in used_batch['num_atoms_per_mol'].detach().cpu().tolist()]
            offset = 0
            for mol_id, pred, size in zip(mol_ids, pred_per_mol, sizes):
                ref = next(refs)
                ref_elems = [atom.GetAtomicNum() for atom in ref.GetAtoms()]
                cache_elems = [int(z) for z in elems_cpu[offset:offset + size]]
                offset += size
                if ref_elems != cache_elems:
                    raise RuntimeError(f'element/order mismatch for molecule {mol_id}')
                pred_indices_ok = all(0 <= i < len(ref_elems) and 0 <= j < len(ref_elems) for i, j in pred)
                if not pred_indices_ok:
                    raise RuntimeError(f'predicted atom index out of range for molecule {mol_id}')
                true = reference_bonds(ref)
                has_charge = any(atom.GetFormalCharge() != 0 for atom in ref.GetAtoms())
                row = {
                    'mol_idx': int(mol_id), 'sigma': sigma,
                    'n_atoms': ref.GetNumAtoms(), 'n_pred_bonds': len(pred),
                    'reference_has_charge': int(has_charge),
                    'reference_full_graph_exact': int(pred == true),
                }
                for mode, use_charge in (('reference_charge', True), ('zero_charge', False)):
                    san, molblock, smiles, canonical_match, reason = validate_graph(
                        ref, pred, use_charge
                    )
                    row[f'{mode}_sanitize_ok'] = int(san)
                    row[f'{mode}_molblock_roundtrip_ok'] = int(molblock)
                    row[f'{mode}_smiles_roundtrip_ok'] = int(smiles)
                    row[f'{mode}_canonical_smiles_match'] = int(canonical_match)
                    row[f'{mode}_failure_reason'] = reason
                rows.append(row)
                n_done += 1
            if n_done % 2560 == 0:
                print(f'[VALIDITY] sigma={sigma:g} {n_done}/{len(dataset)}', flush=True)
        try:
            next(refs)
            raise RuntimeError('reference SDF has more molecules than the cache')
        except StopIteration:
            pass
        if n_done != len(dataset):
            raise RuntimeError(f'processed {n_done}, expected {len(dataset)}')
        s = summarize(rows, sigma)
        print(
            f"[RESULT] sigma={sigma:g} assisted={s['reference_charge_sanitize_ok_rate']:.4f} "
            f"zero_charge={s['zero_charge_sanitize_ok_rate']:.4f} "
            f"full_graph_exact={s['reference_full_graph_exact_rate']:.4f}",
            flush=True,
        )

    output = (ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output / 'per_molecule.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = [summarize(rows, sigma) for sigma in args.noise_levels]
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'protocol.json').write_text(json.dumps({
        'checkpoint': args.checkpoint,
        'stage1': args.stage1, 'stage2': args.stage2, 'cache': args.cache,
        'reference_sdf': args.reference_sdf, 'noise_levels': args.noise_levels,
        'eval_noise_seed': args.eval_noise_seed, 'batch_size': args.batch_size,
        'cutoff': args.cutoff, 'h_cutoff': args.h_cutoff,
        'formal_charge_modes': ['reference_charge', 'zero_charge'],
        'hydrogen_bond_order_policy': 'all predicted H-X bonds are single',
    }, indent=2), encoding='utf-8')
    print(f'[DONE] wrote {output}', flush=True)


if __name__ == '__main__':
    main()
