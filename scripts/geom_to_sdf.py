"""
Convert GEOM drugs_crude.msgpack(.tar.gz) to an SDF file for BondNet.

This script streams the msgpack payload instead of loading the full GEOM-DRUGS
archive into memory. By default it writes one conformer per molecule. Use
--conformers_per_mol to write more conformers from each molecule.

Example:
    python scripts/geom_to_sdf.py ^
        --input data/drugs_crude.msgpack.tar.gz ^
        --output data/geom_drugs_10k.sdf ^
        --max_mols 10000 ^
        --conformers_per_mol 5 ^
        --conformer lowest_energy
"""

import argparse
import random
import sys
import tarfile
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Path to drugs_crude.msgpack or .tar.gz')
    p.add_argument('--output', required=True, help='Output .sdf path')
    p.add_argument('--max_mols', type=int, default=None, help='Maximum unique input molecules to process')
    p.add_argument('--max_samples', type=int, default=None, help='Maximum written conformer samples')
    p.add_argument('--skip_mols', type=int, default=0, help='Skip this many input molecules first')
    p.add_argument('--conformer', choices=['lowest_energy', 'first', 'random', 'boltzmann'], default='lowest_energy')
    p.add_argument('--max_rel_energy', type=float, default=5.0,
                   help='Max relative energy (kcal/mol) above lowest conformer. '
                        'Conformers above this are excluded. Default 5.0 (~8kT at 300K).')
    p.add_argument('--conformers_per_mol', type=int, default=1,
                   help='Number of conformers to write per molecule. Use -1 for all conformers.')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def _open_msgpack_stream(path: str):
    p = Path(path)
    if str(p).endswith('.tar.gz') or p.suffix == '.gz':
        tar = tarfile.open(path, 'r:gz')
        for member in tar:
            if member.isfile() and member.name.endswith('.msgpack'):
                print(f"  Member: {member.name} ({member.size / 1e9:.1f} GB)")
                f = tar.extractfile(member)
                if f is None:
                    tar.close()
                    raise ValueError(f"Could not extract {member.name}")
                return tar, f
        tar.close()
        raise ValueError('No .msgpack file found in tar.gz')

    return None, open(path, 'rb')


def iter_msgpack_entries(path: str):
    """Yield (smiles, mol_data) pairs without materializing the full archive."""
    import msgpack

    owner, f = _open_msgpack_stream(path)
    try:
        unpacker = msgpack.Unpacker(f, raw=False, strict_map_key=False, max_buffer_size=0)

        # GEOM crude is packed as many consecutive map chunks:
        # smiles -> molecule payload. Keep reading chunks until EOF.
        chunk_idx = 0
        while True:
            try:
                n_entries = unpacker.read_map_header()
            except msgpack.OutOfData:
                return
            except Exception:
                break
            print(f"  Map chunk {chunk_idx}: {n_entries} entries")
            for _ in range(n_entries):
                yield unpacker.unpack(), unpacker.unpack()
            chunk_idx += 1

        # Fallback for variants packed as dict/list chunks.
        for chunk in unpacker:
            if isinstance(chunk, dict):
                yield from chunk.items()
            elif isinstance(chunk, list):
                for item in chunk:
                    if isinstance(item, dict):
                        yield item.get('smiles', ''), item
    finally:
        f.close()
        if owner is not None:
            owner.close()


def _sanitize_with_mdl_aromaticity(mol):
    from rdkit import Chem

    Chem.SanitizeMol(mol)
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
    Chem.SanitizeMol(mol)
    return mol


def _coord_from_atom(atom_dict, key, idx):
    values = atom_dict.get(key)
    if isinstance(values, (list, tuple)) and len(values) > idx:
        return float(values[idx])
    return float(atom_dict.get(key, 0.0))


def _get_xyz_rows(conformer_data: dict):
    xyz = conformer_data.get('xyz')
    if isinstance(xyz, list) and xyz:
        return xyz

    atoms = conformer_data.get('atoms', [])
    rows = []
    for atom in atoms:
        if not isinstance(atom, dict):
            return []
        rows.append([
            atom.get('element', atom.get('atomic_num', atom.get('z', 0))),
            _coord_from_atom(atom, 'x', len(rows)),
            _coord_from_atom(atom, 'y', len(rows)),
            _coord_from_atom(atom, 'z', len(rows)),
        ])
    return rows


def _single_bond_graph(mol):
    from rdkit import Chem

    rw = Chem.RWMol()
    for atom in mol.GetAtoms():
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        rw.AddAtom(new_atom)
    for bond in mol.GetBonds():
        rw.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)
    return rw.GetMol()


def _xyz_order_match(mol, xyz_rows):
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    from rdkit.Geometry import Point3D

    rw = Chem.RWMol()
    for row in xyz_rows:
        rw.AddAtom(Chem.Atom(int(row[0])))
    xyz_mol = rw.GetMol()

    conf = Chem.Conformer(len(xyz_rows))
    for i, row in enumerate(xyz_rows):
        conf.SetAtomPosition(i, Point3D(float(row[1]), float(row[2]), float(row[3])))
    xyz_mol.AddConformer(conf, assignId=True)

    charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    rdDetermineBonds.DetermineConnectivity(xyz_mol, charge=int(charge))

    query = _single_bond_graph(mol)
    target = _single_bond_graph(xyz_mol)
    match = target.GetSubstructMatch(query)
    if len(match) != mol.GetNumAtoms():
        return None
    return match


def geom_mol_to_rdkit(smiles: str, conformer_data: dict):
    """Build an RDKit molecule from SMILES and GEOM conformer coordinates."""
    from rdkit import Chem
    from rdkit.Chem import Conformer

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None

    try:
        _sanitize_with_mdl_aromaticity(mol)
    except Exception:
        return None

    mol = Chem.AddHs(mol)

    xyz_rows = _get_xyz_rows(conformer_data)
    if not xyz_rows:
        return None
    if len(xyz_rows) != mol.GetNumAtoms():
        mol = Chem.RemoveHs(mol)
        if len(xyz_rows) != mol.GetNumAtoms():
            return None

    match = _xyz_order_match(mol, xyz_rows)
    if match is None:
        return None

    conf = Conformer(mol.GetNumAtoms())
    for atom_idx, xyz_idx in enumerate(match):
        row = xyz_rows[xyz_idx]
        if not isinstance(row, (list, tuple)) or len(row) < 4:
            return None
        x, y, z = float(row[1]), float(row[2]), float(row[3])
        conf.SetAtomPosition(atom_idx, (x, y, z))

    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)

    # Reject flat/2D conformers (all Z coords near zero)
    z_coords = [conf.GetAtomPosition(i).z for i in range(mol.GetNumAtoms())]
    if max(abs(z) for z in z_coords) < 0.01:
        return None

    return mol


def _boltzmann_weights(conformers: list) -> list:
    """Return normalised Boltzmann weights, falling back to uniform if missing."""
    weights = []
    for c in conformers:
        w = c.get('boltzmannweight')
        weights.append(float(w) if w is not None and float(w) > 0 else None)
    if all(w is None for w in weights):
        return [1.0 / len(conformers)] * len(conformers)
    total = sum(w for w in weights if w is not None)
    return [(w if w is not None else 0.0) / max(total, 1e-12) for w in weights]


def pick_conformer(conformers: list, strategy: str):
    if not conformers:
        return None
    if strategy == 'first':
        return conformers[0]
    if strategy == 'random':
        return random.choice(conformers)
    if strategy == 'boltzmann':
        weights = _boltzmann_weights(conformers)
        return random.choices(conformers, weights=weights, k=1)[0]

    best, best_e = None, float('inf')
    for conf in conformers:
        e = conf.get('totalenergy', conf.get('energy', float('inf')))
        e = float('inf') if e is None else float(e)
        if e < best_e:
            best_e, best = e, conf
    return best if best is not None else conformers[0]


def pick_conformers(conformers: list, strategy: str, k: int):
    if not conformers:
        return []
    if k < 0 or k >= len(conformers):
        k = len(conformers)
    if strategy == 'first':
        return conformers[:k]
    if strategy == 'boltzmann':
        weights = _boltzmann_weights(conformers)
        return random.choices(conformers, weights=weights, k=k)
    if strategy == 'random':
        return random.sample(conformers, k)

    def energy(conf):
        e = conf.get('totalenergy', conf.get('energy', float('inf')))
        return float('inf') if e is None else float(e)

    return sorted(conformers, key=energy)[:k]


def main():
    args = parse_args()
    random.seed(args.seed)

    try:
        import msgpack  # noqa: F401
    except ImportError:
        print('ERROR: msgpack is not installed. Install it with: pip install msgpack')
        sys.exit(1)

    from rdkit import Chem

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(out_path))

    print(f"Streaming GEOM data from: {args.input}")
    n_seen = 0
    n_processed = 0
    n_written = 0
    n_failed = 0
    t0 = time.time()

    for smiles, mol_data in iter_msgpack_entries(args.input):
        n_seen += 1
        if n_seen <= args.skip_mols:
            continue
        if args.max_mols is not None and n_processed >= args.max_mols:
            break
        n_processed += 1
        if args.max_samples is not None and n_written >= args.max_samples:
            break

        if n_processed % 10000 == 0:
            elapsed = time.time() - t0
            rate = n_processed / max(elapsed, 1e-9)
            print(
                f"  seen={n_seen} processed={n_processed} written={n_written} "
                f"failed={n_failed} {rate:.0f} mol/s"
            )

        if not isinstance(mol_data, dict):
            n_failed += 1
            continue

        conformers = mol_data.get('conformers', [])
        if args.max_rel_energy is not None and args.max_rel_energy > 0:
            conformers = [
                c for c in conformers
                if c.get('relativeenergy') is None
                or float(c.get('relativeenergy', 0)) <= args.max_rel_energy
            ]
        selected_confs = pick_conformers(conformers, args.conformer, args.conformers_per_mol)
        if not selected_confs:
            n_failed += 1
            continue

        for conf_rank, conf_data in enumerate(selected_confs):
            if args.max_samples is not None and n_written >= args.max_samples:
                break

            mol = geom_mol_to_rdkit(str(smiles), conf_data)
            if mol is None:
                n_failed += 1
                continue

            geom_id = conf_data.get('geom_id', conf_rank)
            mol.SetProp('_Name', f'geom_{n_seen}_conf_{geom_id}')
            mol.SetIntProp('geom_mol_idx', int(n_seen - 1))
            try:
                mol.SetIntProp('geom_conf_idx', int(geom_id))
            except Exception:
                mol.SetProp('geom_conf_idx', str(geom_id))
            mol.SetIntProp('geom_conf_rank', int(conf_rank))
            mol.SetProp('geom_smiles', str(smiles))
            writer.write(mol)
            n_written += 1

    writer.close()

    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0.0
    print(f"\nDone: seen={n_seen}, written={n_written}, failed={n_failed}, {elapsed:.1f}s")
    print(f"Output: {out_path} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
