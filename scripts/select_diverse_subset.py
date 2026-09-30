#!/usr/bin/env python3
"""
select_diverse_subset.py -- build a small, diverse, representative training
subset for molecular bond perception by pooling several SDF sources.

Strategy
--------
1. Stream-read each source SDF (with a per-source read cap), retaining only
    molecules with finite 3D coordinates and explicit-H representations.
2. Deduplicate by canonical (non-isomeric) SMILES.
3. Tag every molecule with "hard-context" flags that matter for bond typing
   (triple bonds, aromatic heteroatoms, formal charges, nitro/N-oxide/ammonium,
   sulfonyl, phosphorus, each halogen, small/large rings).
4. Selection = two phases:
     (A) quota fill -- guarantee at least ``--quota`` molecules per hard tag
         (diversity-picked within the tag via MaxMin), so rare-but-important
         contexts are never starved;
     (B) global MaxMin diversity pick over the remaining pool to reach
         ``--target`` molecules.
5. Write the selected molecules to an SDF without kekulizing aromatic bonds,
   and print/save a support report with per-molecule provenance.

Diversity uses Morgan (ECFP4, 2048-bit) fingerprints with Tanimoto distance.
"""
import argparse, json, math, sys, random
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
RDLogger.DisableLog('rdApp.*')

HALOGENS = {9: 'F', 17: 'Cl', 35: 'Br', 53: 'I'}

# SMARTS for hard functional contexts
_SMARTS = {
    'nitro':    Chem.MolFromSmarts('[$([NX3](=O)=O),$([NX3+](=O)[O-])]'),
    'sulfonyl': Chem.MolFromSmarts('[$([SX4](=O)(=O))]'),
    'n_oxide':  Chem.MolFromSmarts('[#7+][#8-]'),
}


def _sanitize_with_mdl_aromaticity(mol):
    """Perceive aromaticity with the MDL model, mirroring the training loader.

    BondNet's dataset loader (bondnet.data.dataset._sanitize_with_mdl_aromaticity)
    normalizes every source molecule to the MDL aromaticity model before it reads
    bond types for labels. Applying the identical procedure here ensures the
    ``arom_hetero`` tag, the canonical SMILES used for dedup, and the exported
    molecules all reflect the exact aromaticity BondNet will train on -- not
    RDKit's default model.
    """
    Chem.SanitizeMol(mol)
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
    Chem.SanitizeMol(mol)


def mol_signature(mol):
    """Return a set of hard-context tags for a molecule."""
    tags = set()
    for b in mol.GetBonds():
        if b.GetBondType() == Chem.BondType.TRIPLE:
            tags.add('triple'); break
    for a in mol.GetAtoms():
        z = a.GetAtomicNum()
        if a.GetIsAromatic() and z != 6:
            tags.add('arom_hetero')
        if a.GetFormalCharge() != 0:
            tags.add('charged')
        if z in HALOGENS:
            tags.add('halogen_' + HALOGENS[z])
        if z == 15:
            tags.add('phosphorus')
    for name, patt in _SMARTS.items():
        if patt is not None and mol.HasSubstructMatch(patt):
            tags.add(name)
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        n = len(ring)
        if n <= 4:
            tags.add('small_ring')
        elif n >= 8:
            tags.add('macrocycle')
    return tags

def morgan_fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

def heavy_atoms(mol):
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)

def has_finite_3d_coordinates(mol):
    """Return whether ``mol`` has one conformer with finite 3D coordinates."""
    if mol.GetNumConformers() == 0:
        return False
    conf = mol.GetConformer()
    if not conf.Is3D():
        return False
    for atom_idx in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_idx)
        if not all(math.isfinite(v) for v in (pos.x, pos.y, pos.z)):
            return False
    return True


def has_explicit_hydrogen_representation(mol):
    """Return whether no heavy atom relies on implicit hydrogens.

    Molecules such as fully substituted carbons may legitimately contain no H
    atoms, so checking the number of hydrogen atoms alone would be wrong.
    """
    return all(
        atom.GetNumImplicitHs() == 0
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1
    )


def molblock_roundtrips_with_training_aromaticity(block):
    """Validate that an exported MolBlock can be read by BondNet's loader."""
    mol = Chem.MolFromMolBlock(block, removeHs=False, sanitize=False)
    if mol is None:
        return False
    try:
        _sanitize_with_mdl_aromaticity(mol)
    except Exception:
        return False
    return True


def load_source(path, tag, max_read, stride, pool, seen_smiles, stats,
                require_explicit_h):
    # Read raw (sanitize=False) then apply the MDL-aromaticity sanitization used
    # by the training loader, so aromaticity is identical to BondNet training.
    supp = Chem.ForwardSDMolSupplier(path, removeHs=False, sanitize=False)
    read = 0
    for i, mol in enumerate(supp):
        if max_read and read >= max_read:
            break
        if stride > 1 and (i % stride) != 0:
            continue
        if mol is None:
            stats['unparsed'] += 1
            continue
        try:
            _sanitize_with_mdl_aromaticity(mol)
        except Exception:
            stats['sanitize_fail'] += 1
            continue
        if not has_finite_3d_coordinates(mol):
            stats['invalid_3d'] += 1
            continue
        if require_explicit_h and not has_explicit_hydrogen_representation(mol):
            stats['implicit_h'] += 1
            continue
        try:
            smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        except Exception:
            stats['unparsed'] += 1
            continue
        read += 1
        if smi in seen_smiles:
            stats['dup'] += 1
            continue
        seen_smiles.add(smi)
        try:
            fp = morgan_fp(mol)
            sig = mol_signature(mol)
            # Writing a Kekule form turns aromatic bonds into alternating
            # single/double labels.  BondNet trains aromaticity explicitly, so
            # preserve RDKit's aromatic bond type in the serialized molecule.
            block = Chem.MolToMolBlock(mol, kekulize=False)
            if not molblock_roundtrips_with_training_aromaticity(block):
                stats['roundtrip_fail'] += 1
                continue
        except Exception:
            stats['unparsed'] += 1
            continue
        pool.append({'smi': smi, 'fp': fp, 'sig': sig, 'block': block,
                     'src': tag, 'src_index': i,
                     'nheavy': heavy_atoms(mol)})
    stats['read_' + tag] = read

def maxmin_pick(fps, n, seed, firstpicks=None):
    """Return indices of a MaxMin-diverse pick of size n from fps.

    Uses RDKit's native LazyBitVectorPick (Tanimoto computed in C++), which
    scales to hundreds of thousands of fingerprints; a Python distance callback
    would be orders of magnitude slower.
    """
    n = min(n, len(fps))
    if n <= 0:
        return []
    fp_list = list(fps)
    picker = MaxMinPicker()
    idx = picker.LazyBitVectorPick(fp_list, len(fp_list), n,
                                   firstPicks=list(firstpicks) if firstpicks else [],
                                   seed=seed)
    return list(idx)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', action='append', default=[], metavar='TAG=PATH',
                    help='Repeatable. e.g. --source qm9=data/qm9.sdf')
    ap.add_argument('--target', type=int, default=50000, help='Total molecules to select')
    ap.add_argument('--quota', type=int, default=2000, help='Min molecules guaranteed per hard tag')
    ap.add_argument('--max_read_per_source', type=int, default=60000)
    ap.add_argument('--stride', type=int, default=1, help='Read every k-th molecule to spread sampling')
    ap.add_argument('--max_per_scaffold', type=int, default=0,
                    help='If >0, cap molecules sharing a non-empty Bemis-Murcko scaffold')
    ap.add_argument('--allow_implicit_h', action='store_true',
                    help='Allow source molecules whose heavy atoms have implicit Hs. '
                        'Disabled by default because BondNet explicit-H training requires '
                        'explicit hydrogen coordinates.')
    ap.add_argument('--out', default='selected_subset.sdf')
    ap.add_argument('--report', default='selection_report.json')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--random', action='store_true',
                    help='Equal-size RANDOM baseline: keep identical preprocessing '
                        '(dedup, explicit-H filter, MDL aromaticity, scaffold cap) but '
                        'pick uniformly at random instead of by diversity + quotas. Use '
                        'to isolate the effect of the selection strategy at matched size.')
    args = ap.parse_args()

    random.seed(args.seed)
    pool, seen, stats = [], set(), Counter()
    source_tags = set()
    for s in args.source:
        if '=' not in s:
            ap.error(f'Invalid --source {s!r}; expected TAG=PATH.')
        tag, path = s.split('=', 1)
        if not tag or not path:
            ap.error(f'Invalid --source {s!r}; expected TAG=PATH.')
        if tag in source_tags:
            ap.error(f'Duplicate source tag: {tag!r}.')
        source_tags.add(tag)
        print(f'[read] {tag} <- {path}', file=sys.stderr)
        load_source(
            path, tag, args.max_read_per_source, args.stride, pool, seen, stats,
            require_explicit_h=not args.allow_implicit_h,
        )
    print(f'[pool] {len(pool)} unique molecules after dedup', file=sys.stderr)
    if not pool:
        print('empty pool', file=sys.stderr); sys.exit(1)

    # optional scaffold cap (diversify away from over-represented scaffolds)
    if args.max_per_scaffold > 0:
        # Avoid a source-file-order bias when choosing representatives of a
        # scaffold, while keeping this preprocessing deterministic.
        random.Random(args.seed).shuffle(pool)
        by_scaf = defaultdict(int); kept = []
        for m in pool:
            try:
                scaf = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(m['smi'])
            except Exception:
                scaf = ''
            # RDKit uses an empty scaffold for acyclic molecules.  Capping
            # that shared empty value would accidentally discard nearly all
            # acyclic chemistry, including many carbonyl and nitrile motifs.
            if not scaf or by_scaf[scaf] < args.max_per_scaffold:
                by_scaf[scaf] += 1; kept.append(m)
        stats['scaffold_dropped'] = len(pool) - len(kept)
        pool = kept

    N = len(pool)
    tag_to_idx = defaultdict(list)
    for i, m in enumerate(pool):
        for t in m['sig']:
            tag_to_idx[t].append(i)

    selected = set()
    selected_ordered = []
    if args.random:
        _order = list(range(N))
        random.Random(args.seed).shuffle(_order)
        selected_ordered = _order[:args.target]
        selected = set(selected_ordered)
        stats['random_mode'] = 1
        stats['after_quota'] = 0

    # Phase A: quota fill per hard tag (diversity within each tag)
    for tag, idxs in sorted(tag_to_idx.items(), key=lambda item: (len(item[1]), item[0])):
        if args.random:
            break
        need = min(args.quota, len(idxs))
        already = [k for k in idxs if k in selected]
        if len(already) >= need:
            continue
        sub_fps = [pool[k]['fp'] for k in idxs]
        # firstpicks = local indices already selected
        local_first = [idxs.index(k) for k in already]
        pick_local = maxmin_pick(sub_fps, need, args.seed, firstpicks=local_first)
        for pl in pick_local:
            selected.add(idxs[pl])

    if not args.random:
        if len(selected) > args.target:
            ap.error(
                f'Hard-tag quota selection needs {len(selected)} molecules, exceeding '
                f'--target={args.target}. Increase --target or lower --quota; refusing '
                'to silently discard a guaranteed tag quota.'
            )
        stats['after_quota'] = len(selected)
        quota_selected = sorted(selected)
        selected_ordered = list(quota_selected)

    # Phase B: global MaxMin to reach target
    if not args.random and len(selected) < args.target:
        all_fps = [m['fp'] for m in pool]
        first = quota_selected
        pick = maxmin_pick(all_fps, min(args.target, N), args.seed, firstpicks=first)
        for p in pick:
            if p not in selected:
                selected.add(p)
                selected_ordered.append(p)
            if len(selected_ordered) >= args.target:
                break

    print(f'[select] {len(selected_ordered)} molecules', file=sys.stderr)

    # Write SDF
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(args.out)
    n_written = 0
    for i in selected_ordered:
        m = Chem.MolFromMolBlock(pool[i]['block'], removeHs=False, sanitize=False)
        if m is None:
            w.close()
            raise RuntimeError(f'Selected molecule {i} cannot be read from its MolBlock.')
        try:
            _sanitize_with_mdl_aromaticity(m)
        except Exception as exc:
            w.close()
            raise RuntimeError(
                f'Selected molecule {i} cannot be sanitized with the training aromaticity model.'
            ) from exc
        m.SetProp('BONDNET_SOURCE', pool[i]['src'])
        m.SetProp('BONDNET_SOURCE_INDEX', str(pool[i]['src_index']))
        m.SetProp('BONDNET_CANONICAL_SMILES', pool[i]['smi'])
        m.SetProp('BONDNET_HARD_TAGS', ','.join(sorted(pool[i]['sig'])))
        w.write(m)
        n_written += 1
    w.close()
    if n_written != len(selected_ordered):
        raise RuntimeError(
            f'Wrote {n_written} molecules but selected {len(selected_ordered)}; output is incomplete.'
        )

    # Report
    rep = {'pool_unique': N, 'selected': n_written,
           'read_stats': dict(stats),
           'require_explicit_h': not args.allow_implicit_h,
           'by_source': dict(Counter(pool[i]['src'] for i in selected_ordered)),
           'hard_tag_available': {t: len(idxs) for t, idxs in tag_to_idx.items()},
           'hard_tag_support': {t: sum(1 for i in selected_ordered if t in pool[i]['sig'])
                                for t in tag_to_idx},
           'size_hist': dict(Counter(
               min(pool[i]['nheavy'] // 10 * 10, 60) for i in selected_ordered
           ))}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, 'w') as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))

if __name__ == '__main__':
    main()
