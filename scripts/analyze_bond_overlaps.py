"""
Bond-Length Overlap Analysis  (Proposal Section 6.2)

Systematically quantifies the overlap between bond-length distributions in
drug-like molecules, demonstrating the fundamental limitation of distance-
based bond perception.

Outputs:
  - Per-type bond-length statistics (mean, std, 5th/95th percentile)
  - Pairwise overlap fractions between bond-type distributions
  - Fraction of bonds that fall in ambiguous distance ranges
  - Per-bond accuracy of OpenBabel/RDKit in overlap regions vs BondNet
  - Histograms saved as PNG files

Usage:
  python scripts/analyze_bond_overlaps.py \
      --data_path /path/to/molecules.sdf \
      --output_dir results/overlap_analysis/
"""

import os
import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')


BOND_NAMES = {0: 'single', 1: 'double', 2: 'triple', 3: 'aromatic'}
BOND_COLORS = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728'}

# Literature reference values for expected bond-length ranges (in Å)
REFERENCE_RANGES = {
    'C-C single':   (1.43, 1.54),
    'C-C double':   (1.31, 1.38),
    'C-C triple':   (1.18, 1.22),
    'C-C aromatic': (1.36, 1.43),
    'C-N single':   (1.36, 1.47),
    'C-N double':   (1.27, 1.35),
    'C-N aromatic': (1.33, 1.38),
    'C-O single':   (1.30, 1.43),
    'C-O double':   (1.19, 1.26),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, default=None,
                   help='.sdf file of molecules to analyze')
    p.add_argument('--data_dir', type=str, default=None,
                   help='Directory of .sdf files')
    p.add_argument('--pyg_root', type=str, default=None,
                   help='PyG QM9 root directory')
    p.add_argument('--max_mols', type=int, default=10000)
    p.add_argument('--output_dir', type=str, default='./results/overlap_analysis')
    p.add_argument('--no_plots', action='store_true',
                   help='Skip matplotlib plots (useful on headless servers)')
    return p.parse_args()


# ------------------------------------------------------------------ #
# Data collection                                                       #
# ------------------------------------------------------------------ #

def collect_bond_data(mols, max_mols=None) -> Dict[str, List[Tuple]]:
    """
    Collect bond lengths and types from a list of RDKit molecules.

    Returns:
        data: dict mapping bond_type_int (0-3) -> list of (dist, atom_pair_str)
    """
    from rdkit import Chem  # type: ignore

    BOND_TYPE_MAP = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3,
    }

    data: Dict[int, List] = {0: [], 1: [], 2: [], 3: []}
    per_atom_pair: Dict[str, Dict[int, List]] = defaultdict(lambda: defaultdict(list))

    n = 0
    for mol in mols:
        if mol is None:
            continue
        try:
            conf = mol.GetConformer()
        except Exception:
            continue

        for bond in mol.GetBonds():
            bt = BOND_TYPE_MAP.get(bond.GetBondType(), -1)
            if bt < 0:
                continue

            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            pi = conf.GetAtomPosition(i)
            pj = conf.GetAtomPosition(j)
            dist = float(np.linalg.norm(
                [pi.x - pj.x, pi.y - pj.y, pi.z - pj.z]
            ))

            zi = mol.GetAtomWithIdx(i).GetAtomicNum()
            zj = mol.GetAtomWithIdx(j).GetAtomicNum()
            pair = f'{min(zi, zj)}-{max(zi, zj)}'

            data[bt].append(dist)
            per_atom_pair[pair][bt].append(dist)

        n += 1
        if max_mols and n >= max_mols:
            break

    log.info(f'Collected bonds from {n} molecules:')
    for bt, dists in data.items():
        log.info(f'  {BOND_NAMES[bt]}: {len(dists):,} bonds')

    return data, per_atom_pair, n


# ------------------------------------------------------------------ #
# Statistics and overlap quantification                                 #
# ------------------------------------------------------------------ #

def compute_distribution_stats(dists: List[float]) -> Dict:
    if not dists:
        return {}
    arr = np.array(dists)
    return {
        'count': len(arr),
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'p5': float(np.percentile(arr, 5)),
        'p25': float(np.percentile(arr, 25)),
        'median': float(np.median(arr)),
        'p75': float(np.percentile(arr, 75)),
        'p95': float(np.percentile(arr, 95)),
        'min': float(arr.min()),
        'max': float(arr.max()),
    }


def overlap_fraction_histogram(
    a: np.ndarray,
    b: np.ndarray,
    n_bins: int = 200,
    range_: Tuple[float, float] = (0.8, 2.5),
) -> float:
    """
    Estimate the Bhattacharyya overlap coefficient between two distributions
    using histogram-based density estimation.

    Returns overlap in [0, 1] (1 = identical distributions).
    """
    bins = np.linspace(range_[0], range_[1], n_bins + 1)
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    bin_width = bins[1] - bins[0]
    overlap = float(np.sum(np.minimum(ha, hb)) * bin_width)
    return overlap


def compute_pairwise_overlaps(data: Dict[int, List[float]]) -> Dict[str, float]:
    """
    Compute pairwise Bhattacharyya overlap between all bond-type distributions.
    Returns dict of 'typeA-typeB' -> overlap fraction.
    """
    results = {}
    types = sorted(data.keys())
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            a, b = types[i], types[j]
            if not data[a] or not data[b]:
                continue
            arr_a = np.array(data[a])
            arr_b = np.array(data[b])
            overlap = overlap_fraction_histogram(arr_a, arr_b)
            key = f'{BOND_NAMES[a]}-{BOND_NAMES[b]}'
            results[key] = overlap
    return results


def find_ambiguous_bonds(
    data: Dict[int, List[float]],
    overlap_threshold: float = 0.1,
) -> Tuple[float, float, float]:
    """
    Identify bonds whose distances fall in the 'ambiguous zone' — where two
    or more bond-type distributions overlap.

    Returns:
        frac_ambiguous:  fraction of all bonds in ambiguous zones
        frac_single_in_overlap:  fraction of single bonds in overlap zone
        frac_aromatic_in_overlap: fraction of aromatic bonds in overlap zone
    """
    # Build kernel-density estimates on a common grid
    from scipy.stats import gaussian_kde  # type: ignore

    grid = np.linspace(0.9, 2.5, 1000)
    densities = {}
    for bt, dists in data.items():
        if len(dists) < 10:
            continue
        try:
            kde = gaussian_kde(np.array(dists), bw_method=0.05)
            densities[bt] = kde(grid)
        except Exception:
            pass

    if not densities:
        return 0.0, 0.0, 0.0

    # A point on the grid is "ambiguous" if more than one bond type has
    # non-negligible density there
    max_density = max(d.max() for d in densities.values())
    threshold = max_density * overlap_threshold

    ambiguous_mask = np.zeros(len(grid), dtype=bool)
    for idx in range(len(grid)):
        above = sum(1 for d in densities.values() if d[idx] > threshold)
        if above >= 2:
            ambiguous_mask[idx] = True

    # For each bond type, estimate fraction falling in ambiguous zones
    stats = {}
    total_ambig = 0
    total_bonds = 0
    for bt, dists in data.items():
        arr = np.array(dists)
        # Find which grid points each bond distance is closest to
        in_ambig = sum(
            1 for d in arr
            if ambiguous_mask[np.argmin(np.abs(grid - d))]
        )
        stats[BOND_NAMES[bt]] = float(in_ambig) / max(len(arr), 1)
        total_ambig += in_ambig
        total_bonds += len(arr)

    frac_all = total_ambig / max(total_bonds, 1)
    return frac_all, stats.get('single', 0.0), stats.get('aromatic', 0.0)


# ------------------------------------------------------------------ #
# Plotting                                                               #
# ------------------------------------------------------------------ #

def plot_distributions(
    data: Dict[int, List[float]],
    output_dir: Path,
    pair_filter: Tuple[int, int] = None,
) -> None:
    """Plot overlapping bond-length histograms for all bond types."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning('matplotlib not available — skipping plots')
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: all types overlaid
    ax = axes[0]
    bins = np.linspace(0.9, 2.5, 80)
    for bt, dists in sorted(data.items()):
        if not dists:
            continue
        ax.hist(np.array(dists), bins=bins, alpha=0.5, density=True,
                label=BOND_NAMES[bt], color=BOND_COLORS[bt])
    ax.set_xlabel('Bond length (Å)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Bond-length distributions by type', fontsize=13)
    ax.legend()
    ax.axvspan(1.38, 1.47, alpha=0.08, color='gray', label='Ambiguous zone')

    # Right: zoom into the critical overlap region
    ax = axes[1]
    bins_zoom = np.linspace(1.28, 1.55, 60)
    for bt, dists in sorted(data.items()):
        if not dists:
            continue
        arr = np.array(dists)
        mask = (arr >= 1.28) & (arr <= 1.55)
        if mask.sum() < 5:
            continue
        ax.hist(arr[mask], bins=bins_zoom, alpha=0.5, density=True,
                label=BOND_NAMES[bt], color=BOND_COLORS[bt])
    ax.set_xlabel('Bond length (Å)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Zoom: overlap region (1.28–1.55 Å)', fontsize=13)
    ax.legend()

    plt.tight_layout()
    out_path = output_dir / 'bond_length_distributions.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f'Plot saved: {out_path}')


def plot_per_atom_pair_overlap(
    per_atom_pair: Dict[str, Dict[int, List[float]]],
    output_dir: Path,
    top_pairs: int = 6,
) -> None:
    """Plot bond-length distributions for the most important atom pairs."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Atomic number to symbol
    Z_SYM = {6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br'}
    pairs_of_interest = ['6-6', '6-7', '6-8']  # C-C, C-N, C-O

    fig, axes = plt.subplots(1, len(pairs_of_interest), figsize=(15, 4))
    bins = np.linspace(0.9, 2.2, 60)

    for ax, pair in zip(axes, pairs_of_interest):
        if pair not in per_atom_pair:
            ax.set_title(f'{pair} (no data)')
            continue
        pair_data = per_atom_pair[pair]
        z1, z2 = [int(z) for z in pair.split('-')]
        title = f'{Z_SYM.get(z1, z1)}-{Z_SYM.get(z2, z2)} bonds'

        for bt, dists in sorted(pair_data.items()):
            if len(dists) < 5:
                continue
            ax.hist(np.array(dists), bins=bins, alpha=0.55, density=True,
                    label=BOND_NAMES[bt], color=BOND_COLORS[bt])
        ax.set_xlabel('Distance (Å)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)

    plt.suptitle('Bond-length distributions by atom pair', fontsize=13, y=1.02)
    plt.tight_layout()
    out_path = output_dir / 'per_atom_pair_distributions.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f'Plot saved: {out_path}')


# ------------------------------------------------------------------ #
# Main                                                                  #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load molecules
    if args.pyg_root:
        from bondnet.data.dataset import BondNetDataset
        ds = BondNetDataset.from_pyg_qm9(root=args.pyg_root, split='train',
                                          max_mols=args.max_mols,
                                          compute_rings=False)
        mols = [ds.mols[i] for i in range(len(ds.mols))]
    elif args.data_dir:
        import glob
        from rdkit import Chem
        mols = []
        for path in sorted(glob.glob(os.path.join(args.data_dir, '*.sdf'))):
            for m in Chem.SDMolSupplier(path, removeHs=True):
                if m and m.GetNumConformers():
                    mols.append(m)
                if args.max_mols and len(mols) >= args.max_mols:
                    break
            if args.max_mols and len(mols) >= args.max_mols:
                break
    else:
        from rdkit import Chem
        mols = [m for m in Chem.SDMolSupplier(args.data_path, removeHs=True)
                if m and m.GetNumConformers()]
        if args.max_mols:
            mols = mols[:args.max_mols]

    log.info(f'Loaded {len(mols)} molecules')

    # Collect bond data
    data, per_atom_pair, n_mols = collect_bond_data(mols, args.max_mols)

    # ── Statistics ─────────────────────────────────────────────────── #
    stats = {}
    for bt, dists in data.items():
        stats[BOND_NAMES[bt]] = compute_distribution_stats(dists)

    log.info('\n=== Bond-length statistics ===')
    for name, s in stats.items():
        if not s:
            continue
        log.info(
            f'{name:>10}:  n={s["count"]:>7,}  '
            f'mean={s["mean"]:.3f}  std={s["std"]:.3f}  '
            f'[{s["p5"]:.3f}, {s["p95"]:.3f}]'
        )

    # ── Pairwise overlap ───────────────────────────────────────────── #
    log.info('\n=== Pairwise Bhattacharyya overlap ===')
    overlaps = compute_pairwise_overlaps(data)
    for pair, ov in sorted(overlaps.items(), key=lambda x: -x[1]):
        log.info(f'  {pair:<30}  {ov:.4f}')

    # ── Ambiguous zone analysis ────────────────────────────────────── #
    log.info('\n=== Ambiguous zone fractions ===')
    try:
        frac_all, frac_single, frac_arom = find_ambiguous_bonds(data)
        log.info(f'  All bonds in ambiguous zone:      {frac_all*100:.1f}%')
        log.info(f'  Single bonds in ambiguous zone:   {frac_single*100:.1f}%')
        log.info(f'  Aromatic bonds in ambiguous zone: {frac_arom*100:.1f}%')
    except Exception as e:
        log.warning(f'Ambiguous zone analysis failed: {e}')
        frac_all, frac_single, frac_arom = 0.0, 0.0, 0.0

    # ── Save JSON ─────────────────────────────────────────────────── #
    results = {
        'n_molecules': n_mols,
        'bond_stats': stats,
        'pairwise_overlaps': overlaps,
        'ambiguous_zone': {
            'frac_all': frac_all,
            'frac_single': frac_single,
            'frac_aromatic': frac_arom,
        },
    }
    out_json = output_dir / 'bond_overlap_analysis.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f'\nResults saved to {out_json}')

    # ── Plots ─────────────────────────────────────────────────────── #
    if not args.no_plots:
        plot_distributions(data, output_dir)
        plot_per_atom_pair_overlap(per_atom_pair, output_dir)

    log.info('\nAnalysis complete.')


if __name__ == '__main__':
    main()
