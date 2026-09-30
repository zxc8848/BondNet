"""
BondNetDataset: dataset classes for QM9 and GEOM-DRUGS.

Supports:
  - Loading molecules from SDF files or a directory of SDF files
  - PyTorch Geometric QM9 dataset (when torch_geometric is available)
  - Pre-computed Wiberg bond orders from NPZ files
  - Per-molecule ring feature pre-computation
  - Molecule-level diversity-weighted sampling
"""

import bisect
import os
import torch
import numpy as np
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler
from typing import Optional, List, Dict, Callable, Sequence

from .featurizer import MoleculeFeaturizer
from .noise_augment import GaussianNoiseAugment
from ..utils.ring_detection import detect_rings
from ..utils.chemistry import get_expected_valence


CACHE_FORMAT_VERSION = 14
STAGE3_EXCLUDES_H = False

DEFAULT_VAL_FRAC = 0.1
DEFAULT_TEST_FRAC = 0.1
DEFAULT_SPLIT_SEED = 42


def _canonicalize_edge_diff(sample: Dict) -> Dict:
    """Return a sample with receiver-minus-sender vectors from its coordinates.

    Older feature caches stored reversed heavy-heavy vectors while H-X vectors
    were correct. Recompute at the cache read boundary so both old and new
    caches obey the same convention without rewriting the original cache.
    """
    edge_index = sample['edge_index']
    coord = sample['coord']
    corrected = dict(sample)
    corrected['edge_diff'] = coord[edge_index[:, 1]] - coord[edge_index[:, 0]]
    return corrected


def _stable_unit_hash(value, seed: int = DEFAULT_SPLIT_SEED) -> float:
    """Map a value to a stable pseudo-random unit interval number in [0, 1)."""
    text = f'{value}|{int(seed)}'.encode('utf-8', errors='ignore')
    # FNV-1a 32-bit hash (deterministic across platforms / processes)
    h = 2166136261
    for b in text:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h / float(1 << 32)


def _assign_split_label(
    sample: Dict,
    default_idx: int,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
    seed: int = DEFAULT_SPLIT_SEED,
) -> str:
    """
    Assign a stable split label for a sample.

    If `geom_mol_idx` exists, split by that key to avoid leaking conformers from
    the same molecule across train/val/test. Otherwise fall back to sample index.
    """
    key = sample.get('geom_mol_idx', default_idx)
    u = _stable_unit_hash(key, seed=seed)
    t = max(0.0, min(1.0, float(test_frac)))
    v = max(0.0, min(1.0 - t, float(val_frac)))
    if u < t:
        return 'test'
    if u < t + v:
        return 'val'
    return 'train'


def _sanitize_with_mdl_aromaticity(mol):
    """
    Normalize RDKit chemistry perception using the MDL aromaticity model.

    The second sanitize refreshes dependent properties such as hybridization and
    conjugation after aromaticity has been reassigned.
    """
    from rdkit import Chem  # type: ignore

    Chem.SanitizeMol(mol)
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
    Chem.SanitizeMol(mol)
    return mol


def _load_wiberg_bo_dict(wiberg_npz: Optional[str]) -> Dict[int, np.ndarray]:
    """
    Load xTB / Wiberg BO labels from NPZ.

    Newer files store sparse successful results as (mol_ids, bo_matrices), while
    older caches may only contain bo_matrices and implicitly assume dense order.
    """
    if not wiberg_npz or not os.path.exists(wiberg_npz):
        return {}

    data = np.load(wiberg_npz, allow_pickle=True)
    bo_matrices = data['bo_matrices']
    mol_ids = data['mol_ids'] if 'mol_ids' in data.files else None

    if mol_ids is None:
        return {i: mat for i, mat in enumerate(bo_matrices)}

    if len(mol_ids) != len(bo_matrices):
        raise ValueError(
            f'Invalid Wiberg NPZ: mol_ids has length {len(mol_ids)} but '
            f'bo_matrices has length {len(bo_matrices)}'
        )

    return {
        int(mol_idx): mat
        for mol_idx, mat in zip(mol_ids.tolist(), bo_matrices)
    }


def _validate_cache_payload(data: Dict, cache_path: str) -> None:
    metadata = data.get('metadata')
    if metadata is None:
        raise ValueError(
            f'Cache file is stale or missing metadata: {cache_path}\n'
            'Rebuild it with: python scripts/precompute_features.py '
            '--data_path <sdf> --output ' + cache_path
        )

    version = metadata.get('cache_format_version')
    if version != CACHE_FORMAT_VERSION:
        raise ValueError(
            f'Cache version mismatch for {cache_path}: found {version}, '
            f'expected {CACHE_FORMAT_VERSION}. Rebuild the cache so H-X edges '
            'participate only in connectivity and not in bond-type supervision.'
        )
        import warnings
        warnings.warn(
            f'Cache version mismatch for {cache_path}: found {version}, '
            f'expected {CACHE_FORMAT_VERSION}. Loading anyway — rebuild if issues occur.',
            UserWarning, stacklevel=3,
        )

    if metadata.get('stage3_excludes_h') is not STAGE3_EXCLUDES_H:
        raise ValueError(
            f'Cache semantic mismatch for {cache_path}: '
            f"stage3_excludes_h={metadata.get('stage3_excludes_h')} but "
            f'expected {STAGE3_EXCLUDES_H}.\n'
            'Rebuild it with: python scripts/precompute_features.py '
            '--data_path <sdf> --output ' + cache_path
        )


class BondNetDataset(Dataset):
    """
    Dataset for BondNet training and evaluation.

    Each item is a feature dict produced by MoleculeFeaturizer, optionally
    augmented with Gaussian coordinate noise and ring features.

    Args:
        mols:            List of RDKit molecules (each must have a 3D conformer).
        wiberg_bo_dict:  Optional dict {mol_idx: (N,N) BO matrix}.
        featurizer:      MoleculeFeaturizer instance (defaults to cutoff=5.0 Å).
        noise_augment:   GaussianNoiseAugment instance.  None = no noise.
        compute_rings:   If True, pre-compute ring features for Stage 2.
        max_ring_size:   Maximum ring size for ring detection.
        transform:       Optional additional transform applied to each sample.
    """

    def __init__(
        self,
        mols: List,
        wiberg_bo_dict: Optional[Dict[int, np.ndarray]] = None,
        featurizer: Optional[MoleculeFeaturizer] = None,
        noise_augment: Optional[GaussianNoiseAugment] = None,
        compute_rings: bool = False,
        max_ring_size: int = 8,
        transform: Optional[Callable] = None,
    ):
        self.mols = mols
        self.wiberg_bo_dict = wiberg_bo_dict or {}
        self.featurizer = featurizer or MoleculeFeaturizer()
        self.noise_augment = noise_augment
        self.compute_rings = compute_rings
        self.max_ring_size = max_ring_size
        self.transform = transform

        # Pre-compute diversity weights for weighted sampling (Section 4.4)
        self._diversity_weights = self._compute_diversity_weights()

    # ------------------------------------------------------------------ #
    # Dataset interface                                                     #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.mols)

    def __getitem__(self, idx: int) -> Dict:
        mol = self.mols[idx]
        bo_matrix = self.wiberg_bo_dict.get(idx, None)

        try:
            sample = self.featurizer.featurize(mol, wiberg_bo_matrix=bo_matrix)
        except Exception as e:
            # Skip problematic molecules by returning an empty-ish sample
            # (the collate_fn will filter these)
            return {'_invalid': True, '_error': str(e)}

        # Optional ring feature computation
        if self.compute_rings and sample['edge_index_bond'].shape[0] > 0:
            sample = self._add_ring_features(sample)

        # Noise augmentation
        if self.noise_augment is not None:
            sample = self.noise_augment(sample)

        # Additional transforms
        if self.transform is not None:
            sample = self.transform(sample)

        sample['mol_idx'] = idx
        if mol.HasProp('geom_mol_idx'):
            try:
                sample['geom_mol_idx'] = int(mol.GetProp('geom_mol_idx'))
            except Exception:
                pass
        if mol.HasProp('geom_conf_idx'):
            try:
                sample['geom_conf_idx'] = int(mol.GetProp('geom_conf_idx'))
            except Exception:
                sample['geom_conf_idx'] = mol.GetProp('geom_conf_idx')
        if mol.HasProp('geom_conf_rank'):
            try:
                sample['geom_conf_rank'] = int(mol.GetProp('geom_conf_rank'))
            except Exception:
                pass
        return sample

    # ------------------------------------------------------------------ #
    # Ring feature computation                                              #
    # ------------------------------------------------------------------ #

    def _add_ring_features(self, sample: Dict) -> Dict:
        """
        Detect rings in the ground-truth connectivity graph and store ring metadata.

        ring_edge_indices are stored as indices into the FULL edge_index tensor
        (not the bonded-edge subset), so that collate_fn can offset them with
        edge_offset and bondnet.py can index directly into backbone edge_feat.
        """
        edge_index_bond = sample['edge_index_bond']  # (E_b, 2)
        bond_mask = sample['bond_mask']              # (E,) bool
        num_atoms = int(sample['num_atoms'])

        # detect_rings returns indices into edge_index_bond (bonded-edge space)
        rings, ring_edge_indices_bond = detect_rings(
            edge_index_bond.tolist(), num_atoms, self.max_ring_size
        )

        if not rings:
            sample['ring_atoms'] = []
            sample['ring_edge_indices'] = []
            sample['ring_is_aromatic'] = torch.zeros(0, dtype=torch.float32)
            sample['ring_features'] = None
            return sample

        # Map ring edge indices: bonded-edge space → full-edge space
        # bonded_positions[k] = position of the k-th bonded edge in full edge_index
        bonded_positions = torch.where(bond_mask)[0]
        n_bonded = len(bonded_positions)
        ring_edge_indices_full = [
            [int(bonded_positions[e]) for e in r_edges if e < n_bonded]
            for r_edges in ring_edge_indices_bond
        ]

        # Ground-truth aromaticity: a ring is aromatic if any bond in it is aromatic
        bond_is_aromatic = sample['bond_is_aromatic']  # (E_b,) indexed in bonded space
        ring_is_aromatic = []
        for r_edges_bond in ring_edge_indices_bond:
            is_arom = any(
                bool(bond_is_aromatic[e])
                for e in r_edges_bond if e < len(bond_is_aromatic)
            )
            ring_is_aromatic.append(float(is_arom))

        sample['ring_atoms'] = rings
        sample['ring_edge_indices'] = ring_edge_indices_full  # full-edge space
        sample['ring_is_aromatic'] = torch.tensor(ring_is_aromatic, dtype=torch.float32)
        # ring_features will be recomputed at training time using real edge features;
        # storing the geometric part only
        sample['ring_features'] = None
        return sample

    # ------------------------------------------------------------------ #
    # Diversity-weighted sampling                                           #
    # ------------------------------------------------------------------ #

    def _compute_diversity_weights(self) -> torch.Tensor:
        """
        Weight each molecule by the diversity of its bond types.
        Molecules with rare bond types (double, triple) are upweighted.
        Returns a (N,) weight tensor.
        """
        weights = np.ones(len(self.mols), dtype=np.float32)
        try:
            from rdkit import Chem  # type: ignore
            from rdkit.Chem import BondType  # type: ignore
            for i, mol in enumerate(self.mols):
                if mol is None:
                    continue
                has_double = any(
                    b.GetBondType() == BondType.DOUBLE for b in mol.GetBonds()
                )
                has_triple = any(
                    b.GetBondType() == BondType.TRIPLE for b in mol.GetBonds()
                )
                if has_triple:
                    weights[i] = 4.0
                elif has_double:
                    weights[i] = 2.0
        except Exception:
            pass
        return torch.tensor(weights, dtype=torch.float32)

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """Return a WeightedRandomSampler for rare-bond-type upweighting."""
        return WeightedRandomSampler(
            weights=self._diversity_weights,
            num_samples=len(self),
            replacement=True,
        )

    # ------------------------------------------------------------------ #
    # Factory methods                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_sdf(
        cls,
        sdf_path: str,
        wiberg_npz: Optional[str] = None,
        max_mols: Optional[int] = None,
        **kwargs,
    ) -> 'BondNetDataset':
        """
        Load molecules from a single SDF file.

        Args:
            sdf_path:    Path to .sdf file.
            wiberg_npz:  Optional path to .npz with key 'bo_matrices'. When
                         present, labels are aligned by mol_ids if available.
            max_mols:    Maximum number of molecules to load.
        """
        from rdkit import Chem  # type: ignore
        # Load with sanitize=False first, then sanitize individually to skip
        # problematic molecules (e.g. QM9's charged species with unusual valence)
        supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        mols = []
        for m in supplier:
            if m is None:
                continue
            try:
                _sanitize_with_mdl_aromaticity(m)
                if m.GetNumConformers() > 0:
                    mols.append(m)
            except Exception:
                pass
            if max_mols and len(mols) >= max_mols:
                break

        bo_dict = _load_wiberg_bo_dict(wiberg_npz)

        return cls(mols, wiberg_bo_dict=bo_dict, **kwargs)

    @classmethod
    def from_sdf_dir(
        cls,
        sdf_dir: str,
        pattern: str = '*.sdf',
        wiberg_npz: Optional[str] = None,
        max_mols: Optional[int] = None,
        **kwargs,
    ) -> 'BondNetDataset':
        """Load all SDF files from a directory."""
        import glob
        from rdkit import Chem  # type: ignore

        sdf_files = sorted(glob.glob(os.path.join(sdf_dir, pattern)))
        mols = []
        for path in sdf_files:
            supplier = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
            for m in supplier:
                if m is None:
                    continue
                try:
                    _sanitize_with_mdl_aromaticity(m)
                    if m.GetNumConformers() > 0:
                        mols.append(m)
                except Exception:
                    continue
                if max_mols and len(mols) >= max_mols:
                    break
            if max_mols and len(mols) >= max_mols:
                break

        bo_dict = _load_wiberg_bo_dict(wiberg_npz)

        return cls(mols, wiberg_bo_dict=bo_dict, **kwargs)

    @classmethod
    def from_pyg_qm9(
        cls,
        root: str,
        split: str = 'train',
        noise_sigma: float = 0.0,
        **kwargs,
    ) -> 'BondNetDataset':
        """
        Load QM9 dataset from PyTorch Geometric.

        Args:
            root:   Root directory for PyG QM9.
            split:  'train', 'val', or 'test'.
        """
        try:
            from torch_geometric.datasets import QM9  # type: ignore
        except ImportError:
            raise ImportError(
                "torch_geometric not found. Install with: pip install torch-geometric"
            )
        from rdkit import Chem  # type: ignore

        dataset = QM9(root=root)
        n = len(dataset)
        # Standard 80/10/10 split
        idx = torch.arange(n)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        if split == 'train':
            idx = idx[:n_train]
        elif split == 'val':
            idx = idx[n_train:n_train + n_val]
        elif split == 'test':
            idx = idx[n_train + n_val:]
        # split=None → use all molecules (for cache building)

        mols = []
        for i in idx.tolist():
            data = dataset[i]
            # Convert PyG Data to RDKit mol
            mol = _pyg_qm9_to_rdkit(data)
            if mol is not None:
                mols.append(mol)

        noise = GaussianNoiseAugment(sigma=noise_sigma) if noise_sigma > 0 else None
        return cls(mols, noise_augment=noise, **kwargs)


# ------------------------------------------------------------------ #
# Cached dataset (pre-computed features)                               #
# ------------------------------------------------------------------ #

class CachedBondNetDataset(Dataset):
    """
    Drop-in replacement for BondNetDataset that loads pre-computed features from disk.

    ``__getitem__`` is O(1) — it just indexes a list and optionally applies
    Gaussian noise augmentation.  This eliminates RDKit featurization from the
    training hot-path and fully saturates the GPU data pipeline.

    Build the cache with::

        python scripts/precompute_features.py \\
            --data_path /path/to/gdb9.sdf \\
            --output data/qm9_cache.pt

    Then train with::

        python train.py --cache_path data/qm9_cache.pt ...
    """

    def __init__(
        self,
        cache_path: str,
        noise_augment: Optional['GaussianNoiseAugment'] = None,
    ):
        data = torch.load(cache_path, map_location='cpu', weights_only=False)
        _validate_cache_payload(data, cache_path)
        self.samples: List[Dict] = data['samples']
        self._diversity_weights: torch.Tensor = data['diversity_weights']
        self.metadata: Dict = data['metadata']
        self.noise_augment = noise_augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = _canonicalize_edge_diff(self.samples[idx])
        if self.noise_augment is not None:
            s = self.noise_augment(s)
        s['mol_idx'] = idx
        return s

    @classmethod
    def split(cls, cache_path: str, val_frac: float = 0.1,
              noise_augment: Optional['GaussianNoiseAugment'] = None,
              seed: int = 42, split_key: Optional[str] = None,
              train_groups: Optional[int] = None,
              respect_fixed_split: bool = True):
        """
        Return (train_ds, val_ds) split from a single cache file.
        Train split gets noise_augment; val split does not.
        """
        import random
        data = torch.load(cache_path, map_location='cpu', weights_only=False)
        _validate_cache_payload(data, cache_path)
        samples = data['samples']
        weights = data['diversity_weights']
        metadata = data['metadata']

        n = len(samples)
        rng = random.Random(seed)
        fixed_labels = [s.get('split') for s in samples]
        has_fixed = any(lbl is not None for lbl in fixed_labels)

        # Current caches carry stable molecule-level train/val/test labels.  These
        # labels must take precedence over the legacy split_key path; otherwise a
        # second 90/10 shuffle silently mixes the nominal test groups into model
        # training.  Set respect_fixed_split=False only to reproduce checkpoints
        # trained before this leakage fix.
        if respect_fixed_split and has_fixed and train_groups is None:
            train_idx = sorted(i for i, lbl in enumerate(fixed_labels) if lbl == 'train')
            val_idx = sorted(i for i, lbl in enumerate(fixed_labels) if lbl == 'val')
            if not val_idx:
                raise ValueError(
                    'Fixed split labels found but no validation-labelled samples are present. '
                    'Refusing to use the test split for model selection.'
                )
            if not train_idx or not val_idx:
                raise ValueError(
                    'Fixed split labels found in cache but train/val indices are invalid. '
                    'Rebuild cache or explicitly request the legacy split.'
                )
        elif split_key:
            groups = sorted({
                s.get(split_key)
                for s in samples
                if s.get(split_key) is not None
            })
            if not groups:
                raise ValueError(f'No samples contain split_key={split_key!r}')
            if train_groups is not None:
                train_group_set = set(groups[:train_groups])
                val_group_set = set(groups[train_groups:])
            else:
                shuffled_groups = list(groups)
                rng.shuffle(shuffled_groups)
                n_val_groups = max(1, int(len(shuffled_groups) * val_frac))
                val_group_set = set(shuffled_groups[:n_val_groups])
                train_group_set = set(shuffled_groups[n_val_groups:])
            train_idx = sorted(
                i for i, s in enumerate(samples)
                if s.get(split_key) in train_group_set
            )
            val_idx = sorted(
                i for i, s in enumerate(samples)
                if s.get(split_key) in val_group_set
            )
            if not train_idx or not val_idx:
                raise ValueError(
                    f'Invalid group split: train={len(train_idx)} val={len(val_idx)} '
                    f'groups={len(groups)} train_groups={train_groups}'
                )
        else:
            if respect_fixed_split and has_fixed:
                train_idx = sorted(i for i, lbl in enumerate(fixed_labels) if lbl == 'train')
                val_idx = sorted(i for i, lbl in enumerate(fixed_labels) if lbl == 'val')
                if not val_idx:
                    raise ValueError(
                        'Fixed split labels found but no validation-labelled samples are present. '
                        'Refusing to use the test split for model selection.'
                    )
                if not train_idx or not val_idx:
                    raise ValueError(
                        'Fixed split labels found in cache but train/val indices are invalid. '
                        'Rebuild cache or remove split labels.'
                    )
            else:
                indices = list(range(n))
                rng.shuffle(indices)
                n_val = max(1, int(n * val_frac))
                val_idx = sorted(indices[:n_val])
                train_idx = sorted(indices[n_val:])

        train_ds = cls.__new__(cls)
        train_ds.samples = [samples[i] for i in train_idx]
        train_ds._diversity_weights = weights[train_idx]
        train_ds.metadata = metadata
        train_ds.metadata = dict(metadata)
        train_ds.metadata['split_indices'] = train_idx
        train_ds.noise_augment = noise_augment

        val_ds = cls.__new__(cls)
        val_ds.samples = [samples[i] for i in val_idx]
        val_ds._diversity_weights = weights[val_idx]
        val_ds.metadata = dict(metadata)
        val_ds.metadata['split_indices'] = val_idx
        val_ds.noise_augment = None

        return train_ds, val_ds


class ShardedCachedBondNetDataset(Dataset):
    """
    Cached dataset backed by many shard files.

    This keeps cache construction and training memory bounded for large SDF
    collections such as PubChem3D.  Each worker lazily loads shard files on
    demand and keeps a small LRU cache in memory.
    """

    MANIFEST_NAME = 'manifest.pt'

    def __init__(
        self,
        cache_dir: str,
        indices: Optional[Sequence[int]] = None,
        noise_augment: Optional['GaussianNoiseAugment'] = None,
        max_open_shards: int = 2,
    ):
        if os.path.basename(cache_dir) == self.MANIFEST_NAME:
            cache_dir = os.path.dirname(cache_dir)
        self.cache_dir = os.path.normpath(cache_dir)
        manifest = self._load_manifest(self.cache_dir)
        metadata = manifest.get('metadata', {})
        _validate_cache_payload({'metadata': metadata}, self.cache_dir)

        self.shards: List[Dict] = manifest['shards']
        self.starts: List[int] = [int(s['start']) for s in self.shards]
        self.n_samples = int(metadata.get('n_molecules', manifest.get('n_samples', 0)))
        self.indices = list(indices) if indices is not None else None
        self.noise_augment = noise_augment
        self.metadata = metadata
        self._diversity_weights = manifest.get(
            'diversity_weights',
            torch.ones(self.n_samples, dtype=torch.float32),
        )
        if self.indices is not None:
            self._diversity_weights = self._diversity_weights[self.indices]
        self._sample_meta = manifest.get('sample_meta')
        self._shard_cache: Dict[int, List[Dict]] = {}
        self._shard_lru: List[int] = []
        self.max_open_shards = max(1, int(max_open_shards))

    @staticmethod
    def manifest_path(cache_dir: str) -> str:
        if cache_dir.endswith('.pt') and os.path.isfile(cache_dir):
            return cache_dir
        return os.path.join(cache_dir, ShardedCachedBondNetDataset.MANIFEST_NAME)

    @classmethod
    def is_sharded_cache(cls, cache_path: str) -> bool:
        if os.path.isdir(cache_path):
            return os.path.exists(cls.manifest_path(cache_path))
        if os.path.basename(cache_path) == cls.MANIFEST_NAME and os.path.exists(cache_path):
            return True
        return False

    @classmethod
    def _load_manifest(cls, cache_dir: str) -> Dict:
        path = cls.manifest_path(cache_dir)
        manifest = torch.load(path, map_location='cpu', weights_only=False)
        if 'metadata' not in manifest or 'shards' not in manifest:
            raise ValueError(f'Invalid sharded cache manifest: {path}')
        return manifest

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else self.n_samples

    def _load_shard(self, shard_id: int) -> List[Dict]:
        if shard_id in self._shard_cache:
            if shard_id in self._shard_lru:
                self._shard_lru.remove(shard_id)
            self._shard_lru.append(shard_id)
            return self._shard_cache[shard_id]

        rel_path = self.shards[shard_id]['path']
        shard_path = rel_path if os.path.isabs(rel_path) else os.path.join(self.cache_dir, rel_path)
        payload = torch.load(shard_path, map_location='cpu', weights_only=False)
        samples = payload['samples']
        self._shard_cache[shard_id] = samples
        self._shard_lru.append(shard_id)

        while len(self._shard_lru) > self.max_open_shards:
            old = self._shard_lru.pop(0)
            self._shard_cache.pop(old, None)
        return samples

    def __getitem__(self, idx: int) -> Dict:
        global_idx = self.indices[idx] if self.indices is not None else idx
        shard_id = bisect.bisect_right(self.starts, global_idx) - 1
        if shard_id < 0:
            raise IndexError(idx)
        local_idx = global_idx - int(self.shards[shard_id]['start'])
        samples = self._load_shard(shard_id)
        s = _canonicalize_edge_diff(samples[local_idx])
        if self.noise_augment is not None:
            s = self.noise_augment(s)
        s['mol_idx'] = global_idx
        return s

    @classmethod
    def split(cls, cache_dir: str, val_frac: float = 0.1,
              noise_augment: Optional['GaussianNoiseAugment'] = None,
              seed: int = 42, split_key: Optional[str] = None,
              train_groups: Optional[int] = None,
              max_samples: Optional[int] = None,
              max_open_shards: int = 2,
              respect_fixed_split: bool = True):
        import random

        manifest = cls._load_manifest(cache_dir)
        metadata = manifest['metadata']
        n = int(metadata.get('n_molecules', manifest.get('n_samples', 0)))
        if max_samples is not None:
            n = min(n, int(max_samples))
        rng = random.Random(seed)

        sample_meta = manifest.get('sample_meta')
        fixed_labels = None
        if sample_meta is not None:
            fixed_labels = [meta.get('split') for meta in sample_meta[:n]]
            if not any(lbl is not None for lbl in fixed_labels):
                fixed_labels = None

        if respect_fixed_split and fixed_labels is not None and train_groups is None:
            train_idx = sorted(i for i, lbl in enumerate(fixed_labels) if lbl == 'train')
            val_idx = sorted(i for i, lbl in enumerate(fixed_labels) if lbl == 'val')
            if not val_idx:
                raise ValueError(
                    'Fixed split labels found but no validation-labelled samples are present. '
                    'Refusing to use the test split for model selection.'
                )
            if not train_idx:
                raise ValueError(
                    'Fixed split labels found but no training-labelled samples are present.'
                )
        elif split_key:
            sample_meta = manifest.get('sample_meta')
            if sample_meta is None:
                raise ValueError(
                    f'Sharded cache has no sample_meta; cannot split by {split_key!r}. '
                    'Rebuild the sharded cache with the current code.'
                )
            groups = sorted({
                meta.get(split_key)
                for meta in sample_meta[:n]
                if meta.get(split_key) is not None
            })
            if not groups:
                raise ValueError(f'No samples contain split_key={split_key!r}')
            if train_groups is not None:
                train_group_set = set(groups[:train_groups])
                val_group_set = set(groups[train_groups:])
            else:
                shuffled_groups = list(groups)
                rng.shuffle(shuffled_groups)
                n_val_groups = max(1, int(len(shuffled_groups) * val_frac))
                val_group_set = set(shuffled_groups[:n_val_groups])
                train_group_set = set(shuffled_groups[n_val_groups:])
            train_idx = [
                i for i, meta in enumerate(sample_meta[:n])
                if meta.get(split_key) in train_group_set
            ]
            val_idx = [
                i for i, meta in enumerate(sample_meta[:n])
                if meta.get(split_key) in val_group_set
            ]
        else:
            if respect_fixed_split and fixed_labels is not None:
                train_idx = sorted(i for i, lbl in enumerate(fixed_labels) if lbl == 'train')
                val_idx = sorted(i for i, lbl in enumerate(fixed_labels) if lbl == 'val')
                if not val_idx:
                    raise ValueError(
                        'Fixed split labels found but no validation-labelled samples are present. '
                        'Refusing to use the test split for model selection.'
                    )
                if not train_idx or not val_idx:
                    raise ValueError(
                        'Fixed split labels found in sharded cache but train/val indices are invalid. '
                        'Rebuild cache or remove split labels.'
                    )
            else:
                indices = list(range(n))
                rng.shuffle(indices)
                n_val = max(1, int(n * val_frac))
                val_idx = sorted(indices[:n_val])
                train_idx = sorted(indices[n_val:])

        if not train_idx or not val_idx:
            raise ValueError(f'Invalid split: train={len(train_idx)} val={len(val_idx)}')

        train_ds = cls(
            cache_dir,
            indices=train_idx,
            noise_augment=noise_augment,
            max_open_shards=max_open_shards,
        )
        val_ds = cls(
            cache_dir,
            indices=val_idx,
            noise_augment=None,
            max_open_shards=max_open_shards,
        )
        train_ds.metadata = dict(train_ds.metadata)
        train_ds.metadata['split_indices'] = train_idx
        val_ds.metadata = dict(val_ds.metadata)
        val_ds.metadata['split_indices'] = val_idx
        return train_ds, val_ds


class ShardedBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that keeps batches shard-local.

    Plain DataLoader(shuffle=True) randomly interleaves samples from every
    shard, causing repeated torch.load() calls for large cache files.  This
    sampler shuffles shard order and sample order within each shard while
    yielding batches that mostly come from one shard.
    """

    def __init__(
        self,
        dataset: ShardedCachedBondNetDataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self._groups = self._build_groups()

    def _build_groups(self) -> Dict[int, List[int]]:
        groups: Dict[int, List[int]] = {}
        for local_idx in range(len(self.dataset)):
            global_idx = (
                self.dataset.indices[local_idx]
                if self.dataset.indices is not None else local_idx
            )
            shard_id = bisect.bisect_right(self.dataset.starts, global_idx) - 1
            groups.setdefault(shard_id, []).append(local_idx)
        return groups

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        import random

        rng = random.Random(self.seed + self.epoch)
        shard_ids = list(self._groups)
        if self.shuffle:
            rng.shuffle(shard_ids)

        for shard_id in shard_ids:
            local_indices = list(self._groups[shard_id])
            if self.shuffle:
                rng.shuffle(local_indices)
            for start in range(0, len(local_indices), self.batch_size):
                batch = local_indices[start:start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        total = 0
        for local_indices in self._groups.values():
            n = len(local_indices)
            total += n // self.batch_size
            if n % self.batch_size and not self.drop_last:
                total += 1
        return total


# ------------------------------------------------------------------ #
# Cache building                                                        #
# ------------------------------------------------------------------ #

def build_cache(
    source: 'BondNetDataset',
    output_path: str,
    num_workers: int = 0,
    show_progress: bool = True,
    split_seed: int = DEFAULT_SPLIT_SEED,
    split_val_frac: float = DEFAULT_VAL_FRAC,
    split_test_frac: float = DEFAULT_TEST_FRAC,
) -> None:
    """
    Featurize all molecules in `source` and save to a `.pt` cache file
    that `CachedBondNetDataset` can load.

    This is called automatically by train.py when `--cache_path` points
    to a non-existent file.  After building, all subsequent epochs read
    from the cache — eliminating RDKit featurization from the hot path.

    Args:
        source:       A `BondNetDataset` (without noise augmentation).
        output_path:  Destination `.pt` file.
        num_workers:  Parallel workers (0 = single-process, safe on Windows).
        show_progress: Print a progress bar.
    """
    import os, time
    from pathlib import Path

    log_fn = print  # simple print; logging not set up at import time

    n = len(source)
    log_fn(f'Building feature cache for {n} molecules → {output_path}')
    t0 = time.time()

    if num_workers > 0:
        from torch.utils.data import DataLoader
        # Use a DataLoader just for parallel featurization
        loader = DataLoader(
            source, batch_size=1, shuffle=False,
            num_workers=num_workers, collate_fn=lambda x: x[0],
        )
        samples = []
        for i, s in enumerate(loader):
            if s.get('_invalid'):
                continue
            s = dict(s)
            s['split'] = _assign_split_label(
                s,
                default_idx=len(samples),
                val_frac=split_val_frac,
                test_frac=split_test_frac,
                seed=split_seed,
            )
            samples.append(s)
            if show_progress and (i + 1) % 5000 == 0:
                elapsed = time.time() - t0
                log_fn(f'  {i+1}/{n}  ({elapsed:.0f}s)')
    else:
        samples = []
        for i in range(n):
            s = source[i]
            if s.get('_invalid'):
                continue
            s = dict(s)
            s['split'] = _assign_split_label(
                s,
                default_idx=len(samples),
                val_frac=split_val_frac,
                test_frac=split_test_frac,
                seed=split_seed,
            )
            samples.append(s)
            if show_progress and (i + 1) % 5000 == 0:
                elapsed = time.time() - t0
                log_fn(f'  {i+1}/{n}  ({elapsed:.0f}s)')

    # Compute diversity weights from the collected samples
    weights = torch.ones(len(samples), dtype=torch.float32)
    for k, s in enumerate(samples):
        bt = s.get('bond_type_full')
        if bt is not None:
            has_triple = (bt == 2).any().item()
            has_double = (bt == 1).any().item()
            if has_triple:
                weights[k] = 4.0
            elif has_double:
                weights[k] = 2.0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'samples': samples,
        'diversity_weights': weights,
        'metadata': {
            'cache_format_version': CACHE_FORMAT_VERSION,
            'n_molecules': len(samples),
            'source': str(output_path),
            'stage3_excludes_h': STAGE3_EXCLUDES_H,
            'explicit_h': getattr(source.featurizer, 'explicit_h', False),
            'h_candidate_policy': 'explicit_h_learned_edges',
            'fixed_split': {
                'method': 'stable_hash',
                'seed': int(split_seed),
                'val_frac': float(split_val_frac),
                'test_frac': float(split_test_frac),
            },
        },
    }
    torch.save(payload, output_path)
    elapsed = time.time() - t0
    log_fn(f'Cache saved: {len(samples)} molecules in {elapsed:.1f}s  → {output_path}')


def _sample_diversity_weight(sample: Dict) -> float:
    bt = sample.get('bond_type_full')
    if bt is None:
        return 1.0
    if (bt == 2).any().item():
        return 4.0
    if (bt == 1).any().item():
        return 2.0
    return 1.0


def _sample_metadata(sample: Dict) -> Dict:
    meta = {}
    for key in ('mol_idx', 'geom_mol_idx', 'geom_conf_idx', 'geom_conf_rank'):
        if key in sample:
            value = sample[key]
            if torch.is_tensor(value):
                value = value.item() if value.numel() == 1 else value.tolist()
            meta[key] = value
    if 'split' in sample:
        meta['split'] = sample['split']
    return meta


def build_sharded_cache_from_dataset(
    source: 'BondNetDataset',
    output_dir: str,
    shard_size: int = 50000,
    num_workers: int = 0,
    show_progress: bool = True,
) -> None:
    """
    Build a directory-backed feature cache from an existing Dataset.

    This avoids keeping all featurized samples in memory before saving, but the
    source dataset may still hold all RDKit molecules. For very large SDF files
    prefer build_sharded_cache_from_sdf(), which streams molecules from disk.
    """
    import time
    from pathlib import Path
    from torch.utils.data import DataLoader

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_fn = print
    n = len(source)
    log_fn(f'Building sharded feature cache for {n} molecules -> {output}')
    t0 = time.time()

    if num_workers > 0:
        iterator = enumerate(DataLoader(
            source, batch_size=1, shuffle=False,
            num_workers=num_workers, collate_fn=lambda x: x[0],
        ))
    else:
        iterator = ((i, source[i]) for i in range(n))

    _write_sharded_samples(
        iterator=iterator,
        output_dir=str(output),
        shard_size=shard_size,
        total=n,
        metadata={
            'explicit_h': getattr(source.featurizer, 'explicit_h', False),
            'h_candidate_policy': 'explicit_h_learned_edges',
        },
        show_progress=show_progress,
        t0=t0,
    )


def build_sharded_cache_from_sdf(
    output_dir: str,
    sdf_path: Optional[str] = None,
    sdf_dir: Optional[str] = None,
    pattern: str = '*.sdf',
    wiberg_npz: Optional[str] = None,
    max_mols: Optional[int] = None,
    featurizer: Optional[MoleculeFeaturizer] = None,
    shard_size: int = 50000,
    show_progress: bool = True,
) -> None:
    """
    Stream SDF molecules directly into shard files.

    Unlike BondNetDataset.from_sdf(), this never materializes all RDKit
    molecules or all featurized samples at once, so it is suitable for 1M-scale
    SDF files.
    """
    import glob
    import time
    from pathlib import Path
    from rdkit import Chem  # type: ignore

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    featurizer = featurizer or MoleculeFeaturizer()
    bo_dict = _load_wiberg_bo_dict(wiberg_npz)

    if sdf_dir:
        sdf_files = sorted(glob.glob(os.path.join(sdf_dir, pattern)))
    elif sdf_path:
        sdf_files = [sdf_path]
    else:
        raise ValueError('Provide sdf_path or sdf_dir.')

    def iter_samples():
        written_idx = 0
        seen_idx = 0
        for path in sdf_files:
            supplier = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
            for mol in supplier:
                seen_idx += 1
                if mol is None:
                    continue
                try:
                    _sanitize_with_mdl_aromaticity(mol)
                    if mol.GetNumConformers() == 0:
                        continue
                    sample = featurizer.featurize(
                        mol,
                        wiberg_bo_matrix=bo_dict.get(written_idx),
                    )
                except Exception:
                    continue
                sample['mol_idx'] = written_idx
                if mol.HasProp('geom_mol_idx'):
                    try:
                        sample['geom_mol_idx'] = int(mol.GetProp('geom_mol_idx'))
                    except Exception:
                        pass
                if mol.HasProp('geom_conf_idx'):
                    try:
                        sample['geom_conf_idx'] = int(mol.GetProp('geom_conf_idx'))
                    except Exception:
                        sample['geom_conf_idx'] = mol.GetProp('geom_conf_idx')
                if mol.HasProp('geom_conf_rank'):
                    try:
                        sample['geom_conf_rank'] = int(mol.GetProp('geom_conf_rank'))
                    except Exception:
                        pass
                yield seen_idx - 1, sample
                written_idx += 1
                if max_mols and written_idx >= max_mols:
                    return

    print(f'Building sharded feature cache from SDF -> {output}')
    _write_sharded_samples(
        iterator=iter_samples(),
        output_dir=str(output),
        shard_size=shard_size,
        total=max_mols,
        metadata={
            'explicit_h': getattr(featurizer, 'explicit_h', False),
            'h_candidate_policy': 'explicit_h_learned_edges',
            'source': sdf_dir or sdf_path,
        },
        show_progress=show_progress,
        t0=time.time(),
    )


def _write_sharded_samples(
    iterator,
    output_dir: str,
    shard_size: int,
    total: Optional[int],
    metadata: Dict,
    show_progress: bool,
    t0: float,
) -> None:
    import time
    from pathlib import Path

    output = Path(output_dir)
    shard_size = max(1, int(shard_size))
    shards = []
    sample_meta = []
    weight_chunks = []
    current = []
    current_weights = []
    n_seen = 0
    n_written = 0
    shard_id = 0
    split_seed = int(metadata.get('split_seed', DEFAULT_SPLIT_SEED))
    split_val_frac = float(metadata.get('split_val_frac', DEFAULT_VAL_FRAC))
    split_test_frac = float(metadata.get('split_test_frac', DEFAULT_TEST_FRAC))

    def flush() -> None:
        nonlocal current, current_weights, shard_id
        if not current:
            return
        shard_name = f'shard_{shard_id:05d}.pt'
        torch.save(
            {
                'samples': current,
                'diversity_weights': torch.tensor(current_weights, dtype=torch.float32),
            },
            output / shard_name,
        )
        start = n_written - len(current)
        shards.append({'path': shard_name, 'start': start, 'n': len(current)})
        weight_chunks.append(torch.tensor(current_weights, dtype=torch.float32))
        shard_id += 1
        current = []
        current_weights = []

    for source_idx, sample in iterator:
        n_seen = int(source_idx) + 1
        if sample.get('_invalid'):
            continue
        sample = dict(sample)
        sample['split'] = _assign_split_label(
            sample,
            default_idx=n_written,
            val_frac=split_val_frac,
            test_frac=split_test_frac,
            seed=split_seed,
        )
        current.append(sample)
        current_weights.append(_sample_diversity_weight(sample))
        meta = _sample_metadata(sample)
        if meta or sample_meta:
            if meta and not sample_meta and n_written > 0:
                sample_meta.extend({} for _ in range(n_written))
            sample_meta.append(meta)
        n_written += 1
        if len(current) >= shard_size:
            flush()
        if show_progress and n_written > 0 and n_written % 5000 == 0:
            elapsed = time.time() - t0
            if total:
                print(f'  written={n_written}/{total} seen={n_seen} ({elapsed:.0f}s)')
            else:
                print(f'  written={n_written} seen={n_seen} ({elapsed:.0f}s)')

    flush()
    diversity_weights = (
        torch.cat(weight_chunks) if weight_chunks
        else torch.empty(0, dtype=torch.float32)
    )
    manifest = {
        'shards': shards,
        'n_samples': n_written,
        'diversity_weights': diversity_weights,
        'metadata': {
            'cache_format_version': CACHE_FORMAT_VERSION,
            'n_molecules': n_written,
            'stage3_excludes_h': STAGE3_EXCLUDES_H,
            'fixed_split': {
                'method': 'stable_hash',
                'seed': split_seed,
                'val_frac': split_val_frac,
                'test_frac': split_test_frac,
            },
            **metadata,
        },
    }
    if sample_meta:
        manifest['sample_meta'] = sample_meta
    torch.save(manifest, output / ShardedCachedBondNetDataset.MANIFEST_NAME)
    elapsed = time.time() - t0
    print(
        f'Sharded cache saved: {n_written} molecules in {len(shards)} shards '
        f'in {elapsed:.1f}s -> {output}'
    )


# ------------------------------------------------------------------ #
# Collation                                                            #
# ------------------------------------------------------------------ #

def collate_fn(samples: List[Dict]) -> Dict:
    """
    Collate a list of per-molecule feature dicts into a batched dict.

    Handles variable-size graphs by offsetting atom indices and
    concatenating all tensors along the batch dimension.

    Ring metadata (ring_atoms_flat, ring_edge_indices_flat) are stored as
    flat lists with globally-offset indices so BondNet.forward() can compute
    ring features from backbone outputs at runtime.

    Returns:
        Batched dict suitable for BondNet.forward().
    """
    valid = [s for s in samples if not s.get('_invalid', False)]
    if not valid:
        return {}

    batch: Dict = {}
    atom_offset = 0

    elems_list = []
    coord_list = []
    local_geom_list = []
    edge_index_list = []
    atom_features_list = []
    edge_diff_list = []
    edge_dist_list = []
    bond_exists_list = []
    bond_mask_list = []
    train_edge_mask_list = []
    train_bond_mask_list = []
    h_candidate_edge_mask_list = []
    edge_index_bond_list = []
    bond_type_full_list = []
    bond_type_train_list = []
    wiberg_bo_all_list = []
    is_resonance_all_list = []
    expected_val_list = []
    num_atoms_list = []
    mol_idx_list = []
    bond_mol_idx_list = []
    edge_mol_idx_list = []


    for mol_i, s in enumerate(valid):
        N = int(s['num_atoms'])
        E = s['edge_index'].shape[0]
        E_b = int(s['bond_mask'].sum().item())

        elems_list.append(s['elems'])
        if 'atom_features' in s and s['atom_features'] is not None:
            atom_features_list.append(s['atom_features'])
        coord_list.append(s['coord'])
        if 'local_geom' in s and s['local_geom'] is not None:
            local_geom_list.append(s['local_geom'])

        edge_index_list.append(s['edge_index'] + atom_offset)
        edge_diff_list.append(s['edge_diff'])
        edge_dist_list.append(s['edge_dist'])

        bond_exists_list.append(s['bond_exists'])
        bond_mask_list.append(s['bond_mask'])
        # H-aware fields — optional; absent in heavy-atom-only (v8-style) data
        if 'train_edge_mask' in s:
            train_edge_mask_list.append(s['train_edge_mask'])
        if 'train_bond_mask' in s:
            train_bond_mask_list.append(s['train_bond_mask'])
        if 'h_candidate_edge_mask' in s:
            h_candidate_edge_mask_list.append(s['h_candidate_edge_mask'])
        edge_index_bond_list.append(s['edge_index_bond'] + atom_offset)
        bond_type_full_list.append(s['bond_type_full'])
        if 'bond_type_train' in s:
            bond_type_train_list.append(s['bond_type_train'])
        wiberg_bo_all_list.append(s['wiberg_bo_all'])
        is_resonance_all_list.append(s['is_resonance_all'])
        expected_val_list.append(s['expected_valence'])
        num_atoms_list.append(N)

        # Per-bond and per-edge molecule index (for per-molecule evaluation)
        bond_mol_idx_list.append(torch.full((E_b,), mol_i, dtype=torch.long))
        edge_mol_idx_list.append(torch.full((E,),   mol_i, dtype=torch.long))

        # Prefer the persistent GEOM molecule identifier.  ``mol_idx`` is not
        # present in the fixed-split caches, and using -1 for every molecule
        # makes deterministic paired perturbations and error provenance
        # impossible.
        mol_idx_list.append(
            s.get('geom_mol_idx', s.get('mol_idx', s.get('sample_id', -1)))
        )

        atom_offset += N

    batch['elems'] = torch.cat(elems_list)
    if len(atom_features_list) == len(valid):
        batch['atom_features'] = torch.cat(atom_features_list)
    batch['coord'] = torch.cat(coord_list)
    if len(local_geom_list) == len(valid):
        batch['local_geom'] = torch.cat(local_geom_list)
    batch['edge_index'] = torch.cat(edge_index_list)
    batch['edge_diff'] = torch.cat(edge_diff_list)
    batch['edge_dist'] = torch.cat(edge_dist_list)
    batch['bond_exists'] = torch.cat(bond_exists_list)
    batch['bond_mask'] = torch.cat(bond_mask_list)
    if len(train_edge_mask_list) == len(valid):
        batch['train_edge_mask'] = torch.cat(train_edge_mask_list)
    if len(train_bond_mask_list) == len(valid):
        batch['train_bond_mask'] = torch.cat(train_bond_mask_list)
    if len(h_candidate_edge_mask_list) == len(valid):
        batch['h_candidate_edge_mask'] = torch.cat(h_candidate_edge_mask_list)
    batch['edge_index_bond'] = torch.cat(edge_index_bond_list)
    batch['bond_type_full'] = torch.cat(bond_type_full_list)  # (E_b,) {0,1,2,3}
    if len(bond_type_train_list) == len(valid):
        batch['bond_type_train'] = torch.cat(bond_type_train_list)
    batch['wiberg_bo_all'] = torch.cat(wiberg_bo_all_list)    # (E_b,) BO for all bonds
    batch['is_resonance_all'] = torch.cat(is_resonance_all_list)  # (E_b,) resonance mask
    batch['expected_valence'] = torch.cat(expected_val_list)
    batch['num_atoms'] = sum(num_atoms_list)
    batch['num_mols'] = len(valid)
    batch['num_atoms_per_mol'] = torch.tensor(num_atoms_list, dtype=torch.long)
    batch['mol_idx'] = torch.tensor(mol_idx_list, dtype=torch.long)
    batch['sample_ids'] = batch['mol_idx'].clone()
    batch['bond_mol_idx'] = torch.cat(bond_mol_idx_list)  # (E_b,) mol index per bond
    batch['edge_mol_idx'] = torch.cat(edge_mol_idx_list)  # (E,)  mol index per edge

    return batch


def collate_connectivity_fn(samples: List[Dict]) -> Dict:
    """
    Lightweight collate for Stage 1 connectivity-only training.

    Avoids concatenating bond-type, valence, BO, and per-molecule bookkeeping
    fields that are unused by the connectivity loss/backbone.
    """
    valid = [s for s in samples if not s.get('_invalid', False)]
    if not valid:
        return {}

    atom_offset = 0
    elems_list = []
    coord_list = []
    local_geom_list = []
    edge_index_list = []
    edge_diff_list = []
    edge_dist_list = []
    bond_exists_list = []
    train_edge_mask_list = []
    num_atoms_list = []
    sample_id_list = []

    for s in valid:
        N = int(s['num_atoms'])
        elems_list.append(s['elems'])
        coord_list.append(s['coord'])
        if 'local_geom' in s and s['local_geom'] is not None:
            local_geom_list.append(s['local_geom'])
        edge_index_list.append(s['edge_index'] + atom_offset)
        edge_diff_list.append(s['edge_diff'])
        edge_dist_list.append(s['edge_dist'])
        bond_exists_list.append(s['bond_exists'])
        if 'train_edge_mask' in s:
            train_edge_mask_list.append(s['train_edge_mask'])
        num_atoms_list.append(N)
        sample_id_list.append(
            s.get('geom_mol_idx', s.get('mol_idx', s.get('sample_id', -1)))
        )
        atom_offset += N

    batch: Dict = {
        'elems': torch.cat(elems_list),
        'coord': torch.cat(coord_list),
        'edge_index': torch.cat(edge_index_list),
        'edge_diff': torch.cat(edge_diff_list),
        'edge_dist': torch.cat(edge_dist_list),
        'bond_exists': torch.cat(bond_exists_list),
        'num_atoms': sum(num_atoms_list),
        'num_mols': len(valid),
        'num_atoms_per_mol': torch.tensor(num_atoms_list, dtype=torch.long),
        'sample_ids': torch.tensor(sample_id_list, dtype=torch.long),
        '_connectivity_only': True,
    }
    if len(local_geom_list) == len(valid):
        batch['local_geom'] = torch.cat(local_geom_list)
    if len(train_edge_mask_list) == len(valid):
        batch['train_edge_mask'] = torch.cat(train_edge_mask_list)
    return batch


def collate_stage2_fn(samples: List[Dict]) -> Dict:
    """
    Lightweight collate for Stage 2 topology bond-type training/evaluation.

    Keeps only fields needed by Stage2 teacher-forced training and Stage1->Stage2
    E2E validation.  This avoids BO/valence/ring legacy bookkeeping.
    """
    valid = [s for s in samples if not s.get('_invalid', False)]
    if not valid:
        return {}

    atom_offset = 0
    elems_list = []
    coord_list = []
    edge_index_list = []
    edge_diff_list = []
    edge_dist_list = []
    bond_exists_list = []
    bond_mask_list = []
    train_edge_mask_list = []
    bond_type_full_list = []
    num_atoms_list = []
    sample_id_list = []
    bond_mol_idx_list = []
    edge_mol_idx_list = []

    for mol_i, s in enumerate(valid):
        N = int(s['num_atoms'])
        E = int(s['edge_index'].shape[0])
        E_b = int(s['bond_mask'].sum().item())
        elems_list.append(s['elems'])
        coord_list.append(s['coord'])
        edge_index_list.append(s['edge_index'] + atom_offset)
        edge_diff_list.append(s['edge_diff'])
        edge_dist_list.append(s['edge_dist'])
        bond_exists_list.append(s['bond_exists'])
        bond_mask_list.append(s['bond_mask'])
        if 'train_edge_mask' in s:
            train_edge_mask_list.append(s['train_edge_mask'])
        bond_type_full_list.append(s['bond_type_full'])
        num_atoms_list.append(N)
        sample_id_list.append(
            s.get('geom_mol_idx', s.get('mol_idx', s.get('sample_id', -1)))
        )
        bond_mol_idx_list.append(torch.full((E_b,), mol_i, dtype=torch.long))
        edge_mol_idx_list.append(torch.full((E,), mol_i, dtype=torch.long))
        atom_offset += N

    batch: Dict = {
        'elems': torch.cat(elems_list),
        'coord': torch.cat(coord_list),
        'edge_index': torch.cat(edge_index_list),
        'edge_diff': torch.cat(edge_diff_list),
        'edge_dist': torch.cat(edge_dist_list),
        'bond_exists': torch.cat(bond_exists_list),
        'bond_mask': torch.cat(bond_mask_list),
        'bond_type_full': torch.cat(bond_type_full_list),
        'num_atoms': sum(num_atoms_list),
        'num_mols': len(valid),
        'num_atoms_per_mol': torch.tensor(num_atoms_list, dtype=torch.long),
        'sample_ids': torch.tensor(sample_id_list, dtype=torch.long),
        'bond_mol_idx': torch.cat(bond_mol_idx_list),
        'edge_mol_idx': torch.cat(edge_mol_idx_list),
    }
    if len(train_edge_mask_list) == len(valid):
        batch['train_edge_mask'] = torch.cat(train_edge_mask_list)
    return batch


# ------------------------------------------------------------------ #
# QM9 conversion helper                                                #
# ------------------------------------------------------------------ #

def _pyg_qm9_to_rdkit(data):
    """
    Convert a PyG QM9 Data object to an RDKit molecule with 3D conformer.

    PyG QM9 edge_attr is (E, 4) with one-hot bond type:
        col 0 = single, 1 = double, 2 = triple, 3 = aromatic
    Both directed edges (i→j) and (j→i) are present; we deduplicate by i<j.
    """
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import RWMol  # type: ignore

        _BOND_TYPES = [
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
            Chem.BondType.AROMATIC,
        ]

        z = data.z.tolist()
        pos = data.pos  # (N, 3)

        rw = RWMol()
        for atomic_num in z:
            rw.AddAtom(Chem.Atom(int(atomic_num)))

        # Build bond-type lookup from directed edge list
        edge_idx = data.edge_index.t().tolist()   # (E, 2)
        has_attr = (
            hasattr(data, 'edge_attr')
            and data.edge_attr is not None
            and data.edge_attr.shape[0] == len(edge_idx)
        )

        seen = set()
        for e_idx, (i, j) in enumerate(edge_idx):
            if i >= j:
                continue
            key = (i, j)
            if key in seen:
                continue
            seen.add(key)

            if has_attr:
                bt_idx = int(data.edge_attr[e_idx].argmax())
                bt_idx = min(bt_idx, len(_BOND_TYPES) - 1)
                bond_type = _BOND_TYPES[bt_idx]
            else:
                bond_type = Chem.BondType.SINGLE

            rw.AddBond(i, j, bond_type)

        mol = rw.GetMol()
        try:
            _sanitize_with_mdl_aromaticity(mol)
        except Exception:
            pass

        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, pos[i].tolist())
        mol.AddConformer(conf, assignId=True)

        return mol
    except Exception:
        return None
