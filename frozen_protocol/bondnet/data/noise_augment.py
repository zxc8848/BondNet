"""
Gaussian noise augmentation for BondNet training.

Adds isotropic Gaussian noise to atom coordinates and recomputes
edge_diff and edge_dist in-place.  edge_index (connectivity topology)
is unchanged — only the geometric quantities are updated.
"""

import torch
from typing import Dict, Optional

from ..utils.local_geometry import compute_local_geometry_features


def apply_dynamic_candidate_mask(
    batch: Dict,
    heavy_cutoff: float,
    hydrogen_cutoff: Optional[float] = None,
) -> Dict:
    """Apply the candidate-graph rule to the batch's current coordinates.

    A cache may be built with a larger radius envelope than the model uses. This
    function turns that envelope into the candidate graph for the current (and
    possibly perturbed) coordinates. It updates supervision masks without
    changing tensor shapes, which keeps batched cached training efficient while
    allowing edges to enter or leave the active candidate graph.

    The cache radius must be at least as large as the requested cutoffs. Edges
    outside the cache envelope cannot be recovered by this function.
    """
    if hydrogen_cutoff is None:
        hydrogen_cutoff = heavy_cutoff
    edge_index = batch["edge_index"]
    edge_dist = batch["edge_dist"]
    elems = batch["elems"]
    src = edge_index[:, 0]
    dst = edge_index[:, 1]
    src_h = elems[src] == 1
    dst_h = elems[dst] == 1
    hh = ~src_h & ~dst_h
    ha = src_h ^ dst_h
    active = (hh & (edge_dist <= float(heavy_cutoff))) | (
        ha & (edge_dist <= float(hydrogen_cutoff))
    )

    updated = dict(batch)
    updated["train_edge_mask"] = active
    if "bond_mask" in batch:
        updated["train_bond_mask"] = batch["bond_mask"].bool() & active & hh
        # ``bond_type_full`` is ordered by the true-bond positions in
        # ``bond_mask`` rather than by every candidate edge. When perturbation
        # moves a true HH bond outside the active candidate graph, Stage 3 emits
        # no logit for that edge, so its label must be removed in the same
        # order. The missed bond is still counted as an error by end-to-end
        # evaluation; this only keeps the training tensors aligned.
        if "bond_type_full" in batch:
            true_pos = torch.where(batch["bond_mask"].bool())[0]
            if batch["bond_type_full"].shape[0] != true_pos.shape[0]:
                raise ValueError(
                    "bond_type_full must align with true positions in bond_mask: "
                    f"{batch['bond_type_full'].shape[0]} != {true_pos.shape[0]}"
                )
            true_hh = hh[true_pos]
            active_true_hh = active[true_pos] & true_hh
            updated["bond_type_train_active"] = batch["bond_type_full"][active_true_hh]
    updated["h_candidate_edge_mask"] = active & ha
    if "local_geom" in batch and "coord" in batch:
        updated["local_geom"] = compute_local_geometry_features(
            batch["coord"], edge_index, edge_dist, float(heavy_cutoff)
        )
    return updated


class GaussianNoiseAugment:
    """
    Apply Gaussian noise to 3D atom coordinates.

    Args:
        sigma:      Noise standard deviation (Å).  σ=0.05 simulates typical
                    EDM/GeoLDM generation error.
        sigma_min:  If provided and sigma_max is also set, σ is sampled
                    uniformly in [sigma_min, sigma_max] per molecule.
        sigma_max:  Upper bound for random σ sampling.
    """

    def __init__(
        self,
        sigma: float = 0.05,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        cutoff: float = 5.0,
        clean_prob: float = 0.0,
    ):
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.cutoff = cutoff
        self.clean_prob = clean_prob

    def __call__(self, sample: Dict) -> Dict:
        """
        Apply noise augmentation to a sample dict in-place.

        Args:
            sample: Feature dict returned by MoleculeFeaturizer.featurize().
                    Must contain 'coord', 'edge_index', 'edge_diff', 'edge_dist'.

        Returns:
            The same dict with 'coord', 'edge_diff', 'edge_dist' updated.
        """
        if self.clean_prob > 0.0 and torch.rand(()) < self.clean_prob:
            return sample

        coord = sample['coord']  # (N, 3)

        # Sample σ from range if specified; otherwise use fixed sigma
        sigma = self.sigma
        if self.sigma_min is not None and self.sigma_max is not None:
            sigma = float(
                torch.empty(1).uniform_(self.sigma_min, self.sigma_max).item()
            )

        # σ=0 → no-op (keeps clean coordinates in the training distribution)
        if sigma == 0.0:
            return sample

        # Perturb coordinates
        noise = torch.randn_like(coord) * sigma
        new_coord = coord + noise

        # Recompute geometric edge features
        edge_index = sample['edge_index']   # (E, 2)
        i_idx = edge_index[:, 0]
        j_idx = edge_index[:, 1]
        new_edge_diff = new_coord[j_idx] - new_coord[i_idx]
        new_edge_dist = new_edge_diff.norm(dim=-1)

        sample = dict(sample)  # shallow copy to avoid mutating original
        sample['coord'] = new_coord
        sample['edge_diff'] = new_edge_diff
        sample['edge_dist'] = new_edge_dist
        if 'local_geom' in sample:
            sample['local_geom'] = compute_local_geometry_features(
                new_coord,
                edge_index,
                new_edge_dist,
                self.cutoff,
            )
        return sample

    def augment_batch(
        self,
        batch: Dict,
        sigma: Optional[float] = None,
        per_molecule: bool = True,
        clean_prob: float = 0.0,
        base_seed: Optional[int] = None,
        epoch: int = 0,
        sample_ids_key: str = 'sample_ids',
    ) -> Dict:
        """
        Augment a collated batch dict.

        When ``base_seed`` is supplied, every molecule receives a deterministic
        random stream keyed by (base_seed, epoch, persistent sample id).  This
        makes perturbations independent of batch order/size and, crucially,
        gives the heavy-atom prefix identical noise in explicit-H and
        heavy-only representations of the same molecule.
        """
        coord = batch['coord']
        if per_molecule and 'num_atoms_per_mol' in batch:
            counts = batch['num_atoms_per_mol'].to(device=coord.device)
            n_mols = int(counts.shape[0])
            if base_seed is not None:
                if sample_ids_key not in batch:
                    raise KeyError(
                        f'Keyed noise requires batch[{sample_ids_key!r}]'
                    )
                sample_ids = batch[sample_ids_key].detach().cpu().tolist()
                if len(sample_ids) != n_mols:
                    raise ValueError(
                        'sample_ids and num_atoms_per_mol disagree: '
                        f'{len(sample_ids)} != {n_mols}'
                    )
                if any(int(x) < 0 for x in sample_ids):
                    raise ValueError('Keyed noise requires non-negative persistent sample ids')

                pieces = []
                offset = 0
                modulus = 2**63 - 1
                for mol_id, count_t in zip(sample_ids, counts.detach().cpu().tolist()):
                    count = int(count_t)
                    # Stable integer mixing; do not use Python's randomized hash.
                    mixed_seed = (
                        int(base_seed) * 1_000_003
                        + int(epoch) * 97_409
                        + int(mol_id) * 9_176
                        + 0x5DEECE66D
                    ) % modulus
                    generator = torch.Generator(device=coord.device)
                    generator.manual_seed(mixed_seed)
                    if self.sigma_min is not None and self.sigma_max is not None:
                        u = torch.rand((), generator=generator, device=coord.device)
                        mol_sigma = self.sigma_min + (
                            self.sigma_max - self.sigma_min
                        ) * float(u.item())
                    else:
                        mol_sigma = float(sigma if sigma is not None else self.sigma)
                    if clean_prob > 0.0:
                        clean_u = torch.rand((), generator=generator, device=coord.device)
                        if float(clean_u.item()) < clean_prob:
                            mol_sigma = 0.0
                    mol_noise = torch.randn(
                        (count, coord.shape[-1]),
                        generator=generator,
                        device=coord.device,
                        dtype=coord.dtype,
                    ) * mol_sigma
                    pieces.append(mol_noise)
                    offset += count
                if offset != coord.shape[0]:
                    raise ValueError(
                        'num_atoms_per_mol does not sum to coordinate count: '
                        f'{offset} != {coord.shape[0]}'
                    )
                noise = torch.cat(pieces, dim=0)
            elif self.sigma_min is not None and self.sigma_max is not None:
                sigmas = torch.empty(
                    n_mols, device=coord.device, dtype=coord.dtype
                ).uniform_(self.sigma_min, self.sigma_max)
                if clean_prob > 0.0:
                    clean_mask = torch.rand(n_mols, device=coord.device) < clean_prob
                    sigmas = sigmas.masked_fill(clean_mask, 0.0)
                atom_sigma = torch.repeat_interleave(sigmas, counts).unsqueeze(-1)
                noise = torch.randn_like(coord) * atom_sigma
            else:
                s = sigma if sigma is not None else self.sigma
                sigmas = torch.full(
                    (n_mols,), float(s), device=coord.device, dtype=coord.dtype
                )
                if clean_prob > 0.0:
                    clean_mask = torch.rand(n_mols, device=coord.device) < clean_prob
                    sigmas = sigmas.masked_fill(clean_mask, 0.0)
                atom_sigma = torch.repeat_interleave(sigmas, counts).unsqueeze(-1)
                noise = torch.randn_like(coord) * atom_sigma
        else:
            s = sigma if sigma is not None else self.sigma
            if clean_prob > 0.0 and torch.rand((), device=coord.device) < clean_prob:
                s = 0.0
            noise = torch.randn_like(coord) * s
        new_coord = coord + noise

        edge_index = batch['edge_index']
        i_idx = edge_index[:, 0]
        j_idx = edge_index[:, 1]
        new_edge_diff = new_coord[j_idx] - new_coord[i_idx]
        new_edge_dist = new_edge_diff.norm(dim=-1)

        batch = dict(batch)
        batch['coord'] = new_coord
        batch['edge_diff'] = new_edge_diff
        batch['edge_dist'] = new_edge_dist
        if 'local_geom' in batch:
            batch['local_geom'] = compute_local_geometry_features(
                new_coord,
                edge_index,
                new_edge_dist,
                self.cutoff,
            )
        return batch
