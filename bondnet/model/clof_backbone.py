"""
ClofNet-style backbone adapted for BondNet edge classification.

This is a lightweight in-repo adapter rather than a direct import of the
downloaded ClofNet scripts, because the original code targets fixed-size
physical systems or PyG score networks.  The adapter keeps the relevant idea:
message passing uses scalar edge features plus a local coordinate frame
constructed from centered 3D coordinates, and returns per-edge invariant
features compatible with BondNet's existing connectivity and bond-type heads.
"""

from __future__ import annotations

import torch
from torch import nn

from .painn_backbone import safe_norm, sinc_expansion, sanitize_hidden


def _segment_sum(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    out = data.new_zeros((num_segments, data.shape[-1]))
    out.index_add_(0, segment_ids, data)
    return out


def _segment_mean(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    out = _segment_sum(data, segment_ids, num_segments)
    count = data.new_zeros((num_segments, data.shape[-1]))
    count.index_add_(0, segment_ids, torch.ones_like(data))
    return out / count.clamp_min(1.0)


def _center_by_counts(coord: torch.Tensor, counts: torch.Tensor | None) -> torch.Tensor:
    if counts is None or counts.numel() <= 1:
        return coord - coord.mean(dim=0, keepdim=True)
    counts = counts.to(device=coord.device, dtype=torch.long)
    mol_ids = torch.repeat_interleave(torch.arange(counts.numel(), device=coord.device), counts)
    centers = _segment_mean(coord, mol_ids, int(counts.numel()))
    return coord - centers[mol_ids]


class ClofLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        edge_hidden_size: int,
        act_fn: nn.Module | None = None,
        coords_weight: float = 0.1,
        norm_diff: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        act_fn = act_fn or nn.SiLU()
        # Kept in the signature for checkpoint/CLI compatibility. BondNet uses
        # ClofNet as a scalar edge-feature extractor, so coordinates stay fixed.
        self.coords_weight = float(coords_weight)
        self.norm_diff = bool(norm_diff)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2 + 1 + 8 + edge_hidden_size, hidden_size),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.node_norm = nn.LayerNorm(hidden_size)
        self.edge_norm = nn.LayerNorm(hidden_size)

    def _local_frame(self, edge_index: torch.Tensor, coord: torch.Tensor):
        row, col = edge_index[:, 0], edge_index[:, 1]
        coord_diff = coord[row] - coord[col]
        radial = (coord_diff * coord_diff).sum(dim=1, keepdim=True)
        coord_cross = torch.cross(coord[row], coord[col], dim=-1)
        if self.norm_diff:
            coord_diff = coord_diff / (radial.sqrt() + 1.0)
            coord_cross = coord_cross / (safe_norm(coord_cross, dim=-1, keepdim=True) + 1.0)
        coord_vertical = torch.cross(coord_diff, coord_cross, dim=-1)
        basis = torch.stack([coord_diff, coord_cross, coord_vertical], dim=1)
        coff_i = torch.einsum("eab,eb->ea", basis, coord[row])
        coff_j = torch.einsum("eab,eb->ea", basis, coord[col])
        ni = safe_norm(coff_i, dim=-1, keepdim=True)
        nj = safe_norm(coff_j, dim=-1, keepdim=True)
        cos = (coff_i * coff_j).sum(dim=-1, keepdim=True) / ni.clamp_min(1e-5) / nj.clamp_min(1e-5)
        cos = cos.clamp(min=-1.0, max=1.0)
        sin = (1.0 - cos * cos).clamp_min(0.0).sqrt()
        frame_scalar = torch.cat([sin, cos, coff_i, coff_j], dim=-1)
        return radial, frame_scalar

    def forward(
        self,
        h: torch.Tensor,
        coord: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ):
        row, col = edge_index[:, 0], edge_index[:, 1]
        radial, frame_scalar = self._local_frame(edge_index, coord)
        edge_feat = self.edge_mlp(torch.cat([h[row], h[col], radial, frame_scalar, edge_attr], dim=-1))
        edge_feat = self.edge_norm(sanitize_hidden(edge_feat))

        agg = _segment_sum(edge_feat, row, h.shape[0])
        h = self.node_norm(h + self.node_mlp(torch.cat([h, agg], dim=-1)))
        return sanitize_hidden(h), coord, edge_feat


class ClofBackbone(nn.Module):
    """
    ClofNet-style edge-feature backbone with the same return contract as PainnBackbone.
    """

    def __init__(
        self,
        num_interactions: int = 4,
        hidden_state_size: int = 128,
        cutoff: float = 5.0,
        edge_embedding_size: int = 20,
        atom_feature_size: int = 4,
        coords_weight: float = 0.1,
        norm_diff: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.edge_embedding_size = edge_embedding_size
        self.atom_embedding = nn.Embedding(119, hidden_state_size)
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_embedding_size, hidden_state_size),
            nn.SiLU(),
            nn.Linear(hidden_state_size, hidden_state_size),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList([
            ClofLayer(
                hidden_size=hidden_state_size,
                edge_hidden_size=hidden_state_size,
                coords_weight=coords_weight,
                norm_diff=norm_diff,
                dropout=dropout,
            )
            for _ in range(num_interactions)
        ])
        self.edge_readout = nn.Sequential(
            nn.Linear(hidden_state_size * 3 + edge_embedding_size + 1, hidden_state_size),
            nn.SiLU(),
            nn.Linear(hidden_state_size, hidden_state_size),
        )

    def forward(
        self,
        elems,
        coord,
        edge_index,
        edge_diff,
        edge_dist,
        local_geom=None,
        atom_features=None,
        num_atoms_per_mol=None,
    ):
        coord_centered = _center_by_counts(coord, num_atoms_per_mol)
        h = self.atom_embedding(elems)
        rbf = sinc_expansion(edge_dist, self.edge_embedding_size, self.cutoff)
        edge_attr = self.edge_embedding(rbf)
        last_edge_feat = edge_attr
        for layer in self.layers:
            h, coord_centered, last_edge_feat = layer(h, coord_centered, edge_index, edge_attr)
            edge_attr = last_edge_feat

        row, col = edge_index[:, 0], edge_index[:, 1]
        edge_input = torch.cat(
            [h[row], h[col], last_edge_feat, rbf, edge_dist.unsqueeze(-1)],
            dim=-1,
        )
        edge_feat = sanitize_hidden(self.edge_readout(sanitize_hidden(edge_input)))
        node_vector = coord.new_zeros((coord.shape[0], 3, self.hidden_state_size))
        return h, node_vector, edge_feat
