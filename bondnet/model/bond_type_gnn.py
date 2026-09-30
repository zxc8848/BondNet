"""
Stage 2 bond-type GNN.

This model runs on the bonded topology predicted by Stage 1. Hydrogens can stay
as ordinary nodes in the message graph while readout is restricted to the bonds
the caller asks it to classify.
"""

import torch
from torch import nn

from .painn_backbone import PainnMessage, PainnUpdate, sinc_expansion

_NORM_EPS = 1e-8       # used for float32 safe_norm
_SINC_EPS = 1e-4       # used for sinc division — must be representable in float16
                       # (float16 min normal ~6e-5; 1e-8 underflows to 0 under AMP)


def safe_norm(x: torch.Tensor, dim, keepdim: bool = False, eps: float = _NORM_EPS):
    """Numerically stable L2 norm with an epsilon inside the square root."""
    return torch.sqrt(torch.sum(x * x, dim=dim, keepdim=keepdim) + eps)


def sanitize_hidden(x: torch.Tensor, limit: float = 50.0) -> torch.Tensor:
    """Keep PaiNN hidden states finite under aggressive coordinate noise."""
    return torch.nan_to_num(x, nan=0.0, posinf=limit, neginf=-limit).clamp_(
        min=-limit,
        max=limit,
    )


def limit_vector_norm(node_vector: torch.Tensor, limit: float) -> torch.Tensor:
    """Clip each equivariant vector channel by its 3D norm."""
    if limit <= 0.0 or node_vector.numel() == 0:
        return node_vector
    vec_norm = safe_norm(node_vector, dim=1, keepdim=True)
    scale = (limit / vec_norm.clamp_min(_NORM_EPS)).clamp_max(1.0)
    return node_vector * scale



class BondTypeGNN(nn.Module):
    """Bond type classifier on a bonded molecular topology."""

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 3,
        edge_embedding_size: int = 16,
        bond_cutoff: float = 3.0,
        num_bond_types: int = 4,
        dropout: float = 0.1,
        vector_norm_limit=3,
        random_node_feat_dim: int = 0,
        random_node_feat_std: float = 1.0,
    ):
        super().__init__()
        self.bond_cutoff = bond_cutoff
        self.edge_embedding_size = edge_embedding_size
        self.random_node_feat_dim = random_node_feat_dim
        self.random_node_feat_std = random_node_feat_std

        self.vector_norm_limit = vector_norm_limit

        self.atom_embedding = nn.Embedding(119, hidden_size)
        self.random_node_proj = (
            nn.Linear(random_node_feat_dim, hidden_size, bias=False)
            if random_node_feat_dim > 0 else None
        )

        self.msg_layers = nn.ModuleList([
            PainnMessage(hidden_size, edge_embedding_size, bond_cutoff)
            for _ in range(num_layers)
        ])
        self.upd_layers = nn.ModuleList([
            PainnUpdate(hidden_size) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

        edge_in = 2 * hidden_size + edge_embedding_size
        self.cls_head = nn.Sequential(
            nn.Linear(edge_in, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, num_bond_types),
        )

    def forward(
        self,
        elems: torch.Tensor,
        bond_edge_index: torch.Tensor,
        bond_edge_diff: torch.Tensor,
        bond_edge_dist: torch.Tensor,
        readout_edge_index: torch.Tensor = None,
        readout_edge_diff: torch.Tensor = None,
        readout_edge_dist: torch.Tensor = None,
    ) -> torch.Tensor:
        if readout_edge_index is None:
            readout_edge_index = bond_edge_index
            readout_edge_diff = bond_edge_diff
            readout_edge_dist = bond_edge_dist
        if readout_edge_index.shape[0] == 0:
            return torch.zeros(0, self.cls_head[-1].out_features, device=elems.device)

        s = self.atom_embedding(elems)
        if self.random_node_proj is not None:
            rand_feat = torch.randn(
                elems.shape[0],
                self.random_node_feat_dim,
                device=elems.device,
                dtype=s.dtype,
            ) * self.random_node_feat_std
            s = s + self.random_node_proj(rand_feat)
        v = torch.zeros(s.shape[0], 3, s.shape[1], device=s.device, dtype=s.dtype)

        if bond_edge_index.shape[0] > 0:
            for msg, upd, norm in zip(self.msg_layers, self.upd_layers, self.norms):
                s, v = msg(s, v, bond_edge_index, bond_edge_diff, bond_edge_dist)
                # s, v = upd(s, v)
                s = norm(s)
                v = v.clamp(-1e4, 1e4)   # prevent vector feature explosion across layers
                
                s= sanitize_hidden(s)
                v = sanitize_hidden(v)
                s,v = upd(s,v)
                s= sanitize_hidden(s)
                v = sanitize_hidden(v)
                v = limit_vector_norm(v, self.vector_norm_limit)


        s_i = s[readout_edge_index[:, 0]]
        s_j = s[readout_edge_index[:, 1]]
        rbf = sinc_expansion(readout_edge_dist, self.edge_embedding_size, self.bond_cutoff)
        return self.cls_head(torch.cat([s_i, s_j, rbf], dim=-1))


def compute_bonded_geometry(coord: torch.Tensor, bond_edge_index: torch.Tensor):
    diff = coord[bond_edge_index[:, 1]] - coord[bond_edge_index[:, 0]]
    dist = diff.norm(dim=-1)
    return diff, dist
