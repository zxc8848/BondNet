"""
PaiNN Backbone adapted for BondNet.

Original PaiNN produces per-atom scalar/vector features for energy prediction.
This adaptation outputs per-edge invariant features h_ij for bond type prediction.

Key modifications from the original PaiNN:
  1. Remove energy readout / force computation
  2. After message-passing, construct edge features by concatenating:
     - source node scalar s_i
     - target node scalar s_j
     - distance-encoded edge features (sinc RBF)
     - ||v_i||, ||v_j|| (vector feature norms — invariant scalars)
  3. Edge projection MLP to produce final h_ij
"""

import torch
from torch import nn


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


# def rms_normalize_vector(
#     node_vector: torch.Tensor,
#     scale: float,
#     eps: float = 1e-4,
#     max_gain: float = 10.0,
# ) -> torch.Tensor:
#     """Normalize each node's full vector block by its RMS, then apply scale."""
#     if scale <= 0.0 or node_vector.numel() == 0:
#         return node_vector
#     rms = node_vector.float().pow(2).mean(dim=(1, 2), keepdim=True).add(eps * eps).sqrt()
#     gain = (float(scale) / rms).clamp_max(max_gain).to(node_vector.dtype).detach()
#     return node_vector * gain


def sinc_expansion(edge_dist: torch.Tensor, edge_size: int, cutoff: float):
    """
    Sinc radial basis function:  sin(n * pi * d / d_cut) / d

    Computed in float32 to avoid float16 underflow of the denominator under
    AMP.  The result is cast back to the input dtype before returning.
    """
    n = torch.arange(edge_size, device=edge_dist.device) + 1
    # Promote to float32 for the division so AMP float16 can't underflow d
    d32 = edge_dist.float().clamp_min(_SINC_EPS)   # (E,) float32
    result = torch.sin(
        d32.unsqueeze(-1) * n * torch.pi / cutoff
    ) / d32.unsqueeze(-1)                           # (E, edge_size) float32
    return result.to(edge_dist.dtype)               # cast back to float16 if needed


def cosine_cutoff(edge_dist: torch.Tensor, cutoff: float):
    """Cosine cutoff: 0.5*(cos(pi*d/d_cut)+1) for d < d_cut, else 0"""
    return torch.where(
        edge_dist < cutoff,
        0.5 * (torch.cos(torch.pi * edge_dist / cutoff) + 1),
        torch.tensor(0.0, device=edge_dist.device, dtype=edge_dist.dtype),
    )


class PainnMessage(nn.Module):
    """PaiNN message-passing function."""

    def __init__(self, node_size: int, edge_size: int, cutoff: float):
        super().__init__()
        self.edge_size = edge_size
        self.node_size = node_size
        self.cutoff = cutoff

        self.scalar_message_mlp = nn.Sequential(
            nn.Linear(node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, node_size * 3),
        )
        self.filter_layer = nn.Linear(edge_size, node_size * 3)

    def forward(self, node_scalar, node_vector, edge, edge_diff, edge_dist):
        filter_weight = self.filter_layer(
            sinc_expansion(edge_dist, self.edge_size, self.cutoff)
        )
        filter_weight = filter_weight * cosine_cutoff(edge_dist, self.cutoff).unsqueeze(-1)

        scalar_out = self.scalar_message_mlp(node_scalar)
        filter_out = filter_weight * scalar_out[edge[:, 1]]

        gate_state_vector, gate_edge_vector, message_scalar = torch.split(
            filter_out, self.node_size, dim=1
        )

        message_vector = node_vector[edge[:, 1]] * gate_state_vector.unsqueeze(1)
        safe_edge_dist = edge_dist.clamp_min(0.01)   # 0.01 Å floor prevents unit-vector blow-up under large noise
        edge_vector = gate_edge_vector.unsqueeze(1) * (
            edge_diff / safe_edge_dist.unsqueeze(-1)
        ).unsqueeze(-1)
        message_vector = message_vector + edge_vector

        residual_scalar = torch.zeros_like(node_scalar)
        residual_vector = torch.zeros_like(node_vector)
        message_scalar = message_scalar.to(residual_scalar.dtype)
        message_vector = message_vector.to(residual_vector.dtype)
        residual_scalar.index_add_(0, edge[:, 0], message_scalar)
        residual_vector.index_add_(0, edge[:, 0], message_vector)

        return node_scalar + residual_scalar, node_vector + residual_vector


class PainnUpdate(nn.Module):
    """PaiNN update function."""

    def __init__(self, node_size: int):
        super().__init__()
        self.update_U = nn.Linear(node_size, node_size)
        self.update_V = nn.Linear(node_size, node_size)

        self.update_mlp = nn.Sequential(
            nn.Linear(node_size * 2, node_size),
            nn.SiLU(),
            nn.Linear(node_size, node_size * 3),
        )

    def forward(self, node_scalar, node_vector):
        Uv = self.update_U(node_vector)
        Vv = self.update_V(node_vector)

        Vv_norm = safe_norm(Vv, dim=1)
        mlp_input = torch.cat((Vv_norm, node_scalar), dim=1)
        mlp_output = self.update_mlp(mlp_input)

        a_vv, a_sv, a_ss = torch.split(mlp_output, node_vector.shape[-1], dim=1)

        delta_v = a_vv.unsqueeze(1) * Uv
        inner_prod = torch.sum(Uv * Vv, dim=1)
        delta_s = a_sv * inner_prod + a_ss

        return node_scalar + delta_s, node_vector + delta_v


class PainnBackbone(nn.Module):
    """
    PaiNN backbone adapted for BondNet edge-feature extraction.

    Outputs per-edge invariant features h_ij for bond type prediction.

    Args:
        num_interactions:    Number of message-passing layers.
        hidden_state_size:   Dimension of node scalar features.
        cutoff:              Radius cutoff (Å).
        edge_embedding_size: Number of sinc RBF basis functions.
    """

    def __init__(
        self,
        num_interactions: int = 3,
        hidden_state_size: int = 128,
        cutoff: float = 5.0,
        edge_embedding_size: int = 20,
        atom_feature_size: int = 4,
        vector_norm_limit: float = 0.0
    ):
        super().__init__()

        self.cutoff = cutoff
        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.edge_embedding_size = edge_embedding_size
        self.vector_norm_limit = vector_norm_limit

        self.atom_embedding = nn.Embedding(119, hidden_state_size)

        self.message_layers = nn.ModuleList([
            PainnMessage(hidden_state_size, edge_embedding_size, cutoff)
            for _ in range(num_interactions)
        ])
        self.update_layers = nn.ModuleList([
            PainnUpdate(hidden_state_size)
            for _ in range(num_interactions)
        ])

        # Edge feature: concat(s_i, s_j, rbf, ||v_i||, ||v_j||) → h_ij
        # Input dim: 2*H + edge_embedding + 2
        edge_input_dim = 2 * hidden_state_size + edge_embedding_size + 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_state_size),
            nn.SiLU(),
            nn.Linear(hidden_state_size, hidden_state_size),
        )

    def forward(self, elems, coord, edge_index, edge_diff, edge_dist,
                local_geom=None, atom_features=None, num_atoms_per_mol=None):
        """
        Args:
            elems:         (N,) atomic numbers.
            coord:         (N, 3) atom positions.
            edge_index:    (E, 2) [receiver i, sender j].
            edge_diff:     (E, 3) r_j - r_i.
            edge_dist:     (E,) distances.
            local_geom:    ignored — kept for call-site compatibility.
            atom_features: ignored legacy argument kept for compatibility.

        Returns:
            node_scalar: (N, H)
            node_vector: (N, 3, H)
            edge_feat:   (E, H)
        """
        node_scalar = self.atom_embedding(elems)
        node_vector = torch.zeros(
            (coord.shape[0], 3, self.hidden_state_size),
            device=coord.device, dtype=coord.dtype,
        )

        for msg, upd in zip(self.message_layers, self.update_layers):
            node_scalar, node_vector = msg(
                node_scalar, node_vector, edge_index, edge_diff, edge_dist
            )
            node_scalar = sanitize_hidden(node_scalar)
            node_vector = sanitize_hidden(node_vector)
            node_scalar, node_vector = upd(node_scalar, node_vector)
            node_scalar = sanitize_hidden(node_scalar)
            node_vector = sanitize_hidden(node_vector)
            node_vector = limit_vector_norm(node_vector, self.vector_norm_limit)

        s_i = node_scalar[edge_index[:, 0]]
        s_j = node_scalar[edge_index[:, 1]]
        rbf = sinc_expansion(edge_dist, self.edge_embedding_size, self.cutoff)

        v_i_norm = safe_norm(node_vector[edge_index[:, 0]], dim=1)  # (E, H)
        v_j_norm = safe_norm(node_vector[edge_index[:, 1]], dim=1)  # (E, H)
        v_i_scalar = safe_norm(v_i_norm, dim=-1, keepdim=True)       # (E, 1)
        v_j_scalar = safe_norm(v_j_norm, dim=-1, keepdim=True)       # (E, 1)

        edge_input = torch.cat([s_i, s_j, rbf, v_i_scalar, v_j_scalar], dim=-1)
        edge_input = sanitize_hidden(edge_input)
        edge_feat = sanitize_hidden(self.edge_mlp(edge_input))

        return node_scalar, node_vector, edge_feat
