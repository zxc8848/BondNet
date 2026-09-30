import torch


_NORM_EPS = 1e-8


def compute_local_geometry_features(
    coord: torch.Tensor,
    edge_index: torch.Tensor,
    edge_dist: torch.Tensor,
    cutoff: float,
) -> torch.Tensor:
    """
    Compute per-atom geometry descriptors from a candidate-edge graph.

    Returns (N, 3): [planarity, avg_cos_angle, degree_norm].
    """
    N = coord.shape[0]
    device = coord.device
    dtype = coord.dtype

    if N == 0:
        return torch.zeros((0, 3), device=device, dtype=dtype)

    recv = edge_index[:, 0]
    send = edge_index[:, 1]
    mask = edge_dist < cutoff
    recv = recv[mask]
    send = send[mask]

    degree_count = torch.bincount(recv, minlength=N).to(dtype=dtype)
    degree_norm = degree_count / 6.0
    avg_cos = torch.full((N,), -0.33, device=device, dtype=dtype)
    planarity = torch.zeros((N,), device=device, dtype=dtype)

    if recv.numel() == 0:
        return torch.stack([planarity, avg_cos, degree_norm], dim=1)

    perm = torch.argsort(recv)
    recv_sorted = recv[perm]
    send_sorted = send[perm]
    edge_pos = torch.arange(recv_sorted.shape[0], device=device)
    group_start = torch.ones_like(recv_sorted, dtype=torch.bool)
    group_start[1:] = recv_sorted[1:] != recv_sorted[:-1]
    start_pos = torch.where(group_start, edge_pos, torch.zeros_like(edge_pos))
    start_pos = torch.cummax(start_pos, dim=0).values
    slot = edge_pos - start_pos

    max_degree = int(degree_count.max().item())
    neighbor_slots = torch.full((N, max_degree), -1, device=device, dtype=torch.long)
    neighbor_slots[recv_sorted, slot] = send_sorted

    valid = neighbor_slots >= 0
    safe_slots = neighbor_slots.clamp_min(0)
    neighbor_coords = coord[safe_slots]
    center_coords = coord.unsqueeze(1)
    vecs = (neighbor_coords - center_coords) * valid.unsqueeze(-1)

    norms = vecs.norm(dim=-1, keepdim=True).clamp(min=_NORM_EPS)
    unit = (vecs / norms) * valid.unsqueeze(-1)

    cos_mat = torch.matmul(unit, unit.transpose(1, 2))
    pair_mask = valid.unsqueeze(1) & valid.unsqueeze(2)
    tri_mask = torch.triu(
        torch.ones((max_degree, max_degree), device=device, dtype=torch.bool),
        diagonal=1,
    ).unsqueeze(0)
    pair_mask = pair_mask & tri_mask
    pair_count = pair_mask.sum(dim=(1, 2))
    cos_sum = (cos_mat * pair_mask.to(dtype)).sum(dim=(1, 2))
    has_angle_pairs = pair_count > 0
    avg_cos[has_angle_pairs] = cos_sum[has_angle_pairs] / pair_count[has_angle_pairs].to(dtype)

    has_plane = degree_count >= 3
    if has_plane.any():
        mean = (neighbor_coords * valid.unsqueeze(-1)).sum(dim=1, keepdim=True)
        mean = mean / degree_count.clamp_min(1).view(-1, 1, 1)
        centered = (neighbor_coords - mean) * valid.unsqueeze(-1)
        cov = torch.matmul(centered.transpose(1, 2), centered)
        eigvals = torch.linalg.eigvalsh(cov)
        min_eig = eigvals[:, 0].clamp_min(0)
        planarity[has_plane] = torch.sqrt(
            min_eig[has_plane] / degree_count[has_plane].clamp_min(1)
        )

    return torch.stack([planarity, avg_cos, degree_norm], dim=1)