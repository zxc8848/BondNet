"""
Ring detection for molecular graphs.

Uses BFS-based shortest-cycle finding: for every edge (u, v), find the
shortest path from v back to u that does not traverse (u, v). The resulting
cycle is a fundamental ring. Duplicate rings (same atom set) are deduplicated.

Returns:
  rings             — list of rings, each an ordered list of atom indices
  ring_edge_indices — for each ring, list of edge indices into the input
                      edge_index array that belong to that ring
"""

from collections import defaultdict, deque
from typing import List, Tuple, Optional


def detect_rings(
    edge_index,
    num_atoms: int,
    max_ring_size: int = 8,
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Detect simple rings in a molecular graph.

    Args:
        edge_index: (E, 2) array / list of int pairs — may contain both
                    directions (i→j) and (j→i) for undirected bonds.
        num_atoms:  Total number of atoms.
        max_ring_size: Rings larger than this are ignored.

    Returns:
        rings:             List[List[int]]  — atom indices in ring order.
        ring_edge_indices: List[List[int]]  — indices into edge_index per ring.
    """
    # ------------------------------------------------------------------ #
    # 1. Build adjacency list and deduplicated undirected edge list         #
    # ------------------------------------------------------------------ #
    # adj[i] = list of (j, original_edge_idx)
    adj: List[List[Tuple[int, int]]] = [[] for _ in range(num_atoms)]
    canonical_edges: List[Tuple[int, int]] = []   # (i, j) with i < j
    canonical_set: set = set()

    for eidx, pair in enumerate(edge_index):
        i, j = int(pair[0]), int(pair[1])
        if i == j:
            continue
        adj[i].append((j, eidx))
        adj[j].append((i, eidx))
        key = (min(i, j), max(i, j))
        if key not in canonical_set:
            canonical_set.add(key)
            canonical_edges.append((i, j))

    # ------------------------------------------------------------------ #
    # 2. For each undirected edge, find shortest cycle through it          #
    # ------------------------------------------------------------------ #
    rings: List[List[int]] = []
    ring_edge_indices: List[List[int]] = []
    seen_ring_keys: set = set()

    for src, dst in canonical_edges:
        path = _bfs_path_skip_edge(adj, src=dst, tgt=src,
                                    skip_u=src, skip_v=dst)
        if path is None:
            continue
        # ring_atoms = [src] + path  (path starts at dst, ends at src)
        ring_atoms = [src] + path
        if not (3 <= len(ring_atoms) <= max_ring_size):
            continue

        ring_key = tuple(sorted(ring_atoms))
        if ring_key in seen_ring_keys:
            continue
        seen_ring_keys.add(ring_key)

        edge_idx_list = _ring_edge_indices(adj, ring_atoms)
        rings.append(ring_atoms)
        ring_edge_indices.append(edge_idx_list)

    return rings, ring_edge_indices


def _bfs_path_skip_edge(
    adj: List[List[Tuple[int, int]]],
    src: int,
    tgt: int,
    skip_u: int,
    skip_v: int,
) -> Optional[List[int]]:
    """
    BFS shortest path from src to tgt, skipping the undirected edge (skip_u, skip_v).

    Returns the path [src, ..., tgt] on success, or None if unreachable.
    """
    queue: deque = deque([(src, [src])])
    visited = {src}

    while queue:
        node, path = queue.popleft()
        for neighbor, _ in adj[node]:
            # Skip the forbidden edge in both directions
            if (node == skip_u and neighbor == skip_v) or \
               (node == skip_v and neighbor == skip_u):
                continue
            if neighbor == tgt:
                return path + [tgt]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None


def _ring_edge_indices(
    adj: List[List[Tuple[int, int]]],
    ring_atoms: List[int],
) -> List[int]:
    """
    Return edge indices (into the original edge_index) for consecutive atom
    pairs in a ring.  Each consecutive pair (ring_atoms[k], ring_atoms[k+1])
    and the closing pair (ring_atoms[-1], ring_atoms[0]) are looked up.
    """
    # Build fast lookup: (i, j) -> edge_idx for atoms in this ring
    lookup: dict = {}
    for atom in ring_atoms:
        for neighbor, eidx in adj[atom]:
            lookup[(atom, neighbor)] = eidx

    edge_idx_list: List[int] = []
    n = len(ring_atoms)
    for k in range(n):
        a = ring_atoms[k]
        b = ring_atoms[(k + 1) % n]
        if (a, b) in lookup:
            edge_idx_list.append(lookup[(a, b)])
        elif (b, a) in lookup:
            edge_idx_list.append(lookup[(b, a)])
    return edge_idx_list


def build_ring_edge_map(
    rings: List[List[int]],
    ring_edge_indices: List[List[int]],
    bonded_indices,        # 1-D tensor of edge positions that are bonded
) -> List[List[int]]:
    """
    Remap ring edge indices (into the full edge_index) to positions within
    the bonded-edge array (which has length = len(bonded_indices)).

    Used in the inference pipeline to map aromatic ring predictions to
    bond_types positions.

    Args:
        rings:            List of rings (atom lists, not needed here).
        ring_edge_indices: List of edge-index lists per ring.
        bonded_indices:   1-D int tensor — bonded_indices[k] = original edge pos.

    Returns:
        ring_edge_map: List[List[int]] — indices into bond_types array per ring.
    """
    import torch
    # Invert bonded_indices: original_edge_pos -> bonded_position
    inv_map: dict = {}
    for k, orig in enumerate(bonded_indices.tolist()):
        inv_map[orig] = k

    ring_edge_map: List[List[int]] = []
    for edge_ids in ring_edge_indices:
        mapped = [inv_map[e] for e in edge_ids if e in inv_map]
        ring_edge_map.append(mapped)
    return ring_edge_map
