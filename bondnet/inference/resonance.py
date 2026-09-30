"""
Resonance-aware scoring for symmetric delocalized oxy-groups.

Motivation
----------
For symmetric delocalized groups -- nitro R-N(=O)-O and carboxyl/carboxylate
R-C(=O)-O -- the two X-O bonds are chemically equivalent (bond order ~1.5 each),
but the discrete ground-truth label arbitrarily assigns one single and one double.
A model that sees two near-identical short X-O environments naturally predicts both
as double; under strict per-bond scoring exactly one of them is then counted wrong,
even though the prediction represents the same delocalized system. Empirically these
groups (dominated by N-O) account for the majority of BondNet's residual bond-type
errors on clean drug-like molecules.

This module provides a *resonance-aware* scoring transform: within a detected
symmetric oxy-group it (i) canonicalizes the predicted bond orders to the group's
correct total order and (ii) compares to the true orders as a swap-invariant
multiset, so any assignment that represents the same delocalized system counts as
correct. It is a SCORING aid (it uses the ground-truth labels) and is intended to be
reported *alongside* the strict metric, not as a replacement.

Bond-type labels: 0=single, 1=double, 2=triple, 3=aromatic.
"""
from collections import defaultdict
from typing import Dict, List, Tuple

_ORDER = {0: 1, 1: 2, 2: 3}   # non-aromatic label -> integer bond order


def find_symmetric_oxy_groups(bonds, elems) -> List[Dict]:
    """Detect nitro and carboxyl/carboxylate groups from a heavy-atom bond graph.

    Args:
        bonds: iterable of ``(i, j)`` undirected heavy-heavy bonds (ground truth).
        elems: mapping/sequence atom index -> atomic number.

    Returns:
        list of ``{'center': c, 'edges': [(a,b), ...], 'total': int}`` where
        ``edges`` are sorted atom-index tuples and ``total`` is the correct sum of
        integer bond orders over the group (3 for both nitro and carboxyl).
    """
    # Use sets so that directed edge lists (both i->j and j->i, as produced by
    # the candidate graph) do not double-count neighbours -- otherwise a terminal
    # oxygen would appear to have heavy degree 2 and no group would be detected.
    adj = defaultdict(set)
    for i, j in bonds:
        adj[int(i)].add(int(j))
        adj[int(j)].add(int(i))
    heavy_deg = {a: len(ns) for a, ns in adj.items()}

    groups = []
    for c, ns in adj.items():
        z = int(elems[c])
        o_neigh = [n for n in ns if int(elems[n]) == 8]
        # terminal O = bonded to only the center among heavy atoms
        term_o = [o for o in o_neigh if heavy_deg.get(o, 0) == 1]
        if z == 7 and len(term_o) == 2:          # nitro  R-N(=O)-O
            edges = [tuple(sorted((c, o))) for o in term_o]
            groups.append({'center': c, 'edges': edges, 'total': 3})
        elif z == 6 and len(term_o) == 2:        # carboxyl / carboxylate
            edges = [tuple(sorted((c, o))) for o in term_o]
            groups.append({'center': c, 'edges': edges, 'total': 3})
    return groups


def _canonicalize_two(orders: List[int], total: int) -> List[int]:
    """Force a two-bond group to the given integer total (e.g. 3 -> {2, 1})."""
    if sum(orders) == total:
        return list(orders)
    hi = 0 if orders[0] >= orders[1] else 1
    out = [1, 1]
    out[hi] = total - 1
    return out


def resonance_align(pred_label: Dict[Tuple[int, int], int],
                    true_label: Dict[Tuple[int, int], int],
                    bonds, elems):
    """Return a copy of ``pred_label`` with resonance-correct group bonds set equal
    to ``true_label``.

    A group is resonance-correct when the predicted orders, after canonicalization
    to the group's total, match the true orders as a multiset (swap-invariant).
    Groups containing an aromatic bond are left untouched.
    """
    pred = dict(pred_label)
    groups = find_symmetric_oxy_groups(bonds, elems)
    for g in groups:
        edges = g['edges']
        if any(pred.get(e) == 3 or true_label.get(e) == 3 for e in edges):
            continue
        try:
            p_ord = [_ORDER[pred[e]] for e in edges]
            t_ord = [_ORDER[true_label[e]] for e in edges]
        except KeyError:
            continue
        canon = _canonicalize_two(p_ord, g['total'])
        if sorted(canon) == sorted(t_ord):
            for e in edges:
                pred[e] = true_label[e]
    return pred, groups


def resonance_aware_predictions(pred_for_true, true_types, hh_edge_index, elems):
    """Torch wrapper for use in ``evaluate.py``.

    Args:
        pred_for_true: (E_b,) long tensor of predicted labels for true HH bonds.
        true_types:    (E_b,) long tensor of true labels for the same bonds.
        hh_edge_index: (E_b, 2) long tensor of atom indices for those bonds.
        elems:         (N,) long tensor of atomic numbers.

    Returns:
        (E_b,) long tensor: a copy of ``pred_for_true`` with resonance-correct
        symmetric oxy-group bonds set equal to ``true_types``.
    """
    import torch

    if pred_for_true.numel() == 0:
        return pred_for_true

    ei = hh_edge_index.detach().cpu().tolist()
    p = pred_for_true.detach().cpu().tolist()
    t = true_types.detach().cpu().tolist()
    el = elems.detach().cpu().tolist()

    pred_map, true_map, bonds = {}, {}, []
    key_at = []
    for (i, j), pl, tl in zip(ei, p, t):
        k = (int(i), int(j)) if i < j else (int(j), int(i))
        pred_map[k] = int(pl)
        true_map[k] = int(tl)
        bonds.append(k)
        key_at.append(k)

    aligned, _ = resonance_align(pred_map, true_map, bonds, el)
    out = [aligned.get(k, p[idx]) for idx, k in enumerate(key_at)]
    return torch.tensor(out, dtype=pred_for_true.dtype, device=pred_for_true.device)
