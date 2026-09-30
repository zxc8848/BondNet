"""
Valence-constrained decoding for BondNet Stage-2 bond-type logits.

STATUS: EXPERIMENTAL -- not used for any reported result (default OFF via
``evaluate.py --valence_decode``).  On clean inputs this decoder currently
*reduces* exact-match molecule validity (~94.6% -> ~82% at sigma=0) because
aromatic bonds are scored with an order of 1.5, which over-estimates the
valence load on some correct aromatic/fused-ring atoms and triggers spurious
downgrades of neighbouring double bonds.  It does help at high noise
(sigma>=0.15).  The recommended fix before using it is to skip any atom that
carries an aromatic bond (as the simpler ``ring_rules.ChemicalRuleFilter``
does) so that only non-aromatic over-valence is repaired.  Left in the tree as
a starting point for that future work.

Motivation
----------
Molecule-level validity requires *every* heavy-heavy bond order to be correct.
Because heavy-atom valences are a strong, noise-invariant prior (a neutral
carbon has four bonds, a neutral oxygen two, ...), we can use them to repair the
independent per-bond predictions of Stage 2.  When the argmax assignment makes
an atom exceed its valence budget, at least one incident bond is over-ordered;
downgrading it usually moves the prediction back toward the true (valence-legal)
assignment.

Improvements over the earlier ``ring_rules.ChemicalRuleFilter``:

1. **Confidence-aware.**  When an atom is over its budget we downgrade the bond
   with the *smallest model-confidence loss*, not simply the highest-order bond.
   Because a softmax denominator cancels within a single edge, the log-prob loss
   of moving edge ``e`` from class ``c`` to class ``c'`` is exactly
   ``logit[e, c] - logit[e, c']`` -- so raw logits suffice.
2. **Hydrogen-aware.**  Each heavy atom's budget for heavy-heavy bonds is its
   total valence minus the number of attached hydrogens, so violations that the
   heavy-only sum would hide are caught.
3. **Aromatic-aware without skipping.**  Aromatic bonds contribute ``1.5`` to the
   budget and are never downgraded (they are typically high-confidence ring
   predictions and the ground truth labels them aromatic), but the atom's other
   bonds can still be repaired.

The numeric core (:func:`decode_core`) is pure Python and has no torch or numpy
dependency, so it can be unit-tested in isolation.  :class:`ValenceConstrainedDecoder`
is a thin torch wrapper used inside ``evaluate.py``.

Bond-type encoding: 0=single, 1=double, 2=triple, 3=aromatic.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

# MAXIMUM allowed valences (union over common charge states), NOT the neutral
# valence.  Using the maximum is essential: many drug-like groups place a
# high-order bond on an atom whose neutral valence is lower (nitro/N-oxide/
# ammonium N with 4 bonds, sulfonyl S with 6, phosphoryl P with 5, ...).  A
# neutral table (N=3, O=2) flags these correct bonds as violations and wrongly
# downgrades them, which destroys accuracy on clean inputs.  With the maximum
# valence, only genuinely impossible assignments (exceeding every charge state)
# are repaired, so correct predictions are left untouched.
_VALENCE_TABLE: Dict[int, int] = {
    1: 1,    # H
    5: 4,    # B  (borate)
    6: 4,    # C
    7: 4,    # N  (ammonium / nitro / N-oxide)
    8: 3,    # O  (oxocarbenium / protonated)
    9: 1,    # F
    14: 4,   # Si
    15: 5,   # P  (phosphoryl / phosphate)
    16: 6,   # S  (sulfonyl / sulfate)
    17: 1,   # Cl
    35: 1,   # Br
    53: 1,   # I
}
_DEFAULT_VALENCE = 4

# Bond-order contribution of each class label.
_BO: Dict[int, float] = {0: 1.0, 1: 2.0, 2: 3.0, 3: 1.5}
# Legal single-step downgrades (aromatic is intentionally excluded).
_DOWNGRADE: Dict[int, int] = {2: 1, 1: 0}


def _valence(z: int) -> int:
    return _VALENCE_TABLE.get(int(z), _DEFAULT_VALENCE)


def _max_softmax(logit_row: Sequence[float]) -> float:
    """Max soft-max probability of a logit row (edge confidence). Pure Python."""
    m = max(logit_row)
    exps = [math.exp(float(x) - m) for x in logit_row]
    s = sum(exps)
    return (max(exps) / s) if s > 0 else 0.0


# Sentinel used for a pruned (removed) edge -- caller must drop these from the
# predicted bond set. -1 is also what the evaluate.py pipeline already uses for
# "no predicted bond", so it flows through unchanged.
PRUNED = -1


def decode_core(
    types: List[int],
    logits: Sequence[Sequence[float]],
    edges: Sequence[Tuple[int, int]],
    elems: Sequence[int],
    n_atoms: int,
    h_counts: Optional[Sequence[int]] = None,
    tol: float = 0.4,
    max_passes: Optional[int] = None,
    conf_gate: float = 1.0,
    allow_prune: bool = False,
) -> List[int]:
    """Confidence-aware, hydrogen-aware valence repair (pure Python).

    Two repair operations are applied to atoms whose heavy-heavy bond-order load
    exceeds their valence budget, always choosing the incident edge with the
    smallest model-confidence loss:

    * **downgrade** a non-aromatic multiple bond one step (triple->double->single);
    * **prune** (only if ``allow_prune``) a non-aromatic single bond, i.e. remove a
      spurious edge entirely (label -> :data:`PRUNED`). This is what repairs the
      *extra bonds* that inflate an atom's coordination under coordinate noise, and
      is the operation that lifts the strict full-graph (HH-graph) exact-match rate.

    Only edges whose max-soft-max confidence is **below** ``conf_gate`` are eligible
    to be modified, so confident (typically correct) predictions are never touched;
    ``conf_gate=1.0`` makes every edge eligible (original behaviour). Aromatic bonds
    are never downgraded or pruned.

    Args:
        types:    length-``E`` list of predicted class labels in ``{0,1,2,3}`` for
                  each *directed* heavy-heavy edge (both ``i->j`` and ``j->i``).
        logits:   ``E x 4`` model logits aligned with ``types`` (symmetrized).
        edges:    length-``E`` list of ``(i, j)`` atom indices per directed edge.
        elems:    length-``N`` atomic numbers.
        n_atoms:  number of atoms ``N``.
        h_counts: length-``N`` attached-hydrogen count per atom; ``None`` -> zeros.
        tol:      slack added to each atom's budget before flagging a violation.
        max_passes: iteration cap (defaults to ``n_atoms``).
        conf_gate: only edges with confidence < this value may be modified.
        allow_prune: enable single-bond pruning (extra-bond removal).

    Returns:
        Length-``E`` list of corrected class labels; pruned edges carry
        :data:`PRUNED` (``-1``) and must be dropped from the predicted bond set.
    """
    E = len(types)
    if E == 0:
        return list(types)

    corrected: List[int] = list(types)
    edges = [(int(a), int(b)) for (a, b) in edges]
    if h_counts is None:
        h_counts = [0] * n_atoms

    # Per-edge confidence for gating.
    conf = [_max_softmax(logits[e]) for e in range(E)]

    # Heavy-heavy valence budget per atom = total valence - attached H.
    budget: List[float] = []
    for a in range(n_atoms):
        b = _valence(int(elems[a])) - int(h_counts[a])
        budget.append(float(b if b > 0 else 0))

    # Reverse-edge lookup so both directions of a bond move together.
    edge_pos: Dict[Tuple[int, int], int] = {}
    for pos, (ai, aj) in enumerate(edges):
        edge_pos[(ai, aj)] = pos

    def _apply(e: int, new: int) -> None:
        corrected[e] = new
        ai, aj = edges[e]
        e_rev = edge_pos.get((aj, ai))
        if e_rev is not None:
            corrected[e_rev] = new

    def _eligible(e: int) -> bool:
        return conf[e] < conf_gate

    # Incident directed edges (as source) per atom.
    incident: List[List[int]] = [[] for _ in range(n_atoms)]
    for e, (ai, aj) in enumerate(edges):
        if 0 <= ai < n_atoms:
            incident[ai].append(e)

    def _load(a: int) -> float:
        return sum(_BO[corrected[e]] for e in incident[a] if corrected[e] != PRUNED)

    passes = n_atoms if max_passes is None else max_passes
    for _ in range(max(1, passes)):
        violated = [a for a in range(n_atoms) if _load(a) > budget[a] + tol]
        if not violated:
            break
        # Fix the most-violated atom first for stability.
        violated.sort(key=lambda a: _load(a) - budget[a], reverse=True)

        changed = False
        for a in violated:
            load_a = _load(a)
            while load_a > budget[a] + tol:
                # 1) Prefer a min-confidence-loss downgrade of a multiple bond.
                best = None  # (loss, edge, new_label, delta_load)
                for e in incident[a]:
                    cur = corrected[e]
                    if cur == PRUNED or cur not in _DOWNGRADE or not _eligible(e):
                        continue
                    new = _DOWNGRADE[cur]
                    loss = float(logits[e][cur]) - float(logits[e][new])
                    if best is None or loss < best[0]:
                        best = (loss, e, new, _BO[cur] - _BO[new])
                # 2) Otherwise, if pruning is allowed, remove the least-confident
                #    incident single bond (a spurious extra edge).
                if best is None and allow_prune:
                    prune_best = None  # (confidence, edge)
                    for e in incident[a]:
                        cur = corrected[e]
                        if cur == 0 and _eligible(e):   # only prune single bonds
                            if prune_best is None or conf[e] < prune_best[0]:
                                prune_best = (conf[e], e)
                    if prune_best is not None:
                        _, e = prune_best
                        _apply(e, PRUNED)
                        load_a -= _BO[0]
                        changed = True
                        continue
                if best is None:
                    break  # nothing eligible to repair at this atom
                _, e, new, delta = best
                _apply(e, new)
                load_a -= delta
                changed = True

        if not changed:
            break

    return corrected


class ValenceConstrainedDecoder:
    """Torch wrapper around :func:`decode_core` for use in ``evaluate.py``."""

    def __init__(self, tol: float = 0.4, max_passes: Optional[int] = None,
                 conf_gate: float = 1.0, allow_prune: bool = False):
        self.tol = tol
        self.max_passes = max_passes
        self.conf_gate = conf_gate
        self.allow_prune = allow_prune

    def __call__(
        self,
        bond_types,            # (E,) long  -- argmax labels for directed HH edges
        cls_logits,            # (E, 4) float
        bonded_edge_index,     # (E, 2) long -- atom indices of the HH edges
        elems,                 # (N,) long   -- atomic numbers
        n_atoms: int,
        h_counts=None,         # (N,) long   -- attached H per atom, or None
    ):
        import torch

        if bond_types.numel() == 0:
            return bond_types

        types = bond_types.detach().cpu().tolist()
        logits = cls_logits.detach().cpu().tolist()
        edges = bonded_edge_index.detach().cpu().tolist()
        elems_l = elems.detach().cpu().tolist()
        hc = None if h_counts is None else h_counts.detach().cpu().tolist()

        corrected = decode_core(
            types=types,
            logits=logits,
            edges=edges,
            elems=elems_l,
            n_atoms=int(n_atoms),
            h_counts=hc,
            tol=self.tol,
            max_passes=self.max_passes,
            conf_gate=self.conf_gate,
            allow_prune=self.allow_prune,
        )
        # Pruned edges carry PRUNED (-1); the caller drops them from the bond set.
        return torch.tensor(corrected, dtype=bond_types.dtype, device=bond_types.device)


def attached_hydrogen_counts(pred_edge_index, elems, n_atoms: int):
    """Count predicted attached hydrogens per atom from all predicted bonded edges.

    Args:
        pred_edge_index: (E_all, 2) atom indices of *all* predicted bonds
                         (heavy-heavy and H-X); may contain both directions.
        elems:           (N,) atomic numbers.
        n_atoms:         number of atoms.

    Returns:
        torch LongTensor (N,) with the attached-H count per atom.
    """
    import torch

    counts = [0] * int(n_atoms)
    seen = set()
    ei = pred_edge_index.detach().cpu().tolist()
    el = elems.detach().cpu().tolist()
    for a, b in ei:
        a, b = int(a), int(b)
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        a_is_h = el[a] == 1
        b_is_h = el[b] == 1
        if a_is_h and not b_is_h:
            counts[b] += 1
        elif b_is_h and not a_is_h:
            counts[a] += 1
    return torch.tensor(counts, dtype=torch.long, device=elems.device)
