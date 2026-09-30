"""
BondNet: one-pass PaiNN bond perception.

The shared PaiNN backbone produces edge features for all candidate edges.

Two heads predict connectivity and heavy-heavy 4-class bond type for explicit-H graphs.

Hydrogen-heavy bonds are learned by the connectivity head and assigned single
without bond-type supervision.
"""

import torch
from torch import nn
from typing import Dict, List, Optional

from .painn_backbone import PainnBackbone
from .clof_backbone import ClofBackbone
from .stage1_connectivity import ConnectivityPredictor
from .stage3_bondtype import BondTypePredictor


class BondNet(nn.Module):
    """
    One-pass PaiNN bond perception from 3D molecular geometry.

    Stage 1: Connectivity, binary classifier on all candidate edges.
    Stage 3: Heavy-heavy bond type, 4-class {single=0, double=1, triple=2, aromatic=3}.

    Args:
        num_interactions:    Number of PaiNN message-passing layers.
        hidden_size:         Hidden dimension for node/edge features.
        cutoff:              Radius cutoff for candidate-edge graph (脜).
        edge_embedding_size: Number of sinc RBF basis functions.
        dropout:             Dropout rate for classifier MLPs.
    """

    def __init__(
        self,
        num_interactions: int = 3,
        hidden_size: int = 128,
        cutoff: float = 5.0,
        edge_embedding_size: int = 20,
        atom_feature_size: int = 4,
        dropout: float = 0.1,
        backbone: str = "painn",
        clof_coords_weight: float = 0.1,
        vector_norm_limit: float = 0.0,
        vector_rms_norm_scale: float = 0.0,
        # Legacy / experimental params 鈥?accepted but ignored for compatibility
        max_ring_size: int = 8,
        num_atom_types: int = 10,
        use_raw_edge_distance: bool = False,
        use_directional_projections: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.cutoff = cutoff
        self.backbone_name = str(backbone).lower()

        if self.backbone_name == "painn":
            self.backbone = PainnBackbone(
                num_interactions=num_interactions,
                hidden_state_size=hidden_size,
                cutoff=cutoff,
                edge_embedding_size=edge_embedding_size,
                atom_feature_size=atom_feature_size,
                vector_norm_limit=vector_norm_limit
            )
        elif self.backbone_name == "clof":
            self.backbone = ClofBackbone(
                num_interactions=num_interactions,
                hidden_state_size=hidden_size,
                cutoff=cutoff,
                edge_embedding_size=edge_embedding_size,
                atom_feature_size=atom_feature_size,
                coords_weight=clof_coords_weight,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown BondNet backbone: {backbone!r}")

        self.stage1 = ConnectivityPredictor(
            hidden_size=hidden_size,
            mlp_hidden=hidden_size // 2,
            dropout=dropout,
        )

        self.stage3 = BondTypePredictor(
            hidden_size=hidden_size,
            mlp_hidden=hidden_size // 2,
            num_bond_types=4,   # single / double / triple / aromatic
            dropout=dropout,
        )

    # ------------------------------------------------------------------ #
    # Forward (training mode)                                              #
    # ------------------------------------------------------------------ #

    def forward(self, batch: Dict) -> Dict:
        """
        Training forward pass 鈥?teacher forcing throughout.

        Stage 3 always receives ground-truth learned bonds (train_bond_mask), so
        its supervision is aligned with bond_type_train labels.
        End-to-end sequential inference is done via predict().

        Args:
            batch: dict with elems, coord, edge_index, edge_diff, edge_dist,
                   bond_mask (E,) bool full ground-truth connectivity.

        Returns:
            output dict with conn_logits, cls_logits (E_b, 4), bo_pred (0,).
        """
        elems = batch['elems']
        coord = batch['coord']
        edge_index = batch['edge_index']
        edge_diff = batch['edge_diff']
        edge_dist = batch['edge_dist']
        local_geom = batch.get('local_geom')
        atom_features = batch.get('atom_features')
        num_atoms_per_mol = batch.get('num_atoms_per_mol')
        # 鈹€鈹€ Backbone 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€ #
        node_scalar, node_vector, edge_feat = self.backbone(
            elems, coord, edge_index, edge_diff, edge_dist,
            local_geom=local_geom, atom_features=atom_features,
            num_atoms_per_mol=num_atoms_per_mol,
        )

        output: Dict = {}

        # 鈹€鈹€ Stage 1: Connectivity 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€ #
        conn_logits = self.stage1(edge_feat)
        output['conn_logits'] = conn_logits

        if batch.get('_connectivity_only', False):
            output['cls_logits'] = torch.zeros(0, 4, device=coord.device)
            output['bo_pred'] = torch.zeros(0, device=coord.device)
            output['edge_feat'] = edge_feat
            output['node_scalar'] = node_scalar
            output['node_vector'] = node_vector
            return output

        # 鈹€鈹€ Stage 3: 4-class bond type 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€ #
        # Always use ground-truth train_bond_mask so Stage 3 supervision aligns
        # with bond_type_train labels. Scheduled sampling is intentionally
        # avoided: it causes cls_logits / label size mismatch and produces
        # zero Stage 3 loss when Stage 1 predicts 0 bonds at init.
        train_bond_mask = batch.get('train_bond_mask', batch.get('bond_mask'))
        if train_bond_mask is not None:
            bonded_edge_feat = edge_feat[train_bond_mask]
        else:
            bonded_edge_feat = edge_feat

        if bonded_edge_feat.shape[0] > 0:
            cls_logits, bo_pred = self.stage3(bonded_edge_feat)
            output['cls_logits'] = cls_logits   # (E_b, 4)
            output['bo_pred'] = bo_pred          # (E_b,)
        else:
            output['cls_logits'] = torch.zeros(0, 4, device=coord.device)
            output['bo_pred'] = torch.zeros(0, device=coord.device)

        output['edge_feat'] = edge_feat
        output['node_scalar'] = node_scalar
        output['node_vector'] = node_vector

        return output

    # ------------------------------------------------------------------ #
    # Inference (sequential stage application with pruning)                #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict(
        self,
        batch: Dict,
        conn_threshold: float = 0.5,
        apply_ring_rules: bool = False,
    ) -> Dict:
        """
        End-to-end inference: coordinates 鈫?bond types.

        Returns:
            bond_mask:         (E,) boolean predicted connectivity
            bond_types:        (E_bonded,) int 鈥?0=single,1=double,2=triple,3=aromatic
            bond_orders:       (0,) empty placeholder for backward compatibility
            edge_index:        (E, 2) full candidate-edge indices
            bonded_edge_index: (E_bonded, 2) indices of predicted bonds
        """
        self.eval()

        elems = batch['elems']
        coord = batch['coord']
        edge_index = batch['edge_index']
        edge_diff = batch['edge_diff']
        edge_dist = batch['edge_dist']
        local_geom = batch.get('local_geom')
        atom_features = batch.get('atom_features')
        num_atoms_per_mol = batch.get('num_atoms_per_mol')
        node_scalar, node_vector, edge_feat = self.backbone(
            elems, coord, edge_index, edge_diff, edge_dist,
            local_geom=local_geom, atom_features=atom_features,
            num_atoms_per_mol=num_atoms_per_mol,
        )

        # Stage 1
        train_edge_mask = batch.get('train_edge_mask')
        if train_edge_mask is None:
            train_edge_mask = torch.ones(edge_feat.shape[0], dtype=torch.bool, device=edge_feat.device)
        learned_bond_mask = self.stage1.predict(edge_feat, threshold=conn_threshold) & train_edge_mask
        bond_mask = learned_bond_mask

        if bond_mask.sum() == 0:
            return {
                'bond_mask': bond_mask,
                'bond_types': torch.zeros(0, dtype=torch.long, device=coord.device),
                'bond_orders': torch.zeros(0, device=coord.device),
                'edge_index': edge_index,
                'bonded_edge_index': edge_index[:0],
                'learned_bond_mask': learned_bond_mask,
            }

        bonded_indices = torch.where(bond_mask)[0]
        bonded_edge_index = edge_index[bonded_indices]
        bond_orders = torch.zeros(bonded_indices.shape[0], device=coord.device)
        bond_types_full = torch.full((edge_index.shape[0],), -1, dtype=torch.long, device=coord.device)
        # H-X predicted bonds are connectivity-only and always single.
        bond_types_full[bonded_indices] = 0

        bonded_is_h = (elems[bonded_edge_index[:, 0]] == 1) | (elems[bonded_edge_index[:, 1]] == 1)
        hh_indices = bonded_indices[~bonded_is_h]
        hh_edge_index = edge_index[hh_indices]
        if hh_indices.numel() > 0:
            cls_logits, pred_bo = self.stage3(edge_feat[hh_indices])
            cls_logits = _symmetrize_logits(cls_logits, hh_edge_index)
            predicted_hh_types = cls_logits.argmax(dim=-1)
            bond_types_full[hh_indices] = predicted_hh_types
            if pred_bo.numel() == hh_indices.shape[0]:
                bond_orders[~bonded_is_h] = pred_bo

        if apply_ring_rules and hh_indices.numel() > 0:
            from ..inference.ring_rules import ChemicalRuleFilter
            rule_filter = ChemicalRuleFilter()
            predicted_hh_types = rule_filter(
                bond_types_full[hh_indices],
                hh_edge_index,
                elems=elems,
                n_atoms=int(elems.shape[0]),
            )
            bond_types_full[hh_indices] = predicted_hh_types

        return {
            'bond_mask': bond_mask,
            'bond_types': bond_types_full[bond_mask],
            'bond_orders': bond_orders,
            'edge_index': edge_index,
            'bonded_edge_index': bonded_edge_index,
            'learned_bond_mask': learned_bond_mask,
        }


# ------------------------------------------------------------------ #
# Symmetrization helper                                                 #
# ------------------------------------------------------------------ #

def _symmetrize_logits(
    cls_logits: torch.Tensor,   # (E_hh, 4)  heavy-heavy bonded edges only
    edge_index: torch.Tensor,   # (E_hh, 2)  [i, j] for each edge
) -> torch.Tensor:
    """
    Average cls_logits of i→j and j→i so both directed edges of an
    undirected bond get the same prediction.
    """
    E = edge_index.shape[0]
    if E == 0:
        return cls_logits
    i, j = edge_index[:, 0], edge_index[:, 1]
    n = int(max(i.max(), j.max()).item()) + 1
    # Map each directed edge (a, b) to a unique integer index a*n + b
    fwd_key = i * n + j
    lookup = torch.full((n * n,), -1, dtype=torch.long, device=edge_index.device)
    lookup.scatter_(0, fwd_key, torch.arange(E, device=edge_index.device))
    rev_pos = lookup[j * n + i]   # position of the reverse edge, -1 if absent

    sym = cls_logits.clone()
    has_rev = rev_pos >= 0
    if has_rev.any():
        pos = torch.where(has_rev)[0]
        rev = rev_pos[has_rev]
        avg = (cls_logits[pos] + cls_logits[rev]) * 0.5
        sym[pos] = avg
        sym[rev] = avg
    return sym


# ------------------------------------------------------------------ #
# BO-guided decoding                                                    #
# ------------------------------------------------------------------ #

# Map continuous BO to nearest discrete class: 0=single,1=double,2=triple,3=aromatic
_BO_BREAKPOINTS = torch.tensor([1.25, 1.75, 2.5])   # thresholds between classes
_BO_CLASS_ORDER = [0, 1, 2]   # single / double / triple (aromatic handled separately)


def _bo_guided_decode(
    cls_logits: torch.Tensor,   # (E_b, 4)
    bo_pred: torch.Tensor,      # (E_b,)
    confidence_thresh: float = 0.75,
) -> torch.Tensor:
    """
    Decode bond types using classification for high-confidence bonds and
    BO-snapping for low-confidence bonds.

    Strategy:
      - Compute softmax confidence (max class probability).
      - High confidence (>= threshold) 鈫?argmax (trust the classifier).
      - Low confidence  (<  threshold) 鈫?snap bo_pred to nearest class:
            BO < 1.25  鈫?single   (0)
            BO < 1.75  鈫?single/aromatic ambiguity 鈫?keep argmax
            BO < 2.5   鈫?double   (1)
            BO >= 2.5  鈫?triple   (2)
        Aromatic (class 3) is only assigned by the classifier, never by BO
        snapping, because BO alone cannot distinguish aromatic from delocalized.
    """
    probs = torch.softmax(cls_logits, dim=-1)          # (E_b, 4)
    confidence, argmax_types = probs.max(dim=-1)       # (E_b,), (E_b,)

    bond_types = argmax_types.clone()

    low_conf = confidence < confidence_thresh
    if not low_conf.any():
        return bond_types

    bo_lc = bo_pred[low_conf]

    # Snap to nearest class by BO value (excluding aromatic)
    snapped = torch.zeros_like(bo_lc, dtype=torch.long)
    snapped[bo_lc >= 1.75] = 1   # double
    snapped[bo_lc >= 2.50] = 2   # triple

    # For BO in [1.25, 1.75] (resonance zone): keep argmax 鈥?classifier knows better
    resonance_zone = (bo_lc >= 1.25) & (bo_lc < 1.75)
    snapped[resonance_zone] = argmax_types[low_conf][resonance_zone]

    bond_types[low_conf] = snapped
    return bond_types
