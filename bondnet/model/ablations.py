"""
Ablation model variants for BondNet (Proposal Section 6.5).

Each ablation removes or disables one component to isolate its contribution:

  FlatBondNet          — No hierarchy: single 5-class classifier (no-bond/single/
                         double/triple/aromatic) on all edges.  Tests value of
                         problem decomposition.

  DistanceOnlyBondNet  — Same PaiNN backbone but edge features use ONLY the
                         sinc-RBF distance encoding (no atom-type embeddings,
                         no angular info, no multi-hop context). Tests whether
                         context beyond distance matters.

  NoBOBondNet          — Stage 3 has no continuous BO regression head.
                         Tests multi-task rebalancing effect.

  NoAromaticityBondNet — No Stage 2; aromaticity is predicted at the bond level
                         as a 4th class in Stage 3. Tests Kekulé ambiguity
                         resolution.

All ablations share the same PaiNN backbone and training setup as the full
BondNet model, making comparisons fair.
"""

import torch
from torch import nn
from typing import Dict, Optional

from .painn_backbone import PainnBackbone, sinc_expansion, cosine_cutoff
from .stage1_connectivity import ConnectivityPredictor
from .stage3_bondtype import BondTypePredictor


# ================================================================== #
# 1. Flat 5-class classifier (no hierarchy)                            #
# ================================================================== #

class FlatBondNet(nn.Module):
    """
    Flat 5-class bond type classifier.

    Predicts {no-bond, single, double, triple, aromatic} in a single
    classification step — no hierarchical decomposition.

    The 5 classes:
      0: no bond
      1: single
      2: double
      3: triple
      4: aromatic
    """

    NUM_CLASSES = 5

    def __init__(
        self,
        num_interactions: int = 4,
        hidden_size: int = 128,
        cutoff: float = 5.0,
        edge_embedding_size: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cutoff = cutoff

        self.backbone = PainnBackbone(
            num_interactions=num_interactions,
            hidden_state_size=hidden_size,
            cutoff=cutoff,
            edge_embedding_size=edge_embedding_size,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, self.NUM_CLASSES),
        )

    def forward(self, batch: Dict) -> Dict:
        _, _, edge_feat = self.backbone(
            batch['elems'], batch['coord'],
            batch['edge_index'], batch['edge_diff'], batch['edge_dist'],
        )
        logits = self.classifier(edge_feat)  # (E, 5)
        return {'flat_logits': logits, 'edge_feat': edge_feat}

    @torch.no_grad()
    def predict(self, batch: Dict) -> torch.Tensor:
        out = self.forward(batch)
        return out['flat_logits'].argmax(dim=-1)  # (E,) in {0,1,2,3,4}


class FlatBondNetLoss(nn.Module):
    """
    Cross-entropy loss for FlatBondNet.

    Labels:
      bond_exists = 0 → class 0 (no-bond)
      bond_exists = 1 AND bond_type_full = t → class t+1  (single→1, double→2, triple→3, arom→4)
    """

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        from .stage3_bondtype import BondTypePredictor
        from ..loss.focal_loss import FocalLoss
        self.focal = FocalLoss(gamma=gamma, weight=weight)

    def build_flat_labels(self, batch: Dict, device) -> torch.Tensor:
        """Build (E,) flat class labels from batch."""
        bond_exists = batch['bond_exists'].bool()     # (E,)
        bond_type_full = batch.get('bond_type_full')  # (E_b,) 0/1/2/3
        E = bond_exists.shape[0]

        labels = torch.zeros(E, dtype=torch.long, device=device)

        if bond_type_full is not None:
            # Map bonded edges to classes 1–4
            bonded_idx = torch.where(bond_exists)[0]
            n = min(len(bonded_idx), len(bond_type_full))
            labels[bonded_idx[:n]] = bond_type_full[:n].long().to(device) + 1

        return labels

    def forward(self, output: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        device = output['flat_logits'].device
        labels = self.build_flat_labels(batch, device)
        loss = self.focal(output['flat_logits'], labels)
        return {'total': loss, 'cls': loss}


# ================================================================== #
# 2. Distance-only backbone (no atom types, no angular info)           #
# ================================================================== #

class DistanceOnlyEdgeMLP(nn.Module):
    """
    Edge feature extractor that uses ONLY the sinc-RBF distance encoding.

    Completely removes:
      - Atom-type embeddings
      - Vector (angular) features from PaiNN
      - Any multi-hop context

    This isolates the contribution of local chemical context vs. raw distance.
    """

    def __init__(self, edge_embedding_size: int = 20, hidden_size: int = 128, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff
        self.edge_embedding_size = edge_embedding_size

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, edge_dist: torch.Tensor) -> torch.Tensor:
        rbf = sinc_expansion(edge_dist, self.edge_embedding_size, self.cutoff)
        cutoff_weight = cosine_cutoff(edge_dist, self.cutoff)
        rbf = rbf * cutoff_weight.unsqueeze(-1)
        return self.edge_mlp(rbf)  # (E, hidden_size)


class DistanceOnlyBondNet(nn.Module):
    """
    BondNet variant with distance-only edge features.

    Tests the key hypothesis: does local chemical *context* beyond
    pairwise distance improve bond perception?

    Architecture: same three-stage hierarchy as BondNet, but the backbone
    is replaced by a simple RBF → MLP that sees only d_ij.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        cutoff: float = 5.0,
        edge_embedding_size: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.hidden_size = hidden_size

        self.distance_encoder = DistanceOnlyEdgeMLP(
            edge_embedding_size=edge_embedding_size,
            hidden_size=hidden_size,
            cutoff=cutoff,
        )

        self.stage1 = ConnectivityPredictor(hidden_size, hidden_size // 2, dropout)
        self.stage3 = BondTypePredictor(hidden_size, hidden_size // 2, 3, dropout)

    def forward(self, batch: Dict) -> Dict:
        edge_feat = self.distance_encoder(batch['edge_dist'])  # (E, H)

        output: Dict = {}
        output['conn_logits'] = self.stage1(edge_feat)

        if 'bond_mask' in batch and batch['bond_mask'] is not None:
            bonded_feat = edge_feat[batch['bond_mask']]
        else:
            bonded_feat = edge_feat

        if 'non_aromatic_mask' in batch and batch['non_aromatic_mask'] is not None:
            non_arom_feat = bonded_feat[batch['non_aromatic_mask']]
        else:
            non_arom_feat = bonded_feat

        if non_arom_feat.shape[0] > 0:
            cls_logits, bo_pred = self.stage3(non_arom_feat)
        else:
            cls_logits = torch.zeros(0, 3, device=batch['coord'].device)
            bo_pred = torch.zeros(0, device=batch['coord'].device)

        output['cls_logits'] = cls_logits
        output['bo_pred'] = bo_pred

        if bonded_feat.shape[0] > 0:
            _, bo_pred_all = self.stage3(bonded_feat)
        else:
            bo_pred_all = torch.zeros(0, device=batch['coord'].device)
        output['bo_pred_all'] = bo_pred_all

        output['edge_feat'] = edge_feat
        return output

    @torch.no_grad()
    def predict(self, batch: Dict, conn_threshold: float = 0.5) -> Dict:
        self.eval()
        edge_feat = self.distance_encoder(batch['edge_dist'])
        bond_mask = self.stage1.predict(edge_feat, threshold=conn_threshold)
        bonded_indices = torch.where(bond_mask)[0]

        if bonded_indices.shape[0] == 0:
            return {
                'bond_mask': bond_mask,
                'bond_types': torch.zeros(0, dtype=torch.long),
                'bond_orders': torch.zeros(0),
            }

        bonded_feat = edge_feat[bonded_indices]
        pred_types, bond_orders = self.stage3.predict(bonded_feat)
        return {
            'bond_mask': bond_mask,
            'bond_types': pred_types,
            'bond_orders': bond_orders,
        }


# ================================================================== #
# 3. No continuous BO regression                                       #
# ================================================================== #

class NoBOBondNet(nn.Module):
    """
    BondNet without the continuous bond order regression head.

    Stage 3 only has the discrete classification head.
    Tests the multi-task rebalancing effect of continuous BO.
    """

    def __init__(
        self,
        num_interactions: int = 4,
        hidden_size: int = 128,
        cutoff: float = 5.0,
        edge_embedding_size: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cutoff = cutoff

        self.backbone = PainnBackbone(
            num_interactions=num_interactions,
            hidden_state_size=hidden_size,
            cutoff=cutoff,
            edge_embedding_size=edge_embedding_size,
        )
        self.stage1 = ConnectivityPredictor(hidden_size, hidden_size // 2, dropout)

        # Stage 3: classification only (no regression head)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3),  # single/double/triple
        )

    def forward(self, batch: Dict) -> Dict:
        _, _, edge_feat = self.backbone(
            batch['elems'], batch['coord'],
            batch['edge_index'], batch['edge_diff'], batch['edge_dist'],
        )

        output: Dict = {}
        output['conn_logits'] = self.stage1(edge_feat)

        if 'bond_mask' in batch and batch['bond_mask'] is not None:
            bonded_feat = edge_feat[batch['bond_mask']]
        else:
            bonded_feat = edge_feat

        if 'non_aromatic_mask' in batch and batch['non_aromatic_mask'] is not None:
            non_arom_feat = bonded_feat[batch['non_aromatic_mask']]
        else:
            non_arom_feat = bonded_feat

        output['cls_logits'] = self.cls_head(non_arom_feat) if non_arom_feat.shape[0] > 0 \
            else torch.zeros(0, 3, device=batch['coord'].device)
        output['bo_pred'] = torch.zeros(0, device=batch['coord'].device)
        output['bo_pred_all'] = torch.zeros(0, device=batch['coord'].device)
        output['edge_feat'] = edge_feat
        return output


# ================================================================== #
# 4. Bond-level aromaticity (no ring-level Stage 2)                    #
# ================================================================== #

class NoRingBondNet(nn.Module):
    """
    BondNet without ring-level Stage 2.

    Aromaticity is predicted as a 4th bond type in Stage 3 (single/double/
    triple/aromatic).  No ring detection or ring-level aggregation.

    Tests: does ring-level aromaticity prediction eliminate Kekulé ambiguity
    compared to bond-level aromaticity prediction?
    """

    def __init__(
        self,
        num_interactions: int = 4,
        hidden_size: int = 128,
        cutoff: float = 5.0,
        edge_embedding_size: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cutoff = cutoff

        self.backbone = PainnBackbone(
            num_interactions=num_interactions,
            hidden_state_size=hidden_size,
            cutoff=cutoff,
            edge_embedding_size=edge_embedding_size,
        )
        self.stage1 = ConnectivityPredictor(hidden_size, hidden_size // 2, dropout)

        # Stage 3: 4-class (single/double/triple/aromatic) + BO regression
        self.shared_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 4),   # {single, double, triple, aromatic}
        )
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus(),
        )

    def forward(self, batch: Dict) -> Dict:
        _, _, edge_feat = self.backbone(
            batch['elems'], batch['coord'],
            batch['edge_index'], batch['edge_diff'], batch['edge_dist'],
        )

        output: Dict = {}
        output['conn_logits'] = self.stage1(edge_feat)

        if 'bond_mask' in batch and batch['bond_mask'] is not None:
            bonded_feat = edge_feat[batch['bond_mask']]
        else:
            bonded_feat = edge_feat

        if bonded_feat.shape[0] > 0:
            shared = self.shared_mlp(bonded_feat)
            cls_logits = self.cls_head(shared)   # (E_b, 4)
            bo_pred = self.reg_head(shared).squeeze(-1)
        else:
            cls_logits = torch.zeros(0, 4, device=batch['coord'].device)
            bo_pred = torch.zeros(0, device=batch['coord'].device)

        output['cls_logits'] = cls_logits
        output['bo_pred'] = bo_pred
        output['bo_pred_all'] = bo_pred
        output['edge_feat'] = edge_feat
        return output
