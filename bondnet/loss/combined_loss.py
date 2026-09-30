"""
Combined BondNet training loss.

    L = BCE(conn) + L_focal(4-class discrete)

Continuous BO regression and BO-based valence regularization are disabled.
Historical constructor arguments are retained for CLI compatibility.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Optional

from .focal_loss import FocalLoss


class BondNetLoss(nn.Module):
    """
    Unified multi-task loss for BondNet.

    Args:
        pos_weight:        Positive-class weight for Stage 1 BCE (default 3.0).
        focal_gamma:       Focal loss gamma for Stage 3 (default 2.0).
        cls_weights:       (3,) per-class weight tensor for Stage 3 focal loss.
        lambda_bo:         Legacy BO-loss weight. Ignored.
        mu_valence:        Legacy valence-loss weight. Ignored.
        huber_delta:       Legacy BO-loss hyperparameter. Ignored.
    """

    def __init__(
        self,
        pos_weight: float = 3.0,
        focal_gamma: float = 2.0,
        conn_focal_gamma: float = 0.0,
        cls_weights: Optional[torch.Tensor] = None,
        hard_weight_bonus: float = 0.0,
        connectivity_only: bool = False,
        # Legacy params — accepted but unused
        lambda_bo: float = 1.0,
        mu_valence: float = 0.1,
        huber_delta: float = 0.5,
    ):
        super().__init__()
        self.hard_weight_bonus = hard_weight_bonus
        self.connectivity_only = connectivity_only
        self.conn_focal_gamma = conn_focal_gamma
        self.register_buffer('pos_weight', torch.tensor(pos_weight))
        self.focal = FocalLoss(gamma=focal_gamma, weight=cls_weights)
    def forward(self, output: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components.

        Required keys in `output`:
            conn_logits  (E,)       — Stage 1 connectivity logits
            cls_logits   (E_bond, 4)  — Stage 3 4-class logits {single,double,triple,arom}

        Required keys in `batch`:
            bond_exists       (E,)        — binary ground-truth connectivity
            bond_type_full    (E_bond,)   — discrete {0,1,2,3} labels (all bonds)
        Returns:
            Dict with individual loss components and 'total' loss.
        """
        losses: Dict[str, torch.Tensor] = {}
        device = output['conn_logits'].device

        # -------- Stage 1: Connectivity --------------------------------- #
        conn_logits = output['conn_logits']
        bond_exists = batch['bond_exists'].float().to(device)
        train_edge_mask = batch.get('train_edge_mask')
        if train_edge_mask is not None:
            train_edge_mask = train_edge_mask.to(device)
            conn_logits = conn_logits[train_edge_mask]
            bond_exists = bond_exists[train_edge_mask]

        if conn_logits.numel() > 0:
            conn_loss = F.binary_cross_entropy_with_logits(
                conn_logits,
                bond_exists,
                pos_weight=self.pos_weight.to(device),
                reduction='none',
            )
            if self.conn_focal_gamma > 0:
                with torch.no_grad():
                    prob = torch.sigmoid(conn_logits)
                    pt = prob * bond_exists + (1.0 - prob) * (1.0 - bond_exists)
                    focal = (1.0 - pt).pow(self.conn_focal_gamma)
                l_conn = (conn_loss * focal).mean()
            else:
                l_conn = conn_loss.mean()
        else:
            l_conn = torch.tensor(0.0, device=device)
        losses['conn'] = l_conn

        if self.connectivity_only:
            losses['cls'] = torch.tensor(0.0, device=device)
            losses['total'] = losses['conn']
            return losses

        # -------- Stage 3: 4-class discrete bond type ------------------- #
        cls_logits = output['cls_logits']          # (E_bond, 4)
        bond_type = batch.get(
            'bond_type_train_active',
            batch.get('bond_type_train', batch['bond_type_full']),
        ).long().to(device)

        if cls_logits.numel() > 0 and bond_type.numel() > 0:
            if self.hard_weight_bonus > 0.0:
                with torch.no_grad():
                    pred_cls = cls_logits.argmax(dim=-1)
                    wrong = (pred_cls != bond_type).float()
                    hard_w = 1.0 + self.hard_weight_bonus * wrong
                    hard_w = hard_w / hard_w.mean()
                per_sample = self.focal(cls_logits, bond_type, reduction='none')
                losses['cls'] = (per_sample * hard_w).mean()
            else:
                losses['cls'] = self.focal(cls_logits, bond_type)
        else:
            losses['cls'] = torch.tensor(0.0, device=device)

        losses['total'] = losses['conn'] + losses['cls']
        return losses
