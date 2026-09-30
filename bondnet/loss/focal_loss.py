"""
Multi-class focal loss (weighted-CE variant) for bond-type classification.

Implemented form (per sample, target class y, class weight w_y):

    ce   = -w_y * log(p_y)                  # class-weighted cross-entropy
    p'   = exp(-ce) = p_y ** w_y            # modulating probability
    loss = (1 - p_y ** w_y) ** gamma * ce

The class weight therefore enters both the loss magnitude and the focusing
factor. This differs from the alpha-balanced focal loss of Lin et al. (2017),
    FL = -alpha_y * (1 - p_y) ** gamma * log(p_y),
in which the weight multiplies the loss only once. For w_y > 1 well-classified
samples are down-weighted less than in that form, and for w_y < 1 more.
With weight=None the two forms coincide. ce is clipped at 100 for numerical
stability; reduction='mean' averages over samples (not over summed weights).
All BondNet results reported in the paper use this implemented variant.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Multi-class focal loss.

    Args:
        gamma:   Focusing parameter.  gamma=0 recovers standard cross-entropy.
        weight:  (C,) class weights applied inside the cross-entropy (see module
                 docstring: they also enter the focusing factor).  If None, no
                 per-class weighting.
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if weight is not None:
            self.register_buffer('weight', weight.float())
        else:
            self.weight = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reduction: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits:    (N, C) unnormalized class scores.
            targets:   (N,)   integer class labels in [0, C).
            reduction: override the instance-level reduction if provided.

        Returns:
            Focal loss scalar (or per-sample tensor if reduction='none').
        """
        if logits.numel() == 0:
            return logits.sum() * 0.0

        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        ce = ce.clamp(max=100.0)   # prevent inf when model is over-confident on wrong class
        pt = torch.exp(-ce).clamp(max=1.0)   # clamp to [0,1]: float32 may give pt>1 when ce<0
        focal = (1.0 - pt) ** self.gamma * ce

        red = reduction if reduction is not None else self.reduction
        if red == 'mean':
            return focal.mean()
        if red == 'sum':
            return focal.sum()
        return focal

    @staticmethod
    def class_weights_from_counts(counts: torch.Tensor) -> torch.Tensor:
        """
        Compute tempered inverse-frequency class weights from per-class sample counts.

        Instead of raw inverse-frequency weighting, use a square-root tempering:

            weights[c] = sqrt(total / (num_classes * counts[c]))

        This still upweights rare classes, but avoids over-correcting and pushing
        common single bonds into higher-order classes.

        Args:
            counts: (C,) number of samples per class.

        Returns:
            (C,) normalized weight tensor.
        """
        total = counts.sum().float()
        num_classes = counts.shape[0]
        w = torch.sqrt(total / (num_classes * counts.float().clamp(min=1)))
        return w / w.mean()  # normalize so weights average to 1
