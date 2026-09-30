"""
Evaluation metrics for BondNet.

Tracks per-class F1, molecule-level validity, connectivity accuracy,
and robustness curves across noise levels.
"""

from typing import Dict, List, Optional
import numpy as np


class BondNetMetrics:
    """
    Accumulates predictions and labels across batches, then computes metrics.

    Bond type encoding:  0=single, 1=double, 2=triple, 3=aromatic
    """

    BOND_NAMES = ['single', 'double', 'triple', 'aromatic']
    BOND_LABELS = [0, 1, 2, 3]

    def __init__(self):
        self.reset()

    # ------------------------------------------------------------------ #
    # State management                                                      #
    # ------------------------------------------------------------------ #

    def reset(self):
        self._conn_pred: List[np.ndarray] = []
        self._conn_label: List[np.ndarray] = []
        self._type_pred: List[np.ndarray] = []
        self._type_label: List[np.ndarray] = []
        self._mol_valid: List[bool] = []
        self._bo_pred: List[np.ndarray] = []
        self._bo_label: List[np.ndarray] = []

    # ------------------------------------------------------------------ #
    # Accumulation helpers                                                  #
    # ------------------------------------------------------------------ #

    def update_connectivity(self, pred, label):
        """
        Args:
            pred:  boolean/int array (E,) — predicted bond existence.
            label: boolean/int array (E,) — ground-truth bond existence.
        """
        self._conn_pred.append(_to_numpy(pred).astype(bool))
        self._conn_label.append(_to_numpy(label).astype(bool))

    def update_bond_types(self, pred, label):
        """
        Args:
            pred:  int array (E',) with values in {0,1,2,3}.
            label: int array (E',) with values in {0,1,2,3}.
        """
        self._type_pred.append(_to_numpy(pred).astype(np.int32))
        self._type_label.append(_to_numpy(label).astype(np.int32))

    def update_bond_orders(self, pred, label):
        """
        Args:
            pred:  float array (E',) — predicted continuous bond orders.
            label: float array (E',) — ground-truth Wiberg bond orders.
        """
        self._bo_pred.append(_to_numpy(pred).astype(np.float32))
        self._bo_label.append(_to_numpy(label).astype(np.float32))

    def update_molecule_validity(self, correct: bool):
        """Record whether the full molecule's bond assignment is correct."""
        self._mol_valid.append(bool(correct))

    # ------------------------------------------------------------------ #
    # Metric computation                                                    #
    # ------------------------------------------------------------------ #

    def compute(self) -> Dict[str, object]:
        results: Dict[str, object] = {}

        # --- Connectivity (Stage 1) ---
        if self._conn_pred:
            y_pred = np.concatenate(self._conn_pred)
            y_true = np.concatenate(self._conn_label)
            p, r, f1 = _binary_prf(y_true, y_pred)
            results['conn_f1'] = f1
            results['conn_precision'] = p
            results['conn_recall'] = r
            n_pos = int(y_true.sum())
            n_total = int(y_true.size)
            n_neg = n_total - n_pos
            results['conn_n_candidates_directed'] = n_total
            results['conn_n_positive_directed'] = n_pos
            results['conn_n_negative_directed'] = n_neg
            results['conn_positive_prevalence'] = float(n_pos / n_total) if n_total else 0.0
            results['conn_negative_to_positive_ratio'] = (
                float(n_neg / n_pos) if n_pos else float('inf')
            )

        # --- Bond type per-class F1 (Stages 2+3) ---
        if self._type_pred:
            all_pred = np.concatenate(self._type_pred)
            all_label = np.concatenate(self._type_label)
            f1s = [_class_f1(all_label, all_pred, label) for label in self.BOND_LABELS]
            for i, name in enumerate(self.BOND_NAMES):
                results[f'f1_{name}'] = float(f1s[i]) if i < len(f1s) else 0.0
            results['f1_macro'] = float(np.mean(f1s))
            support = np.array([(all_label == label).sum() for label in self.BOND_LABELS], dtype=np.float64)
            confusion = np.zeros((len(self.BOND_LABELS), len(self.BOND_LABELS)), dtype=np.int64)
            for true_label, pred_label in zip(all_label, all_pred):
                if true_label in self.BOND_LABELS and pred_label in self.BOND_LABELS:
                    confusion[int(true_label), int(pred_label)] += 1
            results['bond_type_support'] = {
                name: int(support[i]) for i, name in enumerate(self.BOND_NAMES)
            }
            results['bond_type_confusion_matrix'] = confusion.tolist()
            results['bond_type_confusion_labels'] = list(self.BOND_NAMES)
            results['f1_weighted'] = float(
                np.sum(np.array(f1s) * support) / support.sum()
            ) if support.sum() > 0 else 0.0

        # --- Continuous BO regression ---
        if self._bo_pred:
            bo_pred = np.concatenate(self._bo_pred)
            bo_label = np.concatenate(self._bo_label)
            valid = np.isfinite(bo_pred) & np.isfinite(bo_label)
            if valid.any():
                diff = bo_label[valid] - bo_pred[valid]
                results['bo_mae'] = float(np.mean(np.abs(diff)))
                results['bo_rmse'] = float(np.sqrt(np.mean(diff * diff)))
                nan_frac = 1.0 - valid.mean()
                if nan_frac > 0:
                    results['bo_nan_frac'] = float(nan_frac)

        # --- Molecule-level validity ---
        if self._mol_valid:
            results['mol_validity'] = float(np.mean(self._mol_valid))

        return results

    def summary(self) -> str:
        m = self.compute()
        lines = []
        if 'conn_f1' in m:
            lines.append(
                f"Connectivity  F1={m['conn_f1']:.4f} "
                f"P={m['conn_precision']:.4f} R={m['conn_recall']:.4f}"
            )
        if 'f1_single' in m:
            per_class = '  '.join(
                f"{n}={m[f'f1_{n}']:.4f}" for n in self.BOND_NAMES
            )
            lines.append(f"Bond F1  {per_class}  macro={m['f1_macro']:.4f}")
        if 'bo_mae' in m:
            lines.append(f"BO regression  MAE={m['bo_mae']:.4f}  RMSE={m['bo_rmse']:.4f}")
        if 'mol_validity' in m:
            lines.append(f"Mol validity  {m['mol_validity']*100:.1f}%")
        return '\n'.join(lines)


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, 'cpu'):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _binary_prf(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = np.logical_and(y_true, y_pred).sum()
    fp = np.logical_and(~y_true, y_pred).sum()
    fn = np.logical_and(y_true, ~y_pred).sum()
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _class_f1(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> float:
    true_pos = y_true == label
    pred_pos = y_pred == label
    tp = np.logical_and(true_pos, pred_pos).sum()
    fp = np.logical_and(~true_pos, pred_pos).sum()
    fn = np.logical_and(true_pos, ~pred_pos).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
