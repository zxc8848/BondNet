import numpy as np
import pytest

from scripts.export_per_molecule_stats import class_counts, undirected_hh_labels


def test_undirected_or_decoding_counts_each_pair_once():
    edges = np.asarray([[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]])
    true = np.asarray([0, 0, 1, 1, -1, -1])
    pred = np.asarray([-1, 0, -1, -1, 2, -1])
    pair_true, pair_pred = undirected_hh_labels(edges, true, pred, mol_id=7)
    assert pair_true.tolist() == [0, 1, -1]
    assert pair_pred.tolist() == [0, -1, 2]
    tp, fp, fn = class_counts(pair_true, pair_pred, include_extra=True)
    assert tp.tolist() == [1, 0, 0, 0]
    assert fp.tolist() == [0, 0, 1, 0]
    assert fn.tolist() == [0, 1, 0, 0]


def test_undirected_decoding_rejects_conflicting_labels():
    with pytest.raises(ValueError, match="Conflicting predicted"):
        undirected_hh_labels(
            np.asarray([[0, 1], [1, 0]]),
            np.asarray([0, 0]),
            np.asarray([0, 1]),
            mol_id=7,
        )
