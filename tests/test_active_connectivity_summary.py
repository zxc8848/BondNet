import pytest

from scripts.summarize_revision_v3_active_connectivity import corrected_metrics


def test_active_connectivity_excludes_only_outside_positive_records():
    # Archived cache-envelope counts: one active TP, one active FP, and two
    # false negatives, of which one reference bond is outside the active graph.
    row = corrected_metrics(tp=1, fp=1, fn_cache=2,
                            excluded_reference_bonds=1)
    assert (row["tp"], row["fp"], row["fn"]) == (1, 1, 1)
    assert row["precision"] == pytest.approx(0.5)
    assert row["recall"] == pytest.approx(0.5)
    assert row["f1"] == pytest.approx(0.5)


def test_active_connectivity_rejects_excess_subtraction():
    with pytest.raises(ValueError, match="Negative confusion count"):
        corrected_metrics(tp=1, fp=0, fn_cache=0,
                          excluded_reference_bonds=1)
