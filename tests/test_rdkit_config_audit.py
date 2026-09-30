from scripts.audit_rdkit_configurations import pipeline_f1, update_pipeline_counts


def test_unordered_pipeline_counts_include_wrong_extra_and_missing_bonds():
    true = {(0, 1): 0, (1, 2): 1, (2, 3): 3}
    pred = {(0, 1): 0, (1, 2): 3, (3, 4): 2}
    tp, fp, fn = [0] * 4, [0] * 4, [0] * 4
    update_pipeline_counts(true, pred, tp, fp, fn)
    assert tp == [1, 0, 0, 0]
    assert fp == [0, 0, 1, 1]
    assert fn == [0, 1, 0, 1]
    assert pipeline_f1(tp, fp, fn) == [1.0, 0.0, 0.0, 0.0]


def test_failure_gets_zero_credit_but_preserves_reference_denominator():
    tp, fp, fn = [0] * 4, [0] * 4, [0] * 4
    update_pipeline_counts({(0, 1): 0, (1, 2): 3}, {}, tp, fp, fn)
    assert tp == [0] * 4
    assert fp == [0] * 4
    assert fn == [1, 0, 0, 1]
