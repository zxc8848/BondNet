from scripts.p0c_yuelbond_compare import _sdf_row_key


def test_shared_sdf_prediction_keys_use_row_order_not_geom_identity():
    geom_mol_idx = [128416, 121556, 108974]
    keys = [_sdf_row_key(0, i) for i in range(len(geom_mol_idx))]
    assert keys == ['0', '1', '2']
    assert keys != list(map(str, geom_mol_idx))
    assert [_sdf_row_key(128, i) for i in range(3)] == ['128', '129', '130']
