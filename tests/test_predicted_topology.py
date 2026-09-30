import torch

from train_stage2 import _predicted_topology_edges


class _DummyBackbone:
    def __call__(self, elems, coord, edge_index, edge_diff, edge_dist, **kwargs):
        edge_feat = torch.arange(edge_index.shape[0], device=coord.device).float().unsqueeze(-1)
        return None, None, edge_feat


class _DummyConnectivity:
    def predict(self, edge_feat, threshold=0.5):
        # Predict two true HH directions, two false-positive HH directions,
        # and two true H--A directions.
        return torch.ones(edge_feat.shape[0], dtype=torch.bool, device=edge_feat.device)


class _DummyStage1:
    cutoff = 2.5
    backbone = _DummyBackbone()
    stage1 = _DummyConnectivity()


def test_predicted_topology_keeps_false_positives_but_does_not_supervise_them():
    edge_index = torch.tensor(
        [[0, 1], [1, 0], [0, 2], [2, 0], [0, 3], [3, 0]], dtype=torch.long
    )
    coord = torch.tensor(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 1.6, 0.0], [0.0, 0.0, 1.0]]
    )
    edge_diff = coord[edge_index[:, 1]] - coord[edge_index[:, 0]]
    edge_dist = edge_diff.norm(dim=-1)
    batch = {
        'elems': torch.tensor([6, 6, 8, 1]),
        'coord': coord,
        'edge_index': edge_index,
        'edge_diff': edge_diff,
        'edge_dist': edge_dist,
        # Edges 0/1 are a true HH bond; 2/3 are false positives; 4/5 are H--A.
        'bond_mask': torch.tensor([True, True, False, False, True, True]),
        'bond_type_full': torch.tensor([1, 1, 0, 0]),
        'num_atoms_per_mol': torch.tensor([4]),
    }

    bond_ei, _, _, readout_ei, _, _, target = _predicted_topology_edges(
        _DummyStage1(), batch, conn_threshold=0.5
    )

    assert torch.equal(bond_ei, edge_index)
    assert torch.equal(readout_ei, edge_index[:2])
    assert torch.equal(target, torch.tensor([1, 1]))
