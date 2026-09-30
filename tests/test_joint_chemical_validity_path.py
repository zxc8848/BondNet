"""The full-molecule validity exporter must use the joint type head correctly."""

from types import SimpleNamespace

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from bondnet.data.dataset import collate_fn
from bondnet.data.featurizer import MoleculeFeaturizer
from bondnet.model.bondnet import BondNet
from scripts.evaluate_chemical_validity import predict_batch
from scripts.export_per_molecule_stats import predict


def test_joint_validity_prediction_matches_main_joint_predictor():
    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    assert AllChem.EmbedMolecule(mol, randomSeed=11) == 0
    sample = MoleculeFeaturizer(
        cutoff=2.5, h_cutoff=2.5, explicit_h=True
    ).featurize(mol)
    sample["geom_mol_idx"] = 11
    batch = collate_fn([sample])
    model = BondNet(
        num_interactions=1, hidden_size=32, cutoff=2.5,
        edge_embedding_size=8,
    ).eval()
    device = torch.device("cpu")

    with torch.no_grad():
        validity_graphs, _ = predict_batch(
            model, None, batch, device, 2.5, 2.5, 0.0, 0.0, 20260921,
        )
        scored_batch, conn, pred_type = predict(
            model, None, batch, device,
            SimpleNamespace(
                cutoff=2.5, h_cutoff=2.5, conn_threshold=0.0,
                eval_noise_seed=20260921,
            ),
            sigma=0.0,
        )
    edge_index = scored_batch["edge_index"].tolist()
    elems = scored_batch["elems"].tolist()
    scored_graph = {}
    for pos, (i, j) in enumerate(edge_index):
        if not bool(conn[pos]):
            continue
        label = int(pred_type[pos])
        if label < 0:
            assert elems[i] == 1 or elems[j] == 1
            label = 0
        pair = tuple(sorted((i, j)))
        assert pair not in scored_graph or scored_graph[pair] == label
        scored_graph[pair] = label

    assert any(elems[i] != 1 and elems[j] != 1 for i, j in scored_graph)
    assert any(elems[i] == 1 or elems[j] == 1 for i, j in scored_graph)
    assert validity_graphs == [scored_graph]
