from argparse import Namespace

import pytest
import torch

from train import load_ck, save_ck


class _Backbone(torch.nn.Module):
    num_interactions = 1
    edge_embedding_size = 4


class _Model(torch.nn.Module):
    hidden_size = 4
    cutoff = 2.5

    def __init__(self):
        super().__init__()
        self.backbone = _Backbone()
        self.weight = torch.nn.Parameter(torch.ones(1))


def test_resume_restores_best_e2e_selection_score(tmp_path):
    path = tmp_path / 'checkpoint.pt'
    model = _Model()
    optimizer = torch.optim.AdamW(model.parameters())
    args = Namespace(num_interactions=1, hidden_size=4, cutoff=2.5,
                     edge_embedding_size=4, explicit_h=True)
    save_ck(path, model, optimizer, None, 9, 0.2, args, best_e2e=0.991)

    restored = _Model()
    restored_optimizer = torch.optim.AdamW(restored.parameters())
    epoch, best_val, best_e2e = load_ck(
        path, restored, restored_optimizer, args=args
    )
    assert epoch == 9
    assert best_val == 0.2
    assert best_e2e == 0.991


def test_legacy_resume_fails_instead_of_overwriting_best_checkpoint(tmp_path):
    path = tmp_path / 'legacy.pt'
    model = _Model()
    torch.save({'epoch': 9, 'model_state_dict': model.state_dict(),
                'best_val_loss': 0.2, 'args': {}}, path)

    with pytest.raises(ValueError, match='does not record best_e2e'):
        load_ck(path, _Model())
