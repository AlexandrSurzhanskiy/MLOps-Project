import torch
from src.models.recsys_nn import RecSysNN
from src.training.seed_utils import set_seed


def test_reproducibility():
    set_seed(123)
    model1 = RecSysNN(n_users=100, n_items=100, embedding_dim=16, hidden_dim=32)
    state1 = [p.clone() for p in model1.parameters()]

    set_seed(123)
    model2 = RecSysNN(n_users=100, n_items=100, embedding_dim=16, hidden_dim=32)
    state2 = [p.clone() for p in model2.parameters()]

    for p1, p2 in zip(state1, state2):
        assert torch.allclose(p1, p2), "Модели при одинаковом seed должны иметь идентичные веса"
