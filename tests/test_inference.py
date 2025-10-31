import torch
from src.models.recsys_nn import RecSysNN


def test_predictions_in_range():
    model = RecSysNN(n_users=10, n_items=10, embedding_dim=4, hidden_dim=8)
    preds = model(torch.tensor([0, 1]), torch.tensor([1, 2]))
    assert torch.all((preds >= 0) & (preds <= 1))
