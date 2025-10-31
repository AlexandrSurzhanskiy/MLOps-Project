import torch
from torch.utils.data import DataLoader, TensorDataset
from src.training.validate import validate
from src.models.recsys_nn import RecSysNN


def test_validate_runs_without_errors():
    model = RecSysNN(n_users=5, n_items=5, embedding_dim=4, hidden_dim=8)
    users = torch.tensor([0, 1, 2, 3, 4])
    items = torch.tensor([1, 2, 3, 4, 0])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
    loader = DataLoader(TensorDataset(users, items, labels), batch_size=2)
    metrics = validate(model, loader, device="cpu")

    assert isinstance(metrics, dict)
    assert "auc" in metrics
    assert 0 <= metrics["auc"] <= 1 or torch.isnan(torch.tensor(metrics["auc"]))
