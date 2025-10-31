import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.recsys_nn import RecSysNN
from src.training.train import train_one_epoch


def test_train_one_epoch_runs():
    model = RecSysNN(n_users=3, n_items=5, embedding_dim=8, hidden_dim=16)
    data = TensorDataset(
        torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]), torch.tensor([1.0, 0.0, 1.0])
    )
    loader = DataLoader(data, batch_size=2)
    loss = train_one_epoch(
        model, loader, torch.nn.BCELoss(), torch.optim.Adam(model.parameters()), "cpu"
    )
    assert loss >= 0
