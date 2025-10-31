import torch
from src.training.optimizer import get_optimizer, get_scheduler


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 1)


def test_get_optimizer_and_scheduler():
    model = DummyModel()
    cfg = {
        "training": {
            "optimizer": {"type": "adam", "lr": 0.001, "weight_decay": 0.01},
            "scheduler": {"type": "step", "step_size": 1, "gamma": 0.9},
        }
    }

    optimizer = get_optimizer(model, cfg)
    assert isinstance(optimizer, torch.optim.Optimizer)

    scheduler = get_scheduler(optimizer, cfg)
    assert scheduler is not None
