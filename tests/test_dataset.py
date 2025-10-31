import torch
from src.data.dataset import InteractionDataset


def test_dataset_shape(sample_df):
    ds = InteractionDataset(sample_df)
    u, i, y = ds[0]
    assert isinstance(u, torch.Tensor)
    assert u.dtype == torch.long
    assert 0 <= y.item() <= 1
