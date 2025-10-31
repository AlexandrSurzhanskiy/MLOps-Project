import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import logging


class InteractionDataset(Dataset):
    def __init__(self, parquet_path):
        logging.info(f"Загрузка датасета: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items = torch.tensor(df["item_idx"].values, dtype=torch.long)
        self.labels = torch.tensor(df["label"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


def get_dataloaders(train_path, test_path, batch_size=256):
    train_ds = InteractionDataset(train_path)
    test_ds = InteractionDataset(test_path)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    logging.info(f"DataLoaders готовы: train={len(train_ds)}, test={len(test_ds)}")
    return train_loader, test_loader
