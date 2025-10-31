import torch.nn as nn
import logging
from tqdm import tqdm
from src.training.validate import validate
from src.training.optimizer import get_optimizer, get_scheduler
from src.training.seed_utils import set_seed


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for users, items, labels in tqdm(dataloader, desc="Training", ncols=90):
        users, items, labels = (
            users.to(device),
            items.to(device),
            labels.to(device),
        )
        optimizer.zero_grad()
        preds = model(users, items)
        preds = preds.view(-1)
        labels = labels.view(-1)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def train_model(model, train_loader, val_loader, cfg, device="cpu"):
    set_seed(cfg["training"].get("random_seed", 42))
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)
    n_epochs = cfg["training"].get("epochs", 10)

    for epoch in range(1, n_epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = validate(model, val_loader, device)
        logging.info(f"[EPOCH {epoch}/{n_epochs}] loss={loss:.4f} | AUC={metrics.get('auc', 0):.4f}")

        if scheduler:
            if "plateau" in str(type(scheduler)).lower():
                scheduler.step(metrics.get("auc", loss))
            else:
                scheduler.step()

    return model
