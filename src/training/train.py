import torch.nn as nn
import logging
from tqdm import tqdm

from src.training.validate import validate
from src.training.optimizer import get_optimizer, get_scheduler
from src.training.seed_utils import set_seed


def _log_mlflow_metrics(epoch: int, train_loss: float, val_metrics: dict, optimizer=None):
    try:
        import mlflow

        if mlflow.active_run() is None:
            return

        mlflow.log_metric("train_loss", float(train_loss), step=int(epoch))

        for k, v in (val_metrics or {}).items():
            if isinstance(v, (int, float)) and v == v:
                mlflow.log_metric(f"val_{k}", float(v), step=int(epoch))

        if optimizer is not None and optimizer.param_groups:
            mlflow.log_metric("lr", float(optimizer.param_groups[0]["lr"]), step=int(epoch))

    except ImportError:
        return


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

    return running_loss / max(len(dataloader), 1)


def train_model(model, train_loader, val_loader, cfg, device="cpu"):
    set_seed(cfg["training"].get("random_seed", 42))
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)

    n_epochs = cfg["training"].get("epochs", 10)

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        metrics = validate(model, val_loader, device=device, criterion=criterion)

        logging.info(
            f"[EPOCH {epoch}/{n_epochs}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={metrics.get('val_loss', 0):.4f} | "
            f"AUC={metrics.get('auc', 0):.4f}"
        )

        _log_mlflow_metrics(epoch=epoch, train_loss=train_loss, val_metrics=metrics, optimizer=optimizer)

        if scheduler:
            if "plateau" in str(type(scheduler)).lower():
                scheduler.step(metrics.get("auc", train_loss))
            else:
                scheduler.step()

    return model
