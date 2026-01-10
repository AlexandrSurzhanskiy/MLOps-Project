import torch
import logging
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def validate(model, loader, device="cpu", criterion=None):
    model.eval()
    y_true, y_pred = [], []

    running_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for users, items, labels in loader:
            users, items, labels = (
                users.to(device),
                items.to(device),
                labels.to(device),
            )

            preds = model(users, items).view(-1)
            labels = labels.view(-1)

            if criterion is not None:
                loss = criterion(preds, labels)
                running_loss += loss.item()
                n_batches += 1

            y_true += labels.cpu().tolist()
            y_pred += preds.cpu().tolist()

    preds_bin = [1 if p >= 0.5 else 0 for p in y_pred]

    acc = accuracy_score(y_true, preds_bin)
    f1 = f1_score(y_true, preds_bin)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = float("nan")

    metrics = {"accuracy": acc, "f1": f1, "auc": auc}

    if criterion is not None:
        metrics["val_loss"] = running_loss / max(n_batches, 1)

    if "val_loss" in metrics:
        logging.info(
            f"[VALIDATION] Loss: {metrics['val_loss']:.4f} | " f"Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}"
        )
    else:
        logging.info(f"[VALIDATION] Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    return metrics
