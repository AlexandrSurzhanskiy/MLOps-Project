import torch
import logging
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def validate(model, loader, device="cpu"):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for users, items, labels in loader:
            users, items, labels = users.to(device), items.to(device), labels.to(device)
            preds = model(users, items)
            y_true += labels.cpu().view(-1).tolist()
            y_pred += preds.cpu().view(-1).tolist()

    preds_bin = [1 if p >= 0.5 else 0 for p in y_pred]

    acc = accuracy_score(y_true, preds_bin)
    f1 = f1_score(y_true, preds_bin)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = float("nan")

    logging.info(f"[VALIDATION] Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    return {"accuracy": acc, "f1": f1, "auc": auc}
