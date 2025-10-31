import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import logging
from src.evaluation.metrics import postprocess_predictions
import numpy as np


def classification_metrics(y_true, y_pred, threshold=0.5):
    y_pred_bin = (np.array(y_pred) >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred_bin)
    f1 = f1_score(y_true, y_pred_bin)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = float('nan')
    logging.info(f"[METRICS] Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    return {"accuracy": acc, "f1": f1, "auc": auc}


def hit_rate_at_k(y_true_lists, y_pred_lists, k=10):
    hits = 0
    for true_items, preds in zip(y_true_lists, y_pred_lists):
        if any(i in true_items for i in preds[:k]):
            hits += 1
    return hits / len(y_true_lists)


def ndcg_at_k(y_true_lists, y_pred_lists, k=10):
    ndcgs = []
    for true_items, preds in zip(y_true_lists, y_pred_lists):
        dcg = 0.0
        for i, item in enumerate(preds[:k]):
            if item in true_items:
                dcg += 1.0 / np.log2(i + 2)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), k)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return np.mean(ndcgs)


def map_at_k(y_true_lists, y_pred_lists, k=10):
    ap_list = []
    for true_items, preds in zip(y_true_lists, y_pred_lists):
        hits = 0
        sum_precisions = 0
        for i, item in enumerate(preds[:k]):
            if item in true_items:
                hits += 1
                sum_precisions += hits / (i + 1)
        ap = sum_precisions / min(len(true_items), k) if true_items else 0
        ap_list.append(ap)
    return np.mean(ap_list)


def compute_metrics(y_true, y_pred, topk_data=None):
    metrics = classification_metrics(y_true, y_pred)

    if topk_data:
        y_true_lists, y_pred_lists = topk_data
        for k in [5, 10, 20]:
            hr = hit_rate_at_k(y_true_lists, y_pred_lists, k=k)
            ndcg = ndcg_at_k(y_true_lists, y_pred_lists, k=k)
            mapk = map_at_k(y_true_lists, y_pred_lists, k=k)
            metrics.update({
                f"hit_rate@{k}": hr,
                f"ndcg@{k}": ndcg,
                f"map@{k}": mapk
            })

    return metrics


def postprocess_predictions(raw_scores, threshold=0.5):
    if (raw_scores < 0).any() or (raw_scores > 1).any():
        raise ValueError("Предсказания выходят за пределы [0,1]")
    return (raw_scores >= threshold).astype(int)


def test_postprocess_predictions():
    preds = np.array([0.1, 0.9])
    out = postprocess_predictions(preds, threshold=0.5)
    assert (out == np.array([0,1])).all()
