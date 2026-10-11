import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score


def compute_metrics(y_true, y_score, thresholds, label_names):
    y_pred = (y_score >= np.asarray(thresholds)[None, :]).astype(int)
    per_label = {}
    aurocs, auprcs, f1s, senss, prevalences = [], [], [], [], []
    for i, name in enumerate(label_names):
        n_pos = int(y_true[:, i].sum())
        prev = n_pos / y_true.shape[0]
        auroc = roc_auc_score(y_true[:, i], y_score[:, i])
        auprc = average_precision_score(y_true[:, i], y_score[:, i])
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        sens = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        aurocs.append(auroc)
        auprcs.append(auprc)
        f1s.append(f1)
        senss.append(sens)
        prevalences.append(prev)
        per_label[name] = {"auroc": auroc, "auprc": auprc, "f1": f1, "sens": sens, "n_pos": n_pos, "prevalence": prev}
    weights = np.array(prevalences) / np.sum(prevalences)
    macro = {
        "auroc": float(np.mean(aurocs)),
        "auprc": float(np.mean(auprcs)),
        "auprc_weighted": float(np.sum(np.array(auprcs) * weights)),
        "f1": float(np.mean(f1s)),
        "sens": float(np.mean(senss)),
    }
    return macro, per_label


def youden_thresholds(y_true, y_score):
    candidates = np.arange(0.05, 0.90, 0.025)
    thresholds = np.full(y_true.shape[1], 0.5)
    for i in range(y_true.shape[1]):
        n_neg = int((1 - y_true[:, i]).sum())
        best_score, best_t = -1.0, 0.5
        for t in candidates:
            pred = (y_score[:, i] >= t).astype(int)
            sens = recall_score(y_true[:, i], pred, zero_division=0)
            tn = int(((1 - y_true[:, i]) * (1 - pred)).sum())
            spec = tn / n_neg if n_neg > 0 else 0.0
            score = sens + spec - 1.0
            if score > best_score:
                best_score, best_t = score, t
        thresholds[i] = best_t
    return thresholds


@torch.no_grad()
def predict(model, loader):
    model.eval()
    device = next(model.parameters()).device
    scores, targets = [], []
    for X_b, y_b in loader:
        scores.append(torch.sigmoid(model(X_b.to(device))["logits"]).cpu().numpy())
        targets.append(y_b.numpy())
    return np.vstack(targets), np.vstack(scores)
