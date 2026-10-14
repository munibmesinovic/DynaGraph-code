import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from .data import load_split, make_loaders
from .importance import RunningImportance
from .losses import DynaGraphLoss
from .metrics import compute_metrics, predict, youden_thresholds
from .model import build_model


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_deterministic():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CheckpointSelector:
    def __init__(self, patience, min_epoch):
        self.patience, self.min_epoch = patience, min_epoch
        self.best, self.no_improve = -1.0, 0

    def step(self, epoch, value):
        if epoch < self.min_epoch:
            return False, False
        if value > self.best:
            self.best, self.no_improve = value, 0
            return True, False
        self.no_improve += 1
        return False, self.no_improve >= self.patience


def evaluate_model(model, cfg, val_dl, test_dl):
    names = cfg["dataset"]["label_names"]
    y_val, s_val = predict(model, val_dl)
    y_test, s_test = predict(model, test_dl)
    thresholds = youden_thresholds(y_val, s_val)
    out = {}
    for split, (y, s) in {"val": (y_val, s_val), "test": (y_test, s_test)}.items():
        macro, per_label = compute_metrics(y, s, thresholds, names)
        out[split] = {"macro": macro, "per_label": per_label}
    out["thresholds"] = thresholds.tolist()
    predictions = {"y_true_val": y_val, "y_score_val": s_val, "y_true_test": y_test, "y_score_test": s_test,
                   "thresholds": thresholds}
    return out, predictions


def train(seed, cfg, device, out_dir, max_epochs=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_deterministic()
    set_seed(seed)
    tc = cfg["training"]
    max_epochs = max_epochs or tc["max_epochs"]

    X_tr, y_tr, X_va, y_va, X_te, y_te = load_split(seed, cfg["data"]["processed_dir"])
    train_dl, val_dl, test_dl = make_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te, tc["batch_size"])
    model = build_model(cfg).to(device)

    pos_counts = y_tr.sum(dim=0)
    pos_weight = (y_tr.size(0) - pos_counts) / pos_counts.clamp(min=1.0)
    criterion = DynaGraphLoss(cfg, pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=tc["lr_patience"], factor=0.5)
    selector = CheckpointSelector(tc["patience"], tc["min_select_epoch"])
    ckpt_path = out_dir / f"best_model_seed{seed}.pt"
    history = []
    importance = RunningImportance(model)
    best_epoch = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        running, terms_sum, n_batches = 0.0, {}, 0
        for step, (X_b, y_b) in enumerate(train_dl):
            X_b, y_b = X_b.to(device), y_b.to(device)
            out = model(X_b)
            out["G"].retain_grad()
            loss, terms = criterion(out, y_b, model, X_b, epoch)
            (loss / tc["grad_accum"]).backward()
            importance.accumulate(out["G"])
            if (step + 1) % tc["grad_accum"] == 0 or (step + 1) == len(train_dl):
                model.interp_matrix.update_from_gradients()
                importance.step()
                nn.utils.clip_grad_norm_(model.parameters(), tc["clip_norm"])
                optimizer.step()
                optimizer.zero_grad()
            running += terms["total"]
            for k, v in terms.items():
                terms_sum[k] = terms_sum.get(k, 0.0) + v
            n_batches += 1

        y_val, s_val = predict(model, val_dl)
        val_m, _ = compute_metrics(y_val, s_val, np.full(y_val.shape[1], 0.5), cfg["dataset"]["label_names"])
        scheduler.step(val_m["auprc_weighted"])
        history.append({
            "epoch": epoch, "train_loss": running / n_batches,
            "val_auroc": val_m["auroc"], "val_auprc_w": val_m["auprc_weighted"],
            "train_terms": {k: v / n_batches for k, v in terms_sum.items()},
        })
        save, stop = selector.step(epoch, val_m["auprc_weighted"])
        if save:
            torch.save(model.state_dict(), ckpt_path)
            best_epoch = epoch
        importance.end_epoch()
        print(f"epoch {epoch} loss {running / n_batches:.4f} val_auroc {val_m['auroc']:.4f} "
              f"val_auprc_w {val_m['auprc_weighted']:.4f}{' saved' if save else ''}", flush=True)
        if stop:
            break

    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    if selector.best < 0:
        torch.save(model.state_dict(), ckpt_path)
        best_epoch = len(history)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    importance.save(out_dir, best_epoch, model.importance)
    metrics, predictions = evaluate_model(model, cfg, val_dl, test_dl)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    np.savez(out_dir / "predictions.npz", **predictions)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-epochs", type=int, default=None)
    a = p.parse_args()
    cfg = load_config(a.config)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    m = train(a.seed, cfg, torch.device(a.device), a.out, a.max_epochs)
    print(json.dumps(m["test"]["macro"]))


if __name__ == "__main__":
    main()
