import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import load_split, make_loaders
from .model import build_model
from .train import evaluate_model, load_config, set_deterministic


def evaluate_checkpoint(cfg, seed, checkpoint, device):
    set_deterministic()
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_split(seed, cfg["data"]["processed_dir"])
    _, val_dl, test_dl = make_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te, cfg["training"]["batch_size"])
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    return evaluate_model(model, cfg, val_dl, test_dl)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    metrics, predictions = evaluate_checkpoint(load_config(a.config), a.seed, a.checkpoint, torch.device(a.device))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    np.savez(out / "predictions.npz", **predictions)
    print(json.dumps(metrics["test"]["macro"]))


if __name__ == "__main__":
    main()
