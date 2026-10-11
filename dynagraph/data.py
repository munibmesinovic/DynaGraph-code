from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_split(seed, data_dir):
    data_dir = Path(data_dir)
    X_tr = np.load(data_dir / f"X_train_{seed}.npy")
    X_va = np.load(data_dir / f"X_val_{seed}.npy")
    X_te = np.load(data_dir / f"X_test_{seed}.npy")
    y_tr = np.load(data_dir / f"y_train_{seed}.npy")
    y_va = np.load(data_dir / f"y_val_{seed}.npy")
    y_te = np.load(data_dir / f"y_test_{seed}.npy")

    F = X_tr.shape[-1]
    flat = X_tr.reshape(-1, F)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-8] = 1.0
    continuous = np.ones(F, dtype=bool)
    for f in range(F):
        uniq = np.unique(flat[:, f])
        if len(uniq) <= 2 and set(uniq).issubset({0.0, 1.0}):
            continuous[f] = False
    mean[~continuous] = 0.0
    std[~continuous] = 1.0

    def to_model(X):
        X = (X - mean) / std
        return torch.from_numpy(X.transpose(0, 2, 1)[:, np.newaxis, :, :]).float()

    return (
        to_model(X_tr), torch.from_numpy(y_tr).float(),
        to_model(X_va), torch.from_numpy(y_va).float(),
        to_model(X_te), torch.from_numpy(y_te).float(),
    )


def make_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te, batch_size):
    train_dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    val_dl = DataLoader(TensorDataset(X_va, y_va), batch_size=batch_size, shuffle=False, pin_memory=True)
    test_dl = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_dl, val_dl, test_dl
