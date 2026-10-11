import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit


def group_stratified_split(y_all, groups, seed):
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    group_label = np.zeros(len(unique_groups), dtype=int)
    np.maximum.at(group_label, inverse, y_all[:, 0].astype(int))

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_g, temp_g = next(sss1.split(unique_groups, group_label))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_rel, test_rel = next(sss2.split(temp_g, group_label[temp_g]))

    def rows(g_idx):
        return np.flatnonzero(np.isin(inverse, g_idx))

    return rows(train_g), rows(temp_g[val_rel]), rows(temp_g[test_rel])


def group_stratified_split_and_save(x_all, y_all, groups, ids, seeds, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        splits = dict(zip(["train", "val", "test"], group_stratified_split(y_all, groups, seed)))
        for a in splits:
            for b in splits:
                if a < b:
                    assert len(set(groups[splits[a]]) & set(groups[splits[b]])) == 0
        assert sum(len(v) for v in splits.values()) == len(x_all)

        for name, idx in splits.items():
            np.save(out_dir / f"X_{name}_{seed}.npy", x_all[idx])
            np.save(out_dir / f"y_{name}_{seed}.npy", y_all[idx])
            np.save(out_dir / f"ids_{name}_{seed}.npy", ids[idx])

        print(f"  seed {seed}: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    print(f"  Saved to {out_dir}/")


def stratified_split_and_save(x_all, y_all, seeds, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    strat_labels = y_all[:, 0].astype(int)

    for seed in seeds:
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        train_idx, temp_idx = next(sss1.split(x_all, strat_labels))
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
        val_rel, test_rel = next(sss2.split(x_all[temp_idx], strat_labels[temp_idx]))
        val_idx, test_idx = temp_idx[val_rel], temp_idx[test_rel]

        assert len(set(train_idx) & set(val_idx)) == 0
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(set(val_idx) & set(test_idx)) == 0
        assert len(train_idx) + len(val_idx) + len(test_idx) == len(x_all)

        np.save(out_dir / f"X_train_{seed}.npy", x_all[train_idx])
        np.save(out_dir / f"X_val_{seed}.npy", x_all[val_idx])
        np.save(out_dir / f"X_test_{seed}.npy", x_all[test_idx])
        np.save(out_dir / f"y_train_{seed}.npy", y_all[train_idx])
        np.save(out_dir / f"y_val_{seed}.npy", y_all[val_idx])
        np.save(out_dir / f"y_test_{seed}.npy", y_all[test_idx])

        print(f"  seed {seed}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    print(f"  Saved to {out_dir}/")
