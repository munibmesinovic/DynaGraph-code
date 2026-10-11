import argparse
import json
from pathlib import Path

import numpy as np

DATASETS = ["mimic", "eicu", "hirid", "ehrshot"]
SEEDS = [42, 1992, 1709, 250, 213]
METRICS = ["auroc", "auprc", "f1", "sens"]


def summarise(runs_dir, dataset):
    values = {m: [] for m in METRICS}
    for seed in SEEDS:
        macro = json.loads((Path(runs_dir) / dataset / f"seed{seed}" / "metrics.json").read_text())["test"]["macro"]
        for m in METRICS:
            values[m].append(macro[m])
    return {m: (float(np.mean(v)), float(np.std(v, ddof=1))) for m, v in values.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--datasets", nargs="*", default=DATASETS)
    a = p.parse_args()
    table = {}
    for ds in a.datasets:
        table[ds] = summarise(a.runs, ds)
        cells = " & ".join(f"{mean:.3f} ({sd:.3f})" for mean, sd in table[ds].values())
        print(f"{ds:8s} {' '.join(METRICS)}: {cells}")
    print(json.dumps(table))


if __name__ == "__main__":
    main()
