import glob
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

from .splits import stratified_split_and_save

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "vm1", "vm3", "vm4", "vm5", "vm13", "vm20", "vm28", "vm62",
    "vm136", "vm146", "vm172", "vm174", "vm176",
    "pm41", "pm42", "pm43", "pm44", "pm87",
]


def run_pipeline(cfg):
    raw_dir = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["data"]["processed_dir"])
    seeds = cfg["data"]["seeds"]
    n_hours = cfg["data"]["n_hours"]
    label_names = cfg["dataset"]["label_names"]

    imputed_dir = raw_dir / "imputed_stage" / "csv"
    labels_path = Path(cfg["data"]["labels_path"])
    general_table_path = raw_dir / "general_table.csv"

    labels_df = pd.read_csv(labels_path)
    print(f"  Labels: {len(labels_df)} patients")

    gen = pd.read_csv(general_table_path)
    gen = gen[["patientid", "age", "sex"]].copy()
    gen["sex_encoded"] = gen["sex"].map({"M": 1.0, "F": 0.0})

    labels_df = labels_df.merge(gen[["patientid", "age", "sex_encoded"]], on="patientid", how="left")
    labels_df = labels_df.dropna(subset=["age", "sex_encoded"]).reset_index(drop=True)
    labels_df = labels_df.sort_values("patientid").reset_index(drop=True)

    cohort_set = set(labels_df["patientid"].values)
    patient_order = labels_df["patientid"].values
    n_patients = len(labels_df)
    print(f"  Cohort after age/sex filter: {n_patients} patients")

    y_all = labels_df[label_names].values.astype(np.float32)
    static_age = labels_df["age"].values.astype(np.float32)
    static_sex = labels_df["sex_encoded"].values.astype(np.float32)

    csv_files = sorted(glob.glob(str(imputed_dir / "*.csv")))
    n_ts_features = len(FEATURE_COLS)
    print(f"  Loading {len(csv_files)} staging CSVs...")

    x_all = np.full((n_patients, n_hours, n_ts_features), np.nan, dtype=np.float32)
    pid_to_idx = {pid: i for i, pid in enumerate(patient_order)}

    for file_i, csv_path in enumerate(csv_files):
        df = pd.read_csv(csv_path)
        df = df[df["patientid"].isin(cohort_set)]
        if len(df) == 0:
            continue

        df = df[df["reldatetime"] <= 86100].copy()
        df["hour_bin"] = (df["reldatetime"] // 3600).astype(int)
        df = df[df["hour_bin"] < n_hours]
        grouped = df.groupby(["patientid", "hour_bin"])[FEATURE_COLS].mean()

        for pid, patient_data in grouped.groupby(level=0):
            if pid not in pid_to_idx:
                continue
            idx = pid_to_idx[pid]
            hours = patient_data.index.get_level_values(1).values
            x_all[idx, hours] = patient_data.values.astype(np.float32)

        if (file_i + 1) % 50 == 0:
            print(f"    {file_i + 1}/{len(csv_files)} files...")

    print(f"  Patients with data: {int(np.sum(~np.isnan(x_all[:, 0, 0])))}")

    for i in range(n_patients):
        for f in range(n_ts_features):
            col = x_all[i, :, f]
            if np.any(np.isnan(col)):
                mask = np.isnan(col)
                idx_arr = np.where(~mask, np.arange(n_hours), 0)
                np.maximum.accumulate(idx_arr, out=idx_arr)
                col_filled = col[idx_arr]
                mask2 = np.isnan(col_filled)
                if np.any(mask2):
                    first_valid = np.argmax(~mask2)
                    col_filled[:first_valid] = col_filled[first_valid]
                x_all[i, :, f] = col_filled

    age_broadcast = np.broadcast_to(static_age[:, None, None], (n_patients, n_hours, 1)).copy()
    sex_broadcast = np.broadcast_to(static_sex[:, None, None], (n_patients, n_hours, 1)).copy()
    x_all = np.concatenate([x_all, age_broadcast, sex_broadcast], axis=2)

    assert np.isnan(x_all).sum() == 0, f"X has NaN"
    assert np.isnan(y_all).sum() == 0, f"y has NaN"
    print(f"  X: {x_all.shape}, y: {y_all.shape}")

    for j, name in enumerate(label_names):
        n_pos = int(y_all[:, j].sum())
        print(f"    {name}: {n_pos}/{n_patients} ({100 * n_pos / n_patients:.1f}%)")

    stratified_split_and_save(x_all, y_all, seeds, out_dir)
