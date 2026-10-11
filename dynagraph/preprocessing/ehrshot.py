import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

BENCHMARK_DIRS = {
    "AMI": "new_acutemi",
    "PANCAN": "new_pancan",
    "HYPERTENSION": "new_hypertension",
    "HYPERLIPIDEMIA": "new_hyperlipidemia",
}

LOINC_MAP = {
    "LOINC/8867-4": "HR", "LOINC/8480-6": "SBP", "LOINC/8462-4": "DBP",
    "LOINC/8310-5": "TEMP", "LOINC/LP21258-6": "OS", "LOINC/9279-1": "RR",
    "LOINC/2345-7": "Glucose", "LOINC/2951-2": "Sodium",
    "LOINC/2823-3": "Potassium", "LOINC/17861-6": "Calcium",
    "LOINC/2028-9": "Bicarbonate", "LOINC/3094-0": "BUN",
    "LOINC/2160-0": "Creatinine", "LOINC/718-7": "Haemoglobin",
    "LOINC/4544-3": "Haematocrit", "LOINC/777-3": "Platelets",
    "LOINC/788-0": "RDW", "LOINC/2075-0": "Chloride",
}
TS_FEATURE_NAMES = list(LOINC_MAP.values())

RACE_CODES = {"Race/2": "Asian", "Race/3": "Black", "Race/4": "Native", "Race/5": "White"}


def _load_main(raw_dir):
    csv_path = raw_dir / "ehrshot.csv"
    df = pd.read_csv(csv_path, low_memory=False)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    print(f"  ehrshot.csv: {len(df):,} rows, {df['patient_id'].nunique():,} patients")
    return df


def _extract_labels(raw_dir, label_names):
    label_dfs = {}
    for cond, dirname in BENCHMARK_DIRS.items():
        path = raw_dir / dirname / "labeled_patients.csv"
        ldf = pd.read_csv(path)
        ldf = ldf[["patient_id", "prediction_time", "value"]].copy()
        ldf["prediction_time"] = pd.to_datetime(ldf["prediction_time"])
        ldf.rename(columns={"value": cond}, inplace=True)
        ldf = ldf.drop_duplicates(["patient_id", "prediction_time"])
        label_dfs[cond] = ldf

    merged = label_dfs["AMI"]
    for cond in list(BENCHMARK_DIRS.keys())[1:]:
        if cond in label_names:
            merged = merged.merge(
                label_dfs[cond], on=["patient_id", "prediction_time"], how="left"
            )

    for cond in label_names:
        merged[cond] = merged[cond].fillna(0).astype(int)

    print(f"  Labels: {len(merged):,} samples, {merged['patient_id'].nunique():,} patients")
    return merged


def _extract_demographics(df_main):
    person = df_main[df_main["omop_table"] == "person"].copy()

    birth = person[person["code"] == "SNOMED/3950001"][["patient_id", "start"]].copy()
    birth["start"] = pd.to_datetime(birth["start"])
    birth["birth_year"] = birth["start"].dt.year
    birth = birth[["patient_id", "birth_year"]].drop_duplicates(subset="patient_id", keep="first")

    sex = person[person["code"].isin(["Gender/M", "Gender/F"])][["patient_id", "code"]].copy()
    sex["sex"] = (sex["code"] == "Gender/M").astype(int)
    sex = sex[["patient_id", "sex"]].drop_duplicates(subset="patient_id", keep="first")

    race = person[person["code"].isin(RACE_CODES.keys())][["patient_id", "code"]].copy()
    race["race_name"] = race["code"].map(RACE_CODES)
    race = race.drop_duplicates(subset="patient_id", keep="first")
    race_dummies = pd.get_dummies(
        race[["patient_id", "race_name"]], columns=["race_name"], prefix="", prefix_sep=""
    )
    race_dummies = race_dummies.groupby("patient_id").max().reset_index()

    demo = birth.merge(sex, on="patient_id", how="left")
    demo = demo.merge(race_dummies, on="patient_id", how="left")
    for rc in ["Asian", "Black", "Native", "White"]:
        if rc not in demo.columns:
            demo[rc] = 0
    race_cols = ["Asian", "Black", "Native", "White"]
    demo[race_cols] = demo[race_cols].fillna(0).astype(int)
    demo["Unknown_Race"] = (demo[race_cols].sum(axis=1) == 0).astype(int)
    demo = demo.fillna(0).drop_duplicates(subset="patient_id")

    print(f"  Demographics: {len(demo):,} patients")
    return demo


def _extract_loinc_features(df_main, cohort_ids):
    cohort_set = set(cohort_ids)
    feature_dfs = {}
    for code, name in LOINC_MAP.items():
        fdf = df_main[
            (df_main["code"] == code) & (df_main["patient_id"].isin(cohort_set))
        ][["patient_id", "start", "value"]].copy()
        fdf.rename(columns={"value": name}, inplace=True)
        fdf[name] = pd.to_numeric(fdf[name], errors="coerce")
        fdf = fdf.groupby(["patient_id", "start"])[[name]].mean().reset_index()
        feature_dfs[name] = fdf
    return feature_dfs


def _merge_and_resample_daily(feature_dfs):
    names = list(LOINC_MAP.values())
    merged = feature_dfs[names[0]]
    for name in names[1:]:
        merged = merged.merge(feature_dfs[name], on=["patient_id", "start"], how="outer")

    for name in names:
        merged[name] = pd.to_numeric(merged[name], errors="coerce")

    merged["start"] = pd.to_datetime(merged["start"], errors="coerce")
    merged = merged.groupby(["patient_id", "start"])[names].mean().reset_index()

    merged = merged.set_index("start")
    parts = []
    patient_ids = sorted(merged["patient_id"].unique())
    for i, pid in enumerate(patient_ids):
        group = merged.loc[merged["patient_id"] == pid, names]
        daily = group.resample("D").mean()
        daily["patient_id"] = pid
        parts.append(daily)
        if (i + 1) % 500 == 0:
            print(f"    Resampled {i + 1:,}/{len(patient_ids):,} patients...")
    ts_daily = pd.concat(parts).reset_index()
    print(f"  Daily grid: {len(ts_daily):,} rows, {ts_daily['patient_id'].nunique():,} patients")
    return ts_daily


def _build_demo_row(demo_idx, pid, year):
    birth_year = int(demo_idx.loc[pid, "birth_year"])
    age = year - birth_year
    return np.array([
        year, age,
        int(demo_idx.loc[pid, "sex"]),
        int(demo_idx.loc[pid, "White"]),
        int(demo_idx.loc[pid, "Asian"]),
        int(demo_idx.loc[pid, "Black"]),
        int(demo_idx.loc[pid, "Native"]),
        int(demo_idx.loc[pid, "Unknown_Race"]),
    ], dtype=np.float32)


def _window_and_build(ts_daily_raw, labels, demo, label_names, n_steps):
    names = TS_FEATURE_NAMES
    demo_idx = demo.set_index("patient_id")
    valid_demo_pids = set(demo_idx.index)

    ts_by_pid = {}
    for pid, group in ts_daily_raw.groupby("patient_id"):
        g = group.sort_values("start")
        ts_by_pid[pid] = {"dates": g["start"].values, "values": g[names].values}

    X_list, y_list, pid_list = [], [], []
    dropped_short, dropped_nan = 0, 0

    for _, row in labels.iterrows():
        pid = row["patient_id"]
        pt = pd.Timestamp(row["prediction_time"])

        if pid not in ts_by_pid or pid not in valid_demo_pids:
            continue

        data = ts_by_pid[pid]
        mask = data["dates"] < np.datetime64(pt)
        valid_vals = data["values"][mask]

        if len(valid_vals) < n_steps:
            dropped_short += 1
            continue

        window = pd.DataFrame(valid_vals[-n_steps:], columns=names).ffill().bfill()
        ts_vals = window.values.astype(np.float32)

        if np.isnan(ts_vals).any():
            dropped_nan += 1
            continue

        year = pt.year
        demo_row = _build_demo_row(demo_idx, pid, year)
        demo_block = np.tile(demo_row, (n_steps, 1))
        X_row = np.concatenate([ts_vals, demo_block], axis=1)
        y_row = np.array([row[cond] for cond in label_names], dtype=np.float32)

        X_list.append(X_row)
        y_list.append(y_row)
        pid_list.append(pid)

    X_all = np.stack(X_list).astype(np.float32)
    y_all = np.stack(y_list).astype(np.float32)
    sample_pids = np.array(pid_list)

    print(f"  Valid samples: {len(X_list):,} from {len(set(pid_list)):,} patients")
    print(f"  Dropped: {dropped_short} short, {dropped_nan} NaN")
    return X_all, y_all, sample_pids


def _group_split_save(X_all, y_all, patient_ids, seeds, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        train_idx, temp_idx = next(gss1.split(X_all, groups=patient_ids))

        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
        val_rel, test_rel = next(gss2.split(
            X_all[temp_idx], groups=patient_ids[temp_idx]
        ))
        val_idx = temp_idx[val_rel]
        test_idx = temp_idx[test_rel]

        assert len(set(patient_ids[train_idx]) & set(patient_ids[val_idx])) == 0
        assert len(set(patient_ids[train_idx]) & set(patient_ids[test_idx])) == 0
        assert len(set(patient_ids[val_idx]) & set(patient_ids[test_idx])) == 0
        assert len(train_idx) + len(val_idx) + len(test_idx) == len(X_all)

        np.save(out_dir / f"X_train_{seed}.npy", X_all[train_idx])
        np.save(out_dir / f"X_val_{seed}.npy", X_all[val_idx])
        np.save(out_dir / f"X_test_{seed}.npy", X_all[test_idx])
        np.save(out_dir / f"y_train_{seed}.npy", y_all[train_idx])
        np.save(out_dir / f"y_val_{seed}.npy", y_all[val_idx])
        np.save(out_dir / f"y_test_{seed}.npy", y_all[test_idx])

        print(f"  seed {seed}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    print(f"  Saved to {out_dir}/")


def run_pipeline(cfg):
    raw_dir = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["data"]["processed_dir"])
    label_names = cfg["dataset"]["label_names"]
    n_steps = cfg["data"]["n_steps"]

    out_dir.mkdir(parents=True, exist_ok=True)

    df_main = _load_main(raw_dir)
    labels = _extract_labels(raw_dir, label_names)
    demo = _extract_demographics(df_main)

    cohort_ids = sorted(labels["patient_id"].unique())
    print(f"  Cohort: {len(cohort_ids):,} patients")

    feature_dfs = _extract_loinc_features(df_main, cohort_ids)
    del df_main

    ts_daily = _merge_and_resample_daily(feature_dfs)
    del feature_dfs

    X_all, y_all, sample_pids = _window_and_build(
        ts_daily, labels, demo, label_names, n_steps
    )

    assert np.isnan(X_all).sum() == 0
    assert np.isnan(y_all).sum() == 0
    print(f"  X: {X_all.shape}, y: {y_all.shape}")

    for j, name in enumerate(label_names):
        n_pos = int(y_all[:, j].sum())
        print(f"    {name}: {n_pos}/{len(X_all)} ({100 * n_pos / len(X_all):.1f}%)")

    _group_split_save(X_all, y_all, sample_pids, cfg["data"]["seeds"], out_dir)
