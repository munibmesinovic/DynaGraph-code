import json

import numpy as np
import pandas as pd
import warnings
from pathlib import Path

from .splits import group_stratified_split_and_save

warnings.filterwarnings("ignore")

LAB_FEATURES_TO_DROP = [
    "urinary specific gravity", "troponin - I", "total bilirubin", "total protein",
    "pH", "paCO2", "paO2", "phosphate", "PTT", "O2 Sat (%)", "HCO3", "FiO2",
    "Base Excess", "-basos", "-eos", "-lymphs", "-monos", "-polys", "ALT (SGPT)",
    "AST (SGOT)", "MPV", "PT", "PT - INR", "lactate", "bedside glucose",
    "magnesium", "albumin", "alkaline phos.", "anion gap", "MCH", "RDW", "bicarbonate",
]


def _or_flag(df, candidate_cols):
    found = [c for c in candidate_cols if c in df.columns]
    df[found] = df[found].astype("bool")
    result = df[found[0]]
    for c in found[1:]:
        result = result | df[c]
    df[found] = df[found].astype("float")
    return result.astype(float)


def _diagnosis_flags(diagnoses, post_diagnoses):
    mi_cols = [
        "cardiovascular|chest pain / ASHD|acute coronary syndrome|acute myocardial infarction (no ST elevation)",
        "cardiovascular|chest pain / ASHD|acute coronary syndrome|acute myocardial infarction (with ST elevation)",
        "cardiovascular|chest pain / ASHD|acute coronary syndrome|s/p PTCA",
        "Acute MI location", "Acute MI location|inferior", "Acute MI location|non-Q",
        "Non-operative|Diagnosis|Cardiovascular|Infarction, acute myocardial (MI)",
        "Cardiovascular (R)|Myocardial Infarction",
        "Cardiovascular (R)|Myocardial Infarction|MI - date unknown",
        "Cardiovascular (R)|Myocardial Infarction|MI - remote",
        "Cardiovascular (R)|Myocardial Infarction|MI - within 6 months",
    ]
    diagnoses = diagnoses.assign(MI=_or_flag(diagnoses, mi_cols))

    arr_cols = [
        "cardiovascular|arrhythmias",
        "cardiovascular|arrhythmias|bradycardia",
        "cardiovascular|arrhythmias|sinus tachycardia",
        "cardiovascular|arrhythmias|ventricular tachycardia",
    ]
    post_diagnoses = post_diagnoses.assign(ARR=_or_flag(post_diagnoses, arr_cols))

    merged = post_diagnoses[["ARR"]].merge(diagnoses, left_index=True, right_index=True)
    merged["ARR"] = merged["ARR"].astype(int)

    af_cols = [
        "cardiovascular|arrhythmias|atrial fibrillation",
        "cardiovascular|arrhythmias|atrial fibrillation|with controlled ventricular response",
        "cardiovascular|arrhythmias|atrial fibrillation|with rapid ventricular response",
    ]
    post_diagnoses = post_diagnoses.assign(AF=_or_flag(post_diagnoses, af_cols))
    merged = post_diagnoses[["AF"]].merge(merged, left_index=True, right_index=True)
    merged["AF"] = merged["AF"].astype(int)

    hf_cols = [
        "cardiovascular|ventricular disorders|congestive heart failure",
        "cardiovascular|shock / hypotension|cardiogenic shock",
    ]
    post_diagnoses = post_diagnoses.assign(HF=_or_flag(post_diagnoses, hf_cols))
    merged = post_diagnoses[["HF"]].merge(merged, left_index=True, right_index=True)
    merged["HF"] = merged["HF"].astype(int)

    pad_cols = ["cardiovascular|vascular disorders|peripheral vascular ischemia"]
    post_diagnoses = post_diagnoses.assign(PAD=_or_flag(post_diagnoses, pad_cols))
    merged = post_diagnoses[["PAD"]].merge(merged, left_index=True, right_index=True)
    merged["PAD"] = merged["PAD"].astype(int)
    merged["MI"] = merged["MI"].astype(int)

    return merged[["MI", "PAD", "HF", "AF", "ARR"]]


def _resample_hourly(ts):
    ts = ts.groupby(level=[0, 1]).mean()
    ts.reset_index(level=1, inplace=True)
    ref = pd.to_datetime("2000-01-01 00:00:00")
    ts.time = pd.to_timedelta(ts.time, errors="coerce") + ref
    ts.time = ts.time.dt.ceil(freq="H")
    ts.time = ts.time - ref
    ts.time = pd.to_timedelta(ts.time, unit="T")
    ts.set_index("time", append=True, inplace=True)
    ts.reset_index(level=0, inplace=True)
    ts = ts.groupby("patient").resample("H", closed="right", label="right").mean()
    ts = ts.drop(columns="patient")
    return ts


def first_window(ts, n_hours):
    hours = ts.index.get_level_values(1) / pd.Timedelta(hours=1)
    keep = np.asarray(hours <= n_hours)
    ts = ts[keep]
    ts.index = pd.MultiIndex.from_arrays(
        [ts.index.get_level_values(0), np.asarray(hours[keep]).round().astype(int)],
        names=["patient", "hour"],
    )

    def one_stay(g):
        g = g.droplevel(0)
        grid = g.reindex(range(min(int(g.index.min()), 1), n_hours + 1))
        return grid.ffill().loc[1:n_hours].bfill()

    out = ts.groupby(level=0, group_keys=True).apply(one_stay)
    out.index.names = ["patient", "hour"]
    complete = ~out.isna().any(axis=1).groupby(level=0).any()
    return out[out.index.get_level_values(0).isin(complete[complete].index)]


def filter_min_stay(labels, n_hours):
    return labels[labels["unitdischargeoffset"] >= n_hours * 60]


def run_pipeline(cfg):
    raw_dir = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["data"]["processed_dir"])
    seeds = cfg["data"]["seeds"]
    n_hours = cfg["data"]["n_hours"]
    label_names = cfg["dataset"]["label_names"]
    log = []

    def note(msg):
        print(f"  {msg}")
        log.append(msg)

    labels = pd.read_csv(raw_dir / "preprocessed_labels.csv", index_col="patient")
    patient_of_stay = labels["uniquepid"]
    labels = labels[["unitdischargeoffset", "predictedhospitalmortality", "actualhospitalmortality"]]

    flats = pd.read_csv(raw_dir / "preprocessed_flat.csv", index_col="patient")
    flats = flats[flats["> 89"] != 1].drop(columns=["> 89", "nullweight", "nullheight"])

    diagnoses = pd.read_csv(raw_dir / "preprocessed_diagnoses.csv", index_col="patient")
    diag_post = pd.read_csv(raw_dir / "preprocessed_diagnoses_post.csv", index_col="patient")

    merged_diag = _diagnosis_flags(diagnoses, diag_post)
    note(f"Diagnosis flags: {len(merged_diag)} stays")

    final = merged_diag.merge(labels, left_index=True, right_index=True)
    final = final[final.MI == 1].drop(columns=["MI"])
    final = final.merge(flats, left_index=True, right_index=True)
    cardiac_cols = ["unittype_CCU-CTICU", "unittype_CSICU", "unittype_CTICU", "unittype_Cardiac ICU"]
    final = final.loc[final[cardiac_cols].any(axis=1)]
    note(f"Cardiac ICU MI stays: {final.index.nunique()}")

    final = final.loc[filter_min_stay(final, n_hours).index]
    note(f"Still in the unit at hour {n_hours}: {final.index.nunique()}")
    cohort = set(final.index)

    def window(ts_raw):
        ts_raw = ts_raw[ts_raw.index.get_level_values(0).isin(cohort)]
        return first_window(_resample_hourly(ts_raw), n_hours)

    ts_lab_raw = pd.read_csv(raw_dir / "timeseries_lab.csv", index_col=["patient", "time"])
    ts_lab_raw = ts_lab_raw.drop(columns=[c for c in LAB_FEATURES_TO_DROP if c in ts_lab_raw.columns])
    final = final.merge(window(ts_lab_raw), left_index=True, right_index=True)
    note(f"Every retained lab measured by hour {n_hours}: {final.index.get_level_values(0).nunique()}")

    ts_aper_raw = pd.read_csv(raw_dir / "timeseries_aperiodic.csv", index_col=["patient", "time"])
    final = final.merge(window(ts_aper_raw), left_index=True, right_index=True)
    note(f"Every aperiodic vital measured by hour {n_hours}: {final.index.get_level_values(0).nunique()}")

    periodic_path = cfg["data"].get("periodic_path")
    periodic_features = cfg["data"].get("periodic_features", [])
    if periodic_path and periodic_features:
        usecols = ["patientunitstayid", "observationoffset"] + periodic_features
        stays_now = set(final.index.get_level_values(0))
        parts = [c[c["patientunitstayid"].isin(stays_now)]
                 for c in pd.read_csv(periodic_path, usecols=usecols, chunksize=5_000_000)]
        ts_per = pd.concat(parts).rename(columns={"patientunitstayid": "patient", "observationoffset": "time"})
        ts_per = ts_per.set_index(["patient", pd.to_timedelta(ts_per["time"], unit="min")])[periodic_features]
        final = final.merge(window(ts_per), left_index=True, right_index=True)
        note(f"Every periodic vital measured by hour {n_hours}: {final.index.get_level_values(0).nunique()}")

    final["predictedhospitalmortality"] = final["predictedhospitalmortality"].replace([-1], [0])

    stay_ids = final.index.get_level_values(0).unique()
    n_stays = len(stay_ids)
    note(f"Final cohort: {n_stays} stays from {patient_of_stay.loc[stay_ids].nunique()} patients")

    label_col_map = {"mortality": "actualhospitalmortality", "PAD": "PAD", "HF": "HF", "AF": "AF", "ARR": "ARR"}
    y_df = final[[label_col_map[ln] for ln in label_names]].droplevel(1)
    y_df = y_df[~y_df.index.duplicated(keep="first")]
    y_df.columns = label_names

    drop_from_x = [c for c in ["unitdischargeoffset", "actualhospitalmortality", "PAD", "HF", "ARR", "AF"] if c in final.columns]
    drop_from_x += [c for c in cfg["data"].get("drop_features", []) if c in final.columns]
    X_df = final.drop(columns=drop_from_x)

    x_all = np.empty((n_stays, n_hours, X_df.shape[1]), dtype=np.float32)
    for i, sid in enumerate(stay_ids):
        sample = X_df.loc[sid]
        assert sample.index.tolist() == list(range(1, n_hours + 1)), f"stay {sid}: hours {sample.index.tolist()}"
        x_all[i] = sample.values
    y_all = y_df.loc[stay_ids].values.astype(np.float32)

    assert np.isnan(x_all).sum() == 0, "X has NaN"
    assert np.isnan(y_all).sum() == 0, "y has NaN"
    note(f"X: {x_all.shape}, y: {y_all.shape}")
    for j, name in enumerate(label_names):
        n_pos = int(y_all[:, j].sum())
        note(f"{name}: {n_pos}/{n_stays} ({100 * n_pos / n_stays:.1f}%)")

    groups = patient_of_stay.loc[stay_ids].to_numpy()
    group_stratified_split_and_save(x_all, y_all, groups, np.asarray(stay_ids), seeds, out_dir)
    (out_dir / "feature_names.json").write_text(json.dumps(list(X_df.columns), indent=1))
    (out_dir / "label_names.json").write_text(json.dumps(list(label_names)))
    (out_dir / "processing_log.txt").write_text("\n".join(log) + "\n")
