import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")

VITALS_LABS = [
    "glucose", "hematocrit", "sodium", "creatinine", "potassium",
    "blood urea nitrogen", "oxygen saturation", "chloride", "hemoglobin",
    "platelets", "bicarbonate", "white blood cell count", "heart rate",
    "diastolic blood pressure", "systolic blood pressure", "mean blood pressure",
    "respiratory rate", "red blood cell count",
    "mean corpuscular hemoglobin concentration", "mean corpuscular hemoglobin",
    "mean corpuscular volume", "anion gap", "temperature", "magnesium",
    "prothrombin time inr", "prothrombin time pt", "partial thromboplastin time",
    "phosphorous",
]

INTERVENTIONS = [
    "vent", "vaso", "adenosine", "dobutamine", "dopamine", "epinephrine",
    "isuprel", "milrinone", "norepinephrine", "phenylephrine", "vasopressin",
    "colloid_bolus", "crystalloid_bolus", "nivdurations",
]

PHENO_LABEL_COLS = [
    "Shock", "Acute cerebrovascular disease", "Acute myocardial infarction",
    "Cardiac dysrhythmias",
    "Chronic kidney disease",
    "Chronic obstructive pulmonary disease and bronchiectasis",
    "Congestive heart failure- nonhypertensive",
]

LABEL_COL_MAP = {
    "mort_icu": "mort_icu",
    "mort_hosp": "mort_hosp",
    "readmission_30": "readmission_30",
    "Shock": "Shock",
    "Acute_cerebrovascular": "Acute cerebrovascular disease",
    "AMI": "Acute myocardial infarction",
    "Cardiac_dysrhythmias": "Cardiac dysrhythmias",
    "CKD": "Chronic kidney disease",
    "COPD": "Chronic obstructive pulmonary disease and bronchiectasis",
    "CHF": "Congestive heart failure- nonhypertensive",
}


def _group_aware_split_and_save(x_all, y_all, subject_ids, seeds, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unique_subjects = np.unique(subject_ids)
    subj_to_rows = {s: [] for s in unique_subjects}
    for i, s in enumerate(subject_ids):
        subj_to_rows[s].append(i)

    subj_strat = np.array(
        [int(y_all[subj_to_rows[s], 0].max()) for s in unique_subjects], dtype=int,
    )

    for seed in seeds:
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        train_subj_idx, temp_subj_idx = next(sss1.split(unique_subjects, subj_strat))

        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
        val_rel, test_rel = next(
            sss2.split(unique_subjects[temp_subj_idx], subj_strat[temp_subj_idx])
        )
        val_subj_idx = temp_subj_idx[val_rel]
        test_subj_idx = temp_subj_idx[test_rel]

        train_rows = np.sort(np.concatenate(
            [subj_to_rows[unique_subjects[i]] for i in train_subj_idx]
        ))
        val_rows = np.sort(np.concatenate(
            [subj_to_rows[unique_subjects[i]] for i in val_subj_idx]
        ))
        test_rows = np.sort(np.concatenate(
            [subj_to_rows[unique_subjects[i]] for i in test_subj_idx]
        ))

        assert len(set(subject_ids[train_rows]) & set(subject_ids[val_rows])) == 0
        assert len(set(subject_ids[train_rows]) & set(subject_ids[test_rows])) == 0
        assert len(set(subject_ids[val_rows]) & set(subject_ids[test_rows])) == 0
        assert len(train_rows) + len(val_rows) + len(test_rows) == len(x_all)

        np.save(out_dir / f"X_train_{seed}.npy", x_all[train_rows])
        np.save(out_dir / f"X_val_{seed}.npy", x_all[val_rows])
        np.save(out_dir / f"X_test_{seed}.npy", x_all[test_rows])
        np.save(out_dir / f"y_train_{seed}.npy", y_all[train_rows])
        np.save(out_dir / f"y_val_{seed}.npy", y_all[val_rows])
        np.save(out_dir / f"y_test_{seed}.npy", y_all[test_rows])

        print(f"  seed {seed}: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")

    print(f"  Saved to {out_dir}/")


def run_pipeline(cfg):
    raw_dir = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["data"]["processed_dir"])
    seeds = cfg["data"]["seeds"]
    n_hours = cfg["data"]["n_hours"]
    label_names = cfg["dataset"]["label_names"]

    h5_path = raw_dir / "MIMIC_EXTRACT_all_hourly_data.h5"
    pheno_path = raw_dir / "phenotypes.csv"

    patients = pd.read_hdf(h5_path, key="patients")
    patients = patients[patients["max_hours"] >= 24]
    patients = patients[patients["age"] >= 18]
    patients = patients.reset_index()
    n_unique = patients["subject_id"].nunique()
    print(f"  Filtered patients: {len(patients)} stays from {n_unique} subjects")

    cohort_icustays = set(patients["icustay_id"].values)

    pheno = pd.read_csv(pheno_path)
    pheno = pheno.rename(columns={"Congestive heart failure; nonhypertensive": "Congestive heart failure- nonhypertensive"})
    pheno = pheno[["icustay_id"] + PHENO_LABEL_COLS].copy()
    patients = patients.merge(pheno, on="icustay_id", how="inner")
    cohort_icustays = set(patients["icustay_id"].values)
    print(f"  After phenotype merge: {len(patients)} stays")

    print("  Loading vitals_labs_mean...")
    vlm = pd.read_hdf(h5_path, key="vitals_labs_mean")
    vlm = vlm[vlm.index.get_level_values("icustay_id").isin(cohort_icustays)]

    print("  Loading interventions...")
    intv = pd.read_hdf(h5_path, key="interventions")
    intv = intv[intv.index.get_level_values("icustay_id").isin(cohort_icustays)]

    hours = vlm.index.get_level_values("hours_in")
    vlm = vlm[(hours >= 0) & (hours < n_hours)]
    hours_intv = intv.index.get_level_values("hours_in")
    intv = intv[(hours_intv >= 0) & (hours_intv < n_hours)]

    vlm_flat = vlm.copy()
    vlm_flat.columns = [c[0] if isinstance(c, tuple) else c for c in vlm_flat.columns]
    vlm_sel = vlm_flat[VITALS_LABS]
    intv_sel = intv[INTERVENTIONS]

    common_idx = vlm_sel.index.intersection(intv_sel.index)
    vlm_sel = vlm_sel.loc[common_idx]
    intv_sel = intv_sel.loc[common_idx]
    combined = pd.concat([vlm_sel, intv_sel], axis=1)

    combined = combined.groupby(level=[0, 1, 2]).ffill()
    combined = combined.groupby(level=[0, 1, 2]).bfill()

    has_nan = combined.isna().any(axis=1)
    stays_with_nan = has_nan.groupby(level=2).any()
    clean_stays = stays_with_nan[~stays_with_nan].index
    combined = combined[combined.index.get_level_values("icustay_id").isin(clean_stays)]

    cohort_icustays = set(clean_stays)
    patients = patients[patients["icustay_id"].isin(cohort_icustays)].reset_index(drop=True)
    patients = patients.sort_values("icustay_id").reset_index(drop=True)
    n_patients = len(patients)
    n_unique = patients["subject_id"].nunique()
    print(f"  After imputation: {n_patients} stays from {n_unique} subjects")

    patient_order = patients["icustay_id"].values
    pid_to_idx = {pid: i for i, pid in enumerate(patient_order)}

    n_ts_features = len(VITALS_LABS) + len(INTERVENTIONS)
    feature_cols = VITALS_LABS + INTERVENTIONS
    x_ts = np.full((n_patients, n_hours, n_ts_features), np.nan, dtype=np.float32)

    for icustay_id in patient_order:
        idx = pid_to_idx[icustay_id]
        mask = combined.index.get_level_values("icustay_id") == icustay_id
        patient_data = combined[mask]
        hours_in = patient_data.index.get_level_values("hours_in").values.astype(int)
        valid = (hours_in >= 0) & (hours_in < n_hours)
        x_ts[idx, hours_in[valid]] = patient_data[feature_cols].values[valid].astype(np.float32)

    if np.isnan(x_ts).sum() > 0:
        for i in range(n_patients):
            for f in range(n_ts_features):
                col = x_ts[i, :, f]
                if np.any(np.isnan(col)):
                    mask_nan = np.isnan(col)
                    idx_arr = np.where(~mask_nan, np.arange(n_hours), 0)
                    np.maximum.accumulate(idx_arr, out=idx_arr)
                    col_filled = col[idx_arr]
                    mask2 = np.isnan(col_filled)
                    if np.any(mask2):
                        first_valid = np.argmax(~mask2)
                        col_filled[:first_valid] = col_filled[first_valid]
                    x_ts[i, :, f] = col_filled

        remaining_nan = np.isnan(x_ts).sum()
        if remaining_nan > 0:
            has_nan_mask = np.isnan(x_ts).any(axis=(1, 2))
            keep_mask = ~has_nan_mask
            x_ts = x_ts[keep_mask]
            patients = patients[keep_mask].reset_index(drop=True)
            n_patients = len(patients)

    static_age = patients["age"].values.astype(np.float32)

    gender_col = None
    for col_name in ["gender", "sex", "gender_M", "is_male"]:
        if col_name in patients.columns:
            gender_col = col_name
            break

    if gender_col is not None:
        sex_vals = patients[gender_col]
        if hasattr(sex_vals, "cat") or sex_vals.dtype == object:
            static_sex = np.array([1.0 if str(s) == "M" else 0.0 for s in sex_vals], dtype=np.float32)
        else:
            static_sex = sex_vals.values.astype(np.float32)
    else:
        static_sex = np.zeros(n_patients, dtype=np.float32)

    age_broadcast = np.broadcast_to(static_age[:, None, None], (n_patients, n_hours, 1)).copy()
    sex_broadcast = np.broadcast_to(static_sex[:, None, None], (n_patients, n_hours, 1)).copy()
    x_all = np.concatenate([x_ts, age_broadcast, sex_broadcast], axis=2)

    y_cols = [LABEL_COL_MAP[name] for name in label_names]
    y_all = patients[y_cols].values.astype(np.float32)

    assert np.isnan(x_all).sum() == 0, f"X has NaN"
    assert np.isnan(y_all).sum() == 0, f"y has NaN"
    print(f"  X: {x_all.shape}, y: {y_all.shape}")

    for j, name in enumerate(label_names):
        n_pos = int(y_all[:, j].sum())
        print(f"    {name}: {n_pos}/{n_patients} ({100 * n_pos / n_patients:.1f}%)")

    subject_ids = patients["subject_id"].values
    _group_aware_split_and_save(x_all, y_all, subject_ids, seeds, out_dir)
