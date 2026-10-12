# DynaGraph

Code for "DynaGraph: Interpretable Dynamic Graph Learning for Temporal Electronic Health Records".

## Data

MIMIC-III, eICU, and HiRID are available on PhysioNet, and EHRSHOT from Stanford, after credentialed access. Put the files of each dataset in `data/<dataset>/raw`:

- MIMIC-III: `all_hourly_data.h5` from MIMIC-Extract (https://github.com/MLforHealth/MIMIC_Extract), saved as `MIMIC_EXTRACT_all_hourly_data.h5`, and `phenotypes.csv`, one row per `icustay_id` with the phenotype labels of the MIMIC-III benchmark (https://github.com/YerevaNN/mimic3-benchmarks).
- eICU: `preprocessed_labels.csv`, `preprocessed_flat.csv`, and `preprocessed_diagnoses.csv` from the eICU preprocessing of Rocheteau et al. (https://github.com/EmmaRocheteau/TPC-LoS-prediction), and their lab and aperiodic vital sign exports after `reconfigure_timeseries` in `eICU_preprocessing/timeseries.py`, saved as `timeseries_lab.csv` and `timeseries_aperiodic.csv`. Add `diagnosis.csv.gz` and `vitalPeriodic.csv.gz` from eICU 2.0.
- HiRID: the imputed stage (`imputed_stage/csv`) and `general_table.csv` of HiRID 1.1.1, and `hirid_labels.csv`, with one row per admission of at least 24 hours: `mortality` from `discharge_status` in `general_table.csv`, and `circulatory_failure` equal to 1 if `Dynamic_CircFailure_12Hours` from the HiRID-ICU-Benchmark (https://github.com/ratschlab/HIRID-ICU-Benchmark) is 1 at any time after hour 24.
- EHRSHOT: `ehrshot.csv` and the `labeled_patients.csv` of each task from the benchmark (https://github.com/som-shahlab/ehrshot-benchmark).

## Usage

```
pip install -r requirements.txt
python -m dynagraph.preprocess --dataset mimic --config configs/preprocessing/mimic.yaml
bash run_seeds.sh mimic
python -m dynagraph.aggregate --runs runs
```

Datasets are `mimic`, `eicu`, `hirid`, and `ehrshot`. `run_seeds.sh` trains the five seeds used in the paper (42, 1992, 1709, 250, and 213), and `dynagraph.aggregate` prints the mean and standard deviation of the test AUROC, AUPRC, F1, and sensitivity for each dataset.
