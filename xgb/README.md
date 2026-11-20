# Tolosa-SW storm surge bias correction with XGBoost

Statistical post-processing of the storm surge forecast by the **Tolosa-SW**
ocean model at French Atlantic and English Channel tide gauges, developed
within the FOCCUS project.

For each tide gauge, an XGBoost model learns the **surge error** of
Tolosa-SW (`surge_obs − surge_mod`) from atmospheric, wave, tidal and IBI
ocean predictors. The corrected surge is then:

```
surge_corrected = surge_mod + XGBoost(predictors)
```

## Pipeline

```
raw per-station CSV ──► build_features.py ──► enriched CSV ──► train_xgb.py ──► models, scores, figures, parquet
                          (feature engineering)                 (nested CV, tuning, evaluation)
```

| File | Role |
|---|---|
| `build_features.py` | Adds derived predictors to each station CSV: wind stress and cross/along-shore projections, tidal phase and range, wave setup proxy, IBI currents, interactions, calendar terms, lags, centred derivatives, rolling statistics. |
| `train_xgb.py` | Per station: chronological train/test split, leave-one-winter-out nested cross-validation with Optuna tuning, final model, scores, storm events, diagnostic figures, export of a validation period. |
| `station_io.py` | Station identification (aliases → canonical identifier) and reading of the enriched CSV files. |
| `stations.json` | Station registry: canonical identifiers and their aliases. |
| `config.yaml` | All settings shared by both scripts. |
| `run_xgb` | SLURM job script running the full chain. |

## Inputs

### 1. One CSV file per station

- Location: `paths.input_csv_dir`; file name = station name or alias
  (e.g. `BREST.csv`, `BOULOGNE_SUR_MER.csv`).
- First column: timestamp (`time`); regular time step of `cadence_minutes`
  (10 min by default).
- Missing values are allowed in the predictors: XGBoost handles them natively.

Columns used (names are set in `config.yaml`, defaults shown):

| Variable | Default column | Config key | Required |
|---|---|---|---|
| Observed surge (tide gauge) | `surge_obs` | `columns.surge_obs` | yes |
| Tolosa-SW modelled surge | `surge_mod` | `columns.surge_mod` | yes |
| Target: `surge_obs − surge_mod` | `surge_error` | `target` | yes |
| Predicted tide | `tide` | `features.tide_col` | recommended |
| 10 m wind components | `u10`, `v10` | `features.u10_col`, `features.v10_col` | recommended |
| …or wind speed + direction (meteorological convention) | `wind_speed`, `wind_dir` | `features.wind_speed_col`, `features.wind_dir_col` | alternative |
| Mean sea-level pressure | `MSL` | `features.msl_col` | recommended |
| Significant wave height | `Hs` | `features.hs_col` | recommended |
| Wave direction (nautical convention) | `wave_dir` | `features.wave_dir_col` | optional |
| IBI sea surface height | `IBI_ssh` | `features.ibi_ssh_col` | optional |
| IBI surface currents | `IBI_u`, `IBI_v` | `features.ibi_u_col`, `features.ibi_v_col` | optional |

A feature whose input column is missing is skipped with a message. Every
other column of the CSV is kept and used as a predictor, **except** the
target and the columns listed in `drop_predictors`. Any column derived from
the observations (observed water level, observed surge, their averages…)
must be listed there, otherwise it leaks the target.

### 2. `config.yaml`

| Section | Content |
|---|---|
| `cadence_minutes` | Time step of the CSV files. |
| `paths` | Raw CSV, enriched CSV and output directories. |
| `dates` | `end_train`, `end_test`, optional `validation` period (see below). |
| `columns`, `target`, `drop_predictors` | Observed / modelled surge, target, excluded columns. |
| `stations` | Per-station `coast_normal_deg`: direction of the coast normal, pointing from sea to land, in degrees counter-clockwise from east. Required for the cross/along-shore features. |
| `features` | Input column names, lags, derivatives and rolling statistics. |
| `sample_weight` | Weighting of the training samples by the observed surge. |
| `cv`, `optuna`, `xgb_fixed`, `xgb_param_space` | Cross-validation and hyperparameter tuning. |
| `events` | Threshold quantile, minimum duration and merge gap of storm events. |
| `importance` | Optional permutation importance. |
| `cache` | Reuse of trained models and per-fold checkpoints. |

### 3. `stations.json`

Registry mapping each station to its aliases, so that differently named files
(`BOUCAU-BAYONNE`, `BAYONNE_BOUCAU`…) resolve to the same identifier. The
identifier names every output file.

## Time periods

| Period | Definition | Use |
|---|---|---|
| Train | start of the data → `end_train` | Nested CV (one fold per winter, October → September by default), hyperparameter tuning, final model, event threshold. |
| Test | `end_train` → `end_test` | Independent evaluation: scores, events, figures. Never used for training or tuning. |
| Validation | `validation.start` → `validation.end` | Inference and parquet export only, keeping time steps without observation (continuous series for downstream tools). |

To compare settings, use the cross-validation scores
(`reports/cv_{station}.json`), not the test scores.

## Outputs

Written under `paths.output_root`:

```
models/XGB_{station}.joblib              final model
reports/cv_{station}.json                CV folds, best hyperparameters, cache context
reports/scores_test_{station}.json       bias, MAE, RMSE (all data and above the event threshold)
reports/events_test_{station}.csv        per-event peak error, peak timing, RMSE
reports/summary_all_stations.csv         one row per station
figs/{importance,scatter,qqplot,events}_test_{station}.pdf
predictions/validation_{station}.parquet obs, Tolosa-SW, Tolosa-SW XGB-corrected
```

## Usage

```bash
# Full chain on a SLURM cluster
sbatch run_xgb config.yaml
sbatch run_xgb config.yaml -s brest calais     # selected stations
sbatch run_xgb config.yaml --no-features       # enriched CSV files already up to date
sbatch run_xgb config.yaml --no-cache          # force re-tuning

# Or step by step
python build_features.py --config config.yaml [-s STATION ...]
python train_xgb.py      --config config.yaml [-s STATION ...] [--no-cache]
```

Environment variables of `run_xgb`: `SITE_CONFIG` (script sourced first),
`CONDA_SH`, `CONDA_ENV`. Adapt the `#SBATCH` directives to your cluster.

A job stopped by its time limit can be resubmitted unchanged: trained stations
are taken from the cache and only the missing CV folds are recomputed.

## Requirements

Python ≥ 3.9 with `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`,
`optuna`, `pyarrow`, `matplotlib` (≥ 3.9), `seaborn`, `tabulate`, `threadpoolctl`,
`joblib`, `pyyaml`.
