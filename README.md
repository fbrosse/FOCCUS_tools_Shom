# FOCCUS – Tolosa-SW bias correction tools

Tools developed at Shom for the Tolosa-SW configuration in the Horizon Europe
FOCCUS project (https://foccus-project.eu/).

They statistically correct the **storm-surge** outputs of the unstructured
barotropic model Tolosa-SW over the Bay of Biscay and the English Channel.
The Tolosa surge is the difference of two runs, **surge = full − ms**:

- **full**: run with atmospheric forcing (wind and pressure), including the
  inverse barometer effect;
- **ms**: tide-only run.

The tools are divided into two complementary parts:

| Part | Target | Scale | Method |
|---|---|---|---|
| 2D correction (`eof_cca/`) | Surge bias over the whole mesh | Low frequency (periods ≥ 20 days), daily | EOF/CCA regression on IBI fields |
| 1D correction (`xgb/`) | Surge error at the tide gauges | All frequencies, 10 min | Gradient boosting (XGBoost) |

The 2D part handles the slow, basin-scale bias. Faster variations, below
20 days, are left to the 1D part at the tide gauges.

## 2D correction: EOF/CCA regression on IBI (`eof_cca/`)

Learns the relationship between the CMEMS IBI regional model and the Tolosa
surge bias, then predicts a correction field at the elements of the Tolosa-SW
mesh.

- **Reference bias**: DUACS L4 SLA (+ AVISO DAC) minus the Tolosa surge,
  daily, with its temporal mean removed.
- **Predictor**: IBI SSH (detided) minus DAC.
- **Method**: band-pass filtering (e.g. 20–128 days, truncated at a forecast
  horizon), then a CCA between the leading EOFs of IBI and of the bias.
- **Inputs** (paths in `config.json`): DUACS L4 SLA, IBI SSH, AVISO DAC,
  Tolosa-SW 2D outputs (full and ms runs) and 1D extractions at the ports,
  and sparse projection matrices between the grids.
- **Programs**:
  - `eof_cca.py` calibrates the model, validates it at the mesh elements and
    exports daily NetCDF fields (model, correction, corrected model) with the
    UGRID mesh;
  - `predict_correction.py` applies a calibrated model to new IBI fields and
    writes the same NetCDF fields over the prediction period.

See `eof_cca/README.md` for the full list of inputs, options and outputs.

## 1D correction: gradient boosting at the tide gauges (`xgb/`)

Predicts the surge error of Tolosa-SW (observed surge − modelled surge) at
the tide gauges with XGBoost, one model per station.

- **Inputs**: one `.csv` file per tide gauge site, at a 10-minute time step.
  Each file holds the observed and modelled surge and the predictors from
  IBI, atmospheric and wave models. Observed sea level comes from REFMAR.
- **Programs**:
  - `build_features.py` builds the predictors (enriched CSV files);
  - `train_xgb.py` trains one XGBoost model per station (leave-one-winter-out
    cross-validation, Optuna hyperparameter search);
  - `station_io.py` and `stations.json`: station metadata and file reading;
  - `config.yaml` holds the stations and the run parameters;
  - `run_xgb` is the SLURM launch script.

## Requirements

Python ≥ 3.9. The 2D part uses `numpy`, `scipy`, `pandas`, `xarray`, `dask`,
`netCDF4`, `scikit-learn`, `joblib` and `matplotlib` (plus `pyarrow` for the
Parquet export). The 1D part uses `xgboost` and `optuna`.
