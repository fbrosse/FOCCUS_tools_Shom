# Surge bias correction of Tolosa-SW from IBI (EOF/CCA)

Statistical correction of the **storm-surge bias** of the unstructured
barotropic model **Tolosa-SW**, using information extracted from the
**CMEMS IBI** regional model. The reference bias is the difference between
the **DUACS L4 SLA** (+ DAC) and the Tolosa surge.

The Tolosa surge is the difference of two runs: **surge = full − ms**.
- **full**: run with atmospheric forcing (wind and pressure), including the
  inverse barometer effect;
- **ms**: tide-only run.

The correction targets one band of periods (e.g. 20–128 days). A regression in
EOF space (CCA between the leading EOFs of IBI and of the bias) learns the
relationship IBI → bias on a training period. It is then applied to new IBI
fields to predict the correction at the elements of the Tolosa-SW mesh.

## Contents

| File | Role |
|---|---|
| `eof_cca.py` | Calibration: reference bias, EOF/CCA fit, prediction over the test period, validation at the mesh elements, figures, NetCDF/Parquet exports, saved model |
| `predict_correction.py` | Prediction: applies a saved model to new native IBI fields and exports model / correction / corrected model |
| `config.json` | Data paths and port metadata used by `eof_cca.py` |

## Method

1. **Reference bias**: (SLA + DAC) − Tolosa surge, daily. Both fields are
   projected onto the IBI grid and onto the Tolosa elements, and the temporal
   mean is removed pixel by pixel.
2. **Predictor**: IBI − DAC (daily DAC interpolated onto the IBI grid).
3. **Filtering**: Butterworth band-pass. It is applied as a zero-phase window
   truncated at a forecast horizon H (7 days by default): day *t* only uses
   data up to *t + H*.
4. **Regression**: EOFs of IBI and of the bias, then CCA between their
   principal components (`EOFRegression`).
5. **Split**: chronological train/test split of the common days (80 % / 20 %
   by default). Scores and exports cover the test period.

## Inputs

### Data (`config.json`)

Paths are relative to the launch directory. Keys whose name starts with `_`
are comments.

| Key | Content |
|---|---|
| `sat_l4_file` | DUACS L4 SLA (CMEMS `cmems_obs-sl_eur_phy-ssh_my_allsat-l4-duacs`), daily, variable `sla` |
| `ibi_file`, `ibi_var` | IBI SSH on its native grid (here detided daily means, `zos_detided`) |
| `dac_dir`, `dac_var` | AVISO DAC files `dac_dif_<CNES day>_<HH>.nc` (directory walked recursively, a file or a glob also works), variable `dac` |
| `tolosa_2d_full`, `tolosa_2d_ms` | Tolosa-SW 2D outputs of the full run (atmospheric forcing + inverse barometer) and of the ms run (tide only), variable `ssh` at the mesh elements. The UGRID connectivity is read from the full file |
| `tolosa_1d_full_dir`, `tolosa_1d_ms_dir` | Tolosa-SW 1D CSV extractions at the ports, full and ms runs (`modeltime, time, ssh, u, v, bathy`, time in days since 1950-01-01). A flat layout and/or per-period sub-folders are both accepted |
| `weights_dir` | Folder of the sparse projection matrices (see below) |
| `ports` | Per port: `tolosa_csv` (CSV file name), `lat`, `lon`. Used for the Parquet export |

### Projection matrices (`weights_dir`)

Precomputed sparse interpolation matrices (`scipy.sparse`, `.npz`):

| File | Projection |
|---|---|
| `proj_tolosa_to_ibi.npz` | Tolosa elements → IBI grid |
| `proj_sat_to_ibi.npz` | DUACS grid → IBI grid |
| `proj_ibi_to_tolosa.npz` | IBI grid → Tolosa elements |
| `proj_sat_to_tolosa.npz` | DUACS grid → Tolosa elements |

The Tolosa side of each matrix must match the **elements** of the Tolosa
field; the script stops if the sizes differ.

### Command line (`eof_cca.py`)

```
python eof_cca.py --band 20 128 [options]
```

| Option | Default | Role |
|---|---|---|
| `--band LOW HIGH` | required | Calibration band (days). Below 20 d the DAC is the MOG2D model: LOW ≥ 20 advised |
| `--ratio-train` | 0.8 | Fraction of the common days used for training |
| `--forecast-horizon` | 7 | Horizon H (days) of the filtering window |
| `--cache-dir` | `cache` | Cache of the expensive intermediates |
| `--port-disk-radius` | 20 | Radius (km) of the disk average at the ports |
| `--port-disk-weighting` | `gaussian` | Disk weighting (`uniform` or `gaussian`) |
| `--plot-stride-days` | 30 | Plot one day out of K |
| `--export-netcdf` / `--no-export-netcdf` | on | NetCDF export of the daily fields |
| `--export-parquet [PATH]` / `--no-export-parquet` | on | Parquet export at the ports |

## Outputs (`eof_cca.py`)

All outputs are written in the launch directory.

- `plots_2D_<label>/`: figures (corrections, daily IBI / Tolosa / sat /
  reference bias).
- `plots_2D_<label>/netcdf_export/`: daily NetCDF-4 files over the test
  period, cropped to the common IBI ∩ Tolosa extent:
  - `sat.nc`, `bias_ref.nc`, `residual.nc` on the IBI grid;
  - `model.nc`, `correction.nc`, `model_corrected.nc` at the Tolosa
    elements, with the UGRID mesh;
  - `skill.nc` (skill per element).
- `models_calibrated/<experiment>_<train period>_<label>/`: fitted model
  (`method_eof.joblib`), `metadata.json` (band, horizon, periods, inputs)
  and `ibi_grid.npz`.
- `corrections_gauges_<label>_<period>.parquet`: model, correction and
  corrected model at the ports (10 min, test period).
- `<cache-dir>/`: cached daily and projected fields. Delete a file to force
  its recomputation.

## Prediction (`predict_correction.py`)

This script applies a model saved by `eof_cca.py` to new IBI fields, on their
native grid and time step. It does not read `config.json`: every input is
given on the command line. It must sit next to `eof_cca.py`, which it imports.

```
python predict_correction.py --model-dir models_calibrated/<run> \
    --ibi IBI.nc [IBI2.nc ...] --dac aviso/DAC/ \
    --w-ibi-to-tolosa projs_long/proj_ibi_to_tolosa.npz \
    --tolosa-full htot_full.nc --tolosa-ms htot_ms.nc \
    --start YYYY-MM-DD [--end YYYY-MM-DD] [--out-dir prediction]
```

**Inputs**

| Input | Content |
|---|---|
| `--model-dir` | A `models_calibrated/<run>/` folder |
| `--ibi` | IBI files on the calibration grid |
| `--dac` | AVISO DAC files covering the whole IBI period |
| `--w-ibi-to-tolosa` | `proj_ibi_to_tolosa.npz` of the calibration |
| `--tolosa-full`, `--tolosa-ms` | Tolosa-SW 2D files over the prediction period |
| `--start`, `--end` | Prediction period |

**Outputs** (`--out-dir`): `model.nc`, `correction.nc` and
`model_corrected.nc`. They are daily, at the Tolosa elements, with the same
format, grids and crop as the calibration export, and cover the prediction
period only.

**Constraints**

- **History**: the band-pass has a long past support (about 280 days for a
  20–128 d band at a daily step). IBI must start well before `--start`; the
  script reports the days computed with a truncated window.
- **DAC**: the predictor is IBI − DAC, and IBI time steps without DAC are
  dropped. The DAC must therefore also cover the forecast days.
- **Grid**: the IBI grid must be the calibration grid.

## Requirements

Python ≥ 3.9 with `numpy`, `scipy`, `pandas`, `xarray`, `dask`, `netCDF4`
(or `h5netcdf`), `scikit-learn`, `joblib` and `matplotlib`. The Parquet
export also needs `pyarrow` (or `fastparquet`).
