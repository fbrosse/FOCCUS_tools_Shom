#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Surge bias correction of a barotropic model by information extraction from IBI
===============================================================================
Correction of the SURGE bias of an unstructured barotropic model
(Tolosa-SW, surge = full − ms) by extracting information from the IBI
model (CMEMS), the reference bias being computed against the DUACS L4
SLA. Calibration is performed on a single spectral band.

Method (fixed configuration, see section 8)
-------------------------------------------
- reference bias: (SLA + DAC) − Tolosa surge, temporal mean removed;
- predictor: IBI − DAC;
- band-pass filtering: zero-phase window truncated at the forecast
  horizon H (--forecast-horizon, 7 days by default);
- extraction method: regression in EOF space (CCA between the EOFs of
  IBI and of the bias, section 3);
- validation at the elements (cells) of the barotropic mesh, where the
  Tolosa-SW SSH is defined;
- parallel 2D plots, NetCDF export; Parquet export of the 1D model, the
  correction and the corrected model at the ports (tide gauge locations).

Bias and correction fields are passed around as xr.DataArray
(dims=('time', 'lat', 'lon')). Data paths and port metadata are read
from config.json.

Outputs (launch directory; <label> = experiment + calibration label)
---------------------------------------------------------------------
- plots_2D_<label>/: figures, and netcdf_export/ (one NetCDF file per
  source);
- models_calibrated/<experiment>_<train period>_<calibration label>/:
  fitted model (joblib) + metadata.json;
- corrections_gauges_<label>_<period>.parquet: model / correction /
  corrected model at the ports (10 min, test period);
- <cache-dir>/: cached intermediates (projected fields, daily IBI, DAC,
  bias_ref).

Usage: python eof_cca.py --band LOW HIGH [options]
       (see --help)
"""

import argparse
import datetime as dt
import gc
import glob
import json
import logging
import math
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import dask
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure
from scipy import signal, sparse
from scipy.spatial import cKDTree
from sklearn.cross_decomposition import CCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

dask.config.set(scheduler='synchronous')

# dtype of the predicted correction fields (float32: halves the memory of
# the large (nt, ny, nx) fields).
PRED_DTYPE = np.float32

# Butterworth band-pass filtering of the EOF calibration and prediction:
# zero-phase filtering on a sliding window truncated at the forecast
# horizon H. Sample t uses the past (depth P set by the filter
# support) and the future up to t+H, legitimately available since IBI is
# a FORECAST. Implemented as an FFT convolution with the equivalent kernel
# (impulse response of sosfiltfilt) truncated to [-P, +H]: phase alignment
# is recovered while staying honest with respect to the operational
# horizon.
# Default value of H (days); set from --forecast-horizon at run time.
FORECAST_HORIZON_DAYS = 7.0


# =============================================================================
# SECTION 0: DATA STRUCTURES & STANDARDISATION
# =============================================================================
#
# Spatio-temporal fields are passed around as xr.DataArray with standard
# coordinates 'lon', 'lat', 'time' (see standardize_coords()). The output
# of the extraction method is a BiasFieldResult wrapping a DataArray
# (dims=('time','lat','lon')) + method name + metadata.


def canonicalize_element_dim(da: xr.DataArray,
                             name: str = 'element') -> xr.DataArray:
    """Rename the mesh dimension of a Tolosa field to 'element' (the SSH
    is defined at the elements, i.e. the cells, of the mesh). Complements
    standardize_coords, which only handles time/lat/lon.

    Two "element" fields with different dims do not raise a shape error:
    xarray broadcasts → (time, N, N), i.e. several PiB.
    Call ONCE when loading any unstructured field; idempotent, no-op for
    structured fields, ValueError if the spatial dim is ambiguous
    (0 or >1 candidates).
    """
    if name in da.dims:
        return da
    if 'lat' in da.dims or 'lon' in da.dims:
        return da  # structured field: nothing to canonicalise
    spatial = [d for d in da.dims if d != 'time']
    if len(spatial) != 1:
        raise ValueError(
            f"canonicalize_element_dim: cannot identify the mesh dimension "
            f"of '{da.name}' (dims={list(da.dims)}, non-time "
            f"candidates: {spatial}). Exactly one spatial dim is "
            f"expected for an unstructured field."
        )
    return da.rename({spatial[0]: name})


def standardize_coords(ds_or_da, ref_var: Optional[str] = None):
    """Rename lat/lon/time coordinates to standard names.

    `ref_var` disambiguates, in a Tolosa SW NetCDF holding several
    coordinate sets at once (nodes + cells), the set matching the
    dimensions of a target variable.
    """
    obj = ds_or_da

    for alias in ('time', 'time_counter', 'Times', 'T'):
        if alias in obj.dims and alias != 'time':
            obj = obj.rename({alias: 'time'})
            break

    ds_for_detect = (obj if isinstance(obj, xr.Dataset)
                     else obj.to_dataset(name='_tmp'))
    if ref_var is None and not isinstance(obj, xr.Dataset):
        ref_var = '_tmp'
    lat_name, lon_name = _detect_latlon_names(ds_for_detect, ref_var=ref_var)

    rename_map = {}
    if lat_name != 'lat' and lat_name in obj.coords:
        rename_map[lat_name] = 'lat'
    if lon_name != 'lon' and lon_name in obj.coords:
        rename_map[lon_name] = 'lon'
    if rename_map:
        obj = obj.rename(rename_map)
    return obj


def _grid_da(values_3d: np.ndarray, time: np.ndarray,
             lon: np.ndarray, lat: np.ndarray, name: str) -> xr.DataArray:
    """(time, lat, lon) DataArray on a regular grid (e.g. the IBI grid)."""
    return xr.DataArray(
        values_3d,
        dims=('time', 'lat', 'lon'),
        coords={'time': time, 'lat': lat, 'lon': lon},
        name=name,
    )


@dataclass
class BiasFieldResult:
    """Output of a correction method.

    Wraps an xr.DataArray (dims=('time','lat','lon')) with the method
    name and a metadata dict. The `.bias_2d`, `.time`, `.lon`, `.lat`
    accessors expose the underlying numpy arrays for consumers that
    expect them.
    """
    da: xr.DataArray
    method_name: str
    metadata: Dict = field(default_factory=dict)

    @property
    def bias_2d(self) -> np.ndarray:
        return self.da.values

    @property
    def time(self) -> np.ndarray:
        return self.da['time'].values

    @property
    def lon(self) -> np.ndarray:
        return self.da['lon'].values

    @property
    def lat(self) -> np.ndarray:
        return self.da['lat'].values

    @classmethod
    def from_arrays(cls, values_3d: np.ndarray, time: np.ndarray,
                    lon: np.ndarray, lat: np.ndarray,
                    method_name: str,
                    metadata: Optional[Dict] = None) -> 'BiasFieldResult':
        """Build a BiasFieldResult from raw numpy arrays."""
        return cls(da=_grid_da(values_3d, time, lon, lat, 'bias_correction'),
                   method_name=method_name, metadata=metadata or {})


# --- xarray-native time helpers ------------------------------------------------

def _to_daily_time(da: xr.DataArray) -> xr.DataArray:
    """Cast the 'time' coord to datetime64[D] (exact daily alignments)."""
    if da['time'].dtype != np.dtype('datetime64[D]'):
        return da.assign_coords(
            time=da['time'].values.astype('datetime64[D]')
        )
    return da


def resample_daily(da: xr.DataArray,
                   min_fraction_complete: float = 0.95) -> xr.DataArray:
    """Daily mean (calendar days) via xarray.resample; incomplete days dropped.

    A day is kept if it holds at least `min_fraction_complete` × the
    largest number of time stamps per day (count independent of NaNs). The
    output 'time' coord is datetime64[D], for exact alignments (xr.align).
    """
    daily = da.resample(time='1D').mean()

    if len(da['time']) > 1:
        ones = xr.DataArray(
            np.ones(len(da['time']), dtype=np.int64),
            dims=('time',), coords={'time': da['time']},
        )
        n_per_day = ones.resample(time='1D').sum()
        n_expected = int(n_per_day.max().values)
        threshold = max(1, int(min_fraction_complete * n_expected))
        keep = n_per_day >= threshold
        n_dropped = int((~keep).sum().values)
        if n_dropped > 0:
            logger.info(
                f"  resample_daily: {n_dropped}/{int(keep.size)} days "
                f"dropped (timestamps < {threshold}/{n_expected})"
            )
        daily = daily.where(keep, drop=True)

    return _to_daily_time(daily)


def align_inner_time(*das: xr.DataArray) -> Tuple[xr.DataArray, ...]:
    """xr.align(..., join='inner') on 'time', after casting to datetime64[D]."""
    das = tuple(_to_daily_time(d) for d in das)
    return xr.align(*das, join='inner')


def apply_temporal_demean(sat_da: xr.DataArray, model_da: xr.DataArray,
                          ) -> Tuple[xr.DataArray, xr.DataArray]:
    """Remove the pixel-wise TEMPORAL mean of the satellite and of the model.

    Both fields must share the same dimensions (typically after projection
    onto IBI or onto the elements of the barotropic mesh). The temporal
    mean over the whole period removes the model datum, the MDT and any
    time-independent, spatially structured offset (anomaly mode). Spatial slopes and the
    pixel-wise temporal correlation are unaffected: only the absolute
    intercept of the correction changes — on purpose, since the model
    datum/MDT is not meant to be corrected. Points outside the common mask
    (valid at least once in both fields) get NaN means.

    .. warning:: IN-PLACE MUTATION of the passed buffers (avoids a full
       copy of the field, ~14 GiB on the element grid): any alias is
       modified. Only pass owned arrays. Fields are materialised once at
       the start, otherwise Dask would re-evaluate the compute graph at
       each use.
    """
    t0 = time.perf_counter()
    spatial_dims = [d for d in sat_da.dims if d != 'time']
    n_spatial = int(np.prod([sat_da.sizes[d] for d in spatial_dims]))
    n_time = sat_da.sizes['time']
    logger.info(
        f"  apply_temporal_demean: {n_time} steps × {n_spatial} points "
        f"(~{n_time*n_spatial/1e6:.0f} M)…"
    )

    # STEP 1 — materialise sat_da AND model_da once.
    # .load() is called on both unconditionally: one may be Dask and the
    # other numpy (e.g. in `compute_bias_at_elements`, sat comes from
    # `apply_projection` → numpy, model from `resample_daily` → Dask).
    # .load() is a no-op on data already in memory.
    t_c = time.perf_counter()
    sat_da = sat_da.load()
    model_da = model_da.load()
    bytes_each = sat_da.nbytes
    logger.info(
        f"    initial load() in {time.perf_counter()-t_c:.1f}s "
        f"({2*bytes_each/1e9:.1f} GB in RAM, sat+model)"
    )

    # STEP 2 — common_mask, computed ONCE (materialised).
    t_m = time.perf_counter()
    common_mask = (sat_da.notnull().any('time')
                   & model_da.notnull().any('time'))
    common_mask = common_mask.load()  # no-op if already numpy
    n_valid = int(common_mask.values.sum())
    logger.info(
        f"    common_mask in {time.perf_counter()-t_m:.1f}s, "
        f"{n_valid}/{n_spatial} valid points "
        f"({100*n_valid/max(n_spatial,1):.1f} %)"
    )

    # STEP 3 — temporal means and in-place subtraction.
    # MEMORY: no sat_da.where(common_mask), which would allocate a full
    # copy of the field just to propagate the mask. The temporal mean
    # (nanmean) already ignores native NaNs; the common mask is applied
    # to the RESULT (per-point means), not to the 3D field. In-place
    # subtraction: no 3D copy.
    t_t = time.perf_counter()
    cm = common_mask.values  # (n_spatial,) bool, already in RAM
    sat_vals = sat_da.values  # numpy view, no copy
    mod_vals = model_da.values
    time_axis = list(sat_da.dims).index('time')
    with np.errstate(invalid='ignore'):
        sat_mean_t = np.nanmean(sat_vals, axis=time_axis,
                                dtype=np.float64).astype(sat_vals.dtype)
        mod_mean_t = np.nanmean(mod_vals, axis=time_axis,
                                dtype=np.float64).astype(mod_vals.dtype)
    # Outside the common mask → NaN.
    nan32 = np.float32(np.nan)
    sat_mean_t = np.where(cm, sat_mean_t, nan32).astype(sat_vals.dtype)
    mod_mean_t = np.where(cm, mod_mean_t, nan32).astype(mod_vals.dtype)
    # IN-PLACE subtraction on the DataArray's numpy buffer.
    sat_da.values -= np.expand_dims(sat_mean_t, axis=time_axis)
    model_da.values -= np.expand_dims(mod_mean_t, axis=time_axis)
    sat_mean_scalar = float(np.nanmean(sat_mean_t))
    mod_mean_scalar = float(np.nanmean(mod_mean_t))
    del sat_vals, mod_vals
    logger.info(
        f"    {time.perf_counter()-t_t:.1f}s: "
        f"⟨sat⟩_pix mean = {sat_mean_scalar:+.4f} m, "
        f"⟨mod⟩_pix mean = {mod_mean_scalar:+.4f} m"
    )
    logger.info(
        f"  apply_temporal_demean total: {time.perf_counter()-t0:.1f}s"
    )
    return sat_da, model_da


# =============================================================================
# Cache of intermediates (projected fields + daily means)
# =============================================================================
# Each expensive intermediate (daily means, projections by W·x onto the IBI
# grid, daily DAC) is saved as NetCDF in the cache directory and read back
# by the next run. Keys encode what affects the content (experiment,
# variables, period) but not the calibration band, so the cache is shared
# by runs with different bands. Sources are not checked for changes:
# delete the relevant file to force a recomputation.

def _time_span_tag(da: xr.DataArray) -> str:
    """'YYYYMMDDYYYYMMDD' tag of the time axis, part of the cache keys:
    a different period gives a different key, hence a recomputation
    (_cache_load does not check the mtimes of the source files)."""
    t = np.asarray(da['time'].values)
    if t.size == 0:
        return "empty"
    d0 = np.datetime_as_string(t[0].astype('datetime64[D]'))
    d1 = np.datetime_as_string(t[-1].astype('datetime64[D]'))
    return f"{d0}-{d1}".replace('-', '')


def _cache_load(cache_dir: Optional[Path], key: str
                ) -> Optional[xr.DataArray]:
    """Try to read a DataArray from the cache. Returns None if missing or KO."""
    if cache_dir is None:
        return None
    path = cache_dir / f"{key}.nc"
    if not path.exists():
        return None
    try:
        ds = xr.open_dataset(path).load()
        data_vars = list(ds.data_vars)
        if len(data_vars) == 1:
            return ds[data_vars[0]]
        raise ValueError(
            f"cache {key}: {len(data_vars)} variables, 1 expected"
        )
    except Exception as e:
        logger.warning(
            f"  cache_load[{key}]: cannot read ({e}), recomputing"
        )
        return None


def _cache_save(cache_dir: Optional[Path], key: str,
                da: xr.DataArray,
                source_paths: Optional[List[str]] = None) -> None:
    """Save a DataArray to the cache (with source metadata).

    The mtimes of the source files are stored in the attrs to allow a
    later check (not automatic: the user has to inspect or `rm` the
    cache if the sources change).
    """
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.nc"
    ds = da.to_dataset() if isinstance(da, xr.DataArray) else da
    if source_paths:
        ds.attrs['source_paths'] = json.dumps(source_paths)
        mtimes = []
        for p in source_paths:
            try:
                mtimes.append(str(Path(p).stat().st_mtime))
            except FileNotFoundError:
                mtimes.append('')
        ds.attrs['source_mtimes'] = json.dumps(mtimes)
    ds.to_netcdf(path)
    logger.info(f"  cache_save[{key}] → {path}")


# --- Projection with precomputed weight matrices ---

def extrapolate_nan_iterative(field_2d: np.ndarray,
                              max_iter: int = 200) -> np.ndarray:
    """Fill the NaNs of a 2D field by iterative diffusion (each NaN pixel
    with a valid neighbour gets the mean of its valid 4-neighbours, repeated
    up to `max_iter` times). Warns if NaNs remain.

    Reference implementation of extrapolate_nan_stack, which calls it for
    slices containing ±inf.
    """
    filled = field_2d.copy()
    converged = False
    for _ in range(max_iter):
        nans = np.isnan(filled)
        if not nans.any():
            converged = True
            break
        padded = np.pad(filled, 1, mode='constant', constant_values=np.nan)
        neighbors = np.stack([
            padded[1:-1, 2:], padded[1:-1, :-2],
            padded[2:, 1:-1], padded[:-2, 1:-1],
        ])
        with np.errstate(all='ignore'):
            neighbor_mean = np.nanmean(neighbors, axis=0)
        fillable = nans & ~np.isnan(neighbor_mean)
        if not fillable.any():
            break
        filled[fillable] = neighbor_mean[fillable]
    if not converged:
        n_remain = int(np.isnan(filled).sum())
        warnings.warn(
            "extrapolate_nan_iterative: NaNs remaining after max_iter "
            "iterations. Consider a larger max_iter."
        )
        logger.debug(
            f"  extrapolate_nan_iterative: {n_remain} NaNs remaining after "
            f"{max_iter} iterations."
        )
    return filled


# Fill plans already built, keyed by (shape, max_iter, NaN mask).
_EXTRAP_PLAN_CACHE: Dict = {}
_EXTRAP_PLAN_CACHE_MAX = 32


def _extrapolation_plan(nan_mask: np.ndarray, max_iter: int):
    """Unroll extrapolate_nan_iterative on the NaN mask alone.

    At each diffusion iteration, the filled pixels and the neighbours read
    depend only on the current mask, not on the values. Returns
    (steps, converged, n_remain): `steps` = list, per iteration, of
    (flat indices of the filled pixels, (4, n) indices of their
    right/left/down/up neighbours; ny*nx = outside the grid).
    """
    ny, nx = nan_mask.shape
    npix = ny * nx
    padded = np.pad(np.arange(npix).reshape(ny, nx), 1, mode='constant',
                    constant_values=npix)
    neighbors = np.stack([
        padded[1:-1, 2:], padded[1:-1, :-2],
        padded[2:, 1:-1], padded[:-2, 1:-1],
    ]).reshape(4, npix)
    cur = nan_mask.ravel().copy()
    valid = np.append(~cur, False)          # last cell: outside the grid
    steps, converged = [], False
    for _ in range(max_iter):
        if not cur.any():
            converged = True
            break
        fill = np.flatnonzero(cur & valid[neighbors].any(axis=0))
        if fill.size == 0:
            break
        steps.append((fill, neighbors[:, fill]))
        cur[fill] = False
        valid[fill] = True
    return steps, converged, int(cur.sum())


def extrapolate_nan_stack(fields: np.ndarray, max_iter: int = 200,
                          block: int = 64) -> np.ndarray:
    """extrapolate_nan_iterative applied to each slice of `fields`
    (nt, ny, nx), with a strictly identical result, once per mask.

    The diffusion sequence only depends on the NaN mask of a slice:
    slices sharing the same mask (usual case: fixed land mask) share a
    fill plan computed only once (and cached across calls), then applied
    to all slices in blocks of `block`, only on the pixels to fill. Each
    filled value is the same np.nanmean of the same 4 neighbours as in
    the iterative version. Slices containing ±inf: iterative version (the
    mask no longer determines the sequence).
    """
    out = np.array(fields, copy=True)
    nt, ny, nx = out.shape
    flat = out.reshape(nt, ny * nx)
    nan_mask = np.isnan(flat)
    has_inf = np.isinf(flat).any(axis=1)
    groups: Dict[bytes, List[int]] = {}
    for t in range(nt):
        if has_inf[t]:
            out[t] = extrapolate_nan_iterative(out[t], max_iter=max_iter)
        elif nan_mask[t].any():
            groups.setdefault(np.packbits(nan_mask[t]).tobytes(), []).append(t)

    n_unconverged, n_remain_total = 0, 0
    for key, ts in groups.items():
        cache_key = (ny, nx, max_iter, key)
        plan = _EXTRAP_PLAN_CACHE.get(cache_key)
        if plan is None:
            plan = _extrapolation_plan(nan_mask[ts[0]].reshape(ny, nx),
                                       max_iter)
            if len(_EXTRAP_PLAN_CACHE) >= _EXTRAP_PLAN_CACHE_MAX:
                _EXTRAP_PLAN_CACHE.pop(next(iter(_EXTRAP_PLAN_CACHE)))
            _EXTRAP_PLAN_CACHE[cache_key] = plan
        steps, converged, n_remain = plan
        if not converged:
            n_unconverged += len(ts)
            n_remain_total += n_remain * len(ts)
        ts = np.asarray(ts)
        for b0 in range(0, ts.size, block):
            tb = ts[b0:b0 + block]
            # Last NaN column = neighbour outside the grid (NaN padding).
            vals = np.concatenate(
                [flat[tb], np.full((tb.size, 1), np.nan, dtype=flat.dtype)],
                axis=1)
            for fill, nb in steps:
                # (4, n_slices, n_pixels): same axis-0 reduction as the np.stack of
                # the 4 neighbours in the iterative version.
                gathered = np.ascontiguousarray(np.moveaxis(vals[:, nb], 1, 0))
                with np.errstate(all='ignore'):
                    vals[:, fill] = np.nanmean(gathered, axis=0)
            flat[tb] = vals[:, :-1]

    if n_unconverged:
        warnings.warn(
            "extrapolate_nan_iterative: NaNs remaining after max_iter "
            "iterations. Consider a larger max_iter."
        )
        logger.debug(
            f"  extrapolate_nan_stack: {n_unconverged} slice(s) not "
            f"converged, {n_remain_total} NaNs remaining after "
            f"{max_iter} iterations."
        )
    return out


def load_and_normalize_weights(path: str, name: str = "W",
                               tolerance: float = 0.01) -> sparse.spmatrix:
    """Load a sparse projection matrix (.npz) and renormalise it if needed.

    A bilinear matrix satisfies Σⱼ Wᵢⱼ = 1 on each non-empty row. If more
    than 1 % of these rows deviate from 1 by more than `tolerance`, a
    warning is logged and the rows are renormalised (W = W / row_sums).
    """
    W = sparse.load_npz(path)
    n_rows = W.shape[0]
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    non_empty = row_sums > 1e-12
    n_non_empty = int(non_empty.sum())

    if n_non_empty == 0:
        logger.warning(f"  [{name}] All rows are empty.")
        return W

    deviations = np.abs(row_sums[non_empty] - 1.0)
    n_bad = int((deviations > tolerance).sum())
    max_dev = float(deviations.max())
    mean_dev = float(deviations.mean())

    logger.info(
        f"  [{name}] {n_non_empty}/{n_rows} non-empty rows, "
        f"Σ_j W_ij: mean={row_sums[non_empty].mean():.4f}, "
        f"min={row_sums[non_empty].min():.4f}, "
        f"max={row_sums[non_empty].max():.4f}"
    )

    if n_bad / n_non_empty > 0.01:
        logger.warning(
            f"  [{name}] NOT NORMALISED: {n_bad}/{n_non_empty} rows "
            f"({100*n_bad/n_non_empty:.1f} %) have |Σ-1| > {tolerance}. "
            f"max={max_dev:.4f}, mean={mean_dev:.4f}."
        )
        W = W.tocsr()
        inv_row_sums = np.zeros(n_rows)
        inv_row_sums[non_empty] = 1.0 / row_sums[non_empty]
        W = sparse.diags(inv_row_sums) @ W
        logger.info(f"  [{name}] Renormalised.")
    else:
        logger.info(f"  [{name}] OK (normalised within {tolerance}).")
    return W


def apply_projection(W: sparse.spmatrix, values: np.ndarray,
                     dst_shape: Tuple[int, ...],
                     fallback_rows: Optional[np.ndarray] = None,
                     fallback_cols: Optional[np.ndarray] = None,
                     tgt_land_mask: Optional[np.ndarray] = None,
                     temporal_chunk: int = 50) -> np.ndarray:
    """Project `values` ((nt, n_src), (nt, ...) or (n_src,)) with the
    sparse matrix W, then reshape to (nt,) + dst_shape.

    Steps: (1) W @ source, in blocks of `temporal_chunk` time steps;
    (2) target rows `fallback_rows` (outside the source domain) take the
    nearest source value `fallback_cols`; (3) NaN on `tgt_land_mask`.
    """
    squeeze = False

    if values.ndim == 1:
        values = values.reshape(1, -1)
        squeeze = True
    elif values.ndim > 2:
        nt = values.shape[0]
        values = values.reshape(nt, -1)

    nt, n_src = values.shape
    n_dst = W.shape[0]

    if W.shape[1] != n_src:
        raise ValueError(
            f"apply_projection: W.shape[1]={W.shape[1]} != n_src={n_src}."
        )
    n_dst_expected = int(np.prod(dst_shape))
    if n_dst != n_dst_expected:
        raise ValueError(
            f"apply_projection: W.shape[0]={n_dst} != "
            f"prod(dst_shape)={n_dst_expected}."
        )

    has_fallback = (fallback_rows is not None and fallback_cols is not None
                    and len(fallback_rows) > 0)
    result = np.empty((nt, n_dst), dtype=np.float64)

    for t0 in range(0, nt, temporal_chunk):
        t1 = min(t0 + temporal_chunk, nt)
        chunk = values[t0:t1].astype(np.float64)
        block = (W @ chunk.T).T
        if has_fallback:
            block[:, fallback_rows] = chunk[:, fallback_cols]
        result[t0:t1] = block

    result = result.reshape((nt,) + tuple(dst_shape))
    if tgt_land_mask is not None:
        result[:, tgt_land_mask] = np.nan
    if squeeze:
        result = result[0]
    return result


# =============================================================================
# SECTION 1: UTILITIES — FILTERS, TEMPORAL INTERPOLATION
# =============================================================================


class FilterUtils:
    """Butterworth band-pass (order 4, SOS) applied as a zero-phase window
    truncated at the forecast horizon (_apply_forecast)."""

    @staticmethod
    def _forecast_kernel(sos: np.ndarray, P: int, H: int) -> np.ndarray:
        """Equivalent kernel of a `sosfiltfilt` truncated to the window
        [-P, +H] and re-centred. Convolving a signal with this kernel
        (output point t using x[t-P .. t+H]) reproduces zero-phase
        filtering in the interior, while using the future only up to
        t+H (forecast horizon).

        Built numerically as the impulse response of sosfiltfilt to a
        centred impulse (the forward-backward kernel is symmetric and
        decays fast), then truncated to [-P, +H]. The length of the
        impulse support (2·P_pad+1) must exceed P to capture the whole
        tail on the past side.
        """
        P_pad = max(P, H) + 4 * sos.shape[0] + 8  # tail margin
        Limp = 2 * P_pad + 1
        imp = np.zeros(Limp)
        imp[P_pad] = 1.0
        h_full = signal.sosfiltfilt(sos, imp)      # symmetric around P_pad
        c = P_pad
        k = h_full[c - P: c + H + 1].copy()         # length P+H+1, centre @ P
        return k

    # Forecast kernel cache: key (sos.tobytes(), H) → (k, P).
    # Avoids rebuilding the impulse response and measuring P at each call
    # (many calls on the same band).
    _forecast_kernel_cache: Dict = {}

    @staticmethod
    def _apply_forecast(sos: np.ndarray, data: np.ndarray, dt_hours: float,
                        axis: int, label: str) -> np.ndarray:
        """Zero-phase filtering on a window truncated at the horizon H,
        by FFT convolution along `axis`.

        P (past depth) is set automatically to the effective support of
        the kernel (residual energy < 1e-6). H (future) is
        FORECAST_HORIZON_DAYS converted to samples at the current dt.
        Edges: the window shrinks (kernel truncation) where past/future
        is missing — accepted approximate regime, logged once.
        """
        n = data.shape[axis]
        H = int(round(FORECAST_HORIZON_DAYS * 24.0 / dt_hours))

        # P + kernel (cached by (sos, H)).
        cache_key = (sos.tobytes(), H)
        cached = FilterUtils._forecast_kernel_cache.get(cache_key)
        if cached is None:
            # Effective support of the kernel (one half) → P.
            P_pad = 8 * sos.shape[0] + 256
            imp = np.zeros(2 * P_pad + 1); imp[P_pad] = 1.0
            h_full = signal.sosfiltfilt(sos, imp)
            c = P_pad
            one_side = h_full[c:]
            cum = np.cumsum(one_side ** 2)
            tot = cum[-1] if cum[-1] > 0 else 1.0
            P = int(np.searchsorted(cum, tot * (1.0 - 1e-6))) + 1
            P = min(P, P_pad)  # safety bound
            k = FilterUtils._forecast_kernel(sos, P=P, H=H)
            FilterUtils._forecast_kernel_cache[cache_key] = (k, P)
            logger.info(
                f"  forecast[{label}]: P={P} samples (past), "
                f"H={H} samples ({FORECAST_HORIZON_DAYS:g} d @ dt={dt_hours:g}h); "
                f"edges shrunk where past/future is incomplete."
            )
        else:
            k, P = cached
        koff = P

        if n < 2:
            logger.error(
                f"  forecast[{label}]: n={n} < 2 — filter disabled."
            )
            return data.copy()

        if H > n - 1:
            logger.warning(
                f"  forecast[{label}]: horizon H={H} samples > n-1={n-1}; "
                f"future window limited by the series length."
            )

        # Work with the time axis first.
        data_m = np.moveaxis(data, axis, 0)
        shp = data_m.shape
        X = data_m.reshape(n, -1).astype(np.float64)  # (n, npix)

        # 'full' FFT convolution along time, for all pixels.
        # Interior: y[t] = sum_j k[j] * x[t - (j - koff)] = (x * k_rev) shifted.
        # scipy.signal.fftconvolve with the reversed kernel.
        k_rev = k[::-1]
        conv = signal.fftconvolve(
            X, k_rev[:, None], mode='full', axes=0)
        # Alignment: the 'full' output has length n + len(k) - 1. The index of
        # the centred output t is shifted by (len(k)-1-koff).
        start = len(k) - 1 - koff
        Y = conv[start:start + n]

        # --- Edges: truncated kernel -------------------------------------------
        # At the edges, part of the kernel extends beyond the signal and the FFT
        # convolution integrated implicit zeros there. No renormalisation by the
        # kernel sum (~0 for a band-pass): the first P and last H points are
        # recomputed exactly as a dot product with the kernel truncated to the
        # available support.
        left = min(P, n)
        right_start = max(0, n - H)
        for t in list(range(left)) + list(range(right_start, n)):
            lo = max(0, t - P); hi = min(n - 1, t + H)
            klo = koff - (t - lo); khi = koff + (hi - t)
            seg = X[lo:hi + 1]                       # (m, npix)
            kk = k[klo:khi + 1]                      # (m,)
            Y[t] = kk @ seg

        Y = Y.reshape(shp)
        return np.moveaxis(Y, 0, axis)

    @staticmethod
    def bandpass_butterworth(data: np.ndarray, dt_hours: float,
                             low_period_days: float,
                             high_period_days: float,
                             order: int = 4, axis: int = 0) -> np.ndarray:
        """Butterworth band-pass filter, keeps periods [low, high] (days).

        `low_period_days` is the shortest period kept (hence the high
        cut-off frequency), `high_period_days` the longest.

        Nyquist guard: if `low_period_days` ≤ 2·dt_days, the high cut-off
        is clamped just below Nyquist (Wn=0.99) with a warning, to avoid
        a rejection by `scipy.signal.butter`, which strictly requires
        0 < Wn < 1.

        For daily data (dt=24 h), pass low_period_days > 2 d (e.g. 2.1)
        for a well-defined behaviour.
        """
        if low_period_days >= high_period_days:
            raise ValueError(
                f"bandpass: low_period_days ({low_period_days}) must be "
                f"< high_period_days ({high_period_days})."
            )
        dt_days = dt_hours / 24.0
        fs = 1.0 / dt_days
        nyq = 0.5 * fs
        w_high = (1.0 / low_period_days) / nyq    # high-frequency cut-off
        w_low = (1.0 / high_period_days) / nyq    # low-frequency cut-off
        if w_high >= 1.0:
            warnings.warn(
                f"bandpass: low_period_days={low_period_days} d ≤ 2·dt="
                f"{2*dt_days:.3f} d (Nyquist); Wn_high clamped to 0.99 "
                f"(~{2*dt_days*1.01:.3f} d)."
            )
            w_high = 0.99
        elif w_high > 0.9:
            # High cut-off very close to Nyquist (e.g. low band 2.1 d sampled
            # daily → Wn_high = 0.952). The filter is numerically valid but its
            # effective response departs from the theoretical template near
            # Nyquist (squashed transition, ripples). Keep this in mind when
            # interpreting the exact content of the band; check if needed with
            # scipy.signal.sosfreqz(sos, fs=fs).
            logger.warning(
                f"  bandpass [{low_period_days}, {high_period_days}] d: "
                f"Wn_high = {w_high:.3f} > 0.9 (close to Nyquist at dt="
                f"{dt_hours:g} h). Effective response of the short edge "
                f"of the band to be checked (sosfreqz) — the actual "
                f"content may differ from the [{low_period_days} d, …] template."
            )
        if w_low <= 0:
            w_low = 1e-6
        if w_low >= w_high:
            raise ValueError(
                f"bandpass: inconsistent bounds after normalisation "
                f"(Wn_low={w_low:.3f} >= Wn_high={w_high:.3f})."
            )
        sos = signal.butter(order, [w_low, w_high],
                            btype='band', output='sos')
        return FilterUtils._apply_forecast(
            sos, data, dt_hours=dt_hours, axis=axis,
            label=f'bandpass [{low_period_days}, {high_period_days}]d',
        )


# ─────────────────────────────────────────────────────────────────────────────
# 1D temporal interpolation (pandas utilities for the 1D model at the ports)
# ─────────────────────────────────────────────────────────────────────────────

def _regular_time_grid(start, end, delta) -> np.ndarray:
    """Times start, start + delta, … ≤ end (datetime64[ns]).

    `delta`: minutes (int) or timedelta. Integer arithmetic in ns: grid
    exactly identical to iterating start + n·delta.
    """
    if isinstance(delta, int):
        delta = dt.timedelta(minutes=int(delta))
    step = (delta // dt.timedelta(microseconds=1)) * 1000
    t0 = int(pd.Timestamp(start).to_datetime64().astype('datetime64[ns]')
             .astype(np.int64))
    t1 = int(pd.Timestamp(end).to_datetime64().astype('datetime64[ns]')
             .astype(np.int64))
    n = math.ceil(((t1 - t0) / 10**9) / (step / 10**9)) + 1
    grid = t0 + step * np.arange(n, dtype=np.int64)
    return grid[grid <= t1].astype('datetime64[ns]')


_UNIT_PER_DAY = {'ns': 86_400 * 10**9, 'us': 86_400 * 10**6}


def _days_since_1950(times: np.ndarray) -> np.ndarray:
    """Days (float) since 1950-01-01, computed like the pandas division
    Timedelta / timedelta: float(value) / float(units per day), in the
    resolution of the dates (at least µs)."""
    unit = np.datetime_data(times.dtype)[0]
    unit = unit if unit in _UNIT_PER_DAY else 'us'
    ticks = (times.astype(f'datetime64[{unit}]')
             - np.datetime64('1950-01-01', unit)).astype(np.int64)
    return ticks.astype(np.float64) / float(_UNIT_PER_DAY[unit])


# dtype inferred by pandas for a column of Python datetimes (depends on
# the pandas version: datetime64[ns] or datetime64[us]).
_PYDATETIME_DTYPE = pd.Series([dt.datetime(2000, 1, 1)]).dtype


def time_interp(df, tdelta):
    """Linear interpolation of the numeric columns of `df` onto a regular
    grid of step `tdelta` (minutes, int, or timedelta), from the first to
    the last date (truncated to µs)."""
    if 'time' not in df.columns:
        df = df.reset_index()
        set_index = True
    else:
        set_index = False
    times = df['time'].values
    grid = _regular_time_grid(times.min().astype('datetime64[us]'),
                              times.max().astype('datetime64[us]'), tdelta)
    # Grid (exact µs): float(µs) / 86400e6, a correctly rounded division,
    # like Python datetime arithmetic.
    x_grid = _days_since_1950(grid.astype('datetime64[us]'))
    x_src = _days_since_1950(times)

    df2 = pd.DataFrame(columns=df.columns)
    for col in df.columns.values.tolist():
        if col == 'time' and len(df.index) > 0:
            df2[col] = grid.astype(_PYDATETIME_DTYPE)
        elif len(df.index) > 0 and type(df[col].iloc[0]) is not str:
            df2[col] = np.interp(x_grid, x_src, df[col].astype(float))
        else:
            df2[col] = df[col]

    return df2.set_index('time') if set_index else df2


# =============================================================================
# SECTION 2: REFERENCE BIAS
# =============================================================================


class BiasComputation:
    """Bias = satellite − barotropic model (xarray-native)."""

    @staticmethod
    def compute_bias_2d_from_unstructured(
        sat: xr.DataArray,
        model: xr.DataArray,
        W_tolosa_to_ibi: sparse.spmatrix,
        ibi_shape: Tuple[int, int],
        ibi_lon: np.ndarray,
        ibi_lat: np.ndarray,
        W_sat_to_ibi: Optional[sparse.spmatrix] = None,
        fallback_tolosa: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        fallback_sat: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        tgt_land_mask: Optional[np.ndarray] = None,
        return_intermediates: bool = False,
        cache_dir: Optional[Path] = None,
        cache_prefix: str = '',
        sat_cache_tag: str = '',
    ) -> Union[xr.DataArray, Tuple[xr.DataArray, Dict]]:
        """Compute the daily 2D bias (sat − barotropic model) on the IBI grid.

        The (unstructured) model is averaged daily at the elements then
        projected onto IBI; the satellite is projected onto IBI if
        W_sat_to_ibi is given. The time intersection is done by xr.align,
        then the temporal mean is removed on common days (see
        `apply_temporal_demean`).

        Parameters
        ----------
        cache_dir : Optional[Path]
            If given, caches (and reads first) the expensive
            intermediates: daily Tolosa at the elements, daily Tolosa
            projected onto IBI, sat projected onto IBI. Independent of the
            mean removal (applied afterwards).
        cache_prefix : str
            Prefix of the cache keys. Must encode the parameters that
            affect the content: experiment, var_mod, var_sat, etc.
        sat_cache_tag : str
            Sat-specific suffix (e.g. '_dacin'): distinguishes the
            projected SLA without touching the Tolosa caches, reused as is.

        Returns
        -------
        bias_ref : xr.DataArray (time, lat, lon)
        intermediates : dict (only if return_intermediates=True)
            'model_elements_daily_da' (time, element), 'model_daily_da'
            (time, lat, lon): daily Tolosa at the elements / projected onto
            IBI; 'sat_on_out_da': sat on the IBI grid on the days of
            bias_ref. All three are raw (temporal mean NOT removed).
        """
        # ---------- 1. Daily mean of the model at the elements ----------
        # Period tags per SOURCE: the elements/tol_ibi key depends on the model
        # file, the sat_ibi key on the sat file — invalidating one does not
        # force the recomputation of the other.
        _mod_tag = _time_span_tag(model)
        _sat_tag = _time_span_tag(sat)
        # The key keeps its historical '_nodes_' name so that existing
        # caches stay valid; their mesh dimension is renamed on read.
        key_elements = f"{cache_prefix}_tolosa_daily_nodes_{_mod_tag}"
        daily_model_elements_da = _cache_load(cache_dir, key_elements)
        if daily_model_elements_da is not None:
            daily_model_elements_da = canonicalize_element_dim(
                daily_model_elements_da)
        if daily_model_elements_da is None:
            logger.info("  Daily mean of the unstructured barotropic model…")
            # Idempotent guard (no-op if the field was canonicalised at load
            # time).
            daily_model_elements_da = resample_daily(
                canonicalize_element_dim(model)
            )
            if hasattr(daily_model_elements_da, 'compute'):
                daily_model_elements_da = daily_model_elements_da.compute()
            _cache_save(cache_dir, key_elements, daily_model_elements_da)
        else:
            logger.info(
                f"  Daily Tolosa at the elements: read from cache "
                f"({key_elements})"
            )
        logger.info(
            f"    {daily_model_elements_da.sizes['time']} days, "
            f"{daily_model_elements_da.sizes['element']} elements"
        )

        # ---------- 2. Projection elements → IBI grid ----------
        key_tol_ibi = f"{cache_prefix}_tolosa_daily_on_ibi_{_mod_tag}"
        daily_model_da = _cache_load(cache_dir, key_tol_ibi)
        if daily_model_da is None:
            logger.info("  Projection barotropic model → IBI grid…")
            fb_rows_t = fallback_tolosa[0] if fallback_tolosa else None
            fb_cols_t = fallback_tolosa[1] if fallback_tolosa else None
            daily_model_2d = apply_projection(
                W_tolosa_to_ibi, daily_model_elements_da.values, ibi_shape,
                fallback_rows=fb_rows_t, fallback_cols=fb_cols_t,
                tgt_land_mask=tgt_land_mask,
            )
            daily_model_da = _grid_da(
                daily_model_2d, daily_model_elements_da['time'].values,
                ibi_lon, ibi_lat, name='model_daily',
            )
            _cache_save(cache_dir, key_tol_ibi, daily_model_da)
        else:
            logger.info(
                f"  Tolosa projected onto IBI: read from cache ({key_tol_ibi})"
            )

        # ---------- 3. Satellite on the IBI grid ----------
        key_sat_ibi = f"{cache_prefix}{sat_cache_tag}_sat_on_ibi_{_sat_tag}"
        sat_on_out_da = _cache_load(cache_dir, key_sat_ibi)
        if sat_on_out_da is None:
            if W_sat_to_ibi is not None:
                logger.info("  Projection satellite → IBI grid…")
                fb_rows_s = fallback_sat[0] if fallback_sat else None
                fb_cols_s = fallback_sat[1] if fallback_sat else None
                sat_on_out = apply_projection(
                    W_sat_to_ibi, sat.values, ibi_shape,
                    fallback_rows=fb_rows_s, fallback_cols=fb_cols_s,
                    tgt_land_mask=tgt_land_mask,
                )
            else:
                sat_on_out = sat.values
            sat_on_out_da = _grid_da(
                sat_on_out, sat['time'].values,
                ibi_lon, ibi_lat, name='sat_on_ibi',
            )
            _cache_save(cache_dir, key_sat_ibi, sat_on_out_da)
        else:
            logger.info(
                f"  Sat projected onto IBI: read from cache ({key_sat_ibi})"
            )

        # ---------- 4. Time alignment ----------
        sat_a, model_a = align_inner_time(sat_on_out_da, daily_model_da)

        # ---------- 5. Temporal mean removal ----------
        # Applied AFTER the cache, so it is not part of the cache key of the
        # intermediates.
        logger.info("  Temporal mean removal…")
        # ALIASING: when the time axes coincide, align returns the input
        # objects → sat_a/model_a ARE the (raw) fields exposed in
        # `intermediates`, which the in-place demean would overwrite. Copy
        # before demean when the intermediates are requested (~1.5 GB on IBI;
        # at the elements, see compute_bias_at_elements, the buffers are
        # owned and the demean stays in place).
        sat_aligned = sat_a   # raw (not demeaned), for the intermediates
        if return_intermediates:
            sat_a = sat_a.copy(deep=True)
            model_a = model_a.copy(deep=True)
        sat_a, model_a = apply_temporal_demean(sat_a, model_a)

        bias_ref = (sat_a - model_a).rename('bias_ref')

        # Check: days without useful data (all values NaN)
        nonempty = bias_ref.notnull().any(dim=('lat', 'lon'))
        n_empty = int((~nonempty).sum().values)
        if n_empty > 0:
            logger.warning(
                f"  {n_empty}/{bias_ref.sizes['time']} days with "
                f"bias_ref entirely NaN (to inspect)"
            )

        if return_intermediates:
            intermediates = {
                'model_elements_daily_da': daily_model_elements_da,
                'model_daily_da': daily_model_da,
                'sat_on_out_da': sat_aligned,   # raw, time axis of bias_ref
            }
            return bias_ref, intermediates

        return bias_ref

    @staticmethod
    def compute_bias_at_elements(
        sat: xr.DataArray, model: xr.DataArray,
        W_sat_to_tolosa: sparse.spmatrix,
    ) -> xr.DataArray:
        """Daily bias directly at the elements of the barotropic mesh, after
        removal of the temporal mean (see `apply_temporal_demean`, applied
        after time alignment; the "common mask" is the set of elements valid
        in both fields).

        Returns
        -------
        bias_at_elements : xr.DataArray (time, element)
            Non-dim coords 'elem_lon'/'elem_lat' on 'element'.
        """
        n_dst = W_sat_to_tolosa.shape[0]
        # .shape on a DataArray does NOT materialise the data (unlike
        # .values.shape, which would load the whole unstructured Tolosa
        # field). Same for the log.
        n_model = model.shape[1]

        # Projection sat → elements (numpy; the sat field is small).
        # MEMORY: float32 along the whole element chain. Fields at the
        # elements are huge (e.g. 731 d × 2.6 M elements = 14 GiB in float64);
        # float32 halves this, with a relative precision ~1e-7, amply
        # sufficient for SSH/bias in metres.
        sat_at_elements = apply_projection(
            W_sat_to_tolosa, sat.values, (n_dst,)
        ).astype(np.float32, copy=False)
        logger.info(f"  sat_at_elements {sat_at_elements.shape}, "
                    f"sat {sat.shape}, model {model.shape}")

        # Wrap sat at the elements into a DataArray
        sat_elements_da = xr.DataArray(
            sat_at_elements, dims=('time', 'element'),
            coords={'time': sat['time'].values},
            name='sat_at_elements',
        )

        # Daily mean of the model. Idempotent guard (no-op if the field was
        # canonicalised at load time).
        daily_model_da = resample_daily(canonicalize_element_dim(model))
        # float32 (see the memory note above). Lazy astype if Dask.
        daily_model_da = daily_model_da.astype(np.float32)

        # The sat → Tolosa matrix must target exactly the points of the
        # Tolosa field (its elements): a matrix built on the mesh nodes
        # (vertices) would not match.
        if n_dst != n_model:
            raise ValueError(
                f"compute_bias_at_elements: W_sat_to_tolosa has {n_dst} rows, "
                f"the Tolosa field has {n_model} elements."
            )

        # Time alignment (xr.align)
        sat_a, model_a = align_inner_time(sat_elements_da, daily_model_da)

        # Temporal mean removal. Owned buffers (apply_projection allocates;
        # resample+astype copy) → intended in-place mutation (avoids 2
        # copies of the field), the input `model` is not modified.
        logger.info("  Temporal mean removal at the elements…")
        sat_a, model_a = apply_temporal_demean(sat_a, model_a)

        bias = (sat_a - model_a).rename('bias_ref_at_elements')

        # Materialisation: a lazy Dask result would re-trigger the whole
        # chain (sat read + projection + resample + align + demean) at each
        # use of .values by the element validation; here it is paid only once.
        t_mat = time.perf_counter()
        bias = bias.load()
        if time.perf_counter() - t_mat > 1.0:
            logger.info(
                f"  bias_at_elements materialised in "
                f"{time.perf_counter()-t_mat:.1f}s "
                f"({bias.nbytes/1e9:.1f} GB) — reusable without recomputation"
            )

        # Attach the element lon/lat as non-dim coords
        return bias.assign_coords({
            'elem_lon': ('element', model.lon.values),
            'elem_lat': ('element', model.lat.values),
        })


# =============================================================================
# SECTION 3: EXTRACTION METHOD (EOF×EOF CCA regression)
# =============================================================================
#
# API:
#   fit(ibi: xr.DataArray, bias_2d: np.ndarray, dt_hours: float)
#   predict(ibi_future: xr.DataArray, dt_hours: float) -> BiasFieldResult


# ---- EOF method: CCA regression EOF×EOF -------------------------------------

class EOFRegression:
    """CCA regression in a truncated EOF space for IBI and the bias.

    1. EOFs of filtered IBI → Φ_x, PC_x.
    2. EOFs of the filtered bias → Φ_y, PC_y.
    3. CCA(PC_x, PC_y): optimal co-varying projections.
    4. Prediction: future IBI → PC_x → CCA → PC_y_hat → 2D bias.

    EOFs before CCA for regularisation (n_pixels >> raw n_times).
    """

    def __init__(self, band: Tuple[float, float], n_modes: int = 10,
                 variance_threshold: float = 0.90,
                 n_canonical: Optional[int] = None):
        self.band = band
        self.n_modes_max = n_modes
        self.variance_threshold = variance_threshold
        self.n_canonical_user = n_canonical
        self.eofs_x = None
        self.eofs_y = None
        self.mean_ibi = None
        self.std_ibi = None
        self.mean_bias = None
        self.mask_x = None
        self.mask_y = None
        self.cca = None
        self.n_modes_x = None
        self.n_modes_y = None
        self.n_canonical = None
        self.variance_explained_x_ = None
        self.variance_explained_y_ = None
        self.dt_fit = None
        self.fitted = False

    @staticmethod
    def _select_n_modes(eigenvalues, variance_threshold, n_modes_max):
        var_cum = np.cumsum(eigenvalues) / eigenvalues.sum()
        n = min(
            n_modes_max,
            int(np.searchsorted(var_cum, variance_threshold)) + 1,
            len(eigenvalues),
        )
        return n, float(var_cum[n - 1])

    def _flatten_and_filter_ibi(self, ibi_values, dt_hours):
        """Filter IBI, flatten to ocean pixels, centre + normalise."""
        ibi_bp = FilterUtils.bandpass_butterworth(
            ibi_values, dt_hours, *self.band, axis=0,
        )
        self.mask_x = ~np.isnan(ibi_bp[0])
        ibi_flat = ibi_bp[:, self.mask_x]

        nan_cols = np.any(np.isnan(ibi_flat), axis=0)
        if nan_cols.any():
            logger.warning(
                f"  IBI: {nan_cols.sum()} pixels with residual NaNs excluded"
            )
            flat_mask = self.mask_x.copy()
            flat_mask_flat = flat_mask[self.mask_x]
            flat_mask_flat[nan_cols] = False
            self.mask_x[self.mask_x] = flat_mask_flat
            ibi_flat = ibi_bp[:, self.mask_x]

        self.mean_ibi = ibi_flat.mean(axis=0)
        self.std_ibi = ibi_flat.std(axis=0)
        self.std_ibi[self.std_ibi < 1e-10] = 1.0
        return (ibi_flat - self.mean_ibi) / self.std_ibi

    def fit(self, ibi: xr.DataArray, bias_2d: np.ndarray,
            dt_hours: float = 1.0):
        """Fit the regression on the training period.

        ibi: (time, lat, lon); bias_2d: (time, ny, nx) on the same days;
        dt_hours: time step of both. Both are band-passed, reduced to
        their leading EOFs (`variance_threshold`, at most `n_modes`), then
        related by a CCA between their principal components.
        """
        logger.info("EOF method: CCA regression (EOF×EOF) — calibration")
        ibi_norm = self._flatten_and_filter_ibi(ibi.values, dt_hours)
        bias_bp = FilterUtils.bandpass_butterworth(
            bias_2d, dt_hours, *self.band, axis=0,
        )
        nt = ibi_norm.shape[0]
        ny, nx = bias_2d.shape[1], bias_2d.shape[2]

        # Valid bias pixels (≥ 50% valid steps, min 30)
        bias_flat = bias_bp.reshape(nt, ny * nx)
        valid_counts = (~np.isnan(bias_flat)).sum(axis=0)
        valid_pixels = valid_counts >= max(int(0.5 * nt), 30)
        self.mask_y = valid_pixels.reshape(ny, nx)
        if not valid_pixels.any():
            raise RuntimeError("EOF method: no valid pixel to calibrate.")

        bias_valid = bias_flat[:, valid_pixels]
        col_means = np.nanmean(bias_valid, axis=0)
        nan_loc = np.isnan(bias_valid)
        if nan_loc.any():
            bias_valid = bias_valid.copy()
            bias_valid[nan_loc] = np.broadcast_to(
                col_means, bias_valid.shape
            )[nan_loc]

        self.mean_bias = bias_valid.mean(axis=0)
        bias_centered = bias_valid - self.mean_bias

        # EOFs of IBI
        Ux, Sx, Vtx = np.linalg.svd(ibi_norm, full_matrices=False)
        eig_x = Sx ** 2 / max(nt - 1, 1)
        n_x, ve_x = self._select_n_modes(
            eig_x, self.variance_threshold, self.n_modes_max
        )
        self.n_modes_x = n_x
        self.variance_explained_x_ = ve_x
        self.eofs_x = Vtx[:n_x]
        PCs_x = Ux[:, :n_x] * Sx[:n_x]

        # EOFs of the bias
        Uy, Sy, Vty = np.linalg.svd(bias_centered, full_matrices=False)
        eig_y = Sy ** 2 / max(nt - 1, 1)
        n_y, ve_y = self._select_n_modes(
            eig_y, self.variance_threshold, self.n_modes_max
        )
        self.n_modes_y = n_y
        self.variance_explained_y_ = ve_y
        self.eofs_y = Vty[:n_y]
        PCs_y = Uy[:, :n_y] * Sy[:n_y]

        logger.info(
            f"  IBI EOFs: {n_x} modes ({ve_x:.1%}), "
            f"bias EOFs: {n_y} modes ({ve_y:.1%})"
        )

        # CCA
        n_canon = (self.n_canonical_user if self.n_canonical_user is not None
                   else min(n_x, n_y))
        n_canon = min(n_canon, n_x, n_y)
        self.n_canonical = n_canon
        self.cca = CCA(n_components=n_canon, scale=True, max_iter=2000)
        self.cca.fit(PCs_x, PCs_y)

        Xc, Yc = self.cca.transform(PCs_x, PCs_y)
        canon_corr = np.array([
            np.corrcoef(Xc[:, k], Yc[:, k])[0, 1] for k in range(n_canon)
        ])
        logger.info(
            f"  CCA: {n_canon} components, correlations = "
            + ", ".join(f"{c:.2f}" for c in canon_corr)
        )
        self.dt_fit = dt_hours
        self.fitted = True

    def predict(self, ibi_future: xr.DataArray,
                dt_hours: float = 1.0) -> BiasFieldResult:
        """Predict the bias on the IBI grid from `ibi_future` (time step
        dt_hours), band-passed like in fit. NaN outside the bias mask."""
        if not self.fitted:
            raise RuntimeError("EOFRegression.predict: call fit() first.")
        ibi_bp = FilterUtils.bandpass_butterworth(
            ibi_future.values, dt_hours, *self.band, axis=0,
        )
        ibi_flat = ibi_bp[:, self.mask_x]

        # Imputation of residual NaNs by the training mean
        nan_loc = np.isnan(ibi_flat)
        if nan_loc.any():
            n_nan = int(nan_loc.sum())
            logger.warning(
                f"  EOF predict: {n_nan}/{nan_loc.size} "
                f"({100*n_nan/nan_loc.size:.2f}%) NaNs in future IBI → "
                f"imputed with the training mean."
            )
            ibi_flat = ibi_flat.copy()
            ibi_flat[nan_loc] = np.broadcast_to(
                self.mean_ibi, ibi_flat.shape
            )[nan_loc]

        ibi_norm = (ibi_flat - self.mean_ibi) / self.std_ibi
        PCs_x_fut = ibi_norm @ self.eofs_x.T
        PCs_y_hat = self.cca.predict(PCs_x_fut)
        bias_valid_hat = PCs_y_hat @ self.eofs_y + self.mean_bias

        nt_fut = bias_valid_hat.shape[0]
        ny, nx = self.mask_y.shape
        bias_pred = np.full((nt_fut, ny, nx), np.nan, dtype=PRED_DTYPE)
        bias_pred[:, self.mask_y] = bias_valid_hat

        return BiasFieldResult.from_arrays(
            bias_pred, ibi_future.time.values,
            ibi_future.lon.values, ibi_future.lat.values,
            method_name="CCA(EOF×EOF) Regression",
            metadata={
                "n_modes_ibi": self.n_modes_x,
                "n_modes_bias": self.n_modes_y,
                "n_canonical": self.n_canonical,
                "variance_explained_ibi": self.variance_explained_x_,
                "variance_explained_bias": self.variance_explained_y_,
                "dt_fit_h": self.dt_fit,
                "dt_predict_h": dt_hours,
            },
        )


# =============================================================================
# SECTION 4: VALIDATION
# =============================================================================


class ValidationMetrics:
    """Validation metrics for the bias correction."""

    @staticmethod
    def rmse(pred: np.ndarray, ref: np.ndarray,
             axis: Optional[int] = None) -> np.ndarray:
        return np.sqrt(np.nanmean((pred - ref) ** 2, axis=axis))

    @staticmethod
    def mae(pred: np.ndarray, ref: np.ndarray,
            axis: Optional[int] = None) -> np.ndarray:
        return np.nanmean(np.abs(pred - ref), axis=axis)

    @staticmethod
    def correlation(pred: np.ndarray, ref: np.ndarray,
                    axis: int = 0) -> np.ndarray:
        """Pearson correlation (axis `axis`), with a PAIRED NaN mask.

        Means and sums over the common mask ~(isnan(pred)|isnan(ref)),
        so that numerator and denominator cover the same samples
        (otherwise r may fall outside [-1, 1]).
        <2 paired points → NaN.
        """
        valid = ~(np.isnan(pred) | np.isnan(ref))
        n_valid = valid.sum(axis=axis)

        # Means over the common mask only.
        pred_z = np.where(valid, pred, 0.0)
        ref_z = np.where(valid, ref, 0.0)
        denom_n = np.maximum(n_valid.astype(float), 1.0)
        pred_mean = pred_z.sum(axis=axis, keepdims=True) / np.expand_dims(
            denom_n, axis
        )
        ref_mean = ref_z.sum(axis=axis, keepdims=True) / np.expand_dims(
            denom_n, axis
        )

        # Anomalies, set to 0 outside the common mask to cancel their
        # contribution to the three sums.
        pred_anom = np.where(valid, pred - pred_mean, 0.0)
        ref_anom = np.where(valid, ref - ref_mean, 0.0)

        num = np.sum(pred_anom * ref_anom, axis=axis)
        den = np.sqrt(
            np.sum(pred_anom ** 2, axis=axis)
            * np.sum(ref_anom ** 2, axis=axis)
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.where(den > 1e-30, num / den, np.nan)
        # Not enough paired points → undefined rather than 0.
        corr = np.where(n_valid >= 2, corr, np.nan)
        return corr

    @staticmethod
    def skill_score(pred: np.ndarray, ref: np.ndarray,
                    baseline: np.ndarray, axis: int = 0) -> np.ndarray:
        """1 − MSE(pred,ref) / MSE(baseline,ref). NaN if the baseline is perfect."""
        mse_pred = np.nanmean((pred - ref) ** 2, axis=axis)
        mse_base = np.nanmean((baseline - ref) ** 2, axis=axis)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(mse_base > 1e-30, 1 - mse_pred / mse_base, np.nan)

    @staticmethod
    def explained_variance(pred: np.ndarray, ref: np.ndarray,
                           axis: int = 0) -> np.ndarray:
        """EV = 1 − var(ref−pred)/var(ref), BOTH on the paired mask
        (otherwise EV is an artefact when the pred and ref masks differ)."""
        valid = ~(np.isnan(pred) | np.isnan(ref))
        diff = np.where(valid, ref - pred, np.nan)
        ref_masked = np.where(valid, ref, np.nan)
        var_error = np.nanvar(diff, axis=axis)
        var_ref = np.nanvar(ref_masked, axis=axis)
        with np.errstate(divide='ignore', invalid='ignore'):
            ev = np.where(var_ref > 1e-30, 1.0 - var_error / var_ref, np.nan)
        return ev

    # ---- Extreme-oriented metrics -------------------------------------

    @staticmethod
    def quantile_error(pred: np.ndarray, ref: np.ndarray,
                       quantiles: Tuple[float, ...] = (0.95, 0.99, 0.999),
                       axis: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Compare the upper quantiles of the pred / ref distributions."""
        qs = np.asarray(quantiles)
        q_ref = np.nanquantile(ref, qs, axis=axis)
        q_pred = np.nanquantile(pred, qs, axis=axis)
        bias = q_pred - q_ref
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = np.where(np.abs(q_ref) > 1e-30, bias / np.abs(q_ref), np.nan)
        return {'quantiles': qs, 'q_ref': q_ref, 'q_pred': q_pred,
                'bias': bias, 'relative_bias': rel}

    @staticmethod
    def peak_over_threshold_metrics(
        pred: np.ndarray, ref: np.ndarray,
        threshold: Optional[float] = None,
        threshold_quantile: float = 0.95,
    ) -> Dict[str, float]:
        """RMSE / bias / correlation of pred vs ref on the points where
        ref >= threshold (default: the `threshold_quantile` of ref)."""
        pred = np.asarray(pred).ravel()
        ref = np.asarray(ref).ravel()
        valid = ~(np.isnan(pred) | np.isnan(ref))
        if not valid.any():
            return {'threshold': np.nan, 'n_peaks': 0,
                    'rmse_pot': np.nan, 'bias_pot': np.nan,
                    'corr_pot': np.nan, 'mean_ref_pot': np.nan}

        ref_v = ref[valid]
        pred_v = pred[valid]
        if threshold is None:
            threshold = float(np.quantile(ref_v, threshold_quantile))
        peak_mask = ref_v >= threshold
        n_peaks = int(peak_mask.sum())
        if n_peaks < 2:
            return {'threshold': threshold, 'n_peaks': n_peaks,
                    'rmse_pot': np.nan, 'bias_pot': np.nan,
                    'corr_pot': np.nan, 'mean_ref_pot': np.nan}

        diff = pred_v[peak_mask] - ref_v[peak_mask]
        rmse_pot = float(np.sqrt(np.mean(diff ** 2)))
        bias_pot = float(np.mean(diff))
        if (np.std(pred_v[peak_mask]) > 1e-30
                and np.std(ref_v[peak_mask]) > 1e-30):
            corr_pot = float(np.corrcoef(
                pred_v[peak_mask], ref_v[peak_mask]
            )[0, 1])
        else:
            corr_pot = np.nan
        return {
            'threshold': threshold, 'n_peaks': n_peaks,
            'rmse_pot': rmse_pot, 'bias_pot': bias_pot,
            'corr_pot': corr_pot,
            'mean_ref_pot': float(np.mean(ref_v[peak_mask])),
        }


def _extremes_block(pred_flat: np.ndarray, ref_flat: np.ndarray,
                    metrics: ValidationMetrics, label: str = '') -> Dict:
    """Extreme-value evaluation of pred vs ref (flattened element × time
    pools): upper quantiles (95, 99, 99.9 %) and POT metrics above the
    q95 and q99 of ref. Empty or all-NaN inputs → {} with a warning.
    """
    out = {}
    # Guard: if the arrays are empty or entirely NaN, any quantile/POT
    # computation is pointless and crashes (e.g. all-NaN projected fields,
    # non-overlapping domains, badly ordered coordinates).
    if pred_flat.size == 0 or ref_flat.size == 0:
        logger.warning(
            f"  Extremes{label}: empty arrays "
            f"(pred={pred_flat.size}, ref={ref_flat.size}) — "
            f"quantiles/POT skipped."
        )
        return out
    n_valid_pred = int(np.isfinite(pred_flat).sum())
    n_valid_ref = int(np.isfinite(ref_flat).sum())
    if n_valid_pred == 0 or n_valid_ref == 0:
        logger.warning(
            f"  Extremes{label}: no finite value "
            f"(valid pred={n_valid_pred}/{pred_flat.size}, "
            f"valid ref={n_valid_ref}/{ref_flat.size}) — "
            f"quantiles/POT skipped."
        )
        return out

    try:
        qe = metrics.quantile_error(
            pred_flat, ref_flat, quantiles=(0.95, 0.99, 0.999)
        )
        # Pathological case: np.nanquantile on a single point returns a 0-d
        # scalar, which cannot be indexed with [i].
        if np.asarray(qe['q_ref']).ndim == 0:
            logger.warning(
                f"  Quantile error{label}: degenerate inputs "
                f"(quantile returns a 0-d scalar) — skipped."
            )
        else:
            out["quantiles"] = {
                f"q{int(q*1000)/10}": {
                    'ref': float(qe['q_ref'][i]),
                    'pred': float(qe['q_pred'][i]),
                    'bias': float(qe['bias'][i]),
                    'rel_bias': float(qe['relative_bias'][i]),
                }
                for i, q in enumerate(qe['quantiles'])
            }
    except Exception as e:
        logger.warning(f"  Quantile error{label} failed: {e}")
    for qthr in (0.95, 0.99):
        try:
            out[f"pot_q{int(qthr*100)}"] = metrics.peak_over_threshold_metrics(
                pred_flat, ref_flat, threshold_quantile=qthr,
            )
        except Exception as e:
            logger.warning(f"  POT q{qthr}{label} failed: {e}")
    return out


def _add_pot_to_summary(summary: Dict, extremes: Dict,
                        extremes_baseline: Optional[Dict] = None) -> None:
    """Add to `summary` the POT stats (thresholds q95 and q99) of `extremes`.

    Naming consistent with mean_rmse / mean_correlation:
      - `mean_rmse_q95`, `mean_corr_q95`, `mean_bias_q95`, `n_peaks_q95`
      - `mean_rmse_q99`, `mean_corr_q99`, `mean_bias_q99`, `n_peaks_q99`

    If `extremes_baseline` is given, also adds the POT stats of the
    baseline (= Tolosa without correction, pred=zero):
      - `mean_rmse_q95_uncorr`, `mean_bias_q95_uncorr` (signed)
    (The baseline Corr Q95 is not computed: corr(zeros, x) is NaN.)

    If the `pot_qXX` key is missing from extremes (the block failed —
    empty array, all-NaN, etc.), the stats are set to NaN.

    Semantics:
      - POT q95: threshold = q95(ref), quantile of the SIGNED bias (not
        of |ref|): metrics on the ~5 % of points where sat ≫ model. Large
        negative biases are NOT covered by this diagnostic.
    """
    for q_str in ('95', '99'):
        pot = extremes.get(f'pot_q{q_str}', {})
        summary[f'mean_rmse_q{q_str}'] = pot.get('rmse_pot', np.nan)
        summary[f'mean_corr_q{q_str}'] = pot.get('corr_pot', np.nan)
        summary[f'mean_bias_q{q_str}'] = pot.get('bias_pot', np.nan)
        summary[f'n_peaks_q{q_str}'] = pot.get('n_peaks', 0)
    if extremes_baseline is not None:
        for q_str in ('95', '99'):
            pot_b = extremes_baseline.get(f'pot_q{q_str}', {})
            summary[f'mean_rmse_q{q_str}_uncorr'] = pot_b.get(
                'rmse_pot', np.nan
            )
            summary[f'mean_bias_q{q_str}_uncorr'] = pot_b.get(
                'bias_pot', np.nan
            )


def _log_q99(extremes: Dict) -> None:
    """Compact log of the 99 / 99.9 quantiles of an extremes block."""
    if "quantiles" not in extremes:
        return
    q99 = extremes["quantiles"].get("q99.0", {})
    q999 = extremes["quantiles"].get("q99.9", {})
    logger.info(
        f"  q99: ref={q99.get('ref', np.nan):.4f}, "
        f"pred={q99.get('pred', np.nan):.4f}, "
        f"bias={q99.get('bias', np.nan):+.4f} ; "
        f"q99.9 bias={q999.get('bias', np.nan):+.4f}"
    )


class Validator:
    """Validation of the correction at the elements of the barotropic mesh."""

    def __init__(self):
        self.metrics = ValidationMetrics()

    def validate_correction_at_elements(
        self, bias_result: BiasFieldResult, bias_ref_at_elements: xr.DataArray,
        W_ibi_to_tolosa: sparse.spmatrix,
        model_at_elements_da: Optional[xr.DataArray] = None,
        baseline_map: Optional[np.ndarray] = None,
    ) -> Dict:
        """Validation at the elements of the unstructured mesh.

        Parameters
        ----------
        bias_result : BiasFieldResult
            Correction on the IBI grid (IBI dt).
        bias_ref_at_elements : xr.DataArray (time, element)
            Reference bias at the elements (coords 'elem_lon'/'elem_lat' on
            'element').
        W_ibi_to_tolosa : sparse (n_elements, ny_ibi*nx_ibi)
        model_at_elements_da : optional
            Daily Tolosa field at the elements (dims time, element). If given,
            computes the additional obs-space scores:
              - `mean_correlation_tolosa`: corr(Tolosa, sat)
              - `mean_explained_variance_tolosa`: EV(Tolosa, sat)
              - `mean_correlation_corrected`: corr(Tolosa+pred, sat)
              - `mean_explained_variance_corrected`: EV(Tolosa+pred, sat)
            where sat is rebuilt as `sat = bias_ref + model` (aligned in
            time). RMSE and MAE are identical in bias space and obs space,
            hence not duplicated.
        baseline_map : optional, np.ndarray (element,)
            Bias climatology at the elements used as the skill-score
            baseline. Recommended convention: temporal mean of the bias
            over the TRAIN period (available operationally). If None
            (fallback), mean of the validation window itself — an a
            posteriori climatology that "sees" the test period.
        """
        elem_lon = bias_ref_at_elements['elem_lon'].values
        elem_lat = bias_ref_at_elements['elem_lat'].values
        t_start = time.perf_counter()
        logger.info(
            f"Validation at the elements — {bias_result.method_name} "
            f"({len(elem_lon)} elements)"
        )
        results = {"method": bias_result.method_name,
                   "per_element": {}, "summary": {}}

        # STEP 1 — Daily correction + projection onto the elements
        t1 = time.perf_counter()
        daily_corr_da = resample_daily(bias_result.da)
        n_dst = W_ibi_to_tolosa.shape[0]
        if n_dst != bias_ref_at_elements.sizes['element']:
            raise ValueError(
                f"validation: W_ibi_to_tolosa has {n_dst} rows, the reference "
                f"bias {bias_ref_at_elements.sizes['element']} elements."
            )
        # daily_corr_da.values forces materialisation (cheap if already numpy
        # after predict(); otherwise a Dask compute).
        corr_at_elements_2d = apply_projection(
            W_ibi_to_tolosa, daily_corr_da.values, (n_dst,)
        )
        corr_elements_da = xr.DataArray(
            corr_at_elements_2d, dims=('time', 'element'),
            coords={'time': daily_corr_da['time'].values},
            name='corr_at_elements',
        )
        logger.info(
            f"  projection IBI→elements in {time.perf_counter()-t1:.1f}s"
        )

        # STEP 2 — Time alignment + materialisation
        # `bias_ref_at_elements` is normally already materialised (see
        # compute_bias_at_elements); .load() for safety.
        t2 = time.perf_counter()
        pred_da, ref_da = align_inner_time(corr_elements_da,
                                          bias_ref_at_elements)
        if pred_da.sizes['time'] == 0:
            logger.warning("  No common day")
            return results
        pred = pred_da.load().values
        ref = ref_da.load().values
        logger.info(
            f"  align + load in {time.perf_counter()-t2:.1f}s "
            f"({pred.nbytes/1e9:.1f} GB pred, {ref.nbytes/1e9:.1f} GB ref)"
        )

        # STEP 3 — Per-element metrics
        t3 = time.perf_counter()
        n_valid = (~(np.isnan(pred) | np.isnan(ref))).sum(axis=0)
        enough = n_valid >= 10

        def _mask(arr):
            return np.where(enough, arr, np.nan)

        rmse_elements = _mask(self.metrics.rmse(pred, ref, axis=0))
        mae_elements = _mask(self.metrics.mae(pred, ref, axis=0))
        corr_elements = _mask(self.metrics.correlation(pred, ref, axis=0))
        ev_elements = _mask(self.metrics.explained_variance(pred, ref, axis=0))
        # Skill baseline: TRAIN climatology if given, otherwise mean of the
        # validation window (fallback).
        if baseline_map is not None:
            base_mean = np.asarray(baseline_map)[None, :].astype(
                pred.dtype, copy=False
            )
        else:
            base_mean = np.nanmean(ref, axis=0, keepdims=True)
        baseline = np.broadcast_to(base_mean, pred.shape)
        skill_elements = _mask(
            self.metrics.skill_score(pred, ref, baseline, axis=0)
        )
        rmse_uncorr_elements = _mask(
            self.metrics.rmse(np.zeros_like(ref), ref, axis=0)
        )
        mae_uncorr_elements = _mask(
            self.metrics.mae(np.zeros_like(ref), ref, axis=0)
        )
        logger.info(
            f"  6 metrics on {pred.shape} in "
            f"{time.perf_counter()-t3:.1f}s"
        )

        results["per_element"] = {
            "rmse": rmse_elements, "mae": mae_elements,
            "correlation": corr_elements,
            "explained_variance": ev_elements, "skill_score": skill_elements,
            "rmse_uncorr": rmse_uncorr_elements,
            "mae_uncorr": mae_uncorr_elements,
            "lon": elem_lon, "lat": elem_lat,
        }
        ocean_mask = ~np.isnan(rmse_elements)
        results["summary"] = {
            "mean_rmse": float(np.nanmean(rmse_elements[ocean_mask])),
            "mean_mae": float(np.nanmean(mae_elements[ocean_mask])),
            "mean_rmse_uncorr": float(np.nanmean(
                rmse_uncorr_elements[ocean_mask]
            )),
            "mean_mae_uncorr": float(np.nanmean(
                mae_uncorr_elements[ocean_mask]
            )),
            "mean_correlation": float(np.nanmean(corr_elements[ocean_mask])),
            "mean_explained_variance": float(
                np.nanmean(ev_elements[ocean_mask])),
            "mean_skill_score": float(np.nanmean(skill_elements[ocean_mask])),
            "n_days": int(pred_da.sizes['time']),
            "n_elements_valid": int(ocean_mask.sum()),
        }

        # ---- Obs-space scores for Tolosa (and corrected) ----------------
        # sat rebuilt as sat = bias_ref + model (model_at_elements_da).
        if model_at_elements_da is not None:
            model_aligned, _ = align_inner_time(model_at_elements_da, ref_da)
            if model_aligned.sizes['time'] == ref_da.sizes['time']:
                model_arr = model_aligned.values
                sat_arr = model_arr + ref
                corrected_arr = model_arr + pred

                corr_tolosa_elements = _mask(self.metrics.correlation(
                    model_arr, sat_arr, axis=0,
                ))
                ev_tolosa_elements = _mask(self.metrics.explained_variance(
                    model_arr, sat_arr, axis=0,
                ))
                corr_corrected_elements = _mask(self.metrics.correlation(
                    corrected_arr, sat_arr, axis=0,
                ))
                ev_corrected_elements = _mask(self.metrics.explained_variance(
                    corrected_arr, sat_arr, axis=0,
                ))

                results["per_element"]["corr_tolosa"] = corr_tolosa_elements
                results["per_element"]["ev_tolosa"] = ev_tolosa_elements
                results["per_element"]["corr_corrected"] = (
                    corr_corrected_elements)
                results["per_element"]["ev_corrected"] = ev_corrected_elements

                results["summary"]["mean_correlation_tolosa"] = float(
                    np.nanmean(corr_tolosa_elements[ocean_mask])
                )
                results["summary"]["mean_explained_variance_tolosa"] = float(
                    np.nanmean(ev_tolosa_elements[ocean_mask])
                )
                results["summary"]["mean_correlation_corrected"] = float(
                    np.nanmean(corr_corrected_elements[ocean_mask])
                )
                results["summary"]["mean_explained_variance_corrected"] = float(
                    np.nanmean(ev_corrected_elements[ocean_mask])
                )
            else:
                logger.warning(
                    f"  model_at_elements_da cannot be aligned with ref_da "
                    f"({model_aligned.sizes['time']} vs "
                    f"{ref_da.sizes['time']} days) — "
                    f"Tolosa-vs-sat scores not computed."
                )

        if ocean_mask.any():
            pred_flat = pred[:, ocean_mask].ravel()
            ref_flat = ref[:, ocean_mask].ravel()
            # NB: element × time pool — distributional comparison only
            # (spatially and temporally dependent samples).
            results["extremes"] = _extremes_block(
                pred_flat, ref_flat, self.metrics, label=" (elements)"
            )
            results["extremes_baseline"] = _extremes_block(
                np.zeros_like(ref_flat), ref_flat, self.metrics,
                label=" baseline (elements)",
            )
            _add_pot_to_summary(
                results["summary"], results["extremes"],
                extremes_baseline=results["extremes_baseline"],
            )
            _log_q99(results["extremes"])

        logger.info(
            f"  RMSE = {results['summary']['mean_rmse']:.4f}, "
            f"Corr = {results['summary']['mean_correlation']:.3f}, "
            f"Skill = {results['summary']['mean_skill_score']:.3f} "
            f"({results['summary']['n_days']} d, "
            f"{ocean_mask.sum()} elements) — total "
            f"{time.perf_counter()-t_start:.1f}s"
        )
        return results


# =============================================================================
# SECTION 5: 2D FIELD PLOTS
# =============================================================================
#
# Main plotting functions:
#   - plot_subdaily_ibi_tolosa(): IBI / Tolosa-SW (scatter at the elements)
#     over the first and last days of the test period, subsampled (6 h);
#   - plot_daily_ibi_tolosa_biasref(): one figure, one row per day,
#     columns IBI / projected Tolosa / native sat / sat on IBI / bias_ref;
#   - plot_corrections_per_timestep(): one figure per day, bias_ref + the
#     correction, common colour scale.
#
# Parallelised with THREADS (ThreadPoolExecutor), not processes: on HPC
# nodes, spawning a child process from a large parent (IBI + results +
# pipeline) forces the kernel to reserve ~the parent size and fails with
# ENOMEM (OSError Errno 12), whatever the number of workers. Threads
# share the parent memory; Agg rendering and PNG writing release the
# GIL, hence real parallelism. Workers MUST use the object-oriented
# matplotlib API (Figure), never plt.* (pyplot has a non thread-safe
# global state). The useful ceiling is the disk throughput.

N_PLOT_WORKERS = 64   # number of plot rendering threads


def _make_output_dir(label: str) -> Path:
    out = Path(f"plots_2D_{label}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _vlim_quantile(data_list, quantile: float = 0.98,
                   symmetric: bool = True) -> Tuple[float, float]:
    """Robust (vmin, vmax) over all the stacked arrays."""
    flat = np.concatenate([np.ravel(a) for a in data_list if a is not None])
    flat = flat[~np.isnan(flat)]
    if flat.size == 0:
        return -1.0, 1.0
    if symmetric:
        v = float(np.quantile(np.abs(flat), quantile))
        if v < 1e-6:
            v = float(np.max(np.abs(flat))) if flat.size > 0 else 1.0
        return -v, v
    q_low = 1.0 - quantile
    return float(np.quantile(flat, q_low)), float(np.quantile(flat, quantile))


def _select_subdaily_times(times_native: np.ndarray,
                           test_start: np.datetime64,
                           test_end: np.datetime64,
                           step_hours: float = 6.0,
                           n_days_each_end: int = 3) -> np.ndarray:
    """Select the time steps of the n_days_each_end first + last days."""
    ts_start = np.datetime64(test_start, 'ns')
    ts_end = np.datetime64(test_end, 'ns')
    day = np.timedelta64(1, 'D')
    head_end = ts_start + n_days_each_end * day
    tail_start = ts_end - n_days_each_end * day

    times_ns = times_native.astype('datetime64[ns]')
    in_head = (times_ns >= ts_start) & (times_ns < head_end)
    in_tail = (times_ns >= tail_start) & (times_ns <= ts_end)
    cand_idx = np.where(in_head | in_tail)[0]
    if len(cand_idx) == 0:
        return cand_idx

    step_ns = np.timedelta64(int(step_hours * 3600 * 1e9), 'ns')
    selected = []
    last_kept_t = None
    for i in cand_idx:
        t = times_ns[i]
        if last_kept_t is None or (t - last_kept_t) >= step_ns:
            selected.append(i)
            last_kept_t = t
    return np.array(selected, dtype=int)


def _run_parallel(worker_fn, tasks, n_workers, label=""):
    """Run worker_fn(task) on each task in parallel.

    Returns
    -------
    (n_ok, n_err): number of successful / failed tasks. Worker
    exceptions are caught and logged individually; CALLERS must use n_ok
    in their final message — announcing ``len(tasks)`` written would
    hide the failures.
    """
    if len(tasks) == 0:
        return 0, 0
    if n_workers <= 1 or len(tasks) == 1:
        n_err = 0
        for i, t in enumerate(tasks):
            try:
                worker_fn(t)
            except Exception as e:
                n_err += 1
                logger.error(f"  [plot {label}] worker error: {e}")
            if (i + 1) % max(1, len(tasks) // 10) == 0:
                logger.info(f"  [plot {label}] {i+1}/{len(tasks)}")
        if n_err:
            logger.warning(
                f"  [plot {label}] ⚠ {n_err}/{len(tasks)} tasks FAILED"
            )
        return len(tasks) - n_err, n_err

    n_workers = min(n_workers, len(tasks))
    logger.info(
        f"  [plot {label}] {n_workers} threads, {len(tasks)} tasks…"
    )
    # Threads rather than processes: see the section header.
    done = 0
    n_err = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(worker_fn, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                n_err += 1
                logger.error(f"  [plot {label}] worker error: {e}")
            done += 1
            if done % max(1, len(tasks) // 10) == 0:
                logger.info(f"  [plot {label}] {done}/{len(tasks)}")
    if n_err:
        logger.warning(
            f"  [plot {label}] ⚠ {n_err}/{len(tasks)} tasks FAILED"
        )
    return len(tasks) - n_err, n_err


# ---- Worker subdaily ------------------------------------------------------

def _draw_field(fig, ax, lon, lat, field, vlim, title=None):
    """Draw a 2D field (pcolormesh RdBu_r, scale `vlim`) + colorbar in m on
    `ax`. Thread-safe (object-oriented matplotlib API, no plt.*)."""
    pm = ax.pcolormesh(lon, lat, field, shading='auto', cmap='RdBu_r',
                       vmin=vlim[0], vmax=vlim[1])
    if title is not None:
        ax.set_title(title)
    fig.colorbar(pm, ax=ax, label="m")
    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")
    return pm


def _finalize_fig(fig, out_path, dpi, suptitle=None, fontsize=13):
    """Optional overall title, tight_layout and write to disk."""
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=fontsize)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')


def _plot_subdaily_one(args):
    """Worker for 1 sub-daily time step (IBI + Tolosa surge).
    Thread-safe: object-oriented matplotlib API only (no plt.*)."""
    (ibi_lon, ibi_lat, ibi_field,
     tol_lon, tol_lat, tol_field,
     t_ibi_str, t_tol_str, var_mod,
     vlim_ibi, vlim_tol,
     out_path, dpi) = args
    fig = Figure(figsize=(13, 5))
    axes = fig.subplots(1, 2)
    pm = axes[0].pcolormesh(
        ibi_lon, ibi_lat, ibi_field, shading='auto',
        cmap='RdBu_r', vmin=vlim_ibi[0], vmax=vlim_ibi[1],
    )
    axes[0].set_title(f"IBI ({var_mod})\n{t_ibi_str}")
    fig.colorbar(pm, ax=axes[0], label="m")
    sc = axes[1].scatter(
        tol_lon, tol_lat, c=tol_field, s=2, cmap='RdBu_r',
        vmin=vlim_tol[0], vmax=vlim_tol[1],
        edgecolors='none', rasterized=True,
    )
    axes[1].set_title(f"Tolosa-SW ({var_mod})\n{t_tol_str}")
    axes[1].set_xlim(ibi_lon.min(), ibi_lon.max())
    axes[1].set_ylim(ibi_lat.min(), ibi_lat.max())
    fig.colorbar(sc, ax=axes[1], label="m")
    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    _finalize_fig(fig, out_path, dpi)


def plot_subdaily_ibi_tolosa(
    ibi_da: xr.DataArray, da_mod: xr.DataArray,
    test_start: np.datetime64, test_end: np.datetime64,
    var_mod: str,
    n_days_each_end: int = 3, step_hours: float = 6.0,
    out_label: Optional[str] = None, dpi: int = 100,
    n_workers: Optional[int] = None,
) -> None:
    """IBI (pcolormesh) + Tolosa-SW (scatter at the elements), subsampled.

    2 columns (IBI / Tolosa-SW) over the first and last `n_days_each_end`
    days of the test period, one step every `step_hours`.
    """
    out_dir = _make_output_dir(out_label if out_label is not None else var_mod)
    if n_workers is None:
        n_workers = N_PLOT_WORKERS
    logger.info(
        f"[plot] subdaily: {n_days_each_end}+{n_days_each_end} d × "
        f"step {step_hours}h → {out_dir}/"
    )

    ibi_times = ibi_da.time.values
    tolosa_times = da_mod.time.values
    ibi_sel_idx = _select_subdaily_times(
        ibi_times, test_start, test_end, step_hours, n_days_each_end
    )
    tolosa_sel_idx = _select_subdaily_times(
        tolosa_times, test_start, test_end, step_hours, n_days_each_end
    )
    if len(ibi_sel_idx) == 0 or len(tolosa_sel_idx) == 0:
        logger.warning("[plot] subdaily: no time step selected")
        return

    tol_lon = da_mod.lon.values
    tol_lat = da_mod.lat.values
    ibi_lon = ibi_da.lon.values
    ibi_lat = ibi_da.lat.values
    n_plots = min(len(ibi_sel_idx), len(tolosa_sel_idx))

    sample_n = min(4, len(ibi_sel_idx))
    ibi_sample = ibi_da.isel(time=ibi_sel_idx[:sample_n]).values
    tol_sample = da_mod.isel(time=tolosa_sel_idx[:sample_n]).values
    vlim_ibi = _vlim_quantile([ibi_sample], quantile=0.98, symmetric=True)
    vlim_tol = _vlim_quantile([tol_sample], quantile=0.98, symmetric=True)

    tasks = []
    for k in range(n_plots):
        i_ibi, i_tol = ibi_sel_idx[k], tolosa_sel_idx[k]
        t_ibi, t_tol = ibi_times[i_ibi], tolosa_times[i_tol]
        ts_str = np.datetime_as_string(t_ibi, unit='h').replace(
            'T', '_').replace(':', '')
        tasks.append((
            ibi_lon, ibi_lat, ibi_da.isel(time=i_ibi).values,
            tol_lon, tol_lat, da_mod.isel(time=i_tol).values,
            np.datetime_as_string(t_ibi, unit='h'),
            np.datetime_as_string(t_tol, unit='h'),
            var_mod, vlim_ibi, vlim_tol,
            out_dir / f"subdaily_ibi_tolosa_{ts_str}.png", dpi,
        ))

    n_ok, n_err = _run_parallel(_plot_subdaily_one, tasks, n_workers,
                                "subdaily")
    logger.info(f"[plot] subdaily: {n_ok} figures written"
                + (f" ({n_err} FAILED)" if n_err else ""))


# ---- Worker corrections ---------------------------------------------------

def _plot_corrections_one(args):
    """Worker for 1 time step (bias_ref + N corrections). Thread-safe:
    object-oriented matplotlib API only (no plt.*)."""
    (day_str, lon_sub, lat_sub, bref_sub, corrs_today_sub,
     method_names, vlim, var_mod, out_path, dpi) = args

    # Dynamic layout: 4 columns, as many rows as needed.
    n_panels = len(method_names) + 1  # bias_ref + methods
    ncols = 4
    nrows = max(1, -(-n_panels // ncols))  # ceil
    fig = Figure(figsize=(18, 4.5 * nrows))
    axes = fig.subplots(nrows, ncols)
    flat_axes = np.asarray(axes).ravel()

    _draw_field(fig, flat_axes[0], lon_sub, lat_sub, bref_sub, vlim,
                title="bias_ref (sat − Tolosa)")
    for k, name in enumerate(method_names):
        ax = flat_axes[k + 1]
        if name in corrs_today_sub:
            _draw_field(fig, ax, lon_sub, lat_sub, corrs_today_sub[name],
                        vlim, title=name)
        else:
            ax.set_title(name)
    for k in range(len(method_names) + 1, len(flat_axes)):
        flat_axes[k].set_visible(False)

    _finalize_fig(fig, out_path, dpi,
                  suptitle=f"Corrections vs bias_ref — {day_str} ({var_mod})",
                  fontsize=14)


def _read_ugrid_mesh(tolosa_path: str) -> Optional[Dict]:
    """Read the UGRID topology of a Tolosa-SW NetCDF (triangular mesh).

    Returns a dict ready to be embedded in an element NetCDF file:
        {
          'face_nodes': (nelements, 3) int32   — connectivity (transposed
                         from (three, nelements) if needed),
          'face_nodes_attrs': dict              — original attributes
                         (including start_index, to keep AS IS),
          'node_lon': (nnodes,) float64,
          'node_lat': (nnodes,) float64,
          'mesh_attrs': dict                    — attributes of the
                         topology variable (cf_role='mesh_topology', …),
        }
    or None if no usable connectivity is found (the export then remains
    a cloud of elements — degraded but non-blocking behaviour).

    Robust detection: first look for the topology variable
    (cf_role='mesh_topology') to read face_node_connectivity /
    node_coordinates; otherwise fall back on usual candidate names. The
    connectivity is returned as (nelements, three) whatever the original
    orientation; indices are NOT renormalised (start_index kept in the
    attributes).
    """
    try:
        ds = xr.open_dataset(tolosa_path, decode_cf=False)
    except Exception as e:
        logger.warning(
            f"  [mesh] cannot open ({tolosa_path}): {e} — "
            f"mesh not exported."
        )
        return None

    try:
        conn_name = None
        node_lon_name = None
        node_lat_name = None
        mesh_attrs: Dict = {}

        # 1) Topology variable (cf_role='mesh_topology').
        for var in ds.variables.values():
            if var.attrs.get('cf_role') == 'mesh_topology':
                mesh_attrs = dict(var.attrs)
                conn_name = var.attrs.get('face_node_connectivity')
                nodecoords = var.attrs.get('node_coordinates', '')
                parts = nodecoords.split()
                if len(parts) == 2:
                    # conventional "lon lat" order
                    node_lon_name, node_lat_name = parts[0], parts[1]
                break

        # 2) Fallbacks on candidate names if the topology did not give everything.
        if conn_name is None or conn_name not in ds.variables:
            for cand in ('Mesh2_face_nodes', 'mesh2d_face_nodes',
                         'face_nodes', 'element', 'nv', 'triangle',
                         'connectivity'):
                if cand in ds.variables:
                    conn_name = cand
                    break
        if conn_name is None or conn_name not in ds.variables:
            logger.warning(
                "  [mesh] no connectivity variable found "
                f"(topology/candidates). Variables: {list(ds.variables)[:12]}… "
                "— mesh not exported."
            )
            return None

        if node_lon_name not in ds.variables or node_lat_name not in ds.variables:
            # node coordinate candidates
            node_lon_name = next(
                (c for c in ('nlon', 'node_lon', 'lon_node', 'mesh2d_node_x')
                 if c in ds.variables), None)
            node_lat_name = next(
                (c for c in ('nlat', 'node_lat', 'lat_node', 'mesh2d_node_y')
                 if c in ds.variables), None)
        if node_lon_name is None or node_lat_name is None:
            logger.warning(
                "  [mesh] node coordinates not found — "
                "mesh not exported."
            )
            return None

        conn_var = ds[conn_name]
        conn = np.asarray(conn_var.values)
        conn_attrs = dict(conn_var.attrs)
        # Orientation → (nelements, three). The axis of size 3 (or 4) is the
        # "vertices" axis; the other one is "elements".
        if conn.ndim != 2:
            logger.warning(
                f"  [mesh] connectivity {conn_name} of rank {conn.ndim} "
                f"(expected 2) — mesh not exported."
            )
            return None
        vert_axis = 0 if conn.shape[0] <= conn.shape[1] else 1
        if vert_axis == 0:
            conn = conn.T  # (three, nelements) → (nelements, three)
        n_vert = conn.shape[1]
        if n_vert != 3:
            logger.warning(
                f"  [mesh] {n_vert} vertices/face (expected 3, triangular) — "
                f"exported as is as (nelements, {n_vert})."
            )
        conn = np.ascontiguousarray(conn.astype(np.int32, copy=False))

        node_lon = np.asarray(ds[node_lon_name].values, dtype=np.float64)
        node_lat = np.asarray(ds[node_lat_name].values, dtype=np.float64)

        logger.info(
            f"  [mesh] connectivity '{conn_name}' {conn.shape} "
            f"(start_index={conn_attrs.get('start_index', '?')}), "
            f"{node_lon.size} nodes ({node_lon_name}/{node_lat_name})."
        )
        return {
            'face_nodes': conn,
            'face_nodes_attrs': conn_attrs,
            'node_lon': node_lon,
            'node_lat': node_lat,
            'mesh_attrs': mesh_attrs,
        }
    except Exception as e:
        logger.warning(
            f"  [mesh] reading the topology failed ({e}) — "
            f"mesh not exported."
        )
        return None
    finally:
        ds.close()


def export_sources_to_netcdf(
    *,
    da_sat: xr.DataArray,
    da_mod: xr.DataArray,
    correction_da: xr.DataArray,
    bias_ref: xr.DataArray,
    correction_label: str,
    out_label: str,
    period_start: np.datetime64,
    period_end: np.datetime64,
    W_ibi_to_tolosa,
    elem_lon: np.ndarray,
    elem_lat: np.ndarray,
    tolosa_path: Optional[str] = None,
    skill_per_method: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Export 6 sources to NetCDF-4 (1 file/source), daily, test period.

    Sources and grids:
      - sat             (SLA + DAC)         → native IBI grid
      - bias_ref        (sat − Tolosa)      → native IBI grid
      - residual        (bias_ref − corr)   → native IBI grid
      - model           (Tolosa surge)      → mesh elements
      - correction      (correction_label)  → elements (projected from IBI)
      - model_corrected (surge + corr)      → elements

    All sources are first brought to DAILY resolution (resample_daily)
    and restricted to [period_start, period_end].

    Interpretation note: `residual` stays on the IBI grid
    (= bias_ref_IBI − correction_IBI), consistent with the figures. It is
    NOT the residual of the corrected model at the elements.

    Output: plots_2D_<out_label>/netcdf_export/<source>.nc

    Storage: NetCDF-4 with one chunk per time step (time=1, full spatial
    dimensions) and zlib compression, for efficient reading time step by
    time step (reading 1 day = reading 1 chunk).

    `skill_per_method`: skill maps PER ELEMENT (test block), one per
    validated method, written to a dedicated `skill.nc` file (static
    fields without a time dimension, on the mesh dimension, with lon/lat
    and the UGRID mesh). The skill compares the error of the correction
    with that of a TRAIN climatological baseline: > 0 = improvement,
    < 0 = degradation.

    All sources are cropped to the INTERSECTION of their extents
    (lon/lat), so that every file covers exactly the same domain. Regular
    grids are cropped with `sel`, element sources with a mask on the mesh
    dimension — the UGRID connectivity is cropped accordingly, node
    coordinates staying complete so that the indices remain valid.
    """
    base = _make_output_dir(out_label)
    root = base / "netcdf_export"
    root.mkdir(parents=True, exist_ok=True)

    t0 = np.datetime64(period_start, 'ns')
    t1 = np.datetime64(period_end, 'ns')

    def _daily_period(da):
        d = resample_daily(da)
        if hasattr(d, 'compute'):
            d = d.compute()
        return d.sel(time=slice(t0, t1))

    logger.info("=" * 70)
    logger.info(f"NETCDF EXPORT (daily, test period) → {root}/")
    logger.info("=" * 70)

    # -- Sources on the native IBI grid --
    sat_d = _daily_period(da_sat)
    bref_d = _daily_period(bias_ref)
    corr_ibi_d = _daily_period(correction_da)
    # residual on the IBI grid: align bias_ref ↔ correction in time
    bref_a, corr_a = align_inner_time(bref_d, corr_ibi_d)
    residual_d = (bref_a - corr_a).rename('residual')

    # -- Element sources --
    mod_d = _daily_period(da_mod)  # daily Tolosa surge, elements
    # Spatial dims must have the same name, otherwise xarray broadcasts to
    # (time, N, N), several PiB. Idempotent guard.
    mod_d = canonicalize_element_dim(mod_d)
    # Correction projected IBI → elements (per daily time step)
    n_dst = W_ibi_to_tolosa.shape[0]
    corr_elements_arr = apply_projection(
        W_ibi_to_tolosa, corr_ibi_d.values, (n_dst,)
    ).astype(np.float32, copy=False)
    corr_elements_d = xr.DataArray(
        corr_elements_arr, dims=('time', 'element'),
        coords={'time': corr_ibi_d['time'].values,
                'lon': ('element', np.asarray(elem_lon)),
                'lat': ('element', np.asarray(elem_lat))},
        name='correction',
    )
    # Corrected model = surge + correction, at the elements, aligned in time.
    # Anti-broadcast guard: both element sizes MUST match (otherwise the W
    # projection does not match the da_mod mesh); then a POSITIONAL sum in
    # numpy — shape-checked semantics, as everywhere else in the pipeline
    # after projection — re-wrapped with the model coords.
    mod_a, corrn_a = align_inner_time(mod_d, corr_elements_d)
    if mod_a.sizes.get('element') != corrn_a.sizes.get('element'):
        raise ValueError(
            f"NetCDF export: incompatible element sizes "
            f"(model={mod_a.sizes.get('element')}, "
            f"correction={corrn_a.sizes.get('element')}). The "
            f"W_ibi_to_tolosa matrix does not match the da_mod mesh."
        )
    mod_corrected_d = xr.DataArray(
        mod_a.values + corrn_a.values,
        dims=mod_a.dims, coords=mod_a.coords,
        name='model_corrected',
    )

    # -- Writing, 1 file per source --
    exports = {
        'sat': sat_d.rename('sat'),
        'bias_ref': bref_d.rename('bias_ref'),
        'residual': residual_d,
        'model': mod_d.rename('model'),
        'correction': corr_elements_d,
        'model_corrected': mod_corrected_d,
    }

    # -- Cropping to the common extent (intersection) --
    elem_mask = None
    bbox = None
    lo_mins, lo_maxs, la_mins, la_maxs = [], [], [], []
    for da in exports.values():
        lo = np.asarray(da['lon'].values, dtype=float)
        la = np.asarray(da['lat'].values, dtype=float)
        lo_mins.append(np.nanmin(lo)); lo_maxs.append(np.nanmax(lo))
        la_mins.append(np.nanmin(la)); la_maxs.append(np.nanmax(la))
    lon_min, lon_max = max(lo_mins), min(lo_maxs)
    lat_min, lat_max = max(la_mins), min(la_maxs)
    if lon_min >= lon_max or lat_min >= lat_max:
        logger.warning(
            f"  [bbox] empty intersection "
            f"(lon {lon_min:.3f}-{lon_max:.3f}, "
            f"lat {lat_min:.3f}-{lat_max:.3f}) → cropping ignored."
        )
    else:
        bbox = (lon_min, lon_max, lat_min, lat_max)
        logger.info(
            f"  [bbox] common extent: lon {lon_min:.3f}→{lon_max:.3f}, "
            f"lat {lat_min:.3f}→{lat_max:.3f}"
        )
        cropped = {}
        for name, da in exports.items():
            if 'element' in da.dims:
                lo = np.asarray(da['lon'].values, dtype=float)
                la = np.asarray(da['lat'].values, dtype=float)
                m = ((lo >= lon_min) & (lo <= lon_max)
                     & (la >= lat_min) & (la <= lat_max))
                if elem_mask is None:
                    elem_mask = m
                elif not np.array_equal(elem_mask, m):
                    logger.warning(
                        f"  [bbox] diverging mesh mask for "
                        f"'{name}' — mask of the first source kept."
                    )
                cropped[name] = da.isel(element=np.flatnonzero(elem_mask))
            else:
                lat_vals = np.asarray(da['lat'].values, dtype=float)
                lat_slice = (slice(lat_max, lat_min)
                             if lat_vals[0] > lat_vals[-1]
                             else slice(lat_min, lat_max))
                cropped[name] = da.sel(
                    lon=slice(lon_min, lon_max), lat=lat_slice)
            logger.info(
                f"    {name:<16} {dict(da.sizes)} → "
                f"{dict(cropped[name].sizes)}"
            )
        exports = cropped

    # UGRID mesh (connectivity + node coords) read only once, to be
    # embedded in each (self-contained) element file. If tolosa_path is
    # missing or the topology is not found, the export remains a cloud of
    # elements.
    mesh = _read_ugrid_mesh(tolosa_path) if tolosa_path else None
    if tolosa_path and mesh is None:
        logger.warning(
            "  [mesh] connectivity not available → element files "
            "exported WITHOUT mesh (point cloud)."
        )

    def _attach_mesh(ds: xr.Dataset) -> Tuple[xr.Dataset, bool]:
        """Add the mesh variables to the dataset (own dims, other than
        'time'/'element'). Returns (dataset, mesh added?)."""
        if mesh is None:
            return ds, False
        # The connectivity is PER ELEMENT: if the elements have been cropped,
        # it must be cropped identically to stay aligned. Node coordinates stay
        # COMPLETE → the indices (start_index kept) remain valid.
        face_nodes = mesh['face_nodes']
        if elem_mask is not None:
            if face_nodes.shape[0] == elem_mask.size:
                face_nodes = face_nodes[np.flatnonzero(elem_mask)]
            else:
                logger.warning(
                    f"  [mesh] connectivity ({face_nodes.shape[0]} elements) "
                    f"not aligned with the mask ({elem_mask.size}) — "
                    f"connectivity NOT cropped."
                )
        ds = ds.assign(
            Mesh2_face_nodes=(('nelements', 'three'), face_nodes),
            node_lon=(('nnodes',), mesh['node_lon']),
            node_lat=(('nnodes',), mesh['node_lat']),
        )
        # Attributes: connectivity (start_index kept), nodes, topology copied
        # onto a traceability scalar.
        ds['Mesh2_face_nodes'].attrs.update(mesh['face_nodes_attrs'])
        ds['node_lon'].attrs.update(
            {'standard_name': 'longitude', 'units': 'degrees_east',
             'long_name': 'Longitude of 2D mesh nodes'})
        ds['node_lat'].attrs.update(
            {'standard_name': 'latitude', 'units': 'degrees_north',
             'long_name': 'Latitude of 2D mesh nodes'})
        if mesh['mesh_attrs']:
            ds.attrs['mesh_topology'] = json.dumps(
                {k: (v.item() if hasattr(v, 'item') else v)
                 for k, v in mesh['mesh_attrs'].items()}
            )
        return ds, True

    for name, da in exports.items():
        path = root / f"{name}.nc"
        # A previous run's file is overwritten.
        ds = da.to_dataset(name=name)
        is_elements = 'element' in da.dims
        # Traceability attributes
        ds.attrs.update({
            'source': name,
            'correction_label': correction_label,
            'period_start': str(period_start),
            'period_end': str(period_end),
            'resolution': 'daily',
            'grid': ('elements' if is_elements else 'ibi_regular'),
        })
        if bbox is not None:
            ds.attrs.update({
                'bbox_lon_min': float(bbox[0]), 'bbox_lon_max': float(bbox[1]),
                'bbox_lat_min': float(bbox[2]), 'bbox_lat_max': float(bbox[3]),
                'bbox_mode': 'intersection of the exported sources',
            })

        # Embed the mesh ONLY in the element files. Mesh variables carry
        # their own dims (nelements/three/nnodes), distinct from
        # 'element'/'time', and are stored contiguous (small, static).
        mesh_written = False
        if is_elements and mesh is not None:
            ds, mesh_written = _attach_mesh(ds)

        # -- Chunking for efficient reading PER TIME STEP --
        # 1 chunk = 1 time step (time=1), FULL spatial dimensions in the
        # chunk: reading one time step = reading exactly one chunk. Only the
        # data variable is chunked and compressed.
        enc_chunks = tuple(1 if d == 'time' else da.sizes[d]
                           for d in da.dims)
        encoding = {name: {'chunksizes': enc_chunks, 'zlib': True,
                           'complevel': 4}}

        try:
            ds.to_netcdf(path, mode='w', format='NETCDF4',
                         encoding=encoding)
            logger.info(
                f"  ✓ {name:<16} {tuple(da.sizes.items())} "
                f"chunks={enc_chunks}"
                f"{' +mesh' if mesh_written else ''} → {path.name}")
        except Exception as e:
            logger.error(f"  ✗ {name}: NetCDF write failed ({e})")

    # -- Dedicated file: skill per element (test block), all methods --
    if skill_per_method:
        ref_elements = exports.get('model')
        if ref_elements is None or 'element' not in ref_elements.dims:
            logger.warning(
                "  [skill] element source missing → skill not exported."
            )
        else:
            n_el = ref_elements.sizes['element']
            data_vars, skill_enc = {}, {}
            for name, arr in skill_per_method.items():
                a = np.asarray(arr, dtype=np.float32)
                if elem_mask is not None and a.size == elem_mask.size:
                    a = a[np.flatnonzero(elem_mask)]
                if a.size != n_el:
                    logger.warning(
                        f"  [skill] '{name}': {a.size} values for "
                        f"{n_el} elements → ignored."
                    )
                    continue
                var = f"skill_{name}"
                data_vars[var] = (('element',), a)
                skill_enc[var] = {'zlib': True, 'complevel': 4}
            if not data_vars:
                logger.warning("  [skill] no usable map.")
            else:
                ds_sk = xr.Dataset(
                    data_vars,
                    coords={'lon': (('element',), ref_elements['lon'].values),
                            'lat': (('element',), ref_elements['lat'].values)},
                )
                for var in data_vars:
                    ds_sk[var].attrs.update({
                        'long_name': f"Skill score per element ({var[6:]})",
                        'description': ("1 - MSE(correction)/MSE(baseline "
                                        "TRAIN climatology); >0 = improvement"),
                        'units': '1',
                    })
                ds_sk.attrs.update({
                    'source': 'skill', 'grid': 'elements',
                    'period_start': str(period_start),
                    'period_end': str(period_end),
                    'resolution': 'static (aggregated over the test block)',
                    'n_methods': len(data_vars),
                })
                if bbox is not None:
                    ds_sk.attrs.update({
                        'bbox_lon_min': float(bbox[0]),
                        'bbox_lon_max': float(bbox[1]),
                        'bbox_lat_min': float(bbox[2]),
                        'bbox_lat_max': float(bbox[3]),
                    })
                ds_sk, mesh_sk = _attach_mesh(ds_sk)
                try:
                    ds_sk.to_netcdf(root / "skill.nc", mode='w',
                                    format='NETCDF4', encoding=skill_enc)
                    logger.info(
                        f"  ✓ {'skill':<16} {len(data_vars)} method(s) × "
                        f"{n_el} elements"
                        f"{' +mesh' if mesh_sk else ''} → skill.nc")
                except Exception as e:
                    logger.error(f"  ✗ skill: NetCDF write failed ({e})")

    logger.info(f"NETCDF EXPORT done: {len(exports)} files in {root}/")


def _build_port_export_frame(df_corrected_ports, var_ssh='surge'):
    """Long table, one row per (time step, port): `model` (1D Tolosa),
    `correction` (extracted at the port, interpolated to 10 min) and
    `model_corrected` = model + correction, over the steps where the
    correction is defined (test period). Returns None if empty."""
    parts = []
    for port, df in df_corrected_ports.items():
        if (df is None or len(df) == 0
                or 'bias_correction' not in df.columns):
            continue
        df = df.dropna(subset=[var_ssh, 'bias_correction', 'ssh_corrected'])
        if df.empty:
            continue
        parts.append(pd.DataFrame({
            'time': df.index.values,
            'port': port,
            'model': df[var_ssh].to_numpy(dtype=np.float32),
            'correction': df['bias_correction'].to_numpy(dtype=np.float32),
            'model_corrected': df['ssh_corrected'].to_numpy(dtype=np.float32),
        }))
    if not parts:
        return None
    out = pd.concat(parts, ignore_index=True)
    out['port'] = out['port'].astype('category')
    return out


def export_gauge_corrections_to_parquet(
    df_corrected_ports, *, var_ssh='surge', out_path=None, out_label='',
):
    """Write the model, the correction and the corrected model at the ports
    to one Parquet file (long format, see _build_port_export_frame; 'port'
    stored as a category). Default name:
    corrections_gauges_<out_label>_<first day>-<last day>.parquet.
    Needs pyarrow or fastparquet (error logged, not fatal, if missing).
    Returns the path written, or None.
    """
    df = _build_port_export_frame(df_corrected_ports, var_ssh=var_ssh)
    if df is None or df.empty:
        logger.warning("  [parquet] nothing to export — skipped")
        return None
    if out_path is None:
        t0 = pd.Timestamp(df['time'].min()).strftime('%Y%m%d')
        t1 = pd.Timestamp(df['time'].max()).strftime('%Y%m%d')
        out_path = f"corrections_gauges_{out_label}_{t0}-{t1}.parquet"
    out_path = str(out_path)
    logger.info(
        f"  [parquet] {len(df):,} rows — {df['port'].nunique()} ports, "
        f"{df['time'].min()} → {df['time'].max()} → {out_path}")
    try:
        df.to_parquet(out_path, index=False)
    except ImportError:
        logger.error(
            "  [parquet] no parquet engine (pyarrow/fastparquet) — "
            "export skipped (pip install pyarrow).")
        return None
    sz = Path(out_path).stat().st_size / 2 ** 20
    logger.info(f"  [parquet] written: {out_path} ({sz:,.1f} MB)")
    return out_path


def plot_corrections_per_timestep(
    pipeline_results: Dict[str, BiasFieldResult],
    bias_ref: xr.DataArray,
    var_mod: str,
    out_label: Optional[str] = None,
    dpi: int = 100,
    n_workers: Optional[int] = None,
) -> None:
    """One figure per day of `bias_ref` (test period): bias_ref + the
    daily correction(s) (layout of 4 columns × ⌈(N+1)/4⌉ rows).

    Common scale per figure (symmetric RdBu_r, q98). Plotted sub-domain =
    non-NaN area of bias_ref. Days are matched by date; a day missing
    from a correction is left out of that panel.
    """
    out_dir = _make_output_dir(out_label if out_label is not None else var_mod)
    if n_workers is None:
        n_workers = N_PLOT_WORKERS
    method_names = list(pipeline_results.keys())
    logger.info(
        f"[plot] corrections per time step: {len(method_names)} methods, "
        f"{bias_ref.sizes['time']} days → {out_dir}/"
    )

    # Daily means of the corrections (xarray.resample)
    corr_daily_das = {name: resample_daily(res.da)
                      for name, res in pipeline_results.items()}
    if not corr_daily_das:
        logger.warning("[plot] Empty pipeline, nothing to plot")
        return

    # IBI grid coordinates (from the first BiasFieldResult)
    first_result = next(iter(pipeline_results.values()))
    lon_grid = first_result.lon
    lat_grid = first_result.lat

    # "Actually corrected" sub-domain: non-NaN pixels of bias_ref
    ref_valid_mask = bias_ref.notnull().any(dim='time').values
    if not ref_valid_mask.any():
        logger.warning("[plot] bias_ref entirely NaN, nothing to plot")
        return
    rows = np.where(ref_valid_mask.any(axis=1))[0]
    cols = np.where(ref_valid_mask.any(axis=0))[0]
    i0, i1 = rows[0], rows[-1] + 1
    j0, j1 = cols[0], cols[-1] + 1
    lon_sub = lon_grid[j0:j1]
    lat_sub = lat_grid[i0:i1]

    bias_ref_D = _to_daily_time(bias_ref)

    tasks = []
    for d in bias_ref_D['time'].values:
        bref = bias_ref_D.sel(time=d).values
        bref_sub = bref[i0:i1, j0:j1]
        corrs_today_sub = {}
        for name, da_corr in corr_daily_das.items():
            try:
                arr = da_corr.sel(time=d).values
            except KeyError:
                continue
            corrs_today_sub[name] = arr[i0:i1, j0:j1]
        if not corrs_today_sub:
            continue
        all_fields = [bref_sub] + list(corrs_today_sub.values())
        vlim = _vlim_quantile(all_fields, quantile=0.98, symmetric=True)
        out_path = out_dir / f"corrections_{str(d)}.png"
        tasks.append((str(d), lon_sub, lat_sub, bref_sub, corrs_today_sub,
                      method_names, vlim, var_mod, out_path, dpi))

    n_ok, n_err = _run_parallel(_plot_corrections_one, tasks, n_workers,
                                "corrections")
    logger.info(
        f"[plot] corrections: {n_ok} figures written"
        + (f" ({n_err} FAILED)" if n_err else "")
    )


# ---- Figure daily IBI/Tolosa/sat/bias_ref -----------------------------------

def plot_daily_ibi_tolosa_biasref(
    ibi_da: xr.DataArray,
    interp_intermediates: Dict, bias_ref: xr.DataArray,
    test_start: np.datetime64, test_end: np.datetime64, var_mod: str,
    ds_sat_native: Optional[xr.Dataset] = None, sat_var_native: str = 'sla',
    n_days_each_end: int = 3, out_label: Optional[str] = None,
    dpi: int = 100,
) -> None:
    """Single figure, one row per day (n_days_each_end first +
    n_days_each_end last days of the test period).

    Columns: IBI / Tolosa projected onto IBI / [native sat, if
    ds_sat_native] / sat on IBI / bias_ref. Tolosa and sat on IBI are
    read from `interp_intermediates` ('model_daily_da', 'sat_on_out_da',
    see BiasComputation.compute_bias_2d_from_unstructured). Days missing
    from a source are skipped.
    """
    out_dir = _make_output_dir(out_label if out_label is not None else var_mod)
    out_path = out_dir / "daily_ibi_tolosa_biasref.png"
    logger.info(
        f"[plot] daily IBI/Tolosa/sat/bias_ref: "
        f"{n_days_each_end}+{n_days_each_end} days → {out_path}"
    )

    ts = np.datetime64(test_start, 'D')
    te = np.datetime64(test_end, 'D')
    day = np.timedelta64(1, 'D')
    head_days = [ts + i * day for i in range(n_days_each_end)]
    tail_days = [te - (n_days_each_end - 1 - i) * day
                 for i in range(n_days_each_end)]
    days_to_plot = head_days + tail_days

    # Tolosa PROJECTED onto the IBI grid (time, lat, lon), drawn with
    # pcolormesh like the other columns.
    model_post_daily_da = interp_intermediates['model_daily_da']
    sat_on_out_da = interp_intermediates['sat_on_out_da']

    # Daily IBI
    if hasattr(ibi_da, 'compute'):
        ibi_da_loaded = ibi_da.compute()
    else:
        ibi_da_loaded = ibi_da
    ibi_daily_da = _to_daily_time(resample_daily(ibi_da_loaded))

    # Native sat (if given)
    sat_native_lon = sat_native_lat = None
    sat_native_daily_da = None
    if ds_sat_native is not None and sat_var_native in ds_sat_native:
        sat_native_da = ds_sat_native[sat_var_native]
        sat_native_lon = (sat_native_da.lon.values if 'lon' in sat_native_da.coords
                         else sat_native_da.longitude.values)
        sat_native_lat = (sat_native_da.lat.values if 'lat' in sat_native_da.coords
                         else sat_native_da.latitude.values)
        if hasattr(sat_native_da, 'compute'):
            sat_native_da = sat_native_da.compute()
        sat_native_daily_da = _to_daily_time(resample_daily(sat_native_da))

    # Select the available days
    bias_ref_D = _to_daily_time(bias_ref)

    def _safe_sel(da: xr.DataArray, d: np.datetime64) -> Optional[np.ndarray]:
        if da is None:
            return None
        try:
            return da.sel(time=d).values
        except KeyError:
            return None

    ibi_samples, tol_post_samples = [], []
    sat_native_samples, sat_samples, bref_samples = [], [], []
    days_with_data = []

    for d in days_to_plot:
        ibi_f = _safe_sel(ibi_daily_da, d)
        tol_post_f = _safe_sel(model_post_daily_da, d)
        sat_f = _safe_sel(sat_on_out_da, d)
        bref_f = _safe_sel(bias_ref_D, d)
        if any(x is None for x in (ibi_f, tol_post_f, sat_f, bref_f)):
            logger.warning(
                f"  [plot daily] day {d}: missing data, row skipped"
            )
            continue
        sat_native_f = (_safe_sel(sat_native_daily_da, d)
                        if sat_native_daily_da is not None else None)

        ibi_samples.append(ibi_f)
        tol_post_samples.append(tol_post_f)
        sat_native_samples.append(sat_native_f)
        sat_samples.append(sat_f)
        bref_samples.append(bref_f)
        days_with_data.append(d)

    if not days_with_data:
        logger.warning("[plot daily] no row to plot")
        return

    # Common scales
    vlim_main = _vlim_quantile(
        ibi_samples + tol_post_samples
        + sat_native_samples + sat_samples,
        quantile=0.98, symmetric=True,
    )
    vlim_bref = _vlim_quantile(bref_samples, quantile=0.98, symmetric=True)

    # Columns
    cols = ['IBI', 'Tolosa (projected onto IBI)']
    if sat_native_daily_da is not None:
        cols.append('Native sat')
    cols += ['Sat (IBI)', 'bias_ref']
    n_cols = len(cols)
    n_rows = len(days_with_data)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    ibi_lon = ibi_da.lon.values
    ibi_lat = ibi_da.lat.values

    for r, d in enumerate(days_with_data):
        title_day = str(np.datetime64(d, 'D'))
        col = 0
        # IBI
        ax = axes[r, col]
        pm = ax.pcolormesh(ibi_lon, ibi_lat, ibi_samples[r],
                            shading='auto', cmap='RdBu_r',
                            vmin=vlim_main[0], vmax=vlim_main[1])
        ax.set_title(f"{cols[col]}\n{title_day} | "
                     f"μ={np.nanmean(ibi_samples[r]):+.3f}")
        plt.colorbar(pm, ax=ax, label="m")
        col += 1
        # Tolosa (projected onto IBI)
        ax = axes[r, col]
        pm = ax.pcolormesh(ibi_lon, ibi_lat, tol_post_samples[r],
                            shading='auto', cmap='RdBu_r',
                            vmin=vlim_main[0], vmax=vlim_main[1])
        ax.set_title(f"{cols[col]}\nμ={np.nanmean(tol_post_samples[r]):+.3f}")
        plt.colorbar(pm, ax=ax, label="m")
        col += 1
        # Native sat
        if sat_native_daily_da is not None:
            ax = axes[r, col]
            if sat_native_samples[r] is not None:
                pm = ax.pcolormesh(sat_native_lon, sat_native_lat,
                                    sat_native_samples[r],
                                    shading='auto', cmap='RdBu_r',
                                    vmin=vlim_main[0], vmax=vlim_main[1])
                ax.set_title(f"{cols[col]}\n"
                             f"μ={np.nanmean(sat_native_samples[r]):+.3f}")
                plt.colorbar(pm, ax=ax, label="m")
            else:
                ax.set_title(f"{cols[col]} — N/A")
            col += 1
        # Sat (IBI)
        ax = axes[r, col]
        pm = ax.pcolormesh(ibi_lon, ibi_lat, sat_samples[r],
                            shading='auto', cmap='RdBu_r',
                            vmin=vlim_main[0], vmax=vlim_main[1])
        ax.set_title(f"{cols[col]}\nμ={np.nanmean(sat_samples[r]):+.3f}")
        plt.colorbar(pm, ax=ax, label="m")
        col += 1
        # bias_ref
        ax = axes[r, col]
        pm = ax.pcolormesh(ibi_lon, ibi_lat, bref_samples[r],
                            shading='auto', cmap='RdBu_r',
                            vmin=vlim_bref[0], vmax=vlim_bref[1])
        ax.set_title(f"{cols[col]}\nμ={np.nanmean(bref_samples[r]):+.3f}")
        plt.colorbar(pm, ax=ax, label="m")

    for ax in axes.ravel():
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
    fig.suptitle(
        f"Daily IBI/Tolosa/sat/bias_ref ({var_mod}) | "
        f"{n_rows} days × {n_cols} columns", fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"[plot] daily: 1 figure ({n_rows}×{n_cols}) → {out_path}")


# =============================================================================
# SECTION 6: PIPELINE
# =============================================================================


class BiasCorrectionPipeline:
    """Pipeline of ONE spectral band: calibration + prediction + 2D validation.

    Usage:
        pipeline = BiasCorrectionPipeline(band=(20.0, 128.0), dt_hours=24.0)
        pipeline.set_data(ibi_train, bias_2d_train)
        pipeline.fit_all()
        results = pipeline.predict_all(ibi_future, dt_hours=24.0)
        vn = pipeline.validate_all_correction_at_elements(bias_ref_elements, W)
    """

    def __init__(self, band: Tuple[float, float], dt_hours: float = 1.0):
        """
        Parameters
        ----------
        band : (low_days, high_days) — calibration band-pass.
        dt_hours : time step of the fit.

        The results are keyed by method name ('eof', the EOF×EOF CCA
        regression of section 3).
        """
        self.dt_hours = dt_hours
        self.band = band
        self.methods = {'eof': EOFRegression(band=band)}

        self.ibi_train: Optional[xr.DataArray] = None
        self.bias_2d: Optional[np.ndarray] = None
        self.results: Dict[str, BiasFieldResult] = {}
        self.validator = Validator()

    def set_data(self, ibi_train: xr.DataArray, bias_2d: np.ndarray):
        """Training data: daily IBI (time, lat, lon) and the reference bias
        (time, ny, nx) on the same days."""
        self.ibi_train = ibi_train
        self.bias_2d = bias_2d
        logger.info(
            f"Data loaded: IBI {ibi_train.shape}, "
            f"bias {bias_2d.shape}"
        )

    def fit_all(self):
        """Calibrate the method(s) on the data given to set_data."""
        logger.info("=" * 60)
        logger.info("CALIBRATION")
        logger.info("=" * 60)
        for name, method in self.methods.items():
            t0 = time.perf_counter()
            method.fit(self.ibi_train, self.bias_2d, dt_hours=self.dt_hours)
            logger.info(f"  {name} fitted in {time.perf_counter() - t0:.1f} s")

    def predict_all(self, ibi_future: xr.DataArray,
                    dt_hours: Optional[float] = None
                    ) -> Dict[str, BiasFieldResult]:
        """Predict the correction from `ibi_future` with each fitted method.

        dt_hours: time step of `ibi_future` (default: that of the fit).
        Returns (and stores in self.results) {method name: BiasFieldResult}.
        """
        dt = dt_hours if dt_hours is not None else self.dt_hours
        logger.info("=" * 60)
        logger.info(f"BIAS PREDICTION (dt_hours = {dt:.2f})")
        logger.info("=" * 60)
        self.results = {}
        for name, method in self.methods.items():
            if not method.fitted:
                logger.warning(f"{name} not calibrated, skipped.")
                continue
            t0 = time.perf_counter()
            self.results[name] = method.predict(ibi_future, dt_hours=dt)
            logger.info(f"  {name}: OK ({time.perf_counter() - t0:.1f} s)")
        return self.results

    # ---- Persistence ----------------------------------------------------

    def save_models(self, output_dir: str, train_period: Tuple,
                    ibi_lon: np.ndarray, ibi_lat: np.ndarray,
                    var_model: str = 'surge',
                    var_satellite: str = 'sla',
                    extra_metadata: Optional[Dict] = None) -> Path:
        """Save the calibrated models (metadata.json + one joblib per method).

        Output layout:
            {output_dir}/
              metadata.json
              ibi_grid.npz
              method_<name>.joblib
        """
        try:
            import joblib
        except ImportError:
            raise RuntimeError("joblib not available. pip install joblib.")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        logger.info("=" * 70)
        logger.info(f"SAVING MODELS → {out}")
        logger.info("=" * 70)

        fitted_methods = {n: m for n, m in self.methods.items()
                          if getattr(m, 'fitted', False)}
        if not fitted_methods:
            logger.warning("  No fitted method — saving cancelled.")
            return out
        logger.info(
            f"  {len(fitted_methods)}/{len(self.methods)} fitted methods: "
            f"{list(fitted_methods.keys())}"
        )

        train_start_str = np.datetime_as_string(
            np.datetime64(train_period[0]), unit='D'
        )
        train_end_str = np.datetime_as_string(
            np.datetime64(train_period[1]), unit='D'
        )
        metadata = {
            "created_at": dt.datetime.now().isoformat(),
            "train_period": {"start": train_start_str, "end": train_end_str},
            "var_model": var_model,
            "var_satellite": var_satellite,
            "ibi_grid": {
                "shape": [int(len(ibi_lat)), int(len(ibi_lon))],
                "lon_range": [float(ibi_lon.min()), float(ibi_lon.max())],
                "lat_range": [float(ibi_lat.min()), float(ibi_lat.max())],
            },
            "hyperparameters": {
                "band": [float(self.band[0]), float(self.band[1])],
                "dt_hours": float(self.dt_hours),
            },
            "methods": list(fitted_methods.keys()),
        }
        if extra_metadata:
            metadata["extra"] = extra_metadata
        with open(out / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info("  ✓ metadata.json")

        np.savez_compressed(out / "ibi_grid.npz", lon=ibi_lon, lat=ibi_lat)
        logger.info("  ✓ ibi_grid.npz")

        for name, fitted_method in fitted_methods.items():
            fname = out / f"method_{name}.joblib"
            try:
                joblib.dump(fitted_method, fname, compress=3)
                size_mb = fname.stat().st_size / (1024 ** 2)
                logger.info(f"  ✓ method_{name}.joblib ({size_mb:.1f} MB)")
            except Exception as e:
                logger.error(f"  ✗ method_{name}.joblib: {e}")
        logger.info(f"  → {len(fitted_methods)} methods saved")
        return out

    @classmethod
    def load_models(cls, input_dir: str) -> 'BiasCorrectionPipeline':
        """Reload a single-band pipeline from a save_models() folder.

        Counterpart of save_models (not called by main): reads
        metadata.json (band, dt_hours, methods) then replaces each fresh
        instance of self.methods by the fitted object deserialised from
        method_<name>.joblib. The returned pipeline is ready for
        predict_all / validate_* (set_data/fit_all not needed).
        """
        try:
            import joblib
        except ImportError:
            raise RuntimeError("joblib not available. pip install joblib.")

        in_dir = Path(input_dir)
        meta_path = in_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.json missing from {in_dir}; this is not a "
                f"folder produced by save_models()."
            )
        with open(meta_path) as f:
            metadata = json.load(f)

        hyper = metadata.get("hyperparameters", {})
        band = hyper.get("band")
        if band is None:
            raise ValueError(
                f"{meta_path}: hyperparameters.band missing or null — "
                f"models cannot be reloaded. Recalibrate with --band."
            )
        band = (float(band[0]), float(band[1]))
        dt_hours = float(hyper.get("dt_hours", 24.0))
        logger.info(
            f"LOADING MODELS ← {in_dir} "
            f"(band {band[0]:g}-{band[1]:g} d, dt={dt_hours:g} h)"
        )
        pipeline = cls(band=band, dt_hours=dt_hours)
        saved = list(metadata.get("methods", []))
        method_names = [n for n in saved if n in pipeline.methods]
        ignored = [n for n in saved if n not in pipeline.methods]
        if ignored:
            logger.warning(f"  methods no longer available, ignored: {ignored}")

        n_loaded = 0
        for name in method_names:
            fname = in_dir / f"method_{name}.joblib"
            if not fname.exists():
                logger.warning(
                    f"  ✗ method_{name}.joblib missing — method skipped"
                )
                continue
            try:
                obj = joblib.load(fname)
            except Exception as e:
                logger.error(f"  ✗ method_{name}.joblib unreadable: {e}")
                continue
            if not getattr(obj, "fitted", False):
                logger.warning(
                    f"  ⚠ method_{name}.joblib: object not fitted — skipped"
                )
                continue
            pipeline.methods[name] = obj
            n_loaded += 1
            logger.info(f"  ✓ method_{name}.joblib")
        if n_loaded == 0:
            raise RuntimeError(
                f"No method reloaded from {in_dir} "
                f"(metadata.methods = {method_names})."
            )
        logger.info(
            f"  → {n_loaded}/{len(method_names)} methods reloaded"
        )
        return pipeline

    # ---- Validation 2D ------------------------------------------------

    def validate_all_correction_at_elements(
        self, bias_ref_at_elements: xr.DataArray,
        W_ibi_to_tolosa: sparse.spmatrix,
        model_at_elements_da: Optional[xr.DataArray] = None,
        baseline_map: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict]:
        """Validation at the elements of the unstructured barotropic mesh.

        If `model_at_elements_da` (daily Tolosa at the elements) is given,
        additional obs-space scores are computed (see
        `validate_correction_at_elements`).
        """
        logger.info("=" * 60)
        logger.info("VALIDATION AT THE ELEMENTS")
        if baseline_map is not None:
            logger.info("  Skill baseline = TRAIN climatology (given)")
        logger.info("=" * 60)
        validations = {
            name: self.validator.validate_correction_at_elements(
                result, bias_ref_at_elements, W_ibi_to_tolosa=W_ibi_to_tolosa,
                model_at_elements_da=model_at_elements_da,
                baseline_map=baseline_map,
            )
            for name, result in self.results.items()
        }
        self._log_validation_summary(validations, scope="ELEMENTS")
        return validations

    @staticmethod
    def _log_validation_summary(validations: Dict[str, Dict],
                                scope: str = "2D") -> None:
        if not validations:
            return
        logger.info("")
        logger.info("=" * 78)
        logger.info(f"VALIDATION SUMMARY {scope}")
        logger.info("=" * 78)
        header = (f"{'method':<22s} {'RMSE':>8s} {'MAE':>8s} {'Corr':>6s} "
                  f"{'EV':>6s} {'Skill':>7s} {'n':>5s}")
        logger.info(header)
        logger.info("-" * 78)
        for name, val in validations.items():
            s = val.get("summary", {})
            n_d = s.get("n_days", 0)
            logger.info(
                f"{name:<22s} "
                f"{s.get('mean_rmse', np.nan):>8.4f} "
                f"{s.get('mean_mae', np.nan):>8.4f} "
                f"{s.get('mean_correlation', np.nan):>6.3f} "
                f"{s.get('mean_explained_variance', np.nan):>6.3f} "
                f"{s.get('mean_skill_score', np.nan):>+7.3f}"
                f" {int(n_d):>5d}"
            )
        logger.info("=" * 78)


# =============================================================================
# SECTION 7: DATA LOADING (1D ports, DAC, DataLoader)
# =============================================================================


def discover_port_csvs(run_dir, port_to_filename, *, label=''):
    """Resolve {port: [csv paths]} in a Tolosa 1D run folder.

    Two layouts, which can be combined:
      - flat    : <run_dir>/<Station>.csv
      - periods : <run_dir>/<period>/<Station>.csv      (1 csv/station/year)
    Flat files come first, then the sub-folders in lexicographic order
    (dated names → chronological); on time overlaps the merge keeps the
    LAST occurrence (see _concat_port_frames), so the latest period wins.
    Ports without any file are left out (one summary warning).
    """
    base = Path(run_dir)
    subdirs = sorted(d for d in base.glob('*') if d.is_dir()) \
        if base.is_dir() else []
    out, missing = {}, []
    for port, fname in port_to_filename.items():
        paths = []
        flat = base / fname
        if flat.is_file():
            paths.append(str(flat))
        paths.extend(str(d / fname) for d in subdirs
                     if (d / fname).is_file())
        if paths:
            out[port] = paths
        else:
            missing.append(port)
    n_multi = sum(1 for v in out.values() if len(v) > 1)
    logger.info(
        f"  [1D {label}] {base}: {len(out)}/{len(port_to_filename)} "
        f"ports, {len(subdirs)} period folder(s)"
        + (f" ({', '.join(d.name for d in subdirs[:6])}"
           + ('…' if len(subdirs) > 6 else '') + ")" if subdirs else "")
        + (f", {n_multi} multi-file port(s)" if n_multi else "")
    )
    if missing:
        logger.warning(f"  [1D {label}] ports without a file: {missing}")
    return out


def _concat_port_frames(frames, port='', label='', gap_warn_days=7.0):
    """Merge the yearly DataFrames of a port: concat → stable time sort →
    deduplication of the index (keep='last': the last source wins on
    overlaps) → warning if an internal gap exceeds gap_warn_days (missing
    period folder?)."""
    if len(frames) == 1:
        return frames[0]
    df = pd.concat(frames)
    df = df.sort_index(kind='stable')
    dup = df.index.duplicated(keep='last')
    n_dup = int(dup.sum())
    if n_dup:
        df = df[~dup]
        logger.info(
            f"    {port:30s}: {n_dup} overlapping steps deduplicated "
            f"(latest period wins)")
    if len(df) > 1:
        gaps = np.diff(df.index.values).astype('timedelta64[s]').astype(float)
        gmax = float(gaps.max()) / 86400.0
        if gmax > gap_warn_days:
            i = int(np.argmax(gaps))
            logger.warning(
                f"    {port:30s}: gap of {gmax:.1f} d in the {label} "
                f"model after {df.index[i]} — missing period "
                f"folder?")
    return df


# =============================================================================
# DAC: adding the atmospheric correction back into the L4 SLA
# =============================================================================
#
# The DUACS SLA (L3 and L4) is distributed with the DAC removed: the
# barotropic response to the atmosphere (MOG2D below 20 d, inverse
# barometer beyond) is absent from it, whereas the Tolosa surge contains
# it. bias_ref = SLA − surge then carries the surge with the opposite
# sign. SLA + DAC ("DAC-in") restores a common atmospheric content. Below
# 20 d the DAC is the MOG2D model: the reference then compares Tolosa with
# another model; beyond, it is the inverse barometer, a static correction.

_DAC_EPOCH = np.datetime64('1950-01-01T00:00', 'ns')
# AVISO name: dac_dif_<CNES day since 1950-01-01>_<HH>.nc
_DAC_NAME_RE = re.compile(r'_(\d{5})_(\d{2})\.nc$')


def _resolve_dac_paths(path: str) -> List[str]:
    """dac_dir (config) → list of files: a file, a glob, or a directory
    walked recursively (yearly AVISO tree)."""
    if any(ch in path for ch in '*?['):
        paths = sorted(glob.glob(path, recursive=True))
    elif os.path.isdir(path):
        paths = sorted(glob.glob(os.path.join(path, '**', '*.nc'),
                                 recursive=True))
    else:
        paths = [path]
    if not paths:
        raise FileNotFoundError(f"dac_dir: no file matches '{path}'")
    return paths


def _dac_file_time(path: str, ds: Optional[xr.Dataset] = None
                   ) -> Optional[np.datetime64]:
    """Time of a DAC field: from the AVISO name, else the single 'time' coordinate."""
    m = _DAC_NAME_RE.search(os.path.basename(path))
    if m:
        return (_DAC_EPOCH + np.timedelta64(int(m.group(1)), 'D')
                + np.timedelta64(int(m.group(2)), 'h'))
    if ds is not None and 'time' in ds.variables and ds['time'].size == 1:
        return np.datetime64(ds['time'].values.ravel()[0], 'ns')
    return None


def _dac_geometry(ds: xr.Dataset, var_dac: str,
                  bbox: Tuple[float, float, float, float]) -> Dict:
    """Cropping indices on `bbox` (lon_min, lon_max, lat_min, lat_max),
    longitudes brought back to [-180, 180] and sorted, unit factor to m."""
    if var_dac not in ds.variables:
        raise ValueError(f"DAC: variable '{var_dac}' missing (variables: "
                         f"{list(ds.data_vars)}); see dac_var in the config")
    la, lo = _detect_latlon_names(ds, ref_var=var_dac)
    lat = np.asarray(ds[la].values, dtype=np.float64)
    lon = np.asarray(ds[lo].values, dtype=np.float64)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("DAC: 1D lon/lat grid expected")
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    ix = np.where((lon180 >= bbox[0]) & (lon180 <= bbox[1]))[0]
    ix = ix[np.argsort(lon180[ix])]
    iy = np.where((lat >= bbox[2]) & (lat <= bbox[3]))[0]
    iy = iy[np.argsort(lat[iy])]
    if ix.size < 2 or iy.size < 2:
        raise ValueError(f"DAC: extent {bbox} outside the DAC grid")
    units = str(ds[var_dac].attrs.get('units', 'm')).strip().lower()
    return dict(lat_dim=ds[la].dims[0], lon_dim=ds[lo].dims[0],
                shape=(lat.size, lon.size), iy=iy, ix=ix,
                lat=lat[iy], lon=lon180[ix],
                scale={'mm': 1e-3, 'cm': 1e-2}.get(units, 1.0))


def _daily_mean_subdaily(times: np.ndarray, fields: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Calendar-day mean of sub-daily fields (same convention as
    resample_daily); incomplete days dropped. Returns (days, fields,
    number of dropped days)."""
    times = np.asarray(times, dtype='datetime64[ns]')
    times, first = np.unique(times, return_index=True)   # duplicates removed
    fields = fields[first]
    days = times.astype('datetime64[D]')
    uniq, start, count = np.unique(days, return_index=True, return_counts=True)
    full = count >= count.max()
    out = np.stack([fields[s:s + c].mean(axis=0)
                    for s, c, f in zip(start, count, full) if f])
    return uniq[full], out, int((~full).sum())


def _linear_interp_matrix(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """(len(dst), len(src)) matrix of 1D linear interpolation, `src`
    increasing; NaN rows outside [src[0], src[-1]]."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    j = np.clip(np.searchsorted(src, dst) - 1, 0, src.size - 2)
    w = (dst - src[j]) / (src[j + 1] - src[j])
    M = np.zeros((dst.size, src.size))
    r = np.arange(dst.size)
    M[r, j] = 1.0 - w
    M[r, j + 1] = w
    M[(dst < src[0]) | (dst > src[-1])] = np.nan
    return M


def load_dac_daily(path: str, var_dac: str,
                   bbox: Tuple[float, float, float, float],
                   time_bounds: Tuple[np.datetime64, np.datetime64],
                   cache_dir: Optional[Path] = None) -> xr.DataArray:
    """Daily DAC (mean of the 4 six-hourly fields of the calendar day), on
    the native DAC grid cropped to `bbox`, days in
    [time_bounds[0], time_bounds[1][. Cached."""
    t0, t1 = (np.datetime64(t, 'ns') for t in time_bounds)
    key = (f"dac_daily_{var_dac}"
           f"_{bbox[0]:.0f}_{bbox[1]:.0f}_{bbox[2]:.0f}_{bbox[3]:.0f}"
           f"_{np.datetime_as_string(t0, 'D')}_{np.datetime_as_string(t1, 'D')}"
           ).replace('-', '')
    cached = _cache_load(cache_dir, key)
    if cached is not None:
        logger.info(f"  Daily DAC: read from cache ({key})")
        return cached

    paths = _resolve_dac_paths(path)
    logger.info(f"  DAC: {len(paths)} candidate file(s)")
    geom, times, fields = None, [], []
    for p in paths:
        t = _dac_file_time(p)
        if t is not None and not (t0 <= t < t1):
            continue
        with xr.open_dataset(p) as ds:
            if t is None:
                t = _dac_file_time(p, ds)
                if t is None:
                    raise ValueError(f"DAC: date not found for {p}")
                if not (t0 <= t < t1):
                    continue
            if geom is None:
                geom = _dac_geometry(ds, var_dac, bbox)
            da = ds[var_dac].squeeze(drop=True).transpose(
                geom['lat_dim'], geom['lon_dim'])
            v = np.asarray(da.values, dtype=np.float32)
        if v.shape != geom['shape']:
            raise ValueError(f"DAC: grid of {p} differs from the first "
                             f"file ({v.shape} vs {geom['shape']})")
        times.append(t)
        fields.append(v[np.ix_(geom['iy'], geom['ix'])] * geom['scale'])
        if len(times) % 500 == 0:
            logger.info(f"  DAC: {len(times)} fields read")
    if not times:
        raise ValueError(f"dac_dir: no DAC field between {t0} and {t1}")

    days, daily, n_drop = _daily_mean_subdaily(np.array(times),
                                               np.stack(fields))
    logger.info(f"  DAC: {len(times)} fields → {days.size} days"
                + (f" ({n_drop} incomplete days dropped)" if n_drop else ""))
    dac = xr.DataArray(
        daily.astype(np.float32), dims=('time', 'lat', 'lon'),
        coords={'time': days.astype('datetime64[ns]'),
                'lat': geom['lat'], 'lon': geom['lon']},
        name='dac', attrs={'units': 'm'},
    )
    _cache_save(cache_dir, key, dac, source_paths=[path])
    return dac


def _dac_on_grid(dac: xr.DataArray, lon_t: np.ndarray, lat_t: np.ndarray,
                 select: Optional[np.ndarray] = None) -> np.ndarray:
    """Daily DAC bilinearly interpolated onto a regular grid (separable
    interpolation, both grids being regular), NaN outside the DAC extent.
    `select`: indices of the dates to process, all by default.

    Land NaNs of the DAC are first filled by diffusion, so that the
    coastal pixels of the target grid get a value.
    """
    Ay = _linear_interp_matrix(dac['lat'].values, lat_t)
    Ax = _linear_interp_matrix(dac['lon'].values, lon_t)
    rows, cols = np.isfinite(Ay[:, 0]), np.isfinite(Ax[:, 0])
    Ay, Ax, sub = Ay[rows], Ax[cols], np.ix_(rows, cols)
    idx = np.arange(dac.sizes['time']) if select is None else np.asarray(select)
    out = np.full((idx.size, lat_t.size, lon_t.size), np.nan, dtype=np.float32)
    vals = dac.values
    for k0 in range(0, idx.size, 64):
        fb = extrapolate_nan_stack(vals[idx[k0:k0 + 64]].astype(np.float64))
        for j in range(fb.shape[0]):
            # copy: same input array (allocated alone) as the slice-by-slice
            # version, for an identical matrix product.
            out[k0 + j][sub] = Ay @ fb[j].copy() @ Ax.T
    logger.info(f"  DAC → target grid: {int(rows.sum())}×{int(cols.sum())} "
                f"pixels within the extent out of {lat_t.size}×{lon_t.size}")
    return out


def subtract_dac_from_ibi(da: xr.DataArray, dac_ibi: xr.DataArray,
                          label: str = 'IBI') -> xr.DataArray:
    """Detided IBI → IBI − DAC: predictor free of the barotropic
    atmospheric response (inverse barometer included).

    Each time step gets the DAC of its calendar day; time steps without
    DAC are removed. The subtraction is done block-wise, the memory not
    exceeding one copy of the field.
    """
    days = pd.DatetimeIndex(da['time'].values).normalize()
    dac_days = pd.DatetimeIndex(dac_ibi['time'].values)
    idx = dac_days.get_indexer(days)
    if (idx < 0).any():
        n = int((idx < 0).sum())
        logger.warning(f"  {label} − DAC: {n} time steps without DAC → removed")
        keep = np.where(idx >= 0)[0]
        da, idx = da.isel(time=keep), idx[keep]
    dims = da.dims
    v = np.asarray(da.transpose('time', dims[-2], dims[-1]).values,
                   dtype=np.float32)
    dac_v = dac_ibi.values
    for k0 in range(0, v.shape[0], 64):
        sl = slice(k0, min(k0 + 64, v.shape[0]))
        v[sl] -= dac_v[idx[sl]]
    out = xr.DataArray(v, dims=('time', dims[-2], dims[-1]),
                       coords={'time': da['time'].values,
                               dims[-2]: da[dims[-2]].values,
                               dims[-1]: da[dims[-1]].values},
                       name=da.name,
                       attrs={**da.attrs, 'dac': 'DAC removed (IBI − DAC)'})
    logger.info(f"  {label} − DAC: {out.sizes['time']} time steps, "
                f"removed std {float(np.nanstd(dac_v)):.3f} m")
    return out


def add_dac_to_sla(sat: xr.DataArray, dac_path: str,
                   bbox: Tuple[float, float, float, float],
                   time_bounds: Tuple[np.datetime64, np.datetime64],
                   var_dac: str = 'dac',
                   cache_dir: Optional[Path] = None) -> xr.DataArray:
    """L4 SLA (DAC-out) → SLA + DAC (DAC-in), on the sat grid.

    The daily DAC of day D (calendar day, like the daily Tolosa) is added
    to the L4 map of day D: an approximation, since DUACS removes the
    instantaneous DAC from each measurement before mapping. L4 maps are
    centred on midnight, but the calendar-day mean is used to match the
    window of the daily Tolosa with which bias_ref is formed; the 12 h
    offset only affects the non-atmospheric, slow part of the SLA.
    Separable bilinear interpolation (regular grids), after filling the
    land NaNs of the DAC by diffusion for the coastal sat pixels.
    Outside `bbox` (Tolosa ∪ IBI extent + margin) the SLA is set to NaN:
    these pixels are not used by any projection. Days of `sat` without a
    complete DAC are removed.
    """
    logger.info("=" * 70)
    logger.info("DAC : SLA L4 (DAC-out) → SLA + DAC (DAC-in)")
    logger.info("=" * 70)
    dac = load_dac_daily(dac_path, var_dac, bbox, time_bounds, cache_dir)

    sat_days = pd.DatetimeIndex(sat['time'].values).normalize()
    dac_days = pd.DatetimeIndex(dac['time'].values)
    keep = np.where(sat_days.isin(dac_days))[0]
    in_window = ((sat_days >= pd.Timestamp(time_bounds[0]))
                 & (sat_days < pd.Timestamp(time_bounds[1])))
    n_missing = int(in_window.sum()) - keep.size
    if keep.size == 0:
        raise ValueError("DAC: no day in common with the SLA")
    if n_missing > 0:
        logger.warning(f"  DAC: {n_missing} SLA day(s) of the window "
                       f"without a complete DAC → removed")
    sat = sat.isel(time=keep)
    idx = dac_days.get_indexer(sat_days[keep])

    lat_s = sat['lat'].values
    lon_s = sat['lon'].values
    lat_s = lat_s if lat_s.ndim == 1 else lat_s[:, 0]
    lon_s = lon_s if lon_s.ndim == 1 else lon_s[0, :]
    out = _dac_on_grid(dac, lon_s, lat_s, select=idx)

    lat_dim, lon_dim, time_dim = sat['lat'].dims[0], sat['lon'].dims[-1], 'time'
    dac_on_sat = xr.DataArray(
        out, dims=(time_dim, lat_dim, lon_dim),
        coords={time_dim: sat[time_dim].values},
    ).transpose(*sat.dims)
    res = sat + dac_on_sat
    res.name = sat.name
    res.attrs = {**sat.attrs,
                 'dac': f"SLA + DAC (DAC-in), daily DAC from {dac_path}"}
    logger.info(
        f"  SLA + DAC: {keep.size} days; mean DAC "
        f"{float(np.nanmean(out)):+.3f} m, std "
        f"{float(np.nanstd(out)):.3f} m"
    )
    return res


class DataLoader:
    """Unified data loading for the pipeline.

    Handles:
      - 2D: sat NetCDF (DUACS SLA) + model NetCDF (TOLOSA surge).
      - 1D: barotropic CSV extractions at the ports (Tolosa full / ms).
      - Correction at the ports: extraction from a BiasFieldResult (coastal
        extrapolation + disk mean), then interpolation onto the 1D model
        time steps (10 min).
    """

    def __init__(self, use_dask: bool = True, chunk_size: int = 10):
        self.use_dask = use_dask
        self.chunk_size = chunk_size

    # ---- 2D -----------------------------------------------------------

    def load_data_2d(
        self,
        model_file_2d: str,
        model_ms_file_2d: str,
        sat_file_2d: str,
        var_model: str = 'surge',
        var_satellite: str = 'sla',
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """Load the model + satellite NetCDF files.

        If var_model='surge', ms is subtracted from full to produce the
        surge.
        """
        logger.info("=" * 70)
        logger.info("LOADING 2D DATA")
        logger.info("=" * 70)

        chunks_config = {'time': self.chunk_size} if self.use_dask else None
        ds_model = xr.open_dataset(model_file_2d, chunks=chunks_config)
        if var_model == 'surge':
            ds_ms_model = xr.open_dataset(
                model_ms_file_2d, chunks=chunks_config
            )
            ds_model['surge'] = ds_model['ssh'] - ds_ms_model['ssh']
        ds_satellite = xr.open_dataset(
            sat_file_2d, chunks=chunks_config
        )

        logger.info(f"  Model     : {dict(ds_model.dims)}")
        logger.info(f"  Satellite : {dict(ds_satellite.dims)}")

        return ds_model[[var_model]], ds_satellite[[var_satellite]]

    # ---- 1D -----------------------------------------------------------

    def load_model_1d(
        self,
        model_port_files: Dict[str, List[str]],
        model_ms_port_files: Dict[str, List[str]],
        dt_minutes: int = 10,
        var_model: str = 'surge',
    ) -> Dict[str, pd.DataFrame]:
        """Load the 1D barotropic model at the ports (Tolosa full and ms
        CSV extractions), interpolated onto a regular grid of step
        `dt_minutes`. In 'surge' mode, surge = full − ms.
        Returns {port: DataFrame}, column `var_model`.
        """
        logger.info("=" * 70)
        logger.info("LOADING 1D MODEL AT THE PORTS")
        logger.info("=" * 70)

        # Barotropic model (full and ms).
        # `files`: {port: path} OR {port: [paths]} (period layout, 1
        # csv/station/year — see discover_port_csvs): each file is read with
        # the same parsing, merged via _concat_port_frames, then interpolated
        # to dt_minutes on the MERGED series.
        def _read_one_barotropic_csv(file_path):
            df = pd.read_csv(
                file_path, header=0,
                names=['modeltime', 'time', 'ssh', 'u', 'v', 'bathy'],
            )
            df['time'] = [
                dt.datetime(1950, 1, 1) + dt.timedelta(days=float(x))
                for x in df['time']
            ]
            return df.set_index('time')

        def _read_barotropic_csvs(files, label: str
                                 ) -> Dict[str, pd.DataFrame]:
            out = {}
            for port, file_list in files.items():
                paths = ([file_list] if isinstance(file_list, str)
                         else list(file_list))
                try:
                    frames = [_read_one_barotropic_csv(f) for f in paths]
                    df = _concat_port_frames(
                        frames, port=port, label=label)
                    out[port] = time_interp(df[['ssh']], dt_minutes)
                    logger.info(
                        f"  Model {label} {port:30s}: {len(df)} pts"
                        + (f" ({len(paths)} files)"
                           if len(paths) > 1 else "")
                        + f" (→ {dt_minutes} min)"
                    )
                except Exception as e:
                    logger.error(
                        f"  Model error {label} port {port}: {e}"
                    )
            return out

        df_model_ports = _read_barotropic_csvs(model_port_files, "full")
        df_model_ms_ports = _read_barotropic_csvs(model_ms_port_files, "ms")

        # Surge = full - ms
        if var_model == 'surge':
            df_model_ports = {
                k: df_model_ports[k] - df_model_ms_ports[k]
                for k in df_model_ports if k in df_model_ms_ports
            }

        df_model_ports = {k: v.rename({'ssh': var_model}, axis=1)
                           for k, v in df_model_ports.items()}
        return df_model_ports

    # ---- Extraction of the correction at the ports ---------------------

    @staticmethod
    def extract_correction_at_ports(
        bias_result: BiasFieldResult,
        coords_ports: Dict[str, Tuple[float, float]],
        max_iter_extrap: int = 50,
        disk_radius_km: float = 20.0,
        disk_weighting: str = 'gaussian',
    ) -> Dict[str, pd.DataFrame]:
        """Extract the 1D correction at the ports from a BiasFieldResult.

        Pipeline: coastal extrapolation (diffusion) → disk mean around
        each port (haversine). The disk mean smooths extraction artefacts
        and represents the oceanographic neighbourhood of the port;
        fallback on the closest (extrapolated) pixel if no pixel falls in
        the disk.

        Parameters
        ----------
        disk_radius_km : float
            Disk radius in km. Typical values:
              - 10-15 km: narrow local capture, ports in bays.
              - 20-30 km: moderate neighbourhood, open coastal ports.
              - 50+ km: large scale, offshore ports.
        """
        logger.info(
            f"2D → 1D correction extraction at the ports "
            f"({bias_result.method_name}, disk {disk_radius_km:g} km, "
            f"weighting={disk_weighting})"
        )

        lon_1d = (bias_result.lon if bias_result.lon.ndim == 1
                  else bias_result.lon[0, :])
        lat_1d = (bias_result.lat if bias_result.lat.ndim == 1
                  else bias_result.lat[:, 0])

        port_names = list(coords_ports.keys())
        lat_pts = np.array([coords_ports[p][0] for p in port_names])
        lon_pts = np.array([coords_ports[p][1] for p in port_names])

        # Coastal extrapolation (coefficients are NaN over land)
        bias_2d = bias_result.bias_2d
        nt = bias_2d.shape[0]
        n_nan_before = int(np.isnan(bias_2d).sum())
        if n_nan_before > 0:
            logger.info(
                f"  Coastal extrapolation: {n_nan_before} NaNs "
                f"({n_nan_before / bias_2d.size * 100:.1f}%)"
            )
            bias_extrap = extrapolate_nan_stack(bias_2d,
                                                max_iter=max_iter_extrap)
            n_nan_after = int(np.isnan(bias_extrap).sum())
            logger.info(
                f"  → {n_nan_after} NaNs remaining "
                f"(filled: {n_nan_before - n_nan_after})"
            )
        else:
            bias_extrap = bias_2d

        # Disk mask of each port: haversine distance of each grid pixel to the
        # port.
        R_earth_km = 6371.0
        lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)
        values_pts = np.empty((nt, len(port_names)), dtype=np.float64)
        n_in_disk_log = []
        for i, port in enumerate(port_names):
            lat_p = lat_pts[i]
            lon_p = lon_pts[i]
            # Haversine distance between (lat_grid, lon_grid) and (lat_p, lon_p)
            dlat = np.deg2rad(lat_grid - lat_p)
            dlon = np.deg2rad(lon_grid - lon_p)
            a = (np.sin(dlat / 2.0) ** 2
                 + np.cos(np.deg2rad(lat_p))
                 * np.cos(np.deg2rad(lat_grid))
                 * np.sin(dlon / 2.0) ** 2)
            dist_km = 2 * R_earth_km * np.arcsin(np.sqrt(a))
            mask_disk = dist_km <= disk_radius_km
            n_in_disk = int(mask_disk.sum())
            n_in_disk_log.append(n_in_disk)
            if n_in_disk == 0:
                # No pixel in the disk → fallback on the (extrapolated) grid pixel
                # closest to the port.
                j_nn, i_nn = np.unravel_index(
                    int(np.argmin(dist_km)), dist_km.shape
                )
                values_pts[:, i] = bias_extrap[:, j_nn, i_nn]
                logger.warning(
                    f"  {port:30s}: no pixel in the "
                    f"{disk_radius_km} km disk → fallback on the closest "
                    f"pixel ({dist_km[j_nn, i_nn]:.1f} km)"
                )
            else:
                # Weighted spatial mean at each time step (NaNs excluded).
                # Weighting:
                #   'uniform'  : plain mean — a pixel at 19 km weighs as much as a
                #                pixel at 2 km.
                #   'gaussian' : w = exp(-d²/2σ²), σ = R/2 → the radius keeps its
                #                meaning of effective support (~2σ) but the close
                #                pixels dominate, with a smooth transition at the edge.
                # Values: the coastally extrapolated field.
                if disk_weighting == 'gaussian':
                    sigma = max(disk_radius_km / 2.0, 1e-6)
                    w = np.exp(
                        -0.5 * (dist_km[mask_disk] / sigma) ** 2
                    )
                elif disk_weighting == 'uniform':
                    w = np.ones(n_in_disk)
                else:
                    raise ValueError(
                        f"unknown disk_weighting='{disk_weighting}' "
                        f"(expected: 'uniform' or 'gaussian')"
                    )

                def _wmean(field_disk):
                    finite = np.isfinite(field_disk)
                    num = (np.where(finite, field_disk, 0.0)
                           * w).sum(axis=1)
                    den = (finite * w).sum(axis=1)
                    with np.errstate(invalid='ignore',
                                     divide='ignore'):
                        return np.where(den > 1e-30, num / den, np.nan)

                values_pts[:, i] = _wmean(bias_extrap[:, mask_disk])
        logger.info(
            f"  Disk {disk_radius_km} km "
            f"(weighting={disk_weighting}): "
            f"min={min(n_in_disk_log)}, "
            f"max={max(n_in_disk_log)}, "
            f"median={int(np.median(n_in_disk_log))} pixels/port"
        )

        time_index = pd.DatetimeIndex(bias_result.time)
        result = {}
        n_nan_ports = 0
        for i, port in enumerate(port_names):
            series = values_pts[:, i]

            n_nan_port = int(np.isnan(series).sum())
            n_nan_ports += n_nan_port
            result[port] = pd.DataFrame(
                {'bias_correction': series}, index=time_index,
            )
            logger.info(
                f"  {port:30s}: {len(time_index)} pts, "
                f"mean correction = {np.nanmean(series):.4f} m"
                + (f"  ⚠ {n_nan_port} NaN" if n_nan_port > 0 else "")
            )

        if n_nan_ports > 0:
            logger.warning(
                f"  {n_nan_ports} residual NaNs out of "
                f"{len(port_names) * len(time_index)} extractions"
            )
        return result

    @staticmethod
    def interpolate_correction_onto_model(
        df_correction_port: pd.DataFrame,
        df_model_port: pd.DataFrame,
        var_correction: str = 'bias_correction',
    ) -> pd.DataFrame:
        """Linear interpolation of the correction (IBI time step) onto the
        index of the 1D model (10 min), over their common time span."""
        idx_model = df_model_port.index
        idx_corr = df_correction_port.index
        t_min = max(idx_corr.min(), idx_model.min())
        t_max = min(idx_corr.max(), idx_model.max())
        idx_target = idx_model[(idx_model >= t_min) & (idx_model <= t_max)]

        ref = idx_corr[0]
        t_src = (idx_corr - ref).total_seconds().values.astype(float)
        t_dst = (idx_target - ref).total_seconds().values.astype(float)
        vals_src = df_correction_port[var_correction].values.astype(float)
        vals_interp = np.interp(t_dst, t_src, vals_src)
        return pd.DataFrame({var_correction: vals_interp}, index=idx_target)

    @classmethod
    def assemble_corrected_model_1d(
        cls,
        df_model_ports: Dict[str, pd.DataFrame],
        df_correction_ports_ibi: Dict[str, pd.DataFrame],
        var_ssh: str = 'ssh',
        var_correction: str = 'bias_correction',
    ) -> Dict[str, pd.DataFrame]:
        """Per port: ssh_corrected = 1D model (10 min) + correction
        interpolated to 10 min, over the period covered by the correction."""
        logger.info("=" * 70)
        logger.info("ASSEMBLING THE CORRECTED 1D MODEL")
        logger.info("=" * 70)

        result = {}
        for port, df_mod in df_model_ports.items():
            if port not in df_correction_ports_ibi:
                logger.warning(f"  {port}: no correction available")
                df = df_mod[[var_ssh]].copy()
                df['ssh_corrected'] = df[var_ssh]
                result[port] = df
                continue

            df_corr_interp = cls.interpolate_correction_onto_model(
                df_correction_ports_ibi[port], df_mod,
                var_correction=var_correction,
            )
            df_merged = df_mod[[var_ssh]].join(df_corr_interp, how='inner')
            df_merged['ssh_corrected'] = (
                df_merged[var_ssh] + df_merged[var_correction]
            )
            result[port] = df_merged
            logger.info(
                f"  {port:30s}: {len(df_merged)} pts, "
                f"mean correction = {df_merged[var_correction].mean():.4f} m"
            )
        return result


# =============================================================================
# Detection of the lat/lon coordinate names
# =============================================================================

def _detect_latlon_names(ds: xr.Dataset,
                         ref_var: Optional[str] = None) -> Tuple[str, str]:
    """Detect the names of the lat/lon variables.

    For unstructured Tolosa SW NetCDFs holding several sets at once
    (nodes: nlon/nlat, cells: clon/cclat), `ref_var` selects the set
    compatible with the dimensions of that variable.
    """
    possible_pairs = [
        ('lat', 'lon'),
        ('latitude', 'longitude'),
        ('nav_lat', 'nav_lon'),
        ('LATITUDE', 'LONGITUDE'),
        ('nlat', 'nlon'),
        ('node_lat', 'node_lon'),
        ('lat_node', 'lon_node'),
        ('mesh2d_node_y', 'mesh2d_node_x'),
        ('cclat', 'clon'),
        ('clat', 'clon'),
        ('face_lat', 'face_lon'),
        ('mesh2d_face_y', 'mesh2d_face_x'),
        ('y', 'x'),
    ]
    available_pairs = [
        (la, lo) for la, lo in possible_pairs
        if la in ds.variables and lo in ds.variables
    ]
    if not available_pairs:
        raise ValueError(f"lat/lon not found. Variables: {list(ds.variables)}")

    if ref_var is not None and ref_var in ds.variables:
        ref_dims = set(ds[ref_var].dims)
        for la, lo in available_pairs:
            if (set(ds[la].dims).issubset(ref_dims)
                    and set(ds[lo].dims).issubset(ref_dims)):
                if len(available_pairs) > 1:
                    logger.info(
                        f"  _detect_latlon_names: {len(available_pairs)} sets, "
                        f"selected ({la!r}, {lo!r}) compatible with "
                        f"{ref_var!r} (dims {ds[ref_var].dims})"
                    )
                return la, lo
        logger.warning(
            f"  _detect_latlon_names: no set compatible with {ref_var!r}. "
            f"Default: {available_pairs[0]}"
        )
    elif len(available_pairs) > 1:
        logger.warning(
            f"  _detect_latlon_names: {len(available_pairs)} sets available "
            f"{available_pairs}, no ref_var. Default: {available_pairs[0]}"
        )
    return available_pairs[0]


# =============================================================================
# SECTION 8: RUN CONFIGURATION & MAIN WORKFLOW
# =============================================================================
#
# Fixed configuration of the run. Only the options of _build_parser() can
# be set from the command line. The other choices of the method are fixed:
#   - reference bias: (SLA + DAC) − Tolosa surge, temporal mean removed
#     (apply_temporal_demean);
#   - predictor: IBI − DAC (DAC removed from the predictor too);
#   - filtering: zero-phase window truncated at the forecast horizon
#     (--forecast-horizon, FORECAST_HORIZON_DAYS at the top of the file);
#   - NetCDF export cropped to the common extent of the sources.
# NetCDF and Parquet exports are on by default (--no-export-netcdf,
# --no-export-parquet to disable them).

EXP_NAME = 'alpha'           # experiment code (output and cache names)
EXP_DESCRIPTION = 'baseline surge: standard Tolosa (full−ms) vs SLA'
CONFIG_FILE = 'config.json'  # data paths and port metadata


def _calibration_label(band: Tuple[float, float]) -> str:
    """Compact label identifying the calibration,
    `band{LOW}-{HIGH}d_demeantemporal_dacin_ibimdac`: suffix of
    `models_dir` and of the output folders, to avoid collisions between
    runs with different bands."""
    return f"band{band[0]:g}-{band[1]:g}d_demeantemporal_dacin_ibimdac"


_CONFIG_KEYS = ('sat_l4_file', 'ibi_file', 'ibi_var', 'dac_dir', 'dac_var',
                'tolosa_2d_full', 'tolosa_2d_ms', 'tolosa_1d_full_dir',
                'tolosa_1d_ms_dir', 'weights_dir', 'ports')
_PORT_KEYS = ('tolosa_csv', 'lat', 'lon')


def _load_config(path: str) -> Dict:
    """Read the JSON configuration file: data paths and port metadata
    (see the example config.json).

    Keys: sat_l4_file, ibi_file, ibi_var, weights_dir; dac_dir (AVISO
    DAC files dac_dif_<CNES day>_<HH>.nc: directory walked recursively,
    file or glob), dac_var (DAC variable); tolosa_2d_full,
    tolosa_2d_ms (Tolosa 2D files, full and ms), tolosa_1d_full_dir,
    tolosa_1d_ms_dir (folders of the Tolosa 1D CSVs); ports = {name:
    {tolosa_csv, lat, lon}}. Other keys are ignored (keys starting with
    '_' can hold comments).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"configuration file not found: {p}")
    with open(p, encoding='utf-8') as f:
        cfg = json.load(f)
    missing = [k for k in _CONFIG_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"{p}: missing key(s) {missing}")
    for port, meta in cfg['ports'].items():
        missing = [k for k in _PORT_KEYS if k not in meta]
        if missing:
            raise ValueError(
                f"{p}: port {port!r}, missing key(s) {missing}")
    return cfg


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Tolosa-SW surge bias correction: EOF regression on IBI − DAC, "
            "reference bias (SLA + DAC) − Tolosa with temporal mean removed, "
            "zero-phase filtering truncated at the forecast horizon "
            f"(--forecast-horizon). Data paths: {CONFIG_FILE}."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- Calibration ---
    parser.add_argument('--band', nargs=2, type=float, required=True,
                        metavar=('LOW', 'HIGH'),
                        help="Calibration band in days (e.g. 20 128); "
                             "band-pass applied before the EOF regression. "
                             "Below 20 d the DAC is the MOG2D model: LOW ≥ 20 "
                             "advised.")
    parser.add_argument('--ratio-train', type=float, default=0.8,
                        metavar='R',
                        help="Train fraction of the chronological split "
                             "(default 0.8).")
    parser.add_argument('--forecast-horizon', type=float,
                        default=FORECAST_HORIZON_DAYS, metavar='H_DAYS',
                        help="Forecast horizon H (days) of the band-pass "
                             "filtering: sample t uses the future up to t+H "
                             f"(default {FORECAST_HORIZON_DAYS:g}). Large H → "
                             "zero-phase filtering.")

    # --- Data ---
    parser.add_argument('--cache-dir', type=str, default='cache',
                        metavar='DIR',
                        help="Cache of the intermediates (daily IBI, "
                             "projected sat/Tolosa, bias_ref, DAC). To "
                             "invalidate, delete the relevant file "
                             "(default: cache).")

    # --- Extraction at the ports (Parquet export) ---
    parser.add_argument('--port-disk-radius', type=float, default=20.0,
                        metavar='KM',
                        help="Radius (km) of the extraction disk at the ports "
                             "(default 20). 10-15 in bays, 20-30 open coast, "
                             "50+ offshore.")
    parser.add_argument('--port-disk-weighting', type=str, default='gaussian',
                        choices=['uniform', 'gaussian'],
                        help="Disk weighting ('gaussian': σ = R/2; default "
                             "gaussian).")

    # --- Plots and exports ---
    parser.add_argument('--plot-stride-days', type=int, default=30,
                        metavar='K',
                        help="Plot 1 day out of K (scores/exports unaffected; "
                             "default 30).")
    parser.add_argument('--export-netcdf',
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Export 6 daily sources (sat, bias_ref, "
                             "residual on the IBI grid; model, correction, "
                             "model_corrected on the mesh, with the UGRID "
                             "connectivity) + the skill map to NetCDF-4 over "
                             "the test period, cropped to their common extent. "
                             "On by default; --no-export-netcdf to disable.")
    parser.add_argument('--export-parquet', nargs='?', const='auto',
                        default='auto', metavar='PATH',
                        help="Export the 1D model, the correction and the "
                             "corrected model at the ports to Parquet over "
                             "the test period (long format: time, port, "
                             "model, correction, model_corrected). On by "
                             "default, with an automatic file name; PATH "
                             "sets the output file.")
    parser.add_argument('--no-export-parquet', dest='export_parquet',
                        action='store_const', const=None,
                        help="Disable the Parquet export.")
    return parser


def main():
    """End-to-end run: loading → reference bias → calibration (train) →
    prediction (test) → plots → validation at the elements → NetCDF and
    Parquet exports → summary table."""
    # ------------------------------------------------------------------
    # ARGPARSE
    # ------------------------------------------------------------------
    parser = _build_parser()
    args = parser.parse_args()

    # Forecast horizon of the filtering (module-level, read by FilterUtils).
    global FORECAST_HORIZON_DAYS
    if args.forecast_horizon <= 0:
        parser.error(
            f"--forecast-horizon must be > 0 (got {args.forecast_horizon})."
        )
    FORECAST_HORIZON_DAYS = float(args.forecast_horizon)
    logger.info(
        f"  Butterworth filtering: zero-phase window truncated at the "
        f"forecast horizon H={FORECAST_HORIZON_DAYS:g} d"
    )

    # --- Calibration band from --band (required) ---
    low, high = float(args.band[0]), float(args.band[1])
    if low <= 0:
        parser.error(f"--band: low={low} must be > 0")
    if low >= high:
        parser.error(f"--band: low ({low}) must be strictly < high ({high})")
    band = (low, high)
    logger.info(f"  Calibration band: {low:g}-{high:g} d")

    # Calibration label used as a suffix of the output folders
    # (`models_calibrated/...`, `plots_2D_<out_label>`) to avoid collisions
    # between runs with different bands.
    calib_label = _calibration_label(band)
    logger.info(f"  Calibration: {calib_label}")

    # Cache of the intermediates
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Cache of the intermediates: {cache_dir}/")

    exp_name = EXP_NAME
    var_mod = 'surge'
    var_sat = 'sla'

    logger.info("=" * 70)
    logger.info(f"EXPERIMENT: {exp_name}")
    logger.info(f"  {EXP_DESCRIPTION}")
    logger.info(f"  var_mod={var_mod}, var_sat={var_sat}")
    logger.info("=" * 70)

    ratio_train = args.ratio_train
    if not (0.0 < ratio_train < 1.0):
        parser.error(f"--ratio-train must be in ]0,1[ (got {ratio_train})")

    # ------------------------------------------------------------------
    # PATHS AND PORT METADATA (configuration file)
    # ------------------------------------------------------------------
    cfg = _load_config(CONFIG_FILE)
    logger.info(f"  Configuration: {CONFIG_FILE}")
    model_file_2d = cfg['tolosa_2d_full']
    model_ms_file_2d = cfg['tolosa_2d_ms']
    for f in (model_file_2d, model_ms_file_2d):
        if not Path(f).exists():
            raise FileNotFoundError(f"Missing Tolosa 2D file: {f}")
    sat_file_2d = cfg['sat_l4_file']
    ibi_file = cfg['ibi_file']
    dac_dir, dac_var = cfg['dac_dir'], cfg['dac_var']
    weights_dir = cfg['weights_dir']

    ports = cfg['ports']
    port_to_csv = {p: m['tolosa_csv'] for p, m in ports.items()}
    coords_ports = {p: (m['lat'], m['lon']) for p, m in ports.items()}

    # ------------------------------------------------------------------
    # STEP 1: LOADING
    # ------------------------------------------------------------------
    loader = DataLoader(use_dask=True, chunk_size=100)

    ds_mod_2d, ds_sat_2d = loader.load_data_2d(
        model_file_2d, model_ms_file_2d, sat_file_2d,
        var_model=var_mod, var_satellite=var_sat,
    )

    # IBI (native resolution)
    ibi_ds = xr.open_dataset(ibi_file)
    ibi_da = standardize_coords(ibi_ds)[cfg['ibi_var']]
    # Native IBI time step (24 h for the daily product), used to filter the
    # test period at its native resolution.
    ibi_steps_s = (np.diff(ibi_da['time'].values)
                   .astype('timedelta64[s]').astype(float))
    dt_hours_ibi = (float(np.median(ibi_steps_s)) / 3600.0
                    if ibi_steps_s.size else 24.0)
    da_sat = standardize_coords(ds_sat_2d)[var_sat]
    da_mod = canonicalize_element_dim(
        standardize_coords(ds_mod_2d, ref_var=var_mod)[var_mod]
    )
    logger.info(
        f"  da_mod: canonical dims {dict(da_mod.sizes)} "
        f"(mesh dimension → 'element')"
    )

    # --- DAC: L4 SLA DAC-out → SLA + DAC (DAC-in) ---
    # Extent: Tolosa ∪ IBI + 2° (what the sat projections read); period:
    # SLA ∩ Tolosa overlap (the only days used).
    if band[0] < 20.0:
        logger.warning(
            f"  --band LOW={band[0]:g} d < 20 d: below 20 d the DAC is the "
            f"MOG2D model, so the reference compares Tolosa with another "
            f"model there."
        )
    margin_deg = 2.0
    tol_lon = np.asarray(da_mod.lon.values, dtype=np.float64)
    tol_lat = np.asarray(da_mod.lat.values, dtype=np.float64)
    dac_bbox = (
        min(np.nanmin(tol_lon), float(ibi_da.lon.min())) - margin_deg,
        max(np.nanmax(tol_lon), float(ibi_da.lon.max())) + margin_deg,
        min(np.nanmin(tol_lat), float(ibi_da.lat.min())) - margin_deg,
        max(np.nanmax(tol_lat), float(ibi_da.lat.max())) + margin_deg,
    )
    sat_times = pd.DatetimeIndex(da_sat['time'].values)
    mod_times = pd.DatetimeIndex(da_mod['time'].values)
    dac_window = (
        np.datetime64(max(sat_times[0], mod_times[0]).normalize(), 'ns'),
        np.datetime64(min(sat_times[-1], mod_times[-1]).normalize()
                      + pd.Timedelta(days=1), 'ns'),
    )
    da_sat = add_dac_to_sla(
        da_sat, dac_dir, bbox=dac_bbox, time_bounds=dac_window,
        var_dac=dac_var, cache_dir=cache_dir,
    )

    # Same DAC, on the IBI grid: the predictor becomes IBI − DAC,
    # non-atmospheric like the target SLA + DAC − Tolosa.
    dac_daily = load_dac_daily(dac_dir, dac_var, dac_bbox,
                               dac_window, cache_dir=cache_dir)
    dac_ibi = xr.DataArray(
        _dac_on_grid(dac_daily, np.asarray(ibi_da.lon.values, np.float64),
                     np.asarray(ibi_da.lat.values, np.float64)),
        dims=('time', 'lat', 'lon'),
        coords={'time': dac_daily['time'].values,
                'lat': ibi_da.lat.values, 'lon': ibi_da.lon.values},
    )

    # ------------------------------------------------------------------
    # STEP 2: CALIBRATION + PREDICTION (IBI grid)
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("CALIBRATION + PREDICTION (IBI grid)")
    logger.info("=" * 70)

    ny_ibi, nx_ibi = len(ibi_da.lat.values), len(ibi_da.lon.values)
    ibi_shape = (ny_ibi, nx_ibi)

    W_tolosa_to_ibi = load_and_normalize_weights(
        f'{weights_dir}/proj_tolosa_to_ibi.npz', "W_tolosa→ibi"
    )
    W_sat_to_ibi = load_and_normalize_weights(
        f'{weights_dir}/proj_sat_to_ibi.npz', "W_sat→ibi"
    )
    W_ibi_to_tolosa = load_and_normalize_weights(
        f'{weights_dir}/proj_ibi_to_tolosa.npz', "W_ibi→tolosa"
    )
    W_sat_to_tolosa = load_and_normalize_weights(
        f'{weights_dir}/proj_sat_to_tolosa.npz', "W_sat→tolosa"
    )
    logger.info(
        f"  W: tolosa→ibi {W_tolosa_to_ibi.shape}, "
        f"sat→ibi {W_sat_to_ibi.shape}, ibi→tolosa {W_ibi_to_tolosa.shape}"
    )

    def _build_fallback(W_proj, src_lon, src_lat, tgt_lon, tgt_lat):
        """Target points with an empty row in W_proj (outside the source
        domain) → (their row indices, index of the nearest source point),
        or None if there are none."""
        row_sums = np.array(W_proj.sum(axis=1)).ravel()
        tgt_valid = row_sums > 0
        lon_2d_tgt, lat_2d_tgt = np.meshgrid(tgt_lon, tgt_lat)
        invalid_mask = ~tgt_valid
        if not invalid_mask.any():
            return None
        invalid_pts = np.column_stack([
            lon_2d_tgt.ravel()[invalid_mask],
            lat_2d_tgt.ravel()[invalid_mask],
        ])
        tree = cKDTree(np.column_stack([src_lon.ravel(), src_lat.ravel()]))
        _, nearest_idx = tree.query(invalid_pts)
        fb_rows = np.where(invalid_mask)[0]
        logger.info(f"    Fallback: {len(fb_rows)} points outside the domain")
        return (fb_rows, nearest_idx)

    fallback_tolosa = _build_fallback(
        W_tolosa_to_ibi, da_mod.lon.values, da_mod.lat.values,
        ibi_da.lon.values, ibi_da.lat.values,
    )
    sat_lon_1d = (da_sat.lon.values if da_sat.lon.values.ndim == 1
                  else da_sat.lon.values[0, :])
    sat_lat_1d = (da_sat.lat.values if da_sat.lat.values.ndim == 1
                  else da_sat.lat.values[:, 0])
    sat_lon_2d, sat_lat_2d = np.meshgrid(sat_lon_1d, sat_lat_1d)
    fallback_sat = _build_fallback(
        W_sat_to_ibi, sat_lon_2d, sat_lat_2d,
        ibi_da.lon.values, ibi_da.lat.values,
    )

    tgt_land_mask = np.isnan(ibi_da.isel(time=0).values)
    logger.info(
        f"  IBI land mask: {int(tgt_land_mask.sum())} pixels / "
        f"{tgt_land_mask.size}"
    )

    # --- Daily reference bias (time, lat, lon) on the IBI grid ---
    # cache_prefix encodes the parameters affecting the content of the
    # projected fields, not those of the calibration (band), which may vary
    # while reusing the same cache. The '_extrap0' suffix (no coastal
    # extrapolation of the sat field) keeps the names of existing caches.
    cache_prefix = f"exp{exp_name}_{var_mod}_{var_sat}_extrap0"
    bias_ref_da, interp_intermediates = (
        BiasComputation.compute_bias_2d_from_unstructured(
            da_sat, da_mod,
            W_tolosa_to_ibi=W_tolosa_to_ibi,
            ibi_shape=ibi_shape,
            ibi_lon=ibi_da.lon.values, ibi_lat=ibi_da.lat.values,
            W_sat_to_ibi=W_sat_to_ibi,
            fallback_tolosa=fallback_tolosa,
            fallback_sat=fallback_sat,
            tgt_land_mask=tgt_land_mask,
            return_intermediates=True,
            cache_dir=cache_dir, cache_prefix=cache_prefix,
            sat_cache_tag='_dacin',
        )
    )
    logger.info(
        f"  bias_ref (IBI grid): {bias_ref_da.sizes['time']} days, "
        f"({bias_ref_da.sizes['lat']}, {bias_ref_da.sizes['lon']})"
    )

    # --- Saving bias_ref for external use (diagnostics, etc.) ---
    bias_ref_path = cache_dir / (
        f"bias_ref_{exp_name}_{var_mod}_demeantemporal_dacin"
        f"_{_time_span_tag(bias_ref_da)}.nc"
    )
    bias_ref_da.to_dataset().to_netcdf(bias_ref_path)
    logger.info(f"  bias_ref saved → {bias_ref_path}")

    # --- Daily mean of IBI (cached) ---
    key_ibi_daily = (
        f"ibi_daily_{Path(ibi_file).stem}_{_time_span_tag(ibi_da)}"
    )
    daily_ibi_da = _cache_load(cache_dir, key_ibi_daily)
    if daily_ibi_da is None:
        logger.info("  Daily IBI: computing…")
        daily_ibi_da = resample_daily(ibi_da)
        if hasattr(daily_ibi_da, 'compute'):
            daily_ibi_da = daily_ibi_da.compute()
        _cache_save(cache_dir, key_ibi_daily, daily_ibi_da,
                    source_paths=[ibi_file])
    else:
        logger.info(f"  Daily IBI: read from cache ({key_ibi_daily})")

    daily_ibi_da = subtract_dac_from_ibi(daily_ibi_da, dac_ibi,
                                         label='Daily IBI')
    # Write the predictor as it is actually used: external studies
    # (coherence, diagnostics) must read IBI − DAC, not the raw IBI cache
    # whose name does not change.
    _pred_path = cache_dir / f"{key_ibi_daily}_minusdac.nc"
    if not _pred_path.exists():
        daily_ibi_da.to_netcdf(_pred_path)
    logger.info(f"  IBI − DAC predictor written: {_pred_path}")

    # --- Time intersection daily IBI ↔ bias_ref ---
    daily_ibi_da, bias_ref_da = align_inner_time(daily_ibi_da, bias_ref_da)
    n_common = daily_ibi_da.sizes['time']
    logger.info(f"  Common days IBI ↔ bias_ref: {n_common}")
    if n_common == 0:
        raise RuntimeError("No common day between IBI and bias_ref.")

    # --- Chronological train / test split ---
    n_train = int(ratio_train * n_common)
    ibi_train_da = daily_ibi_da.isel(time=slice(0, n_train))
    bias_ref_train_da = bias_ref_da.isel(time=slice(0, n_train))
    bias_ref_test_da = bias_ref_da.isel(time=slice(n_train, None))

    train_start = ibi_train_da['time'].values[0]
    train_end   = ibi_train_da['time'].values[-1]
    test_start_dt = bias_ref_test_da['time'].values[0]
    test_end_dt   = bias_ref_test_da['time'].values[-1]

    logger.info(
        f"  Train: {n_train} days ({train_start} → {train_end})"
    )
    logger.info(
        f"  Test : {n_common - n_train} days ({test_start_dt} → {test_end_dt})"
    )

    # ------------------------------------------------------------------
    # PLOTS EARLY
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("PLOTS EARLY (before fit/predict)")
    logger.info("=" * 70)

    plot_subdaily_ibi_tolosa(
        ibi_da=ibi_da, da_mod=da_mod,
        test_start=test_start_dt, test_end=test_end_dt,
        var_mod=var_mod,
        out_label=f"{exp_name}_{calib_label}",
    )

    plot_daily_ibi_tolosa_biasref(
        ibi_da=ibi_da,
        interp_intermediates=interp_intermediates,
        bias_ref=bias_ref_test_da,
        test_start=test_start_dt, test_end=test_end_dt,
        var_mod=var_mod,
        ds_sat_native=ds_sat_2d, sat_var_native=var_sat,
        out_label=f"{exp_name}_{calib_label}",
    )

    # ------------------------------------------------------------------
    # CALIBRATION
    # ------------------------------------------------------------------
    pipeline = BiasCorrectionPipeline(band=band, dt_hours=24.0)
    pipeline.set_data(ibi_train_da, bias_ref_train_da.values)
    pipeline.fit_all()

    # --- Saving the models ---
    train_start_d = np.datetime64(train_start, 'D')
    train_end_d   = np.datetime64(train_end, 'D')
    models_dir = (
        f"models_calibrated/{exp_name}_{train_start_d}_to_{train_end_d}"
        f"_{calib_label}"
    )
    pipeline.save_models(
        output_dir=models_dir,
        train_period=(train_start_d, train_end_d),
        ibi_lon=ibi_da.lon.values, ibi_lat=ibi_da.lat.values,
        var_model=var_mod, var_satellite=var_sat,
        extra_metadata={
            "experiment_code": exp_name,
            "experiment_description": EXP_DESCRIPTION,
            "tolosa_full_file": model_file_2d,
            "tolosa_ms_file": model_ms_file_2d,
            "ibi_file": ibi_file, "sat_file": sat_file_2d,
            "dac": dac_dir,
            "dac_predictor": True,
            "forecast_horizon_days": FORECAST_HORIZON_DAYS,
            "n_train_days": int(n_train),
        },
    )

    # ------------------------------------------------------------------
    # PREDICTION over the test period at the native IBI dt (always).
    # ------------------------------------------------------------------
    # The upper bound is slightly widened to include the end of the day.
    test_start_ns = np.datetime64(test_start_dt, 'ns')
    test_end_ns   = (np.datetime64(test_end_dt, 'ns')
                    + np.timedelta64(1, 'D'))
    ibi_test_native = ibi_da.sel(time=slice(test_start_ns, test_end_ns))
    ibi_test_native = subtract_dac_from_ibi(ibi_test_native, dac_ibi,
                                            label='IBI test')
    logger.info(
        f"  Native test IBI: {ibi_test_native.sizes['time']} time "
        f"steps, dt = {dt_hours_ibi:g} h"
    )
    results = pipeline.predict_all(
        ibi_test_native, dt_hours=dt_hours_ibi,
    )
    eof_result = results['eof']   # correction field of the EOF method
    # ibi_test_native (large at the native dt) is no longer used after the
    # prediction: released immediately.
    del ibi_test_native
    gc.collect()

    # ------------------------------------------------------------------
    # PLOT corrections per time step
    # ------------------------------------------------------------------
    # Optional stride of the per-time-step plots (scores/exports unaffected).
    bias_ref_plot_da = bias_ref_test_da
    if args.plot_stride_days > 1:
        bias_ref_plot_da = bias_ref_test_da.isel(
            time=slice(None, None, args.plot_stride_days)
        )
        logger.info(
            f"  Per-time-step plots: 1 day out of "
            f"{args.plot_stride_days} → "
            f"{bias_ref_plot_da.sizes['time']} time steps plotted "
            f"(scores and exports unaffected)"
        )
    logger.info("=" * 70)
    logger.info("PLOT corrections per timestep")
    logger.info("=" * 70)
    plot_corrections_per_timestep(
        pipeline_results=results,
        bias_ref=bias_ref_plot_da,
        var_mod=var_mod, out_label=f"{exp_name}_{calib_label}",
    )

    # ------------------------------------------------------------------
    # 2D VALIDATION (elements of the barotropic mesh)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("2D VALIDATION — ELEMENTS OF THE BAROTROPIC MESH")
    logger.info("=" * 70)
    bias_ref_elements_da = BiasComputation.compute_bias_at_elements(
        da_sat, da_mod, W_sat_to_tolosa=W_sat_to_tolosa,
    )
    bias_ref_elements_test_da, _ = align_inner_time(
        bias_ref_elements_da, bias_ref_test_da,
    )
    logger.info(
        f"  bias_ref at the elements (test): "
        f"{bias_ref_elements_test_da.sizes['time']} days, "
        f"{bias_ref_elements_test_da.sizes['element']} elements"
    )
    model_elements_test_da, _ = align_inner_time(
        interp_intermediates['model_elements_daily_da'],
        bias_ref_elements_test_da,
    )
    # Skill baseline = TRAIN climatology of the bias at the elements (mean
    # over the days ≤ end of the train period; bias_ref_elements_da covers the
    # whole common period).
    _elements_train = bias_ref_elements_da.sel(
        time=slice(None, np.datetime64(train_end))
    )
    baseline_train_elements = np.nanmean(_elements_train.values, axis=0)
    logger.info(
        f"  Skill baseline = TRAIN climatology at the elements "
        f"({_elements_train.sizes['time']} days)"
    )
    del _elements_train
    validation_main = pipeline.validate_all_correction_at_elements(
        bias_ref_elements_test_da, W_ibi_to_tolosa=W_ibi_to_tolosa,
        model_at_elements_da=model_elements_test_da,
        baseline_map=baseline_train_elements,
    )
    main_grid_label = "BAROTROPIC (Tolosa mesh elements)"

    # ------------------------------------------------------------------
    # NETCDF EXPORT (6 sources + skill, daily, test period)
    # ------------------------------------------------------------------
    if args.export_netcdf:
        export_sources_to_netcdf(
            da_sat=da_sat,
            da_mod=da_mod,
            correction_da=eof_result.da,
            bias_ref=bias_ref_test_da,
            correction_label='eof',
            out_label=f"{exp_name}_{calib_label}",
            period_start=test_start_dt,
            period_end=test_end_dt,
            W_ibi_to_tolosa=W_ibi_to_tolosa,
            elem_lon=da_mod.lon.values,
            elem_lat=da_mod.lat.values,
            tolosa_path=model_file_2d,
            skill_per_method={
                name: v['per_element']['skill_score']
                for name, v in validation_main.items()
                if isinstance(v, dict) and 'per_element' in v
                and 'skill_score' in v['per_element']
            },
        )

    # ------------------------------------------------------------------
    # PARQUET EXPORT: model / correction / corrected model at the ports
    # ------------------------------------------------------------------
    if args.export_parquet is not None:
        logger.info("=" * 70)
        logger.info("PARQUET EXPORT — model and correction at the ports")
        logger.info("=" * 70)
        # Tolosa 1D CSVs: flat layout (<type>/<Station>.csv) and/or period
        # sub-folders (<type>/<period>/<Station>.csv), merged per port at
        # load time.
        model_port_files = discover_port_csvs(
            cfg['tolosa_1d_full_dir'], port_to_csv, label='full')
        model_ms_port_files = discover_port_csvs(
            cfg['tolosa_1d_ms_dir'], port_to_csv, label='ms')
        df_model_ports = loader.load_model_1d(
            model_port_files, model_ms_port_files,
            dt_minutes=10, var_model=var_mod,
        )
        # Correction (test period) extracted at the ports, added to the 1D
        # Tolosa series (10 min).
        df_correction_ports = DataLoader.extract_correction_at_ports(
            eof_result, coords_ports,
            disk_radius_km=args.port_disk_radius,
            disk_weighting=args.port_disk_weighting,
        )
        df_corrected_ports = DataLoader.assemble_corrected_model_1d(
            df_model_ports, df_correction_ports, var_ssh=var_mod,
        )
        try:
            export_gauge_corrections_to_parquet(
                df_corrected_ports, var_ssh=var_mod,
                out_path=(None if args.export_parquet == 'auto'
                          else args.export_parquet),
                out_label=f"{exp_name}_{calib_label}",
            )
        except Exception as e:
            logger.error(f"  Parquet export failed (not fatal): {e}")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("SUMMARY (2D validation at the mesh elements)")
    logger.info("=" * 70)
    # Tolosa (without correction) is shown as the first row for comparison.
    # The Corr 2D and EV 2D columns are in obs space (corrected model vs
    # sat), falling back to bias space if the model is not given to the
    # validation.

    def _fmt(x, w, d):
        """Format a float on width w with d decimals, '—' if None/NaN."""
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return f"{'—':>{w}}"
        return f"{x:>{w}.{d}f}"

    def _print_2d_table(grid_label: str, validation_2d_grid: Dict[str, Dict]):
        """Print a 2D section for a given grid."""
        if not validation_2d_grid:
            logger.info(f"\n— Grid {grid_label}: no validation —")
            return
        sep = 114
        header = (
            f"  {'Method':<32} | "
            f"{'RMSE 2D':>8} {'Corr 2D':>8} {'EV 2D':>7} {'Skill 2D':>8} || "
            f"{'RMSE Q95':>9} {'Corr Q95':>9} {'Bias Q95':>9}"
        )
        logger.info(f"\n— Grid {grid_label} —")
        logger.info(header)
        logger.info("-" * sep)
        # Tolosa row (read from the first method; the "uncorr" values are
        # identical across methods — it is the baseline)
        first_method = next(iter(validation_2d_grid), None)
        if first_method is not None:
            v2d_ref = validation_2d_grid[first_method].get('summary', {})
            rmse_2d_tol = v2d_ref.get('mean_rmse_uncorr', np.nan)
            corr_2d_tol = v2d_ref.get('mean_correlation_tolosa', np.nan)
            ev_2d_tol = v2d_ref.get('mean_explained_variance_tolosa', np.nan)
            # Baseline Q95 (Tolosa without correction) read from the summary
            rmse_q95_tol = v2d_ref.get('mean_rmse_q95_uncorr', np.nan)
            bias_q95_tol = v2d_ref.get('mean_bias_q95_uncorr', np.nan)
            # The baseline Corr Q95 is undefined (corr(0, x) is NaN)
            logger.info(
                f"  {'Tolosa (no correction)':<32} | "
                f"{_fmt(rmse_2d_tol, 8, 4)} {_fmt(corr_2d_tol, 8, 3)} "
                f"{_fmt(ev_2d_tol, 7, 3)} {0.000:>8.3f} || "
                f"{_fmt(rmse_q95_tol, 9, 4)} {'—':>9} "
                f"{_fmt(bias_q95_tol, 9, 4)}"
            )
            logger.info("-" * sep)
        for method_name in validation_2d_grid:
            v2d = validation_2d_grid[method_name].get('summary', {})
            rmse_2d = v2d.get('mean_rmse', np.nan)
            corr_2d = v2d.get('mean_correlation_corrected',
                              v2d.get('mean_correlation', np.nan))
            ev_2d = v2d.get('mean_explained_variance_corrected',
                            v2d.get('mean_explained_variance', np.nan))
            skill_2d = v2d.get('mean_skill_score', np.nan)
            # Q95 stats (extremes): performance on the ~5 % of points
            # (pixels × time) where the bias is the largest.
            rmse_q95 = v2d.get('mean_rmse_q95', np.nan)
            corr_q95 = v2d.get('mean_corr_q95', np.nan)
            bias_q95 = v2d.get('mean_bias_q95', np.nan)
            logger.info(
                f"  {method_name:<32} | "
                f"{_fmt(rmse_2d, 8, 4)} {_fmt(corr_2d, 8, 3)} "
                f"{_fmt(ev_2d, 7, 3)} {_fmt(skill_2d, 8, 3)} || "
                f"{_fmt(rmse_q95, 9, 4)} {_fmt(corr_q95, 9, 3)} "
                f"{_fmt(bias_q95, 9, 4)}"
            )

    _print_2d_table(main_grid_label, validation_main)

    logger.info("\n✅ Full pipeline completed.")


if __name__ == "__main__":
    main()
