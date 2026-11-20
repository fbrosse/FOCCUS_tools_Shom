#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction of the Tolosa-SW surge bias correction from native IBI fields
=======================================================================
Applies a model calibrated by eof_cca.py (folder
models_calibrated/<...>/: metadata.json, ibi_grid.npz, method_eof.joblib)
to new IBI fields, reusing the functions of eof_cca.py (no
config.json: every input is given on the command line).

Steps
-----
1. Load the calibrated model and its metadata (band, forecast horizon H).
2. Read IBI on its native grid and time step; the grid must be the
   calibration grid.
3. Predictor IBI − DAC (daily DAC interpolated onto the IBI grid), as in
   the calibration.
4. Band-pass filtering (zero-phase window truncated at H) + EOF/CCA
   regression → correction on the IBI grid, at the IBI time step.
5. Export over the prediction period [--start, --end], with the code of
   the calibration export (copied from eof_cca.export_sources_to_netcdf):
   daily means, correction projected onto the Tolosa-SW elements, crop to
   the common IBI ∩ Tolosa extent, UGRID mesh.

Outputs (--out-dir), same format and grids as the calibration export
---------------------------------------------------------------------
- model.nc           : daily Tolosa surge (full − ms) at the elements;
- correction.nc      : daily correction projected onto the elements;
- model_corrected.nc : model + correction.

Important
---------
- HISTORY: the band-pass has a long past support (P samples, ~280 days for
  a 20-128 d band at a daily step). IBI must start well before --start;
  output days with an incomplete past are reported (edge regime).
- FUTURE: day t uses IBI up to t+H. On the last H days of the IBI series
  the future window shrinks (forecast regime, as in the calibration test).
- DAC: the predictor is IBI − DAC. IBI time steps without DAC are dropped,
  so the DAC must cover the whole IBI period, forecast days included.
- The model must have been saved by the current eof_cca.py (class
  EOFRegression).

Usage
-----
python predict_correction.py --model-dir models_calibrated/<run> \\
    --ibi IBI.nc [IBI2.nc ...] --dac aviso/DAC/ \\
    --w-ibi-to-tolosa projs_long/proj_ibi_to_tolosa.npz \\
    --tolosa-full htot_full.nc --tolosa-ms htot_ms.nc \\
    --start YYYY-MM-DD [--end YYYY-MM-DD] [--out-dir prediction]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

import eof_cca as core

logger = logging.getLogger("predict_correction")

# Correction label written in the exported files (method of the model).
CORRECTION_LABEL = 'eof'


# =============================================================================
# Model
# =============================================================================

def load_calibrated_model(model_dir: str) -> Tuple[core.BiasCorrectionPipeline,
                                                    Dict, np.ndarray,
                                                    np.ndarray]:
    """Load a model saved by eof_cca.py → (pipeline, metadata,
    calibration IBI lon, calibration IBI lat)."""
    # Models saved by `python eof_cca.py` (or by the former eof_cca.py)
    # pickle their class as __main__.EOFRegression: it is made visible under
    # that name before unpickling. A model pickled from the module under its
    # former name (eof_cca.EOFRegression) is resolved the same way.
    main_mod = sys.modules['__main__']
    if not hasattr(main_mod, 'EOFRegression'):
        main_mod.EOFRegression = core.EOFRegression
    sys.modules.setdefault('ant_2d_seas', core)

    pipeline = core.BiasCorrectionPipeline.load_models(model_dir)
    with open(Path(model_dir) / "metadata.json") as f:
        metadata = json.load(f)
    grid = np.load(Path(model_dir) / "ibi_grid.npz")
    return pipeline, metadata, grid['lon'], grid['lat']


def check_ibi_grid(ibi: xr.DataArray, lon_cal: np.ndarray,
                   lat_cal: np.ndarray) -> None:
    """The IBI grid must be the calibration grid (EOF masks and patterns
    are defined pixel by pixel)."""
    lon, lat = ibi['lon'].values, ibi['lat'].values
    if (lon.shape != lon_cal.shape or lat.shape != lat_cal.shape
            or not np.allclose(lon, lon_cal, atol=1e-6)
            or not np.allclose(lat, lat_cal, atol=1e-6)):
        raise ValueError(
            f"IBI grid ({lat.size}×{lon.size}) differs from the calibration "
            f"grid ({lat_cal.size}×{lon_cal.size}): same extent and "
            f"resolution required."
        )


# =============================================================================
# Inputs
# =============================================================================

def open_ibi(paths, var: str) -> xr.DataArray:
    """IBI field (time, lat, lon) at its native time step, from one or
    several files (concatenated along time)."""
    ds = (xr.open_dataset(paths[0]) if len(paths) == 1
          else xr.open_mfdataset(paths, combine='by_coords'))
    da = core.standardize_coords(ds)[var]
    return da.sortby('time')


def native_dt_hours(da: xr.DataArray) -> float:
    """Median time step (hours) of a field."""
    steps = (np.diff(da['time'].values)
             .astype('timedelta64[s]').astype(float))
    return float(np.median(steps)) / 3600.0 if steps.size else 24.0


def ibi_minus_dac(ibi: xr.DataArray, dac_path: str, dac_var: str,
                  cache_dir: Optional[Path]) -> xr.DataArray:
    """Predictor IBI − DAC, with the daily DAC interpolated onto the IBI
    grid (same functions as in the calibration)."""
    margin_deg = 2.0
    lon = np.asarray(ibi['lon'].values, np.float64)
    lat = np.asarray(ibi['lat'].values, np.float64)
    bbox = (lon.min() - margin_deg, lon.max() + margin_deg,
            lat.min() - margin_deg, lat.max() + margin_deg)
    times = pd.DatetimeIndex(ibi['time'].values)
    window = (np.datetime64(times[0].normalize(), 'ns'),
              np.datetime64(times[-1].normalize() + pd.Timedelta(days=1),
                            'ns'))
    dac_daily = core.load_dac_daily(dac_path, dac_var, bbox, window,
                                    cache_dir=cache_dir)
    dac_ibi = xr.DataArray(
        core._dac_on_grid(dac_daily, lon, lat),
        dims=('time', 'lat', 'lon'),
        coords={'time': dac_daily['time'].values,
                'lat': ibi['lat'].values, 'lon': ibi['lon'].values},
    )
    return core.subtract_dac_from_ibi(ibi, dac_ibi, label='IBI')


def open_tolosa_surge(full_path: str, ms_path: str,
                      var: str) -> xr.DataArray:
    """Tolosa-SW surge = full − ms at the elements (time, element), lazily
    read (as DataLoader.load_data_2d in the calibration)."""
    chunks = {'time': 24}
    ds = xr.open_dataset(full_path, chunks=chunks)
    ds_ms = xr.open_dataset(ms_path, chunks=chunks)
    ds = ds.assign(surge=ds[var] - ds_ms[var])
    return core.canonicalize_element_dim(
        core.standardize_coords(ds, ref_var='surge')['surge'])


def in_days(da: xr.DataArray, t0: np.datetime64,
            t1: np.datetime64) -> xr.DataArray:
    """Time steps of the calendar days t0 … t1 (inclusive)."""
    t = da['time'].values
    keep = np.flatnonzero((t >= t0) & (t < t1 + np.timedelta64(1, 'D')))
    return da.isel(time=keep)


# =============================================================================
# Export (copy of the export of eof_cca.py, which is left unchanged)
# =============================================================================

def daily_window(da: xr.DataArray, t0: np.datetime64,
                  t1: np.datetime64) -> xr.DataArray:
    """Daily mean (resample_daily, computed) restricted to [t0, t1], as
    _daily_period in eof_cca.export_sources_to_netcdf."""
    d = core.resample_daily(da)
    if hasattr(d, 'compute'):
        d = d.compute()
    return d.sel(time=slice(t0, t1))


def build_element_sources(mod_d: xr.DataArray, corr_ibi_d: xr.DataArray,
                          W_ibi_to_tolosa,
                          elem_lon: np.ndarray, elem_lat: np.ndarray,
                          ) -> Dict[str, xr.DataArray]:
    """Daily sources at the Tolosa elements, as exported to NetCDF by
    eof_cca.export_sources_to_netcdf (same code):
    'model' (daily surge `mod_d`), 'correction' (daily correction on the
    IBI grid `corr_ibi_d` projected onto the elements by W_ibi_to_tolosa)
    and 'model_corrected' (model + correction on their common days)."""
    # Spatial dims must have the same name, otherwise xarray broadcasts to
    # (time, N, N), several PiB. Idempotent guard.
    mod_d = core.canonicalize_element_dim(mod_d)
    # Correction projected IBI → elements (per daily time step)
    n_dst = W_ibi_to_tolosa.shape[0]
    corr_elements_arr = core.apply_projection(
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
    # projection does not match the model mesh); then a POSITIONAL sum in
    # numpy — shape-checked semantics, as everywhere else in the pipeline
    # after projection — re-wrapped with the model coords.
    mod_a, corrn_a = core.align_inner_time(mod_d, corr_elements_d)
    if mod_a.sizes.get('element') != corrn_a.sizes.get('element'):
        raise ValueError(
            f"NetCDF export: incompatible element sizes "
            f"(model={mod_a.sizes.get('element')}, "
            f"correction={corrn_a.sizes.get('element')}). The "
            f"W_ibi_to_tolosa matrix does not match the model mesh."
        )
    mod_corrected_d = xr.DataArray(
        mod_a.values + corrn_a.values,
        dims=mod_a.dims, coords=mod_a.coords,
        name='model_corrected',
    )
    return {
        'model': mod_d.rename('model'),
        'correction': corr_elements_d,
        'model_corrected': mod_corrected_d,
    }


def write_sources_netcdf(exports: Dict[str, xr.DataArray], root: Path, *,
                         correction_label: str,
                         period_start: np.datetime64,
                         period_end: np.datetime64,
                         tolosa_path: Optional[str] = None,
                         extent_sources: Tuple[xr.DataArray, ...] = (),
                         ) -> None:
    """Write daily sources to NetCDF-4 in `root` (one <name>.nc per
    source), cropped to the intersection of the
    extents of `exports` and of `extent_sources` (fields not exported but
    defining the common domain). Element files embed the UGRID mesh of
    `tolosa_path`. Copy of the writing part of
    eof_cca.export_sources_to_netcdf (same layout and attributes)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # -- Cropping to the common extent (intersection) --
    elem_mask = None
    bbox = None
    lo_mins, lo_maxs, la_mins, la_maxs = [], [], [], []
    for da in list(exports.values()) + list(extent_sources):
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
    mesh = core._read_ugrid_mesh(tolosa_path) if tolosa_path else None
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

    logger.info(f"NETCDF EXPORT done: {len(exports)} files in {root}/")


# =============================================================================
# Filtering diagnostics
# =============================================================================

def filter_past_support_days(dt_hours: float) -> Optional[float]:
    """Past support P of the band-pass (days), read from the kernel cache of
    FilterUtils filled by the prediction (None if unavailable)."""
    cache = core.FilterUtils._forecast_kernel_cache
    if not cache:
        return None
    P = max(p for _, p in cache.values())
    return P * dt_hours / 24.0


# =============================================================================
# Main
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prediction of the Tolosa-SW surge bias correction from "
                    "native IBI fields, with a model calibrated by "
                    "eof_cca.py; exports model / correction / "
                    "model_corrected as in the calibration.")
    p.add_argument('--model-dir', required=True, metavar='DIR',
                   help="Calibrated model folder (models_calibrated/<run>).")
    p.add_argument('--ibi', required=True, nargs='+', metavar='FILE',
                   help="IBI file(s), native grid and time step (same grid "
                        "as the calibration), with enough history before "
                        "--start for the band-pass.")
    p.add_argument('--ibi-var', default='zos_detided',
                   help="IBI variable (default zos_detided).")
    p.add_argument('--dac', required=True, metavar='PATH',
                   help="AVISO DAC files covering the IBI period: directory "
                        "(walked recursively), file or glob (quoted).")
    p.add_argument('--dac-var', default='dac',
                   help="DAC variable (default dac).")
    p.add_argument('--w-ibi-to-tolosa', required=True, metavar='NPZ',
                   help="Projection matrix IBI → Tolosa elements "
                        "(proj_ibi_to_tolosa.npz of the calibration).")
    p.add_argument('--tolosa-full', required=True, metavar='FILE',
                   help="Tolosa-SW 2D file (full) over the prediction "
                        "period; also gives the element coordinates and the "
                        "UGRID mesh.")
    p.add_argument('--tolosa-ms', required=True, metavar='FILE',
                   help="Tolosa-SW 2D file (ms): surge = full − ms.")
    p.add_argument('--tolosa-var', default='ssh',
                   help="Tolosa variable (default ssh).")
    p.add_argument('--start', required=True,
                   help="First day of the prediction period (YYYY-MM-DD).")
    p.add_argument('--end',
                   help="Last day of the prediction period (YYYY-MM-DD; "
                        "default: last IBI day).")
    p.add_argument('--forecast-horizon', type=float, metavar='H_DAYS',
                   help="Override the horizon H of the calibration (not "
                        "advised).")
    p.add_argument('--out-dir', default='prediction', metavar='DIR',
                   help="Output folder (default prediction).")
    p.add_argument('--cache-dir', metavar='DIR',
                   help="Cache of the daily DAC (optional).")
    return p


def main():
    args = _build_parser().parse_args()
    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    # 1. Calibrated model ----------------------------------------------------
    pipeline, metadata, lon_cal, lat_cal = load_calibrated_model(
        args.model_dir)
    extra = metadata.get('extra', {})
    horizon = args.forecast_horizon or extra.get('forecast_horizon_days')
    if horizon is None:
        horizon = core.FORECAST_HORIZON_DAYS
        logger.warning(f"  forecast horizon not in the metadata → "
                       f"{horizon:g} d")
    elif args.forecast_horizon:
        logger.warning(f"  forecast horizon overridden: {horizon:g} d "
                       f"(calibration: "
                       f"{extra.get('forecast_horizon_days', '?')} d)")
    core.FORECAST_HORIZON_DAYS = float(horizon)
    logger.info(f"  Band {pipeline.band[0]:g}-{pipeline.band[1]:g} d, "
                f"horizon H = {horizon:g} d, train period "
                f"{metadata['train_period']['start']} → "
                f"{metadata['train_period']['end']}")

    # 2. IBI (native) ----------------------------------------------------------
    ibi = open_ibi(args.ibi, args.ibi_var)
    check_ibi_grid(ibi, lon_cal, lat_cal)
    dt_hours = native_dt_hours(ibi)
    logger.info(f"  IBI: {ibi.sizes['time']} time steps, dt = {dt_hours:g} h, "
                f"{ibi['time'].values[0]} → {ibi['time'].values[-1]}")

    # 3. Predictor IBI − DAC -------------------------------------------------
    if extra.get('dac_predictor', True):
        ibi = ibi_minus_dac(ibi, args.dac, args.dac_var, cache_dir)

    # 4. Prediction on the IBI grid (whole IBI series: history needed) -------
    results = pipeline.predict_all(ibi, dt_hours=dt_hours)
    corr = results[CORRECTION_LABEL].da
    times = corr['time'].values

    # Prediction period (calendar days, inclusive)
    t0 = np.datetime64(args.start, 'D').astype('datetime64[ns]')
    t1 = (np.datetime64(args.end, 'D') if args.end
          else times[-1].astype('datetime64[D]')).astype('datetime64[ns]')
    if t1 < t0:
        raise SystemExit(f"--end ({t1}) is before --start ({t0}).")
    logger.info(f"  Prediction period: {t0.astype('datetime64[D]')} → "
                f"{t1.astype('datetime64[D]')}")

    P_days = filter_past_support_days(dt_hours)
    if P_days is not None:
        support = np.timedelta64(int(round(P_days * 86400)), 's')
        if t0 < times[0] + support:
            logger.warning(
                f"  History: the band-pass needs ~{P_days:.0f} d of past "
                f"data; days before {times[0] + support} use a truncated "
                f"past window (edge regime). Provide IBI from "
                f"{(t0 - support).astype('datetime64[D]')} for a full window."
            )

    corr_win = in_days(corr, t0, t1)
    if corr_win.sizes['time'] == 0:
        raise SystemExit("No IBI time step in the prediction period.")

    # 5. Exports, as in the calibration -----------------------------------------
    W = core.load_and_normalize_weights(args.w_ibi_to_tolosa, "W_ibi→tolosa")
    model = open_tolosa_surge(args.tolosa_full, args.tolosa_ms,
                              args.tolosa_var)
    if model.sizes['element'] != W.shape[0]:
        raise ValueError(
            f"W_ibi_to_tolosa has {W.shape[0]} rows, the Tolosa field "
            f"{model.sizes['element']} elements.")
    model_win = in_days(model, t0, t1)
    if model_win.sizes['time'] == 0:
        raise SystemExit("No Tolosa time step in the prediction period.")

    corr_d = daily_window(corr_win, t0, t1)
    exports = build_element_sources(
        daily_window(model_win, t0, t1), corr_d, W,
        model['lon'].values, model['lat'].values,
    )
    n_days = exports['model_corrected'].sizes['time']
    logger.info("=" * 70)
    logger.info(f"NETCDF EXPORT (daily, prediction period, {n_days} days) "
                f"→ {out_dir}/")
    logger.info("=" * 70)
    # The IBI grid (correction before projection) takes part in the common
    # extent, as the IBI-grid sources of the calibration export.
    write_sources_netcdf(
        exports, out_dir, correction_label=CORRECTION_LABEL,
        period_start=t0.astype('datetime64[D]'),
        period_end=t1.astype('datetime64[D]'), tolosa_path=args.tolosa_full,
        extent_sources=(corr_d,),
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
