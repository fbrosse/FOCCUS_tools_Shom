# -*- coding: utf-8 -*-
"""
build_features.py — enrichment of the per-station CSV files for the XGBoost
bias correction of the Tolosa storm surge.

For each station CSV (timestamp in the first column, regular time step
`cadence_minutes`), the following columns are added, in this order:
    1. the IBI SSH − Tolosa surge difference, the wind (u, v) components when
       they are derived from (speed, direction), and the wind speed;
    2. tidal features (time since high water, semi-diurnal phase, 24 h tidal
       range, time derivative);
    3. wind features (direction, quadratic stress, cross-shore / along-shore
       projection on the station's coast normal);
    4. wave features (wave setup proxy, wave/coast alignment);
    5. IBI current features (speed, cross/along-shore projection);
    6. explicit interactions (tide × surge, tide × wind², …);
    7. calendar features (sin/cos of the day of year and of the hour);
    8. configurable lags, centred derivatives and rolling statistics.

Durations (lags, windows) are given in hours and converted into time steps
using `cadence_minutes`.

Usage:
    python build_features.py --config config.yaml [-s BREST CALAIS]
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.signal import find_peaks

import station_io


# ---------------------------------------------------------------------------
# Time step: sph = number of time steps per hour (6 for 10 min).
# ---------------------------------------------------------------------------

def n_steps(hours: float, sph: float) -> int:
    return int(round(hours * sph))


def _check_regular_cadence(df: pd.DataFrame, expected_minutes: int = 10) -> None:
    dt = df.index.to_series().diff().dropna()
    n_bad = int((dt != pd.Timedelta(minutes=expected_minutes)).sum())
    if n_bad:
        warnings.warn(
            f"Irregular index: {n_bad} intervals ≠ {expected_minutes} min. "
            "Lags, derivatives and rolling windows are computed in number of "
            "time steps: reindex onto a regular grid before build_features."
        )


def _skip_missing(kind: str, col: str) -> None:
    print(f"  [{kind}] missing column, skipped: {col}")


def _centered_diff(x: pd.Series, dt_hours: float, sph: float) -> pd.Series:
    """Centred difference over ±dt_hours, in units per hour."""
    step = n_steps(dt_hours, sph)
    return (x.shift(-step) - x.shift(step)) / (2.0 * dt_hours)


# ---------------------------------------------------------------------------
# Lags, derivatives, rolling statistics
# ---------------------------------------------------------------------------

def add_lags(df: pd.DataFrame, cols: Iterable[str],
             lags_hours: Iterable[float], sph: float) -> pd.DataFrame:
    out = {}
    for col in cols:
        if col not in df.columns:
            _skip_missing("lag", col)
            continue
        for h in lags_hours:
            out[f"{col}_lag{int(h)}h"] = df[col].shift(n_steps(h, sph))
    return pd.DataFrame(out, index=df.index)


def add_derivatives(df: pd.DataFrame, cols: Iterable[str], sph: float,
                    dt_hours: float = 1.0) -> pd.DataFrame:
    out = {}
    for col in cols:
        if col not in df.columns:
            _skip_missing("derivative", col)
            continue
        out[f"d{col}_dt"] = _centered_diff(df[col], dt_hours, sph)
    return pd.DataFrame(out, index=df.index)


def add_rolling_aggregates(df: pd.DataFrame, specs: list[dict],
                           sph: float) -> pd.DataFrame:
    """Rolling statistics (backward-looking window).

    specs: list of dicts {col, window_h, agg, abs (optional)}, with
    agg ∈ {'max', 'min', 'mean', 'std', 'sum'}.
    Example: {col: 'wind_speed', window_h: 6, agg: 'max'} → wind_speed_max_6h.
    """
    out = {}
    for s in specs:
        col = s["col"]
        if col not in df.columns:
            _skip_missing("rolling", col)
            continue
        w = n_steps(s["window_h"], sph)
        x = df[col].abs() if s.get("abs", False) else df[col]
        roll = x.rolling(window=w, min_periods=max(1, w // 2))
        out[f"{col}_{s['agg']}_{int(s['window_h'])}h"] = getattr(roll, s["agg"])()
    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------------------
# Tide: phase, range, derivative
# ---------------------------------------------------------------------------

def add_tidal_features(df: pd.DataFrame, tide_col: str | None,
                       sph: float) -> pd.DataFrame:
    if tide_col is None or tide_col not in df.columns:
        return pd.DataFrame(index=df.index)

    tide = df[tide_col]
    n = len(tide)

    # High waters: peaks at least 10 h apart, i.e. one per semi-diurnal cycle
    # (avoids double peaks caused by M2/S2/N2 interactions).
    peaks, _ = find_peaks(tide.values, distance=n_steps(10.0, sph))

    hours_since = np.full(n, np.nan)
    if peaks.size:
        idx_prev = np.searchsorted(peaks, np.arange(n), side="right") - 1
        valid = idx_prev >= 0
        hours_since[valid] = (np.arange(n)[valid] - peaks[idx_prev[valid]]) / sph
    hours_since_s = pd.Series(hours_since, index=df.index)

    # Semi-diurnal phase (M2 period ≈ 12.42 h).
    T = 12.42
    phase = 2 * np.pi * (hours_since_s % T) / T

    out = pd.DataFrame(index=df.index)
    out["tide_hours_since_HW"] = hours_since_s
    out["tide_phase_sin"] = np.sin(phase)
    out["tide_phase_cos"] = np.cos(phase)

    w = n_steps(24, sph)
    roll = tide.rolling(window=w, min_periods=w // 2)
    out["tide_range_24h"] = roll.max() - roll.min()

    out["dtide_dt"] = _centered_diff(tide, 1.0, sph)
    return out


# ---------------------------------------------------------------------------
# Wind
# ---------------------------------------------------------------------------

def wind_components(df: pd.DataFrame, cfg: dict):
    """Return (u, v, from_speed_dir), or None if no wind source is available.

    The (u10_col, v10_col) columns take precedence. Otherwise, conversion from
    (wind_speed_col, wind_dir_col), the direction being in meteorological
    convention (where the wind blows from, in degrees). u is positive
    eastwards, v northwards.
    """
    u_col, v_col = cfg.get("u10_col"), cfg.get("v10_col")
    if u_col and v_col and u_col in df.columns and v_col in df.columns:
        return df[u_col], df[v_col], False

    speed_col, dir_col = cfg.get("wind_speed_col"), cfg.get("wind_dir_col")
    if speed_col and dir_col and speed_col in df.columns and dir_col in df.columns:
        offset = cfg.get("wind_dir_offset_deg", 270.0)
        rad = np.deg2rad(offset - df[dir_col])
        return df[speed_col] * np.cos(rad), df[speed_col] * np.sin(rad), True

    return None


def add_wind_features(u: pd.Series, v: pd.Series, cfg: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=u.index)
    speed = np.sqrt(u ** 2 + v ** 2)

    # Direction in meteorological convention (where the wind blows from:
    # 0 = N, 90 = E).
    wind_dir_meteo = (270.0 - np.rad2deg(np.arctan2(v, u))) % 360.0
    out["wind_dir_meteo"] = wind_dir_meteo

    # Cyclic encoding: a threshold-based model does not know that 359° ≈ 1°.
    rad = np.deg2rad(wind_dir_meteo)
    out["wind_dir_sin"] = np.sin(rad)
    out["wind_dir_cos"] = np.cos(rad)

    # Quadratic wind stress (τ ∝ |U|·U).
    out["wind_stress_u"] = u * speed
    out["wind_stress_v"] = v * speed
    out["wind_speed_sq"] = speed ** 2

    # Projection on the station's coast normal (mathematical convention:
    # 0° = east, counter-clockwise, normal pointing from sea to land).
    cn = cfg.get("coast_normal_deg")
    if cn is not None:
        t = np.deg2rad(cn)
        nx, ny = np.cos(t), np.sin(t)
        out["wind_cross"] = u * nx + v * ny   # > 0: onshore wind
        out["wind_along"] = -u * ny + v * nx
        out["wind_stress_cross"] = out["wind_cross"] * speed
        out["wind_stress_along"] = out["wind_along"] * speed

    return out


# ---------------------------------------------------------------------------
# Waves
# ---------------------------------------------------------------------------

def add_wave_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    hs_col = cfg.get("hs_col")
    have_hs = bool(hs_col) and hs_col in df.columns
    if have_hs:
        out["wave_setup_proxy"] = df[hs_col] ** 2

    # Alignment = cos(angle between the wave propagation direction and the
    # coast normal). wave_dir_col is in nautical convention (where the waves
    # come from, 0 = N, clockwise); 270 − dir gives the propagation direction
    # in mathematical convention, like coast_normal_deg.
    # 1: offshore swell hitting the coast head-on; 0: swell parallel to the coast.
    wdir_col = cfg.get("wave_dir_col")
    cn = cfg.get("coast_normal_deg")
    if wdir_col and wdir_col in df.columns and cn is not None:
        align = np.cos(np.deg2rad(270.0 - df[wdir_col] - cn))
        out["wave_alignment"] = align
        if have_hs:
            out["wave_setup_aligned"] = (df[hs_col] ** 2) * align

    return out


# ---------------------------------------------------------------------------
# IBI surface currents
# ---------------------------------------------------------------------------

def add_ibi_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    ibi_u, ibi_v = cfg.get("ibi_u_col"), cfg.get("ibi_v_col")
    if not (ibi_u and ibi_v and ibi_u in df.columns and ibi_v in df.columns):
        return out

    u, v = df[ibi_u], df[ibi_v]
    out["ibi_cur_speed"] = np.sqrt(u ** 2 + v ** 2)
    cn = cfg.get("coast_normal_deg")
    if cn is not None:
        t = np.deg2rad(cn)
        nx, ny = np.cos(t), np.sin(t)
        out["ibi_cur_cross"] = u * nx + v * ny
        out["ibi_cur_along"] = -u * ny + v * nx
    return out


# ---------------------------------------------------------------------------
# Explicit interactions
# ---------------------------------------------------------------------------

def add_interactions(df: pd.DataFrame, cfg: dict, sph: float) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    def has(key):
        col = cfg.get(key)
        return bool(col) and col in df.columns

    tide, surge_mod = cfg.get("tide_col"), cfg.get("surge_mod_col")
    speed, msl, hs = cfg.get("wind_speed_col"), cfg.get("msl_col"), cfg.get("hs_col")

    if has("tide_col") and has("surge_mod_col"):
        out["tide_x_surge_mod"] = df[tide] * df[surge_mod]
    if has("tide_col") and has("wind_speed_col"):
        out["tide_x_wind_sq"] = df[tide] * df[speed] ** 2
    if has("msl_col"):
        # Pressure anomaly w.r.t. a 7-day rolling mean (inverse barometer:
        # −1 hPa ≈ +1 cm of sea level).
        out["msl_anomaly"] = df[msl] - df[msl].rolling(
            n_steps(24 * 7, sph), min_periods=n_steps(24, sph)).mean()
    if has("hs_col") and has("wind_speed_col"):
        out["hs_x_wind_sq"] = df[hs] * df[speed] ** 2

    return out


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    doy = df.index.dayofyear + df.index.hour / 24.0 + df.index.minute / 1440.0
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    h = df.index.hour + df.index.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * h / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * h / 24.0)
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply every enrichment (the input DataFrame is not modified)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame must have a DatetimeIndex.")
    cadence = cfg.get("cadence_minutes", 10)
    _check_regular_cadence(df, expected_minutes=cadence)
    sph = 60.0 / cadence
    df = df.copy()

    # Columns added directly to the base DataFrame (they can also be used as
    # inputs of the configured lags / rolling statistics).
    surge_mod, ibi = cfg.get("surge_mod_col"), cfg.get("ibi_ssh_col")
    if ibi and surge_mod and ibi in df.columns and surge_mod in df.columns:
        df["zos_minus_ssh"] = df[ibi] - df[surge_mod]

    wind = wind_components(df, cfg)
    if wind is not None:
        u, v, from_speed_dir = wind
        if from_speed_dir:
            df["u10"] = u
            df["v10"] = v
        if "wind_speed" not in df.columns:
            df["wind_speed"] = np.sqrt(u ** 2 + v ** 2)

    parts = [
        df,
        add_tidal_features(df, cfg.get("tide_col"), sph),
        add_wind_features(u, v, cfg) if wind is not None else None,
        add_wave_features(df, cfg),
        add_ibi_features(df, cfg),
        add_interactions(df, cfg, sph),
        add_calendar_features(df),
    ]
    if cfg.get("lag_cols"):
        parts.append(add_lags(df, cfg["lag_cols"], cfg.get("lag_hours", [1, 3, 6, 12]), sph))
    if cfg.get("derivative_cols"):
        parts.append(add_derivatives(df, cfg["derivative_cols"], sph))
    if cfg.get("rolling_aggregates"):
        parts.append(add_rolling_aggregates(df, cfg["rolling_aggregates"], sph))

    out = pd.concat([p for p in parts if p is not None], axis=1)
    # On duplicated names, the first occurrence (original column) is kept.
    return out.loc[:, ~out.columns.duplicated()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "-s", "--stations", nargs="+", default=None, metavar="STATION",
        help="Restrict to these stations (name or alias from stations.json, "
             "case- and separator-insensitive). Default: every CSV file.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    in_dir = Path(cfg["paths"]["input_csv_dir"])
    out_dir = Path(cfg["paths"]["enriched_csv_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_cfg = {
        "cadence_minutes": cfg.get("cadence_minutes", 10),
        "surge_mod_col": cfg["columns"]["surge_mod"],
        **cfg["features"],
    }
    # `stations` keys mapped to the canonical identifier (as for the files).
    stations_cfg = {station_io.canonical(k): v
                    for k, v in (cfg.get("stations") or {}).items()}

    csv_files = sorted(in_dir.glob("*.csv"))
    if args.stations is not None:
        wanted = {station_io.canonical(s) for s in args.stations}
        csv_files = [p for p in csv_files
                     if station_io.canonical(station_io.station_id(p)) in wanted]
    if not csv_files:
        print(f"No CSV file found in {in_dir}")
        return

    for csv_path in csv_files:
        station = station_io.station_id(csv_path)
        print(f"[{station}] reading {csv_path.name}")
        # First column = timestamp.
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

        # Global settings + station-specific overrides (coast_normal_deg…).
        sta_override = stations_cfg.get(station_io.canonical(station))
        if sta_override is None:
            print(f"[{station}] WARNING: no {station} entry in `stations` "
                  "→ no cross/along-shore features")
        sta_cfg = {**feat_cfg, **(sta_override or {})}
        enriched = build_features(df, sta_cfg)

        out_path = out_dir / csv_path.name
        enriched.index.name = "time"  # column expected by station_io
        enriched.to_csv(out_path)
        print(f"[{station}] {enriched.shape[0]} rows × {enriched.shape[1]} cols → {out_path.name}")


if __name__ == "__main__":
    main()
