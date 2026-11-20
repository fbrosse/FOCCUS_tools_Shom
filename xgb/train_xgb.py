# -*- coding: utf-8 -*-
"""
train_xgb.py — XGBoost bias correction of the Tolosa storm surge.

For each station, the model learns the error surge_obs − surge_mod from the
features produced by build_features.py. Main steps:
    - chronological train / test split (`dates` in the config);
    - nested cross-validation: leave-one-winter-out outer loop, Optuna (TPE)
      tuning + early stopping in the inner loop;
    - final model refitted on the whole training set;
    - sample weighting by the observed surge (`sample_weight`);
    - overall and per-event scores, diagnostic figures (MDI importance,
      optional permutation importance, scatter, QQ plot, event time series);
    - per-station cache and per-fold checkpoints to resume an interrupted run;
    - inference + parquet export over an optional validation period.

Dependencies: xgboost, optuna, scikit-learn, pandas, pyarrow, seaborn,
tabulate, threadpoolctl; station_io.py and stations.json (same directory).

Usage:
    python train_xgb.py --config config.yaml [-s BREST CALAIS] [--no-cache]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

# Number of OpenMP threads set BEFORE any import that initialises OpenMP/BLAS
# (taken from SLURM when available), with passive waiting: avoids
# oversubscription and OpenMP × fork deadlocks.
_n_threads = (os.environ.get("SLURM_CPUS_PER_TASK")
              or os.environ.get("OMP_NUM_THREADS") or "8")
os.environ.setdefault("OMP_NUM_THREADS", _n_threads)
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (HPC without display)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import scipy.stats
import seaborn as sns
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tabulate import tabulate
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

import station_io

warnings.filterwarnings("ignore", category=FutureWarning)


def _rmse(a, b) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))


# ---------------------------------------------------------------------------
# Sample weights
# ---------------------------------------------------------------------------

def compute_sample_weights(y_obs_surge: np.ndarray, scheme: dict) -> np.ndarray:
    """Per-sample weights, computed from the *observed* surge.

    Large surges get more weight during training. Weights depend on the
    observation, not on the error, so that samples are not up-weighted merely
    because they are poorly predicted.

    Schemes (`kind` key):
        - 'flat'                       : weight = 1.
        - 'quantile_boost' (q, alpha)  : alpha if surge_obs > quantile q, else 1.
        - 'power' (p, eps)             : (|surge_obs| + eps) ** p.
        - 'exp' (beta, clip)           : exp(beta · (surge_obs − q50) / (q95 − q50)),
                                         clipped at clip.
    Weights are normalised to a mean of 1.
    """
    y = np.asarray(y_obs_surge, dtype=float)
    kind = scheme.get("kind", "flat")

    if kind == "flat":
        w = np.ones_like(y)
    elif kind == "quantile_boost":
        q = np.quantile(y, scheme.get("q", 0.95))
        w = np.where(y > q, scheme.get("alpha", 5.0), 1.0)
    elif kind == "power":
        w = (np.abs(y) + scheme.get("eps", 0.05)) ** scheme.get("p", 2.0)
    elif kind == "exp":
        q50 = np.quantile(y, 0.5)
        q95 = np.quantile(y, 0.95)
        scale = max(q95 - q50, 1e-6)
        w = np.exp(scheme.get("beta", 5.0) * (y - q50) / scale)
        w = np.clip(w, 0, scheme.get("clip", 1e3))
    else:
        raise ValueError(f"Unknown sample_weight scheme: {kind}")

    return w / w.mean()


# ---------------------------------------------------------------------------
# Split by extended winter season (Oct → Sep)
# ---------------------------------------------------------------------------

def winter_labels(index: pd.DatetimeIndex, start_month: int = 10) -> np.ndarray:
    """Start year of the season of each time step: Oct(Y) → Sep(Y+1) ↦ Y."""
    years = index.year.to_numpy()
    return np.where(index.month.to_numpy() >= start_month, years, years - 1)


def winter_folds(labels: np.ndarray):
    """Leave-one-winter-out: one fold (train_idx, val_idx) per season.

    Returns (folds, winters), winters being the sorted list of seasons.
    """
    winters = [int(w) for w in np.unique(labels)]
    folds = []
    for w in winters:
        val = np.where(labels == w)[0]
        trn = np.where(labels != w)[0]
        if len(val) and len(trn):
            folds.append((trn, val))
    return folds, winters


# ---------------------------------------------------------------------------
# Event detection and scoring
# ---------------------------------------------------------------------------

def detect_events(obs: pd.Series, threshold: float, min_duration_h: float = 3.0,
                  merge_gap_h: float = 12.0, cadence_minutes: float = 10):
    """Periods where obs > threshold, dips of at most merge_gap_h being merged.

    obs is first placed on a regular time grid: missing time steps count as
    below the threshold, so data gaps are accounted for in durations and in
    the gaps between events.
    Returns a list of (start, end) timestamps, end included.
    """
    step = pd.Timedelta(minutes=cadence_minutes)
    sph = 60.0 / cadence_minutes
    grid = obs.resample(step).max()
    above = (grid.values > threshold).astype(int)  # NaN → below
    if not above.any():
        return []
    diff = np.diff(np.concatenate(([0], above, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    merge_gap = int(merge_gap_h * sph)
    merged_s = [int(starts[0])]
    merged_e = [int(ends[0])]
    for s, e in zip(starts[1:], ends[1:]):
        if s - merged_e[-1] <= merge_gap:
            merged_e[-1] = int(e)
        else:
            merged_s.append(int(s))
            merged_e.append(int(e))

    min_dur = int(min_duration_h * sph)
    return [(grid.index[s], grid.index[e])
            for s, e in zip(merged_s, merged_e) if (e - s + 1) >= min_dur]


def _between(x: pd.Series, t0, t1, step) -> pd.Series:
    """Values of x in the time steps [t0, t1] (t1 being a grid step label)."""
    return x[(x.index >= t0) & (x.index < t1 + step)]


def score_event(t0, t1, obs: pd.Series, mod: pd.Series, corr: pd.Series,
                step: pd.Timedelta) -> dict:
    o = _between(obs, t0, t1, step)
    m = _between(mod, t0, t1, step)
    c = _between(corr, t0, t1, step)
    o_peak_t, m_peak_t, c_peak_t = o.idxmax(), m.idxmax(), c.idxmax()

    return {
        "start":            o.index[0],
        "end":              o.index[-1],
        "duration_h":       (o.index[-1] - o.index[0]).total_seconds() / 3600.0,
        "obs_peak":         float(o.max()),
        "obs_peak_time":    o_peak_t,
        "mod_peak":         float(m.max()),
        "corr_peak":        float(c.max()),
        "peak_err_raw":     float(m.max() - o.max()),
        "peak_err_corr":    float(c.max() - o.max()),
        "peak_lag_raw_h":   (m_peak_t - o_peak_t).total_seconds() / 3600.0,
        "peak_lag_corr_h":  (c_peak_t - o_peak_t).total_seconds() / 3600.0,
        "rmse_raw":         _rmse(o, m),
        "rmse_corr":        _rmse(o, c),
        "mae_raw":          float(mean_absolute_error(o, m)),
        "mae_corr":         float(mean_absolute_error(o, c)),
    }


# ---------------------------------------------------------------------------
# XGBoost tuning: Optuna (TPE) + early stopping
# ---------------------------------------------------------------------------

def _sample_params(trial: "optuna.Trial", param_space: dict) -> dict:
    """Translate the YAML search space into trial.suggest_* calls.

    {choices: [...]} → categorical; {low, high} → integer if both bounds are
    integers (unless `float: true`), float otherwise; `log: true` → log scale.
    """
    params = {}
    for name, spec in param_space.items():
        if "choices" in spec:
            params[name] = trial.suggest_categorical(name, spec["choices"])
            continue
        low, high = spec["low"], spec["high"]
        log = spec.get("log", False)
        is_int = (isinstance(low, int) and isinstance(high, int)
                  and not spec.get("float", False))
        if is_int:
            params[name] = trial.suggest_int(name, low, high, log=log)
        else:
            params[name] = trial.suggest_float(name, float(low), float(high), log=log)
    return params


def tune_xgb_optuna(X, y, w, param_space: dict, fixed_params: dict,
                    optuna_cfg: dict, random_state: int):
    """TPE search with early stopping, then refit on the whole (X, y).

    The data are split chronologically: the first (1 − inner_val_fraction)
    are used to fit each trial, the last inner_val_fraction for early stopping
    and for the trial score (unweighted RMSE). The final refit uses
    n_estimators = max(50, round(best_iteration × n_estimators_refit_scale)).

    Returns
    -------
    (final_model, best_params, best_score, best_iteration, n_estimators_refit)
    """
    cut = int(len(X) * (1 - optuna_cfg["inner_val_fraction"]))
    X_tr, X_va = X[:cut], X[cut:]
    y_tr, y_va = y[:cut], y[cut:]
    w_tr = w[:cut]
    xgb_n_jobs = optuna_cfg.get("xgb_n_jobs", -1)

    def objective(trial: "optuna.Trial") -> float:
        params = _sample_params(trial, param_space)
        params.update(fixed_params)
        model = XGBRegressor(
            **params,
            n_estimators=optuna_cfg["n_estimators_max"],
            random_state=random_state,
            n_jobs=xgb_n_jobs,
            early_stopping_rounds=optuna_cfg["early_stopping_rounds"],
            eval_metric="rmse",
            verbosity=0,
        )
        model.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_va, y_va)], verbose=False)
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        return _rmse(y_va, model.predict(X_va))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    # timeout_s bounds the duration of this study: once reached, Optuna starts
    # no new trial and keeps the best one found.
    timeout_s = optuna_cfg.get("timeout_s")
    study.optimize(
        objective,
        n_trials=optuna_cfg["n_trials"],
        timeout=timeout_s,
        n_jobs=optuna_cfg.get("trial_n_jobs", 1),
        show_progress_bar=False,
    )
    n_done = len(study.trials)
    if timeout_s and n_done < optuna_cfg["n_trials"]:
        print(f"    [optuna] timeout {timeout_s}s reached: "
              f"{n_done}/{optuna_cfg['n_trials']} trials completed")

    best_params = study.best_params
    best_iter = study.best_trial.user_attrs["best_iteration"]
    refit_scale = optuna_cfg.get("n_estimators_refit_scale", 1.25)
    n_est_refit = max(int(round(best_iter * refit_scale)), 50)

    final = XGBRegressor(
        **{**fixed_params, **best_params},
        n_estimators=n_est_refit,
        random_state=random_state,
        n_jobs=xgb_n_jobs,
        verbosity=0,
    )
    final.fit(X, y, sample_weight=w)
    return final, best_params, float(study.best_value), int(best_iter), int(n_est_refit)


# ---------------------------------------------------------------------------
# Per-station cache
# ---------------------------------------------------------------------------
# A previous training is reused if models/XGB_{station}.joblib exists and the
# `cache_context` block of reports/cv_{station}.json matches the current
# context. Otherwise, full tuning. Downstream outputs (scores, events,
# figures, parquet) are always regenerated.

def _build_cache_context(df_train, target, predictors, cfg):
    """Everything that determines the trained model: training period and
    size, predictors (in column order, since the model is positional),
    weighting, seed, winter split, Optuna and XGBoost settings.
    Normalised through a JSON round trip so that it compares equal to the
    context read back from the cache."""
    ctx = {
        "train_start":        str(df_train.index.min()),
        "train_end":          str(df_train.index.max()),
        "n_train":            int(len(df_train)),
        "target":             target,
        "predictors_used":    list(predictors),
        "sample_weight":      cfg["sample_weight"],
        "random_state":       cfg["random_state"],
        "winter_start_month": cfg["cv"].get("winter_start_month", 10),
        "optuna":             cfg["optuna"],
        "xgb_fixed":          cfg["xgb_fixed"],
        "xgb_param_space":    cfg["xgb_param_space"],
    }
    return json.loads(json.dumps(ctx, default=str))


def _try_load_cache(station, current_context, out_dirs):
    """Return (xgb_model, cv_data) if a valid cache exists, otherwise None."""
    xgb_path = out_dirs["models"] / f"XGB_{station}.joblib"
    cv_json_path = out_dirs["reports"] / f"cv_{station}.json"
    if not xgb_path.exists() or not cv_json_path.exists():
        return None

    try:
        with open(cv_json_path) as f:
            cv_data = json.load(f)
    except Exception:
        return None

    cached_context = cv_data.get("cache_context")
    if cached_context is None:
        return None

    diff = [k for k in sorted(set(cached_context) | set(current_context))
            if cached_context.get(k) != current_context.get(k)]
    if diff:
        head = ", ".join(diff[:3]) + (", …" if len(diff) > 3 else "")
        print(f"[{station}] cache invalidated (changed field(s): {head})")
        return None

    try:
        final_model = joblib.load(xgb_path)
    except Exception as e:
        print(f"[{station}] failed to load cache: {e!r}")
        return None
    return final_model, cv_data


# ---------------------------------------------------------------------------
# Per-fold checkpoints
# ---------------------------------------------------------------------------
# The result of each outer CV fold is written to reports/checkpoints/ as soon
# as it is done, and reloaded on restart if the context is unchanged: an
# interrupted job only recomputes the missing folds. Checkpoints are deleted
# once the station is complete.

def _fold_ckpt_path(out_dirs, station, k):
    return out_dirs["checkpoints"] / f"fold_{station}_{k}.json"


def _load_fold_checkpoint(out_dirs, station, k, current_context):
    p = _fold_ckpt_path(out_dirs, station, k)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            data = json.load(f)
    except Exception:
        return None
    if data.get("cache_context") != current_context:
        return None
    return data.get("result")


def _save_fold_checkpoint(out_dirs, station, k, current_context, result):
    """Best effort: a failed write does not stop the training."""
    try:
        with open(_fold_ckpt_path(out_dirs, station, k), "w") as f:
            json.dump({"cache_context": current_context, "result": result},
                      f, indent=2, default=str)
    except Exception as exc:
        print(f"[{station}]   checkpoint of fold {k} not written: {exc!r}")


def _clear_fold_checkpoints(out_dirs, station):
    for p in out_dirs["checkpoints"].glob(f"fold_{station}_*.json"):
        try:
            p.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Training of one station: nested CV + final model
# ---------------------------------------------------------------------------

def _tune(X, y, w, cfg):
    return tune_xgb_optuna(X, y, w, cfg["xgb_param_space"], cfg["xgb_fixed"],
                           cfg["optuna"], cfg["random_state"])


def _train_station(station, df_train, X_train, y_train, w_train, cfg, out_dirs,
                   current_context):
    """Leave-one-winter-out nested CV, then final tuning on the whole
    training set.

    Saves models/XGB_{station}.joblib and reports/cv_{station}.json.
    Returns the final model.
    """
    cache_cfg = cfg.get("cache", {}) or {}
    use_fold_ckpt = (bool(cache_cfg.get("enabled", True))
                     and bool(cache_cfg.get("fold_checkpoints", True)))

    labels = winter_labels(df_train.index, cfg["cv"].get("winter_start_month", 10))
    folds, winters = winter_folds(labels)
    # A missing or very short winter here usually comes from the dropna on
    # the target: check this first if the nested CV is disabled.
    counts = {w: int((labels == w).sum()) for w in winters}
    print(f"[{station}] time steps per winter in the training set: {counts}")

    cv_results = []
    if len(folds) < 2:
        print(f"[{station}] only {len(folds)} winter(s) in the training set — "
              "nested CV disabled, going straight to the final tuning.")
    else:
        print(f"[{station}] nested CV — {len(folds)} folds (winters: {winters})")
        for k, (trn_idx, val_idx) in enumerate(folds):
            if use_fold_ckpt:
                cached_fold = _load_fold_checkpoint(out_dirs, station, k,
                                                    current_context)
                if cached_fold is not None:
                    cv_results.append(cached_fold)
                    print(f"[{station}]   fold {k + 1}/{len(folds)} "
                          f"(winter {winters[k]}) → resumed from checkpoint "
                          f"(RMSE_out = {cached_fold.get('outer_rmse', float('nan')):.4f})")
                    continue

            t0 = time.time()
            est, params, inner_score, best_iter, n_est_refit = _tune(
                X_train[trn_idx], y_train[trn_idx], w_train[trn_idx], cfg)

            y_va_pred = est.predict(X_train[val_idx])
            y_va_true = y_train[val_idx]
            fold_res = {
                "fold": k,
                "holdout_winter": int(winters[k]),
                "best_params": params,
                "best_iteration": best_iter,
                "n_estimators_refit": n_est_refit,
                "inner_eval_rmse": inner_score,
                "outer_rmse": _rmse(y_va_true, y_va_pred),
                "outer_mae": float(mean_absolute_error(y_va_true, y_va_pred)),
                "elapsed_s": round(time.time() - t0, 1),
            }
            cv_results.append(fold_res)
            if use_fold_ckpt:
                _save_fold_checkpoint(out_dirs, station, k, current_context, fold_res)
            print(f"[{station}]   fold {k + 1}/{len(folds)} "
                  f"(winter {winters[k]}) → RMSE_out = {fold_res['outer_rmse']:.4f} "
                  f"[n_est = {best_iter}, {fold_res['elapsed_s'] / 60:.0f} min]")

    # ----- Final model: tuning on the whole training set -----
    print(f"[{station}] final tuning on the whole training set")
    final_model, final_params, final_score, final_best_iter, final_n_est = _tune(
        X_train, y_train, w_train, cfg)

    joblib.dump(final_model, out_dirs["models"] / f"XGB_{station}.joblib")
    with open(out_dirs["reports"] / f"cv_{station}.json", "w") as f:
        json.dump({
            "cv_folds":                 cv_results,
            "final_params":             final_params,
            "final_best_iteration":     final_best_iter,
            "final_n_estimators_refit": final_n_est,
            "final_inner_eval_rmse":    final_score,
            "cache_context":            current_context,
        }, f, indent=2, default=str)

    # cv_{station}.json + model are enough for the cache: checkpoints redundant.
    if use_fold_ckpt:
        _clear_fold_checkpoints(out_dirs, station)

    return final_model


# ---------------------------------------------------------------------------
# Evaluation on the test block
# ---------------------------------------------------------------------------

def _evaluate_block(station, block_label, df_block, predictors, model,
                    surge_obs_col, surge_mod_col, target, q95_train, cfg,
                    out_dirs):
    """Overall scores, events and figures for one block.

    block_label is used as output file suffix (e.g. scores_test_BREST.json).
    Returns the dict of aggregated scores.
    """
    X_block = df_block[predictors].values
    y_block = df_block[target].values
    obs = df_block[surge_obs_col]
    mod = df_block[surge_mod_col]
    corrected = pd.Series(model.predict(X_block) + mod.values,
                          index=df_block.index, name="corrected")

    high = obs > q95_train
    nan = float("nan")
    agg = {
        "block":       block_label,
        "start":       str(df_block.index.min()),
        "end":         str(df_block.index.max()),
        "n":           int(len(df_block)),
        "bias_raw":    float(np.mean(obs - mod)),
        "bias_corr":   float(np.mean(obs - corrected)),
        "mae_raw":     float(mean_absolute_error(obs, mod)),
        "mae_corr":    float(mean_absolute_error(obs, corrected)),
        "rmse_raw":    _rmse(obs, mod),
        "rmse_corr":   _rmse(obs, corrected),
        "bias_raw_q":  float(np.mean(obs[high] - mod[high])) if high.any() else nan,
        "bias_corr_q": float(np.mean(obs[high] - corrected[high])) if high.any() else nan,
        "rmse_raw_q":  _rmse(obs[high], mod[high]) if high.any() else nan,
        "rmse_corr_q": _rmse(obs[high], corrected[high]) if high.any() else nan,
    }

    with open(out_dirs["reports"] / f"scores_{block_label}_{station}.json", "w") as f:
        json.dump({"agg": agg, "q95_train": q95_train}, f, indent=2)

    print(f"[{station}] {block_label} scores: "
          f"{df_block.index.min()} → {df_block.index.max()} "
          f"({len(df_block)} time steps)")
    print(tabulate(pd.DataFrame([agg]), headers="keys",
                   tablefmt="pretty", floatfmt=".3f", showindex=False))

    # ----- Events -----
    cadence = cfg.get("cadence_minutes", 10)
    step = pd.Timedelta(minutes=cadence)
    events = detect_events(
        obs, threshold=q95_train,
        min_duration_h=cfg["events"]["min_duration_h"],
        merge_gap_h=cfg["events"]["merge_gap_h"],
        cadence_minutes=cadence,
    )
    per_event_df = pd.DataFrame([score_event(t0, t1, obs, mod, corrected, step)
                                 for t0, t1 in events])
    if not per_event_df.empty:
        per_event_df = per_event_df.sort_values("obs_peak", ascending=False)
        per_event_df.to_csv(
            out_dirs["reports"] / f"events_{block_label}_{station}.csv", index=False)
        print(f"[{station}] {len(events)} events ({block_label}, "
              f"threshold obs > {q95_train:.2f} m)")
        cols_show = ["start", "obs_peak", "mod_peak", "corr_peak",
                     "peak_err_raw", "peak_err_corr",
                     "peak_lag_raw_h", "peak_lag_corr_h",
                     "rmse_raw", "rmse_corr"]
        print(tabulate(per_event_df[cols_show], headers="keys",
                       tablefmt="pretty", floatfmt=".3f", showindex=False))

    make_diagnostic_plots(
        station, model, predictors, X_block, y_block, obs, mod, corrected,
        events, step, out_dirs["figs"], block_label,
        imp_cfg=cfg.get("importance", {}),
    )
    return agg


# ---------------------------------------------------------------------------
# Per-station pipeline
# ---------------------------------------------------------------------------

def run_station(station: str, df: pd.DataFrame, cfg: dict, out_dirs: dict):
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    # df_raw keeps every row (including missing observations) for the export
    # of the validation period.
    df_raw = df.sort_index()

    target = cfg["target"]
    surge_obs_col = cfg["columns"]["surge_obs"]
    surge_mod_col = cfg["columns"]["surge_mod"]

    # Only rows without target, surge_mod (needed to rebuild the corrected
    # surge) or surge_obs (weights, scores) are dropped. NaNs in the other
    # predictors are handled natively by XGBoost.
    essential = [c for c in (target, surge_mod_col, surge_obs_col)
                 if c in df_raw.columns]
    df = df_raw.dropna(subset=essential)
    n_dropped = len(df_raw) - len(df)
    if n_dropped:
        print(f"[{station}] targeted dropna: {n_dropped} rows removed "
              f"on {len(essential)} essential columns; {len(df)} kept "
              f"(NaNs of the other predictors left to XGBoost)")

    # Chronological split: train = [start, end_train[ ; test = [end_train, end_test[.
    end_train = pd.to_datetime(cfg["dates"]["end_train"])
    end_test = pd.to_datetime(cfg["dates"]["end_test"])
    df_train = df.loc[df.index < end_train]
    df_test = df.loc[(df.index >= end_train) & (df.index < end_test)]
    if df_train.empty or df_test.empty:
        print(f"[{station}] empty training or test block — skipped")
        return None

    # Predictors: every column except the target and the columns that would
    # leak it (drop_predictors).
    drop = set(cfg.get("drop_predictors", [])) | {target}
    predictors = [c for c in df_train.columns if c not in drop]

    X_train = df_train[predictors].values
    y_train = df_train[target].values
    w_train = compute_sample_weights(df_train[surge_obs_col].values,
                                     cfg["sample_weight"])

    # ----- Model: cache or full tuning -----
    cache_enabled = bool((cfg.get("cache", {}) or {}).get("enabled", True))
    current_context = _build_cache_context(df_train, target, predictors, cfg)
    cached = (_try_load_cache(station, current_context, out_dirs)
              if cache_enabled else None)

    if cached is not None:
        final_model, cv_data = cached
        print(f"[{station}] cache HIT — model reused, tuning skipped "
              f"({len(predictors)} predictors, "
              f"n_est = {cv_data.get('final_n_estimators_refit')})")
    else:
        if cache_enabled:
            print(f"[{station}] cache MISS — full tuning")
        final_model = _train_station(
            station, df_train, X_train, y_train, w_train, cfg, out_dirs,
            current_context)

    q95_train = float(np.quantile(df_train[surge_obs_col], cfg["events"]["quantile"]))
    agg_test = _evaluate_block(
        station, "test", df_test, predictors, final_model,
        surge_obs_col, surge_mod_col, target, q95_train, cfg, out_dirs,
    )

    # ----- Validation period: inference + parquet export -----
    # Period outside the train/test split, predicted with the final model for
    # downstream analyses.
    val_cfg = cfg["dates"].get("validation")
    if val_cfg:
        val_start = pd.to_datetime(val_cfg["start"])
        val_end = pd.to_datetime(val_cfg["end"])
        # Unfiltered rows (missing observations kept), except those with no
        # predictor at all.
        df_val = df_raw.loc[(df_raw.index >= val_start) & (df_raw.index < val_end)]
        df_val = df_val.dropna(subset=predictors, how="all")
        predict_and_export(
            df_val, predictors, final_model, surge_mod_col, station,
            out_dirs["predictions"], period_label="validation",
            surge_obs_col=surge_obs_col,
        )

    return {"station": station, "q95_train": q95_train, "agg_test": agg_test}


# ---------------------------------------------------------------------------
# Inference and parquet export
# ---------------------------------------------------------------------------

def predict_and_export(df_period: pd.DataFrame, predictors: list, model,
                       surge_mod_col: str, station: str, output_dir: Path,
                       period_label: str, surge_obs_col: str | None = None) -> None:
    """Predict over a period and export obs / raw / corrected to parquet.

    Columns: obs (if surge_obs_col is given, NaNs kept), Tolosa-SW,
    Tolosa-SW XGB-corrected. NaNs in the predictors are handled natively by
    XGBoost, as during training. If surge_obs_col is given, scores are
    printed over the rows where obs and surge_mod are defined.
    """
    if df_period.empty:
        print(f"[{station}] empty '{period_label}' period — no export")
        return

    surge_mod = df_period[surge_mod_col].values
    corrected_vals = model.predict(df_period[predictors].values) + surge_mod
    has_obs = surge_obs_col is not None and surge_obs_col in df_period.columns

    cols = {}
    if has_obs:
        cols["obs"] = df_period[surge_obs_col].values
    cols["Tolosa-SW"] = surge_mod
    cols["Tolosa-SW XGB-corrected"] = corrected_vals

    out = pd.DataFrame(cols, index=df_period.index)
    out.index.name = "time"
    out_path = output_dir / f"{period_label}_{station}.parquet"
    out.to_parquet(out_path)
    print(f"[{station}] '{period_label}': {len(out)} time steps exported "
          f"({df_period.index.min()} → {df_period.index.max()}) → {out_path.name}")

    if not has_obs:
        return
    obs = df_period[surge_obs_col]
    ok = obs.notna().values & np.isfinite(surge_mod) & np.isfinite(corrected_vals)
    n_ok = int(ok.sum())
    if n_ok < 2:
        print(f"[{station}] '{period_label}': too few usable time steps "
              f"({n_ok}) — no scores")
        return
    o, m, c = obs.values[ok], surge_mod[ok], corrected_vals[ok]
    print(f"[{station}] '{period_label}' scores "
          f"({df_period.index.min()} → {df_period.index.max()}, "
          f"{n_ok}/{len(obs)} usable time steps, "
          f"{int(obs.notna().sum())} with obs): "
          f"RMSE raw={_rmse(o, m):.4f} → corr={_rmse(o, c):.4f} | "
          f"bias raw={float(np.mean(o - m)):+.4f} → corr={float(np.mean(o - c)):+.4f}")


# ---------------------------------------------------------------------------
# Diagnostic figures
# ---------------------------------------------------------------------------

def make_diagnostic_plots(station, model, predictors, X, y, obs, mod, corrected,
                          events, step, fig_dir: Path, block_label: str,
                          imp_cfg: dict | None = None):
    """Feature importance, scatter plot, QQ plot and per-event time series."""
    imp_cfg = imp_cfg or {}
    sfx = f"_{block_label}"

    # --- Importance: MDI (always), permutation (optional) ---
    fi = model.feature_importances_
    sorted_mdi = np.argsort(fi)
    pos = np.arange(len(sorted_mdi)) + 0.5

    perm = None
    if imp_cfg.get("permutation", False):
        # n_jobs=1 and a single BLAS/OpenMP thread: a joblib fork combined with
        # the OpenMP threads of XGBoost can deadlock.
        with threadpool_limits(limits=1):
            perm = permutation_importance(
                model, X, y,
                n_repeats=imp_cfg.get("perm_n_repeats", 10),
                max_samples=imp_cfg.get("perm_max_samples", 1.0),
                random_state=42, n_jobs=1,
            )

    n_panels = 1 + int(perm is not None)
    with PdfPages(fig_dir / f"importance{sfx}_{station}.pdf") as pdf:
        fig = plt.figure(figsize=(9 * n_panels, max(8, 0.25 * len(predictors))))

        ax = plt.subplot(1, n_panels, 1)
        ax.barh(pos, fi[sorted_mdi], align="center")
        ax.set_yticks(pos)
        ax.set_yticklabels(np.array(predictors)[sorted_mdi], fontsize=7)
        ax.set_title("Feature importance (MDI)")

        if perm is not None:
            sorted_perm = perm.importances_mean.argsort()
            ax = plt.subplot(1, n_panels, 2)
            ax.boxplot(perm.importances[sorted_perm].T, vert=False,
                       tick_labels=np.array(predictors)[sorted_perm])
            ax.tick_params(axis="y", labelsize=7)
            ax.set_title(f"Permutation importance ({block_label} block)")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # --- Scatter plot obs vs model ---
    sns.set_theme()
    df_sc = pd.concat([
        pd.DataFrame({"obs": obs.values, "model": mod.values, "kind": "Tolosa raw"}),
        pd.DataFrame({"obs": obs.values, "model": corrected.values, "kind": "XGB corrected"}),
    ], ignore_index=True)

    with PdfPages(fig_dir / f"scatter{sfx}_{station}.pdf") as pdf:
        g = sns.relplot(data=df_sc, x="obs", y="model", hue="kind",
                        height=8, linewidth=0, alpha=0.5)
        plt.axline((0, 0), slope=1, color="black")
        tmin = float(min(df_sc["obs"].min(), df_sc["model"].min(), 0.0))
        tmax = float(max(df_sc["obs"].max(), df_sc["model"].max()))
        plt.xlim(tmin, tmax * 1.05)
        plt.ylim(tmin, tmax * 1.05)
        g.fig.suptitle(f"{station} — observed vs modelled surge")
        g.fig.subplots_adjust(top=0.92)
        pdf.savefig(g.fig)
        plt.close(g.fig)

    # --- QQ plot ---
    _, oq = scipy.stats.probplot(obs.values, fit=False)
    _, mq = scipy.stats.probplot(mod.values, fit=False)
    _, cq = scipy.stats.probplot(corrected.values, fit=False)
    df_qq = pd.concat([
        pd.DataFrame({"obs_q": oq, "model_q": mq, "kind": "Tolosa raw"}),
        pd.DataFrame({"obs_q": oq, "model_q": cq, "kind": "XGB corrected"}),
    ], ignore_index=True)

    with PdfPages(fig_dir / f"qqplot{sfx}_{station}.pdf") as pdf:
        g = sns.relplot(data=df_qq, x="obs_q", y="model_q", hue="kind",
                        height=8, linewidth=0)
        plt.axline((0, 0), slope=1, color="black")
        g.fig.suptitle(f"{station} — QQ plot")
        g.fig.subplots_adjust(top=0.92)
        pdf.savefig(g.fig)
        plt.close(g.fig)

    # --- Per-event time series (one page per event, by decreasing obs peak) ---
    if not events:
        return
    sns.set_theme(style="whitegrid")
    order = sorted(range(len(events)),
                   key=lambda i: -_between(obs, *events[i], step).max())

    with PdfPages(fig_dir / f"events{sfx}_{station}.pdf") as pdf:
        for rank, idx in enumerate(order):
            t0, t1 = events[idx]
            # ±25 % of the event duration as context.
            pad = int(0.25 * ((t1 - t0) / step + 1)) * step
            o = obs[(obs.index >= t0 - pad) & (obs.index <= t1 + pad)]
            m = mod[(mod.index >= t0 - pad) & (mod.index <= t1 + pad)]
            c = corrected[(corrected.index >= t0 - pad) & (corrected.index <= t1 + pad)]

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(o.index, o.values, color="black", lw=1.6, label="Obs")
            ax.plot(m.index, m.values, color="tab:blue", lw=1.2, label="Tolosa")
            ax.plot(c.index, c.values, color="tab:red", lw=1.2, label="XGB corrected")
            ax.axvspan(t0, t1, color="orange", alpha=0.1)
            ax.set_title(
                f"{station} — event #{rank + 1} "
                f"({t0.strftime('%Y-%m-%d %H:%M')} → "
                f"{t1.strftime('%Y-%m-%d %H:%M')}) — "
                f"obs peak = {_between(obs, t0, t1, step).max():.2f} m"
            )
            ax.set_ylabel("surge (m)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            fig.autofmt_xdate()
            pdf.savefig(fig)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-station summary
# ---------------------------------------------------------------------------

def _write_summary(rows: list, path: Path, partial: bool) -> pd.DataFrame:
    """Write the multi-station summary.

    Full run: the file is rewritten. Partial run (-s): only the rows of the
    stations processed in this run are replaced; the other rows are kept.
    """
    summary = pd.DataFrame(rows)
    if partial and path.exists():
        previous = pd.read_csv(path)
        if "block" in previous.columns:
            previous = previous[previous["block"] == "test"]
        previous = previous[~previous["station"].isin(summary["station"])]
        summary = pd.concat([previous.reindex(columns=summary.columns), summary],
                            ignore_index=True)
    front = [c for c in ["station", "block", "start", "end", "n", "q95_train"]
             if c in summary.columns]
    summary = summary[front + [c for c in summary.columns if c not in front]]
    summary = summary.sort_values("station").reset_index(drop=True)
    summary.to_csv(path, index=False)
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Line-buffered output: under SLURM, stdout redirected to a file is
    # block-buffered, which can make the job look stuck.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        pass  # stream replaced by an object without reconfigure()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable the cache and the checkpoints: re-tune every station. "
             "Overrides cache.enabled from the YAML.",
    )
    parser.add_argument(
        "-s", "--stations", nargs="+", default=None, metavar="STATION",
        help="Restrict the run to these stations (name or alias from the "
             "stations.json registry, case- and separator-insensitive: "
             "'boulogne-sur-mer' = 'BOULOGNE_SUR_MER'). Default: every CSV "
             "file in paths.enriched_csv_dir.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.no_cache:
        cfg.setdefault("cache", {})["enabled"] = False
        print("[CLI] cache.enabled = False (forced re-tuning)")

    np.random.seed(cfg["random_state"])

    out_root = Path(cfg["paths"]["output_root"])
    out_dirs = {
        "models":      out_root / "models",
        "figs":        out_root / "figs",
        "reports":     out_root / "reports",
        "predictions": out_root / "predictions",
        "checkpoints": out_root / "reports" / "checkpoints",
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    data = station_io.read_station_csvs(cfg["paths"]["enriched_csv_dir"],
                                        args.stations)
    if not data:
        raise SystemExit(f"No station read from {cfg['paths']['enriched_csv_dir']}")
    print(f"[stations] {len(data)} station(s): {sorted(data)}")

    all_summaries = []
    for station, df in data.items():
        print(f"\n{'=' * 24} {station} {'=' * 24}")
        try:
            res = run_station(station, df, cfg, out_dirs)
        except Exception as exc:
            print(f"[{station}] FAILED: {exc!r}")
            continue
        finally:
            # Release figures and memory between stations.
            plt.close("all")
            gc.collect()
        if res is not None:
            base = {"station": res["station"], "q95_train": res["q95_train"]}
            all_summaries.append({**base, **res["agg_test"]})

    if all_summaries:
        summary = _write_summary(all_summaries,
                                 out_dirs["reports"] / "summary_all_stations.csv",
                                 partial=args.stations is not None)
        print("\n=== multi-station summary ===")
        print(tabulate(summary, headers="keys",
                       tablefmt="pretty", floatfmt=".3f", showindex=False))


if __name__ == "__main__":
    main()
