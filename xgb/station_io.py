# -*- coding: utf-8 -*-
"""
station_io.py — station identification and reading of enriched CSV files.

The same station may appear under several file names (aliases, e.g.
"BAYONNE-BOUCAU" / "BOUCAU-BAYONNE"). The stations.json registry maps each
alias to a canonical identifier. This identifier names every output of
train_xgb.py (models, reports, figures, cache).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

REGISTRY_PATH = Path(__file__).with_name("stations.json")

# Replacements applied after upper-casing. Regex alternatives are tried in
# this order: "ST-" is handled before "-".
_SYMBOLS = {"ST_": "SAINT_", "ST-": "SAINT-", "'": "", " ": "_", "/": "_",
            "-": "_", "(": "_", ")": ""}
_SYMBOLS_RE = re.compile("|".join(re.escape(k) for k in _SYMBOLS))

# Naming prefixes / suffixes stripped from file names (various data sources:
# tide gauges, WW3, CMEMS…).
_PREFIXES = [
    "ssh_", "lssh_", "slev_", "Candhis_", "ww3.", "ww3_Hs_", "ww3_T0m1_",
    "cmems_mod_ibi_phy_anfc_0.027deg-3D_P1D-m_multi-vars-",
    "cmems_mod_ibi_phy-ssh_anfc_detided-0.027deg_P1D-m-",
    "cmems_mod_glo_phy_anfc_0.083deg_P1D-m-",
    "cmems_mod_ibi_phy_anfc_0.027deg-2D_PT15M-i-",
    "cmems_mod_ibi_phy_anfc_0.027deg-2D_PT1H-m-",
    "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m-",
    "cmems_mod_glo_phy_anfc_merged-sl_PT1H-i-",
    "cmems_mod_glo_phy_anfc_0.083deg_static_1741768856637-",
    "cmems_mod_ibi_phy_anfc_0.027deg-3D_static_mdtPT15M-i-",
    "NEATL36_1h-m_2DT-oce_20240915-20240915PT1H-m-",
]
_SUFFIXES = [
    "_ms.nc", "_full.nc", ".nc", ".hfs", ".nc4", ".csv", ".pts",
    "_arch", "_spec", "_tab", "_30m_averaged", "-30m_averaged",
    "_surface", "-surface", "_from_grib", "_near", "_visu", "_ww3",
    *[f"_test{i}" for i in range(10)],
    "_refmar_imp.job_histGBregressor",
]
# Trailing patterns: date range, date, year, digit.
_TRAILING_PATTERNS = [
    r"_[0-9]{4}-[0-1][0-9]-[0-3][0-9]_[0-9]{4}-[0-1][0-9]-[0-3][0-9]$",
    r"_[0-9]{4}[0-1][0-9][0-3][0-9]",
    r"_[0-9]{4}",
    r"[0-9]",
]


def normalize_name(name: str) -> str:
    """Upper case, separators unified as "_", "ST" expanded to "SAINT"."""
    return _SYMBOLS_RE.sub(lambda m: _SYMBOLS[m.group(0)], name.upper())


@lru_cache(maxsize=None)
def _alias_map() -> dict:
    """{normalized alias: normalized canonical identifier}."""
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    return {normalize_name(alias): normalize_name(station)
            for station, info in registry.items()
            for alias in info.get("names", [])}


def _strip_name(basename: str) -> str:
    """Raw station name extracted from a file name."""
    res = basename
    for p in _PREFIXES:
        res = res.removeprefix(p)
    for s in _SUFFIXES:
        res = res.removesuffix(s)
    for pattern in _TRAILING_PATTERNS:
        m = re.search(pattern, res)
        if m is not None:
            res = res.removesuffix(m.group())
    return res


def canonical(name: str) -> str:
    """Canonical identifier of a station name or alias (e.g. 'boulogne-sur-mer'
    → 'BOULOGNE_SUR_MER'). Name unknown to the registry → normalized name."""
    key = normalize_name(name)
    return _alias_map().get(key, key)


def station_id(path) -> str:
    """Canonical identifier of the station a file belongs to.

    Name absent from the registry → name extracted from the file, unchanged.
    """
    name = _strip_name(Path(path).name)
    return _alias_map().get(normalize_name(name)) or name


def read_station_csvs(data_dir, stations=None) -> dict:
    """Read the CSV files of data_dir → {station identifier: DataFrame}.

    The CSV files must have a `time` column; the index is sorted and
    duplicated timestamps are dropped (first occurrence kept).
    stations: names or aliases to keep (case- and separator-insensitive);
    None → every station in the directory.
    """
    requested = None
    if stations is not None:
        requested = {canonical(s) for s in stations}

    data = {}
    for path in sorted(Path(data_dir).glob("*.csv")):
        sid = station_id(path)
        if requested is not None and canonical(sid) not in requested:
            continue
        if sid in data:
            print(f"[stations] {path.name}: identifier {sid} already read, file skipped")
            continue
        df = pd.read_csv(path, header=0, engine="c")
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        data[sid] = df[~df.index.duplicated(keep="first")]

    if requested is not None:
        missing = requested - {canonical(s) for s in data}
        if missing:
            print(f"[stations] not found: {sorted(missing)}")
    return data
