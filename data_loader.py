"""
========================================================================
data_loader.py — Centralised Data Loading & Caching Layer
========================================================================
Sits between raw storage (Excel) and all analysis modules. Responsible
for every read, write, deduplication, and format conversion in the
project. No other module reads files directly — they call this module.

Responsibilities
----------------
  - Load price data from Excel, write Parquet mirror on first load or
    when the Excel file is newer than the Parquet file.
  - Load sentiment history from the Excel sentiment_history sheet and
    mirror it to sentiment_history.parquet.
  - Serve cached DataFrames so repeated calls within one process do not
    re-read files.
  - Enforce deduplication on every load.
  - Provide typed, clean DataFrames with consistent dtypes to all callers.

Storage layout
--------------
  most_active_stocks_dataset.xlsx   — human-readable source of truth
  most_active_stocks_dataset.parquet— fast-read mirror of Sheet1
  sentiment_history.parquet         — fast-read mirror of sentiment_history sheet

Parquet strategy
----------------
  Excel is written by the scraper after every daily run. On each
  dashboard or historical_metrics run, load_price_data() compares the
  mtime of the Excel file against the Parquet file. If Excel is newer,
  it rebuilds the Parquet mirror. Otherwise it reads Parquet directly.
  This gives openpyxl overhead only once per day instead of on every run.

Pip dependencies
----------------
  pip install pandas openpyxl pyarrow
========================================================================
"""

import logging
import os
from typing import Optional

import pandas as pd

from config import PATHS
from sentiment import SENT_SHEET, normalise_dates, empty_sentiment_df

logger = logging.getLogger(__name__)

# ── File paths — single authoritative definition ──────────────────────
EXCEL_FILE        = PATHS.EXCEL_FILE
PRICE_PARQUET     = PATHS.PRICE_PARQUET
SENTIMENT_PARQUET = PATHS.SENTIMENT_PARQUET
TECHNICAL_PARQUET = PATHS.TECHNICAL_PARQUET
FORECAST_PARQUET  = PATHS.FORECAST_PARQUET
SIGNAL_PARQUET    = PATHS.SIGNAL_PARQUET
ANOMALY_PARQUET   = PATHS.ANOMALY_PARQUET

# ── Required columns and dtypes for each dataset ─────────────────────
_PRICE_DTYPES: dict = {
    "Symbol":       "str",
    "Company Name": "str",
    "Price":        "float64",
    "Change":       "float64",
    "% Change":     "float64",
}

_SENTIMENT_DTYPES: dict = {
    "Symbol":          "str",
    "Sentiment_Score": "float64",
    "Headlines_Used":  "int32",
    "Sentiment_Label": "str",
    "Top_Headline":    "str",
}

# ── In-process cache — invalidated when files change ─────────────────
_cache: dict = {}

_EMPTY_PHASE2_COLUMNS = {
    "technical": [
        "Date", "avg_pct", "volatility", "breadth_pct", "sma_avg_7d",
        "sma_avg_30d", "ema_avg_7d", "ema_avg_30d", "mom_avg_7d",
        "vol_zscore_7d", "breadth_sma_7d", "breadth_ema_7d",
        "breadth_mom_7d", "signal_strength", "trend_7_30",
    ],
    "forecast": ["Date", "metric", "forecast", "lower_bound", "upper_bound", "method"],
    "signal": ["Date", "Symbol", "signal_type", "direction", "strength", "score", "reason"],
    "anomaly": ["Date", "Symbol", "anomaly_type", "severity", "z_score", "value", "reason"],
}

_PHASE2_NUMERIC_COLS = {
    "technical": [
        "avg_pct", "volatility", "breadth_pct", "sma_avg_7d", "sma_avg_30d",
        "ema_avg_7d", "ema_avg_30d", "mom_avg_7d", "vol_zscore_7d",
        "breadth_sma_7d", "breadth_ema_7d", "breadth_mom_7d", "signal_strength",
    ],
    "forecast": ["forecast", "lower_bound", "upper_bound"],
    "signal": ["strength", "score"],
    "anomaly": ["z_score", "value"],
}


# ══════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════

def _mtime(path: str) -> float:
    """Return file modification time or 0.0 if file does not exist."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _parquet_is_fresh(excel_path: str, parquet_path: str) -> bool:
    """
    Return True if the Parquet file exists and is at least as new as
    the Excel file, meaning no rebuild is needed.
    """
    return (
        os.path.exists(parquet_path)
        and _mtime(parquet_path) >= _mtime(excel_path)
    )


def _makedirs_for(path: str) -> None:
    """
    Create parent directories for a file path.
    Guards against empty dirname when file lives in the project root.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _enforce_price_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast price DataFrame columns to canonical dtypes."""
    df["Date"] = pd.to_datetime(df["Date"])
    for col, dtype in _PRICE_DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype, errors="ignore")
    return df


def _load_parquet_cache(path: str, cache_key: str, columns: list[str]) -> pd.DataFrame:
    """Load an optional Phase 2 parquet cache with a safe empty fallback."""
    if cache_key in _cache:
        return _cache[cache_key]

    if not os.path.exists(path):
        logger.info("data_loader: optional cache not found: %s", path)
        df = pd.DataFrame(columns=columns)
        _cache[cache_key] = df
        return df

    try:
        df = pd.read_parquet(path, engine="pyarrow")
    except Exception as exc:
        logger.warning("data_loader: could not read %s (%s). Returning empty.", path, exc)
        df = pd.DataFrame(columns=columns)

    missing = [c for c in columns if c not in df.columns]
    if missing:
        logger.warning(
            "data_loader: %s cache missing columns %s; adding empty typed columns.",
            cache_key, missing,
        )
        for col in missing:
            df[col] = pd.NA

    df = df[[c for c in columns if c in df.columns] + [c for c in df.columns if c not in columns]]

    if "Date" in df.columns and not df.empty:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in _PHASE2_NUMERIC_COLS.get(cache_key, []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    _cache[cache_key] = df
    return df


def _save_parquet_cache(df: pd.DataFrame, path: str, cache_key: str) -> None:
    """Persist a Phase 2 cache and invalidate its in-process copy."""
    out = df.copy()
    for col in _EMPTY_PHASE2_COLUMNS.get(cache_key, []):
        if col not in out.columns:
            out[col] = pd.NA
    if "Date" in out.columns and not out.empty:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for col in _PHASE2_NUMERIC_COLS.get(cache_key, []):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    _makedirs_for(path)
    out.to_parquet(path, index=False, engine="pyarrow")
    _cache.pop(cache_key, None)
    logger.info("data_loader: saved %d rows to %s.", len(out), path)


def _enforce_sentiment_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast sentiment DataFrame columns to canonical dtypes."""
    df = normalise_dates(df)  # Date → plain "YYYY-MM-DD" string
    for col, dtype in _SENTIMENT_DTYPES.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, TypeError):
                pass
    return df


def _dedup(df: pd.DataFrame, label: str = "rows") -> pd.DataFrame:
    """Drop duplicate Date+Symbol rows and warn if any were removed."""
    before = len(df)
    df = df.drop_duplicates(subset=["Date", "Symbol"]).reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        logger.warning("data_loader: removed %d duplicate %s.", removed, label)
    return df


# Convenience aliases kept for call-site readability
def _dedup_price(df: pd.DataFrame) -> pd.DataFrame:
    return _dedup(df, "Date+Symbol price rows")


def _dedup_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    return _dedup(df, "Date+Symbol sentiment rows")


# ══════════════════════════════════════════════════════════════════════
# PARQUET MIRROR BUILDERS
# ══════════════════════════════════════════════════════════════════════

def _build_price_parquet() -> pd.DataFrame:
    """
    Read price data from Excel, clean it, and write the Parquet mirror.
    Returns the cleaned DataFrame.
    """
    logger.info("data_loader: reading price data from Excel (building Parquet mirror)...")
    df = pd.read_excel(EXCEL_FILE, engine="openpyxl")
    df = _enforce_price_dtypes(df)
    df = _dedup_price(df)
    df.to_parquet(PRICE_PARQUET, index=False, engine="pyarrow")
    logger.info(
        "data_loader: Parquet mirror written — %d rows, %d days.",
        len(df), df["Date"].nunique(),
    )
    return df


def _build_sentiment_parquet() -> pd.DataFrame:
    """
    Read sentiment history from Excel sheet, clean it, and write the
    Parquet mirror. Returns the cleaned DataFrame.
    """
    logger.info("data_loader: reading sentiment history from Excel sheet...")
    try:
        df = pd.read_excel(EXCEL_FILE, sheet_name=SENT_SHEET, engine="openpyxl")
    except Exception:
        logger.info("data_loader: no sentiment_history sheet found — returning empty.")
        return empty_sentiment_df()

    df = _enforce_sentiment_dtypes(df)
    df = _dedup_sentiment(df)
    df.to_parquet(SENTIMENT_PARQUET, index=False, engine="pyarrow")
    logger.info(
        "data_loader: sentiment Parquet written — %d rows, %d days.",
        len(df), df["Date"].nunique(),
    )
    return df


# ══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════

def load_price_data(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Load the full historical price dataset.

    Strategy
    --------
    1. Return from in-process cache if available and not force_rebuild.
    2. Read from Parquet if it is at least as new as the Excel file.
    3. Rebuild Parquet from Excel otherwise (happens once per day after
       the scraper runs).

    Parameters
    ----------
    force_rebuild : if True, always rebuild Parquet from Excel regardless
                    of file modification times.

    Returns
    -------
    pd.DataFrame with columns:
        Date (datetime64), Symbol, Company Name, Price, Change, % Change
    Sorted ascending by Date.
    """
    cache_key = "price"

    if not force_rebuild and cache_key in _cache:
        return _cache[cache_key]

    if not force_rebuild and _parquet_is_fresh(EXCEL_FILE, PRICE_PARQUET):
        logger.info("data_loader: loading price data from Parquet.")
        df = pd.read_parquet(PRICE_PARQUET, engine="pyarrow")
        df = _enforce_price_dtypes(df)
    else:
        df = _build_price_parquet()

    df = df.sort_values("Date").reset_index(drop=True)
    _cache[cache_key] = df
    return df


def load_sentiment_history(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Load the full sentiment history dataset.

    Strategy
    --------
    Same Parquet-first strategy as load_price_data(). The sentiment
    Parquet is rebuilt whenever the Excel file is newer.

    Returns
    -------
    pd.DataFrame with columns:
        Date (str "YYYY-MM-DD"), Symbol, Sentiment_Score,
        Sentiment_Label, Headlines_Used, Top_Headline
    Sorted ascending by Date.
    """
    cache_key = "sentiment"

    if not force_rebuild and cache_key in _cache:
        return _cache[cache_key]

    if not force_rebuild and _parquet_is_fresh(EXCEL_FILE, SENTIMENT_PARQUET):
        logger.info("data_loader: loading sentiment from Parquet.")
        df = pd.read_parquet(SENTIMENT_PARQUET, engine="pyarrow")
        df = _enforce_sentiment_dtypes(df)
    else:
        df = _build_sentiment_parquet()

    df = df.sort_values("Date").reset_index(drop=True)
    _cache[cache_key] = df
    return df


def load_latest_day(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Return only the latest trading day's price rows.

    This is the slice used by all latest-day charts (Overview, Top
    Movers, Analytics, Screener, Story). Calling this instead of
    load_price_data() and slicing manually is slightly faster because
    the Parquet read uses column pruning when possible.

    Returns
    -------
    pd.DataFrame — same schema as load_price_data(), single date.
    """
    df_full = load_price_data(force_rebuild=force_rebuild)
    latest  = df_full["Date"].max()
    return df_full[df_full["Date"] == latest].copy().reset_index(drop=True)


def load_sentiment_today(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Return only today's sentiment rows from sentiment history.

    Returns empty sentiment DataFrame if today is not yet stored.
    """
    history   = load_sentiment_history(force_rebuild=force_rebuild)
    today_str = str(pd.Timestamp.today().date())
    result    = history[history["Date"] == today_str].copy().reset_index(drop=True)
    if result.empty:
        logger.info("data_loader: no sentiment rows found for %s.", today_str)
    return result


def save_price_data(df: pd.DataFrame) -> None:
    """
    Save the full price DataFrame to both Excel (Sheet1) and Parquet.

    Called by scraper.py after building the combined DataFrame. Writing
    both formats here keeps storage logic in one place.
    """
    df = _enforce_price_dtypes(df)
    df = _dedup_price(df)
    df = df.sort_values("Date").reset_index(drop=True)

    # Excel — write only Sheet1, preserving any other sheets (e.g. sentiment_history)
    if os.path.exists(EXCEL_FILE):
        with pd.ExcelWriter(
            EXCEL_FILE, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            df.to_excel(writer, sheet_name=PATHS.PRICE_SHEET, index=False)
    else:
        df.to_excel(EXCEL_FILE, sheet_name=PATHS.PRICE_SHEET, index=False, engine="openpyxl")

    # Parquet — fast-read mirror
    df.to_parquet(PRICE_PARQUET, index=False, engine="pyarrow")

    # Invalidate cache so next load_price_data() call reflects new data
    _cache.pop("price", None)

    logger.info(
        "data_loader: saved %d price rows (%d days) to Excel + Parquet.",
        len(df), df["Date"].nunique(),
    )


def save_sentiment_history(df: pd.DataFrame) -> None:
    """
    Save the full sentiment history DataFrame to both the Excel
    sentiment_history sheet and sentiment_history.parquet.

    Called by scraper.py and dashboard.py after appending new rows.
    """
    df = _enforce_sentiment_dtypes(df)
    df = _dedup_sentiment(df)
    df = df.sort_values("Date").reset_index(drop=True)

    # Excel sheet — human-readable
    with pd.ExcelWriter(
        EXCEL_FILE, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        df.to_excel(writer, sheet_name=SENT_SHEET, index=False)

    # Parquet — fast-read mirror
    df.to_parquet(SENTIMENT_PARQUET, index=False, engine="pyarrow")

    # Invalidate cache
    _cache.pop("sentiment", None)

    logger.info(
        "data_loader: saved %d sentiment rows (%d days) to Excel + Parquet.",
        len(df), df["Date"].nunique(),
    )


def load_technical_data() -> pd.DataFrame:
    """Load cached Phase 2 technical metrics, returning an empty schema if absent."""
    return _load_parquet_cache(
        TECHNICAL_PARQUET, "technical", _EMPTY_PHASE2_COLUMNS["technical"]
    )


def save_technical_data(df: pd.DataFrame) -> None:
    """Persist Phase 2 technical metrics to parquet."""
    _save_parquet_cache(df, TECHNICAL_PARQUET, "technical")


def load_forecast_data() -> pd.DataFrame:
    """Load cached Phase 2 forecast metrics, returning an empty schema if absent."""
    return _load_parquet_cache(
        FORECAST_PARQUET, "forecast", _EMPTY_PHASE2_COLUMNS["forecast"]
    )


def save_forecast_data(df: pd.DataFrame) -> None:
    """Persist Phase 2 forecast metrics to parquet."""
    _save_parquet_cache(df, FORECAST_PARQUET, "forecast")


def load_signal_data() -> pd.DataFrame:
    """Load cached Phase 2 signal metrics, returning an empty schema if absent."""
    return _load_parquet_cache(
        SIGNAL_PARQUET, "signal", _EMPTY_PHASE2_COLUMNS["signal"]
    )


def save_signal_data(df: pd.DataFrame) -> None:
    """Persist Phase 2 signal metrics to parquet."""
    _save_parquet_cache(df, SIGNAL_PARQUET, "signal")


def load_anomaly_data() -> pd.DataFrame:
    """Load cached Phase 2 anomaly metrics, returning an empty schema if absent."""
    return _load_parquet_cache(
        ANOMALY_PARQUET, "anomaly", _EMPTY_PHASE2_COLUMNS["anomaly"]
    )


def save_anomaly_data(df: pd.DataFrame) -> None:
    """Persist Phase 2 anomaly metrics to parquet."""
    _save_parquet_cache(df, ANOMALY_PARQUET, "anomaly")


def invalidate_cache() -> None:
    """
    Clear the in-process cache. Call this if you have written new data
    and need subsequent load calls to see the updated files.
    """
    _cache.clear()
    logger.info("data_loader: cache invalidated.")


def dataset_summary() -> dict:
    """
    Return a lightweight summary dict without loading full data into
    memory. Useful for health-checks and dashboard header stats.

    Returns
    -------
    dict with keys:
        n_price_rows, n_price_days, latest_date,
        n_sentiment_rows, n_sentiment_days,
        price_parquet_exists, sentiment_parquet_exists
    """
    try:
        df_p = load_price_data()
        n_p_rows = len(df_p)
        n_p_days = df_p["Date"].nunique()
        latest   = str(df_p["Date"].max().date())
    except Exception:
        n_p_rows = 0
        n_p_days = 0
        latest   = "N/A"

    try:
        df_s     = load_sentiment_history()
        n_s_rows = len(df_s)
        n_s_days = df_s["Date"].nunique()
    except Exception:
        n_s_rows = 0
        n_s_days = 0

    return {
        "n_price_rows":           n_p_rows,
        "n_price_days":           n_p_days,
        "latest_date":            latest,
        "n_sentiment_rows":       n_s_rows,
        "n_sentiment_days":       n_s_days,
        "price_parquet_exists":   os.path.exists(PRICE_PARQUET),
        "sentiment_parquet_exists": os.path.exists(SENTIMENT_PARQUET),
    }


# ======================================================================
# HISTORICAL OHLCV DATA — LOAD / SAVE
# Parquet-first, one file per ticker under data/historical/.
# These functions mirror the pattern of the Phase 2 cache helpers but
# are organised per-symbol rather than per-metric-type.
# ======================================================================

def _ohlcv_path(symbol: str) -> str:
    """Return the canonical parquet path for a symbol's OHLCV data."""
    from config import HISTORICAL_PATHS
    return os.path.join(HISTORICAL_PATHS.OHLCV_DIR, f"{symbol.upper()}.parquet")


def _features_path(symbol: str) -> str:
    """Return the canonical parquet path for a symbol's feature data."""
    from config import HISTORICAL_PATHS
    return os.path.join(HISTORICAL_PATHS.FEATURES_DIR, f"{symbol.upper()}_features.parquet")


def _macro_path(series_id: str) -> str:
    """Return the canonical parquet path for a FRED macro series."""
    from config import HISTORICAL_PATHS
    return os.path.join(HISTORICAL_PATHS.FRED_DIR, f"{series_id}.parquet")


_OHLCV_REQUIRED_COLS: list[str] = [
    "Date", "Open", "High", "Low", "Close", "Volume"
]


def load_ohlcv(symbol: str, force_rebuild: bool = False) -> pd.DataFrame:
    """
    Load OHLCV data for a single ticker from the local parquet cache.

    Returns an empty DataFrame with the OHLCV schema if the symbol has
    not yet been fetched (safe first-run behaviour).

    Parameters
    ----------
    symbol        : ticker string, e.g. "AAPL"
    force_rebuild : bypass in-process cache (forces parquet re-read)

    Returns
    -------
    pd.DataFrame with columns: Date, Open, High, Low, Close, Volume
    Sorted ascending by Date.
    """
    cache_key = f"ohlcv_{symbol.upper()}"
    if not force_rebuild and cache_key in _cache:
        return _cache[cache_key]

    path = _ohlcv_path(symbol)
    if not os.path.exists(path):
        logger.debug("data_loader: OHLCV not found for %s — run fetch_historical.", symbol)
        return pd.DataFrame(columns=_OHLCV_REQUIRED_COLS)

    try:
        df = pd.read_parquet(path, engine="pyarrow")
        _d = pd.to_datetime(df["Date"], errors="coerce")
        df["Date"] = _d.dt.tz_convert(None) if _d.dt.tz is not None else _d
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)
        _cache[cache_key] = df
        logger.info("data_loader: loaded OHLCV for %s (%d rows).", symbol, len(df))
        return df
    except Exception as exc:
        logger.warning("data_loader: OHLCV read failed for %s (%s).", symbol, exc)
        return pd.DataFrame(columns=_OHLCV_REQUIRED_COLS)


def save_ohlcv(df: pd.DataFrame, symbol: str) -> None:
    """
    Persist OHLCV data for a single ticker to parquet.

    Deduplicates on Date before writing. Invalidates the in-process
    cache so the next load reflects the new file.

    Parameters
    ----------
    df     : OHLCV DataFrame (must contain Date, Open, High, Low, Close, Volume)
    symbol : ticker string
    """
    if df is None or df.empty:
        logger.warning("data_loader: save_ohlcv called with empty DataFrame for %s.", symbol)
        return

    out = df.copy()
    _d = pd.to_datetime(out["Date"], errors="coerce")
    out["Date"] = _d.dt.tz_convert(None) if _d.dt.tz is not None else _d
    out = out.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    path = _ohlcv_path(symbol)
    _makedirs_for(path)
    out.to_parquet(path, index=False, engine="pyarrow")
    _cache.pop(f"ohlcv_{symbol.upper()}", None)
    logger.info("data_loader: saved OHLCV for %s (%d rows) → %s.", symbol, len(out), path)


def load_features(symbol: str) -> pd.DataFrame:
    """
    Load pre-computed feature/indicator DataFrame for a ticker.

    Returns empty DataFrame if features have not yet been computed.
    """
    cache_key = f"features_{symbol.upper()}"
    if cache_key in _cache:
        return _cache[cache_key]

    path = _features_path(symbol)
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        df = pd.read_parquet(path, engine="pyarrow")
        _d = pd.to_datetime(df["Date"], errors="coerce")
        df["Date"] = _d.dt.tz_convert(None) if _d.dt.tz is not None else _d
        df = df.sort_values("Date").reset_index(drop=True)
        _cache[cache_key] = df
        return df
    except Exception as exc:
        logger.warning("data_loader: features read failed for %s (%s).", symbol, exc)
        return pd.DataFrame()


def save_features(df: pd.DataFrame, symbol: str) -> None:
    """Persist feature/indicator DataFrame for a ticker to parquet."""
    if df is None or df.empty:
        return
    path = _features_path(symbol)
    _makedirs_for(path)
    df.copy().to_parquet(path, index=False, engine="pyarrow")
    _cache.pop(f"features_{symbol.upper()}", None)
    logger.info("data_loader: saved features for %s (%d rows).", symbol, len(df))


def load_macro(series_alias: str) -> pd.DataFrame:
    """
    Load a FRED macro series from local parquet cache.

    Parameters
    ----------
    series_alias : the local alias key from FRED_SERIES dict
                   (e.g. "fed_funds_rate", "vix")

    Returns
    -------
    pd.DataFrame with columns: Date, value, series_id
    Empty DataFrame if not yet fetched.
    """
    cache_key = f"macro_{series_alias}"
    if cache_key in _cache:
        return _cache[cache_key]

    path = _macro_path(series_alias)
    if not os.path.exists(path):
        logger.info("data_loader: macro series '%s' not found — run fetch_historical.", series_alias)
        return pd.DataFrame(columns=["Date", "value", "series_id"])

    try:
        df = pd.read_parquet(path, engine="pyarrow")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)
        _cache[cache_key] = df
        return df
    except Exception as exc:
        logger.warning("data_loader: macro read failed for '%s' (%s).", series_alias, exc)
        return pd.DataFrame(columns=["Date", "value", "series_id"])


def save_macro(df: pd.DataFrame, series_alias: str) -> None:
    """Persist a FRED macro series to parquet."""
    if df is None or df.empty:
        return
    path = _macro_path(series_alias)
    _makedirs_for(path)
    df.copy().to_parquet(path, index=False, engine="pyarrow")
    _cache.pop(f"macro_{series_alias}", None)
    logger.info("data_loader: saved macro '%s' (%d rows).", series_alias, len(df))


def list_available_ohlcv() -> list[str]:
    """
    Return a list of ticker symbols that have local OHLCV parquet files.

    Useful for the dashboard and pipeline orchestrator to know which
    symbols have historical data without attempting to load each file.
    """
    from config import HISTORICAL_PATHS
    ohlcv_dir = HISTORICAL_PATHS.OHLCV_DIR
    if not os.path.isdir(ohlcv_dir):
        return []
    return [
        f.replace(".parquet", "").upper()
        for f in os.listdir(ohlcv_dir)
        if f.endswith(".parquet")
    ]


def list_available_macro() -> list[str]:
    """Return a list of macro series aliases that have local parquet files."""
    from config import HISTORICAL_PATHS
    macro_dir = HISTORICAL_PATHS.FRED_DIR
    if not os.path.isdir(macro_dir):
        return []
    return [
        f.replace(".parquet", "")
        for f in os.listdir(macro_dir)
        if f.endswith(".parquet")
    ]