"""
========================================================================
pipelines/preprocess.py — OHLCV Data Preprocessing
========================================================================
Cleans and normalises raw OHLCV DataFrames before indicator computation
and feature engineering. Called by the indicator and feature pipelines;
never called directly by dashboard or analytics modules.

Design
------
- Pure transformation layer. No I/O, no network calls.
- All functions accept a DataFrame and return a DataFrame.
- Designed to be chained: preprocess_ohlcv() runs the full standard
  sequence, or callers can apply individual steps selectively.

Integration
-----------
  pipelines/indicators.py calls preprocess_ohlcv() before computing
  indicators. The output is a clean OHLCV DataFrame ready for
  technical_indicators.compute_ohlcv_indicators().
========================================================================
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns that must be present and positive in a valid OHLCV frame
_REQUIRED_NUMERIC = ["Open", "High", "Low", "Close", "Volume"]


def validate_ohlcv(df: pd.DataFrame, symbol: str = "") -> tuple[bool, list[str]]:
    """
    Validate a raw OHLCV DataFrame before preprocessing.

    Returns (is_valid, list_of_issues). Does not raise.

    Checks
    ------
    - Required columns present
    - No all-NaN Close column
    - No negative Close prices
    - High >= Low on all rows
    - Minimum row count (HIST_FETCH.MIN_OHLCV_ROWS)
    """
    from config import HIST_FETCH
    issues = []

    missing = [c for c in _REQUIRED_NUMERIC if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
        return False, issues

    if df["Close"].isna().all():
        issues.append("Close column is all NaN")
        return False, issues

    if (df["Close"].dropna() <= 0).any():
        n = int((df["Close"].dropna() <= 0).sum())
        issues.append(f"{n} row(s) with Close <= 0")

    if "High" in df.columns and "Low" in df.columns:
        inversion = (df["High"] < df["Low"]).sum()
        if inversion:
            issues.append(f"{inversion} row(s) where High < Low")

    if "Volume" in df.columns:
        zero_vol = int((df["Volume"].fillna(0) == 0).sum())
        if zero_vol:
            issues.append(f"{zero_vol} row(s) with zero volume (possible halt/data gap)")

    if "Open" in df.columns:
        prev_close = df["Close"].shift(1)
        gap_pct    = ((df["Open"] - prev_close) / prev_close.replace(0, np.nan)).abs()
        big_gaps   = int((gap_pct > 0.15).sum())
        if big_gaps:
            issues.append(f"{big_gaps} row(s) with overnight gap >15% (split or data error?)")

    if len(df) < HIST_FETCH.MIN_OHLCV_ROWS:
        issues.append(
            f"Only {len(df)} rows — minimum {HIST_FETCH.MIN_OHLCV_ROWS} required "
            "for meaningful indicators"
        )

    is_valid = not any("Missing" in i or "all NaN" in i for i in issues)
    if issues:
        label = symbol or "unknown"
        for iss in issues:
            logger.warning("preprocess [%s]: %s", label, iss)

    return is_valid, issues


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard cleaning steps to a raw OHLCV DataFrame.

    Steps
    -----
    1. Parse and normalise Date column.
    2. Cast numeric columns to float64.
    3. Drop rows where Close is NaN.
    4. Clip negative Open/High/Low/Volume to zero (data errors).
    5. Fix High < Low inversions by swapping.
    6. Remove duplicate Date rows (keep last).
    7. Sort ascending by Date.
    """
    df = df.copy()

    # 1. Date normalisation
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    elif df.index.name == "Date" or hasattr(df.index, "date"):
        df = df.reset_index()
        df = df.rename(columns={"index": "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()

    # 2. Numeric casting
    for col in _REQUIRED_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Drop rows with no Close price
    df = df.dropna(subset=["Close"])

    # 4. Clip negatives
    for col in ["Open", "High", "Low"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].clip(lower=0)

    # 5. Fix High < Low inversions
    if "High" in df.columns and "Low" in df.columns:
        inv = df["High"] < df["Low"]
        if inv.any():
            df.loc[inv, ["High", "Low"]] = df.loc[inv, ["Low", "High"]].values

    # 6. Deduplicate on Date
    if "Date" in df.columns:
        df = df.drop_duplicates(subset=["Date"], keep="last")

    # 7. Sort
    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)

    return df


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add percentage return columns to a clean OHLCV DataFrame.

    Columns added
    -------------
    returns_pct      : daily % change of Close
    log_returns      : log(Close / prev_Close)
    gap_pct          : overnight gap % (Open vs prev_Close)
    intraday_range_pct: (High - Low) / prev_Close * 100
    """
    df = df.copy()
    close = df["Close"]

    df["returns_pct"]       = close.pct_change().mul(100).round(4)
    df["log_returns"]       = np.log(close / close.shift(1)).round(6)

    if "Open" in df.columns:
        df["gap_pct"] = ((df["Open"] - close.shift(1)) / close.shift(1).replace(0, np.nan) * 100).round(4)

    if "High" in df.columns and "Low" in df.columns:
        df["intraday_range_pct"] = (
            (df["High"] - df["Low"]) / close.shift(1).replace(0, np.nan) * 100
        ).round(4)

    return df


def flag_outliers(
    df: pd.DataFrame,
    column: str = "returns_pct",
    n_sigma: float = None,
    window: int = None,
) -> pd.DataFrame:
    """
    Flag rows where `column` deviates more than n_sigma standard deviations
    from a rolling mean baseline. Adds a boolean `{column}_outlier` column.

    Uses a rolling window so detection adapts to regime changes rather than
    comparing against the full-history mean (which would miss regime shifts).
    Window and sigma defaults come from config.HIST_FETCH to avoid hardcoding.

    Parameters
    ----------
    df      : OHLCV DataFrame after compute_returns()
    column  : column to test (default: "returns_pct")
    n_sigma : detection threshold in std devs (default: HIST_FETCH.OUTLIER_N_SIGMA)
    window  : rolling baseline window in days (default: HIST_FETCH.OUTLIER_WINDOW)
    """
    from config import HIST_FETCH
    n_sigma = n_sigma if n_sigma is not None else HIST_FETCH.OUTLIER_N_SIGMA
    window  = window  if window  is not None else HIST_FETCH.OUTLIER_WINDOW

    df = df.copy()
    if column not in df.columns:
        logger.debug("flag_outliers: column '%s' not found — skipping.", column)
        return df

    roll_mean = df[column].rolling(window, min_periods=5).mean()
    roll_std  = df[column].rolling(window, min_periods=5).std().replace(0, np.nan)
    z         = ((df[column] - roll_mean) / roll_std).abs()
    flag_col  = f"{column}_outlier"
    df[flag_col] = z > n_sigma
    n_flagged = int(df[flag_col].sum())
    if n_flagged:
        logger.info(
            "flag_outliers: %d outlier(s) in '%s' (>%.1fσ, %dd rolling window).",
            n_flagged, column, n_sigma, window,
        )
    return df


def detect_splits(df: pd.DataFrame) -> list:
    """
    Return a list of dates where a likely stock split, reverse split, or
    unadjusted corporate action was detected.

    Detection criterion: close-to-close ratio < 0.60 or > 1.67 (a >40%
    single-day move). That magnitude is extremely rare in normal trading
    and almost always indicates an unadjusted price series or data error.

    Parameters
    ----------
    df : OHLCV DataFrame with Date and Close columns

    Returns
    -------
    List of dates (same type as df["Date"] values) where splits were found.
    Empty list if none detected.
    """
    if df.empty or "Close" not in df.columns or len(df) < 2:
        return []

    close = pd.to_numeric(df["Close"], errors="coerce")
    ratio = close / close.shift(1)
    mask  = (ratio < 0.60) | (ratio > 1.67)
    mask  = mask.fillna(False)

    if not mask.any():
        return []

    dates = (
        df.loc[mask, "Date"].tolist()
        if "Date" in df.columns
        else df.index[mask].tolist()
    )

    date_strs = [str(d.date() if hasattr(d, "date") else d) for d in dates[:5]]
    logger.warning(
        "detect_splits: %d likely unadjusted corporate action(s) found: %s",
        len(dates), date_strs,
    )
    return dates


def fill_trading_gaps(df: pd.DataFrame, max_gap_days: int = 3) -> pd.DataFrame:
    """
    Forward-fill short gaps in a daily OHLCV series caused by market holidays
    or trading halts. Gaps longer than max_gap_days are left as NaN.

    Volume is set to 0 on filled rows to indicate no real trading occurred,
    so downstream indicators can distinguish filled rows from live data.

    Parameters
    ----------
    df           : cleaned OHLCV DataFrame with a Date column
    max_gap_days : max consecutive missing business days to fill (default: 3)
    """
    if "Date" not in df.columns or df.empty:
        return df

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    full_range = pd.bdate_range(df["Date"].min(), df["Date"].max())

    df_idx = df.set_index("Date").reindex(full_range)

    # Only fill gaps no wider than max_gap_days
    is_missing   = df_idx["Close"].isna()
    gap_id       = (~is_missing).cumsum()
    gap_size     = is_missing.groupby(gap_id).transform("sum")
    fill_mask    = is_missing & (gap_size <= max_gap_days)

    df_filled           = df_idx.copy()
    df_filled[fill_mask] = df_idx.ffill()[fill_mask]
    df_filled.loc[fill_mask, "Volume"] = 0

    n_filled = int(fill_mask.sum())
    if n_filled:
        logger.info("fill_trading_gaps: forward-filled %d missing business day(s).", n_filled)

    df_filled.index.name = "Date"
    return df_filled.reset_index()


def normalise_series(series: pd.Series, method: str = "minmax") -> pd.Series:
    """
    Normalise a numeric series for model-readiness.

    Parameters
    ----------
    series : numeric pd.Series
    method : "minmax"  — scale to [0,1]  (distorted by outliers)
             "zscore"  — zero mean, unit std
             "robust"  — median-centred, IQR-scaled  (outlier-resistant)

    Returns
    -------
    pd.Series of normalised values
    """
    if method == "zscore":
        mu    = series.mean()
        sigma = series.std()
        if sigma == 0:
            return series - mu
        return ((series - mu) / sigma).round(6)

    if method == "robust":
        median = series.median()
        iqr    = series.quantile(0.75) - series.quantile(0.25)
        if iqr == 0:
            return (series - median).round(6)
        return ((series - median) / iqr).round(6)

    # minmax default
    lo = series.min()
    hi = series.max()
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return ((series - lo) / (hi - lo)).round(6)


def preprocess_ohlcv(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """
    Full standard preprocessing pipeline for a raw OHLCV DataFrame.

    Runs: validate → clean → compute_returns.
    Returns the cleaned DataFrame even if validation finds warnings,
    so downstream indicators can still run on partial data.

    Parameters
    ----------
    df     : raw OHLCV DataFrame
    symbol : ticker label for log messages

    Returns
    -------
    Cleaned pd.DataFrame ready for compute_ohlcv_indicators().
    Empty DataFrame if the input fails hard validation.
    """
    if df is None or df.empty:
        logger.warning("preprocess [%s]: received empty DataFrame.", symbol or "unknown")
        return pd.DataFrame()

    is_valid, _ = validate_ohlcv(df, symbol)
    if not is_valid:
        # Hard failure (missing columns / all-NaN Close) — return empty
        return pd.DataFrame()

    df = clean_ohlcv(df)
    df = compute_returns(df)
    df = flag_outliers(df, column="returns_pct")

    logger.info(
        "preprocess [%s]: clean — %d rows, %d columns.",
        symbol or "unknown", len(df), len(df.columns),
    )
    return df
