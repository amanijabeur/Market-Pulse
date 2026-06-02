"""
========================================================================
time_series.py -- Market Pulse Time Series Analysis Engine
========================================================================
Handles decomposition, trend extraction, rolling window analysis, and
cycle detection on market time series. Keeps all time-series logic
separate from forecasting (forecasting.py) and visualisation (dashboard.py).

Architecture decisions
----------------------
- Receives DataFrames from data_loader or historical_metrics. No I/O.
- Returns clean DataFrames that forecasting.py and dashboard.py consume.
- Uses only numpy/pandas -- no external time-series libraries required.
  This keeps the dependency footprint minimal while covering the
  analytics the platform needs at Phase 2.
- Trend extraction uses centred rolling mean (robust, no assumptions).
- Seasonality is not modelled -- daily market data at this sample size
  does not have meaningful weekly seasonality without multi-year history.

Public API
----------
  decompose(series, window)          -> dict(trend, residual, smoothed)
  rolling_stats(series, window)      -> pd.DataFrame
  detect_trend_change(series)        -> pd.DataFrame
  compute_market_cycles(hist_df)     -> pd.DataFrame
  rolling_correlation(s1, s2, window)-> pd.Series
  compute_breadth_trend(hist_df)     -> pd.DataFrame
========================================================================
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

from config import ROLLING

logger = logging.getLogger(__name__)


# ======================================================================
# DECOMPOSITION
# ======================================================================

def decompose(series: pd.Series, window: int) -> Dict[str, pd.Series]:
    """
    Simple trend-residual decomposition using centred rolling mean.

    The centred rolling mean (center=True) is used as the trend
    estimate -- it does not introduce lag bias the way a trailing
    window does, making it a fairer baseline for residual calculation.

    Parameters
    ----------
    series : numeric pd.Series with a DatetimeIndex or integer index
    window : smoothing window for trend extraction

    Returns
    -------
    dict with keys:
        original  : input series (unchanged)
        trend     : centred rolling mean
        residual  : original - trend (detrended series)
        smoothed  : trailing rolling mean (for display alongside trend)
    """
    trend    = series.rolling(window=window, center=True, min_periods=1).mean()
    residual = series - trend
    smoothed = series.rolling(window=window, min_periods=1).mean()

    return {
        "original": series.round(4),
        "trend":    trend.round(4),
        "residual": residual.round(4),
        "smoothed": smoothed.round(4),
    }


# ======================================================================
# ROLLING STATISTICS
# ======================================================================

def rolling_stats(series: pd.Series, window: int) -> pd.DataFrame:
    """
    Compute a full set of rolling statistics for a series.

    Returns a DataFrame with one column per statistic, aligned to the
    same index as the input series.

    Columns produced
    ----------------
    mean, std, min, max, range, skew, pct_positive
    """
    df = pd.DataFrame(index=series.index)
    roll = series.rolling(window=window, min_periods=1)

    df["mean"]         = roll.mean().round(4)
    df["std"]          = roll.std().round(4)
    df["min"]          = roll.min().round(4)
    df["max"]          = roll.max().round(4)
    df["range"]        = (df["max"] - df["min"]).round(4)
    df["skew"]         = roll.skew().round(4)

    # Proportion of positive observations in the window
    df["pct_positive"] = (
        series.gt(0).rolling(window=window, min_periods=1).mean() * 100
    ).round(2)

    return df


# ======================================================================
# TREND CHANGE DETECTION
# ======================================================================

def detect_trend_change(
    series: pd.Series,
    short_window: int = None,
    long_window:  int = None,
    min_gap:      float = 0.005,
) -> pd.DataFrame:
    """
    Detect crossover points where the short MA crosses the long MA.

    These crossovers are classical trend change signals:
    - Short crosses above long -> potential bullish regime shift
    - Short crosses below long -> potential bearish regime shift

    Parameters
    ----------
    series       : numeric pd.Series (e.g. avg_pct from hist_df)
    short_window : fast MA period (default: ROLLING.SHORT_WINDOW)
    long_window  : slow MA period (default: ROLLING.LONG_WINDOW)
    min_gap      : minimum relative gap to count as a real signal

    Returns
    -------
    pd.DataFrame with columns:
        index (same as series), short_ma, long_ma, crossover
        where crossover is "Bullish", "Bearish", or ""
    """
    short_window = short_window or ROLLING.SHORT_WINDOW
    long_window  = long_window  or ROLLING.LONG_WINDOW

    short_ma = series.rolling(short_window, min_periods=1).mean()
    long_ma  = series.rolling(long_window,  min_periods=1).mean()

    gap       = (short_ma - long_ma) / long_ma.abs().replace(0, np.nan)
    prev_gap  = gap.shift(1)

    crossover = pd.Series("", index=series.index)
    bullish   = (gap > min_gap) & (prev_gap <= min_gap)
    bearish   = (gap < -min_gap) & (prev_gap >= -min_gap)
    crossover[bullish] = "Bullish"
    crossover[bearish] = "Bearish"

    return pd.DataFrame({
        "short_ma":  short_ma.round(4),
        "long_ma":   long_ma.round(4),
        "gap":       gap.round(4),
        "crossover": crossover,
    }, index=series.index)


# ======================================================================
# MARKET CYCLE DETECTION
# ======================================================================

def compute_market_cycles(hist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify bull and bear market cycles from the historical aggregate.

    A cycle is defined as a contiguous run of days where the 7-day
    rolling average % change stays above (bull) or below (bear) zero.
    Each cycle is labelled with its start date, end date, length,
    direction, and average gain/loss.

    Parameters
    ----------
    hist_df : output of historical_metrics.compute_all()["hist_df"]
              Must contain: Date, avg_pct, rolling_avg_7d

    Returns
    -------
    pd.DataFrame with columns:
        start_date, end_date, length_days, direction, avg_return, peak_return
    Sorted by start_date ascending.
    """
    if hist_df.empty or "rolling_avg_7d" not in hist_df.columns:
        return pd.DataFrame(columns=[
            "start_date", "end_date", "length_days",
            "direction", "avg_return", "peak_return",
        ])

    df = hist_df[["Date", "avg_pct", "rolling_avg_7d"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Direction based on rolling avg (smoother than raw avg_pct)
    df["sign"]  = np.sign(df["rolling_avg_7d"])
    df["group"] = (df["sign"] != df["sign"].shift()).cumsum()

    cycles = []
    for _, grp in df.groupby("group"):
        sign = grp["sign"].iloc[0]
        if sign == 0:
            continue
        cycles.append({
            "start_date":   grp["Date"].iloc[0],
            "end_date":     grp["Date"].iloc[-1],
            "length_days":  len(grp),
            "direction":    "Bull" if sign > 0 else "Bear",
            "avg_return":   round(float(grp["avg_pct"].mean()), 4),
            "peak_return":  round(float(
                grp["avg_pct"].max() if sign > 0 else grp["avg_pct"].min()
            ), 4),
        })

    if not cycles:
        return pd.DataFrame(columns=[
            "start_date", "end_date", "length_days",
            "direction", "avg_return", "peak_return",
        ])

    result = pd.DataFrame(cycles).sort_values("start_date").reset_index(drop=True)
    logger.info(
        "time_series: %d market cycles detected (%d bull, %d bear).",
        len(result),
        (result["direction"] == "Bull").sum(),
        (result["direction"] == "Bear").sum(),
    )
    return result


# ======================================================================
# ROLLING CORRELATION
# ======================================================================

def rolling_correlation(
    s1: pd.Series,
    s2: pd.Series,
    window: int,
) -> pd.Series:
    """
    Compute rolling Pearson correlation between two series.

    Useful for tracking whether sentiment and price performance stay
    correlated over time, or whether they diverge (a signal the
    anomaly detection module uses).

    Parameters
    ----------
    s1, s2 : aligned numeric pd.Series (same index)
    window : rolling window size

    Returns
    -------
    pd.Series of correlation values in [-1, +1]
    """
    return s1.rolling(window=window, min_periods=max(3, window // 2)).corr(s2).round(4)


# ======================================================================
# BREADTH TREND
# ======================================================================

def compute_breadth_trend(hist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling breadth trend metrics from hist_df.

    Breadth trend measures the consistency of market participation --
    whether broad groups of stocks are moving in the same direction or
    whether moves are concentrated in a few names.

    Parameters
    ----------
    hist_df : output of historical_metrics.compute_all()["hist_df"]

    Returns
    -------
    pd.DataFrame with columns:
        Date, breadth_pct,
        breadth_sma_7d, breadth_ema_7d,
        breadth_trend  (Expanding | Contracting | Stable),
        net_breadth    (gainers - losers)
    """
    if hist_df.empty:
        return pd.DataFrame()

    df = hist_df[["Date", "breadth_pct", "n_gainers", "n_losers"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    w = ROLLING.SHORT_WINDOW

    df["breadth_sma_7d"] = df["breadth_pct"].rolling(w, min_periods=1).mean().round(2)
    df["breadth_ema_7d"] = df["breadth_pct"].ewm(span=w, adjust=False, min_periods=1).mean().round(2)
    df["net_breadth"]    = df["n_gainers"] - df["n_losers"]

    # Breadth trend: compare current SMA to its own 1-period lag
    prev_sma = df["breadth_sma_7d"].shift(1)
    df["breadth_trend"] = np.where(
        df["breadth_sma_7d"] > prev_sma + 1.0, "Expanding",
        np.where(df["breadth_sma_7d"] < prev_sma - 1.0, "Contracting", "Stable"),
    )

    return df
