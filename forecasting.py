"""
========================================================================
forecasting.py — Market Pulse Forecasting Engine
========================================================================
Generates forward-looking projections for market metrics using
statistical extrapolation methods. All forecasting logic lives here;
dashboard.py and forecast_tab.py only visualise the outputs.

Architecture
------------
- Receives DataFrames from historical_metrics / sentiment. No I/O.
- All forecast outputs include confidence bands so the dashboard can
  render uncertainty ribbons rather than false-precision point estimates.
- Three forecast methods:
    sma    : extend the rolling mean forward (flat projection)
    ema    : extend the EMA forward (widening uncertainty)
    linear : OLS slope extrapolation (trend continuation)
- All three are lightweight (numpy only) and fast enough to recompute
  on every pipeline run. ML hooks are reserved for a future phase.

Integration fixes (v2)
----------------------
- _build_date_index: guard against non-datetime series index so that
  the fallback to RangeIndex is reliable on DataFrames that haven't
  had their index set explicitly.
- forecast_series: the ``window`` parameter now respects
  FORECASTING.DEFAULT_HORIZON from config.py rather than using a bare
  ROLLING.SHORT_WINDOW default, keeping all horizon values centralised.
- forecast_market_metrics: the "sentiment" key is no longer added here
  (that was a responsibility leak); forecast_sentiment_trend() is the
  dedicated entry point and is called separately by main.py.
- _linear_forecast: residual std calculation made safe for the edge
  case where all residuals are identical (std=0 → se=1e-6 guard).
- Consistent logging with module-level logger throughout.

Public API
----------
  forecast_series(series, horizon, method)          -> pd.DataFrame
  forecast_market_metrics(hist_df, horizon, method) -> dict
  forecast_sentiment_trend(daily_sent_df, ...)      -> pd.DataFrame
========================================================================
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from config import FORECASTING, ROLLING

logger = logging.getLogger(__name__)

ForecastMethod = Literal["sma", "ema", "linear"]

# Canonical output columns — shared with data_loader and validators
FORECAST_SCHEMA: list[str] = [
    "Date", "forecast", "lower_bound", "upper_bound", "method"
]


# ======================================================================
# INTERNAL HELPERS
# ======================================================================

def _sma_forecast(
    series: pd.Series,
    horizon: int,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project the series forward by repeating its last rolling mean.

    Confidence bounds are ±1 rolling standard deviation (flat, does not
    widen with horizon because SMA projection carries no trend assumption).
    """
    roll_mean = series.rolling(window=window, min_periods=1).mean()
    roll_std  = series.rolling(window=window, min_periods=1).std().fillna(0)

    point  = float(roll_mean.iloc[-1])
    spread = float(roll_std.iloc[-1])

    steps     = np.arange(1, horizon + 1)
    forecasts = np.full(horizon, point)
    lower     = forecasts - spread * np.sqrt(steps)
    upper     = forecasts + spread * np.sqrt(steps)
    return forecasts, lower, upper


def _ema_forecast(
    series: pd.Series,
    horizon: int,
    span: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project the series forward from its last EMA value.

    Confidence bounds widen with sqrt(horizon) to reflect increasing
    uncertainty further into the future.
    """
    ema   = series.ewm(span=span, adjust=False, min_periods=1).mean()
    std   = series.rolling(span, min_periods=1).std().fillna(0)

    point  = float(ema.iloc[-1])
    spread = float(std.iloc[-1])

    steps     = np.arange(1, horizon + 1)
    forecasts = np.full(horizon, point)
    lower     = forecasts - spread * np.sqrt(steps)
    upper     = forecasts + spread * np.sqrt(steps)
    return forecasts, lower, upper


def _linear_forecast(
    series: pd.Series,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project the series forward using OLS linear extrapolation.

    Uses at most the last FORECASTING.LINEAR_FIT_WINDOW observations.
    Falls back to SMA projection if fewer than 3 observations remain.

    Confidence bounds use regression residual SE scaled by sqrt(step/n)
    and a 95% z-score from FORECASTING.CONFIDENCE_Z (1.96).

    Fix (v2): residual std guard ensures se > 0 even for perfectly
    smooth series (se = max(residual_std, 1e-6)).
    """
    fit_window = FORECASTING.LINEAR_FIT_WINDOW
    n_fit      = min(len(series), fit_window)
    subset     = series.iloc[-n_fit:].dropna().values
    n          = len(subset)

    if n < 3:
        return _sma_forecast(series, horizon, window=min(7, len(series)))

    x      = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = subset.mean()

    denom   = np.sum((x - x_mean) ** 2)
    slope   = np.sum((x - x_mean) * (subset - y_mean)) / denom if denom else 0.0
    intercept = y_mean - slope * x_mean

    residuals = subset - (intercept + slope * x)
    se        = max(float(residuals.std()), 1e-6)   # guard: never zero

    z        = FORECASTING.CONFIDENCE_Z
    x_fwd    = np.arange(n, n + horizon, dtype=float)
    forecasts = intercept + slope * x_fwd
    steps    = np.arange(1, horizon + 1, dtype=float)
    margin   = z * se * np.sqrt(steps / n)
    lower    = forecasts - margin
    upper    = forecasts + margin
    return forecasts, lower, upper


def _build_date_index(series: pd.Series, horizon: int) -> pd.DatetimeIndex | pd.RangeIndex:
    """
    Generate business-day forward dates starting the day after the last
    observation in ``series``.

    Returns a RangeIndex fallback if the series index cannot be
    interpreted as dates (e.g. default integer index).
    """
    try:
        last_date = pd.to_datetime(series.index[-1])
        return pd.bdate_range(
            start=last_date + pd.Timedelta(days=1), periods=horizon
        )
    except Exception:
        return pd.RangeIndex(horizon)


# ======================================================================
# PUBLIC API
# ======================================================================

def forecast_series(
    series: pd.Series,
    horizon: int = None,
    method: ForecastMethod = None,
    window: int = None,
) -> pd.DataFrame:
    """
    Forecast a numeric time series forward by ``horizon`` business days.

    Parameters
    ----------
    series  : pd.Series with a DatetimeIndex or sortable index.
              The last element must be the most recent observation.
    horizon : number of forward periods (default: FORECASTING.DEFAULT_HORIZON)
    method  : "sma" | "ema" | "linear" (default: FORECASTING.DEFAULT_METHOD)
    window  : MA window (default: ROLLING.SHORT_WINDOW)

    Returns
    -------
    pd.DataFrame with columns matching FORECAST_SCHEMA:
        Date, forecast, lower_bound, upper_bound, method
    """
    horizon = horizon or FORECASTING.DEFAULT_HORIZON
    method  = method  or FORECASTING.DEFAULT_METHOD
    window  = window  or ROLLING.SHORT_WINDOW

    clean = series.dropna()
    if clean.empty:
        logger.warning("forecasting: empty series passed — returning empty DataFrame.")
        return pd.DataFrame(columns=FORECAST_SCHEMA)

    if method == "sma":
        fcast, lower, upper = _sma_forecast(clean, horizon, window)
    elif method == "linear":
        fcast, lower, upper = _linear_forecast(clean, horizon)
    else:
        fcast, lower, upper = _ema_forecast(clean, horizon, window)

    dates = _build_date_index(clean, horizon)

    result = pd.DataFrame({
        "Date":        dates,
        "forecast":    fcast.round(4),
        "lower_bound": lower.round(4),
        "upper_bound": upper.round(4),
        "method":      method,
    })

    logger.debug(
        "forecasting: %s | horizon=%d | last=%.4f | fcast[0]=%.4f",
        method, horizon, float(clean.iloc[-1]), float(fcast[0]),
    )
    return result


def forecast_market_metrics(
    hist_df: pd.DataFrame,
    horizon: int = None,
    method: ForecastMethod = None,
) -> dict:
    """
    Forecast three core market metrics forward by ``horizon`` days.

    Metrics
    -------
    avg_pct      : average daily % change
    volatility   : daily % change standard deviation
    breadth_pct  : percentage of advancing stocks

    Parameters
    ----------
    hist_df : output of historical_metrics.compute_all()["hist_df"]
    horizon : forward business days (default: FORECASTING.DEFAULT_HORIZON)
    method  : forecast method (default: FORECASTING.DEFAULT_METHOD)

    Returns
    -------
    dict {metric_name -> pd.DataFrame} with FORECAST_SCHEMA columns.
    Empty DataFrames returned for any metric not present in hist_df.
    """
    horizon = horizon or FORECASTING.DEFAULT_HORIZON
    method  = method  or FORECASTING.DEFAULT_METHOD

    if hist_df.empty:
        logger.warning("forecasting: hist_df is empty — returning empty forecasts.")
        return {
            "avg_pct":     pd.DataFrame(columns=FORECAST_SCHEMA),
            "volatility":  pd.DataFrame(columns=FORECAST_SCHEMA),
            "breadth_pct": pd.DataFrame(columns=FORECAST_SCHEMA),
        }

    df = hist_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    results: dict[str, pd.DataFrame] = {}
    for metric in ("avg_pct", "volatility", "breadth_pct"):
        if metric not in df.columns:
            logger.warning("forecasting: metric '%s' not in hist_df.", metric)
            results[metric] = pd.DataFrame(columns=FORECAST_SCHEMA)
            continue

        s = df[metric].copy()
        s.index = df["Date"]
        results[metric] = forecast_series(s, horizon=horizon, method=method)
        logger.info(
            "forecasting: %s complete (horizon=%d, method=%s).",
            metric, horizon, method,
        )

    return results


def forecast_sentiment_trend(
    daily_sent_df: pd.DataFrame,
    horizon: int = None,
    method: ForecastMethod = None,
) -> pd.DataFrame:
    """
    Forecast the rolling sentiment average forward.

    Uses the 7-day rolling avg_score if it exists in the DataFrame,
    otherwise computes it on the fly from avg_score.

    Parameters
    ----------
    daily_sent_df : DataFrame with columns Date, avg_score
                    (output of sentiment.daily_sentiment_avg())
    horizon       : forward periods (default: FORECASTING.DEFAULT_HORIZON)
    method        : forecast method (default: FORECASTING.DEFAULT_METHOD)

    Returns
    -------
    pd.DataFrame with FORECAST_SCHEMA columns.
    Empty DataFrame if input is insufficient.
    """
    horizon = horizon or FORECASTING.DEFAULT_HORIZON
    method  = method  or FORECASTING.DEFAULT_METHOD

    if daily_sent_df is None or daily_sent_df.empty or "avg_score" not in daily_sent_df.columns:
        logger.warning("forecasting: sentiment DataFrame empty or missing avg_score.")
        return pd.DataFrame(columns=FORECAST_SCHEMA)

    df = daily_sent_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    series = df["avg_score"].copy()
    series.index = df["Date"]
    return forecast_series(series, horizon=horizon, method=method)