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
- Forecast methods:
    mean   : long-run mean with empirical-quantile bands (noise-like series)
    har    : HAR(1,5,22) in log space (persistent, positive series: volatility)
    sma    : extend the rolling mean forward (flat projection)
    ema    : extend the EMA forward (widening uncertainty)
    linear : OLS slope extrapolation (trend continuation)
    auto   : per-metric choice from FORECASTING.METRIC_METHODS
- All are lightweight (numpy only) and fast enough to recompute on
  every pipeline run. backtest() measures accuracy and band coverage
  so method choices can be re-validated as history grows.

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
  backtest(series, method, horizon, n_origins)      -> dict
========================================================================
"""

import logging
import math
from typing import Literal

import numpy as np
import pandas as pd

from config import FORECASTING, ROLLING

logger = logging.getLogger(__name__)

ForecastMethod = Literal["auto", "mean", "har", "sma", "ema", "linear"]

_HAR_LAGS = (1, 5, 22)   # daily, weekly, monthly averages

# Physical range of each forecast metric; forecasts are clipped to these
METRIC_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "avg_pct":     (None, None),
    "volatility":  (0.0, None),
    "breadth_pct": (0.0, 100.0),
    "sentiment":   (-1.0, 1.0),
}

# Canonical output columns — shared with data_loader and validators
FORECAST_SCHEMA: list[str] = [
    "Date", "forecast", "lower_bound", "upper_bound", "method"
]


# ======================================================================
# INTERNAL HELPERS
# ======================================================================

def _critical_value(dof: int | None = None) -> float:
    """
    Two-sided critical value for the configured confidence level.

    FORECASTING.CONFIDENCE_Z defines the level (1.96 -> 95%). When ``dof``
    is given, the Student-t quantile at that level is returned instead,
    which widens the band appropriately for small fit windows.
    """
    z = FORECASTING.CONFIDENCE_Z
    if dof is None:
        return z
    try:
        from scipy import stats
        level = 2 * stats.norm.cdf(z) - 1
        return float(stats.t.ppf((1 + level) / 2, dof))
    except ImportError:
        return z


def _one_step_sigma(series: pd.Series, fitted: pd.Series) -> float:
    """
    Std of one-step-ahead errors: each value minus the fitted level known
    the day before. Uses the last FORECASTING.LINEAR_FIT_WINDOW errors so
    the band reflects recent forecast accuracy, not just recent dispersion.
    """
    errors = (series - fitted.shift(1)).dropna().iloc[-FORECASTING.LINEAR_FIT_WINDOW:]
    if len(errors) < 2:
        return 0.0
    return float(errors.std())


def _quantile_band(errors: np.ndarray) -> tuple[float, float]:
    """
    Empirical (lower, upper) error quantiles at the configured level.

    Distribution-free: market metrics have fat tails (e.g. one 45% vol
    day), which inflate a normal ±z·sigma band while still under-covering
    the tails. Quantiles of past forecast errors fit both ends directly.
    """
    level = math.erf(FORECASTING.CONFIDENCE_Z / math.sqrt(2))
    alpha = (1 - level) / 2
    return float(np.quantile(errors, alpha)), float(np.quantile(errors, 1 - alpha))


def _mean_forecast(
    series: pd.Series,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project the long-run mean of the last FORECASTING.MEAN_WINDOW values.

    Best for series with ~zero autocorrelation (daily avg % change,
    breadth, sentiment), where any recent trend or level is mostly noise.
    Bounds are empirical quantiles of each day's deviation from the mean
    known the day before; flat, since noise does not compound.
    """
    window = FORECASTING.MEAN_WINDOW
    prior  = series.rolling(window, min_periods=ROLLING.SHORT_WINDOW).mean().shift(1)
    errors = (series - prior).dropna().values[-window:]
    if len(errors) < 2 * ROLLING.SHORT_WINDOW:
        return _sma_forecast(series, horizon, ROLLING.SHORT_WINDOW)

    lo, hi    = _quantile_band(errors)
    forecasts = np.full(horizon, float(series.iloc[-window:].mean()))
    return forecasts, forecasts + lo, forecasts + hi


def _har_features(path: np.ndarray, i: int) -> np.ndarray:
    """Intercept plus the mean of the last 1/5/22 values ending at index i."""
    return np.array([1.0, *(path[i - lag + 1:i + 1].mean() for lag in _HAR_LAGS)])


def _har_path(history: np.ndarray, beta: np.ndarray, horizon: int) -> np.ndarray:
    """Iterate the HAR recursion ``horizon`` steps past the end of history."""
    path = list(history[-max(_HAR_LAGS):])
    for _ in range(horizon):
        arr = np.asarray(path)
        path.append(float(_har_features(arr, len(arr) - 1) @ beta))
    return np.asarray(path[-horizon:])


def _har_forecast(
    series: pd.Series,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Heterogeneous AutoRegressive (HAR) forecast fitted in log space.

    Next value is regressed on the average of the last 1, 5 and 22 days
    (Corsi, 2009), the standard model for volatility persistence. Logs
    keep forecasts positive and damp outlier days. Bounds are empirical
    quantiles of in-sample h-step errors, so they widen per step only as
    much as the data shows. The point forecast is the median, exp(log f).
    """
    max_lag = max(_HAR_LAGS)
    values  = series.values.astype(float)
    if (values <= 0).any() or len(values) < max_lag + 3 * ROLLING.SHORT_WINDOW:
        return _ema_forecast(series, horizon, ROLLING.SHORT_WINDOW)

    y      = np.log(values[-(FORECASTING.MEAN_WINDOW + max_lag):])
    rows   = range(max_lag - 1, len(y) - 1)
    X      = np.array([_har_features(y, i) for i in rows])
    beta   = np.linalg.lstsq(X, y[max_lag:], rcond=None)[0]

    # h-step in-sample errors, one row per origin
    errors = np.array([
        y[i + 1:i + 1 + horizon] - _har_path(y[:i + 1], beta, horizon)
        for i in range(max_lag - 1, len(y) - horizon)
    ])
    bands  = np.array([_quantile_band(errors[:, k]) for k in range(horizon)])

    log_f  = _har_path(y, beta, horizon)
    return np.exp(log_f), np.exp(log_f + bands[:, 0]), np.exp(log_f + bands[:, 1])


def _sma_forecast(
    series: pd.Series,
    horizon: int,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project the series forward by repeating its last rolling mean.

    Confidence bounds are ±z × one-step-ahead error std. They are flat:
    a level projection carries no trend, so uncertainty does not compound.
    """
    roll_mean = series.rolling(window=window, min_periods=1).mean()
    sigma     = _one_step_sigma(series, roll_mean)

    forecasts = np.full(horizon, float(roll_mean.iloc[-1]))
    margin    = _critical_value() * sigma
    return forecasts, forecasts - margin, forecasts + margin


def _ema_forecast(
    series: pd.Series,
    horizon: int,
    span: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project the series forward from its last EMA value.

    Simple exponential smoothing: the h-step forecast error variance is
    sigma^2 * (1 + (h-1) * alpha^2), so bounds widen slowly with horizon.
    """
    ema   = series.ewm(span=span, adjust=False, min_periods=1).mean()
    sigma = _one_step_sigma(series, ema)
    alpha = 2 / (span + 1)

    steps     = np.arange(1, horizon + 1)
    forecasts = np.full(horizon, float(ema.iloc[-1]))
    margin    = _critical_value() * sigma * np.sqrt(1 + (steps - 1) * alpha ** 2)
    return forecasts, forecasts - margin, forecasts + margin


def _linear_forecast(
    series: pd.Series,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project the series forward using OLS linear extrapolation.

    Uses at most the last FORECASTING.LINEAR_FIT_WINDOW observations.
    Falls back to SMA projection if fewer than 3 observations remain.

    Confidence bounds are the OLS prediction interval for a new
    observation: s * sqrt(1 + 1/n + (x - x_mean)^2 / Sxx), where s is the
    residual standard error with n-2 degrees of freedom, scaled by the
    Student-t quantile for the level set by FORECASTING.CONFIDENCE_Z.
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
    dof       = n - 2
    s         = max(float(np.sqrt(np.sum(residuals ** 2) / dof)), 1e-6)   # guard: never zero

    x_fwd     = np.arange(n, n + horizon, dtype=float)
    forecasts = intercept + slope * x_fwd
    # New-observation noise plus uncertainty in the fitted intercept and
    # slope, which grows the further we extrapolate from x_mean
    se_pred   = s * np.sqrt(1 + 1 / n + (x_fwd - x_mean) ** 2 / denom)
    margin    = _critical_value(dof) * se_pred
    return forecasts, forecasts - margin, forecasts + margin


def _resolve_method(method: str | None, metric: str | None) -> str:
    """Turn None/"auto" into a concrete method, per metric when known."""
    method = method or FORECASTING.DEFAULT_METHOD
    if method != "auto":
        return method
    return dict(FORECASTING.METRIC_METHODS).get(metric, "mean")


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
    bounds: tuple[float | None, float | None] | None = None,
) -> pd.DataFrame:
    """
    Forecast a numeric time series forward by ``horizon`` business days.

    Parameters
    ----------
    series  : pd.Series with a DatetimeIndex or sortable index.
              The last element must be the most recent observation.
    horizon : number of forward periods (default: FORECASTING.DEFAULT_HORIZON)
    method  : "mean" | "har" | "sma" | "ema" | "linear"
              (default: FORECASTING.DEFAULT_METHOD; "auto" means "mean"
              here, since the per-metric choice needs the metric name)
    window  : MA window (default: ROLLING.SHORT_WINDOW)
    bounds  : optional (min, max) the metric can physically take, e.g.
              (0, None) for volatility; forecast and bands are clipped to it

    Returns
    -------
    pd.DataFrame with columns matching FORECAST_SCHEMA:
        Date, forecast, lower_bound, upper_bound, method
    """
    horizon = horizon or FORECASTING.DEFAULT_HORIZON
    method  = _resolve_method(method, metric=None)
    window  = window  or ROLLING.SHORT_WINDOW

    clean = series.dropna()
    if clean.empty:
        logger.warning("forecasting: empty series passed — returning empty DataFrame.")
        return pd.DataFrame(columns=FORECAST_SCHEMA)

    if method == "mean":
        fcast, lower, upper = _mean_forecast(clean, horizon)
    elif method == "har":
        fcast, lower, upper = _har_forecast(clean, horizon)
    elif method == "sma":
        fcast, lower, upper = _sma_forecast(clean, horizon, window)
    elif method == "linear":
        fcast, lower, upper = _linear_forecast(clean, horizon)
    else:
        fcast, lower, upper = _ema_forecast(clean, horizon, window)

    if bounds is not None:
        fcast, lower, upper = (np.clip(a, *bounds) for a in (fcast, lower, upper))

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
        metric_method = _resolve_method(method, metric)
        results[metric] = forecast_series(
            s, horizon=horizon, method=metric_method, bounds=METRIC_BOUNDS[metric],
        )
        logger.info(
            "forecasting: %s complete (horizon=%d, method=%s).",
            metric, horizon, metric_method,
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
    return forecast_series(
        series, horizon=horizon, method=_resolve_method(method, "sentiment"),
        bounds=METRIC_BOUNDS["sentiment"],
    )


def backtest(
    series: pd.Series,
    method: ForecastMethod,
    horizon: int = None,
    n_origins: int = 250,
) -> dict:
    """
    Rolling-origin evaluation of one method on one series.

    At each of the last ``n_origins`` dates, forecasts ``horizon`` days
    using only data available then, and compares against what happened.

    Returns
    -------
    dict with keys:
        mae      : mean absolute error across all origins and steps
        coverage : share of actual values inside [lower_bound, upper_bound]
        width    : mean band width
        n        : number of forecast/actual pairs scored
    """
    horizon = horizon or FORECASTING.DEFAULT_HORIZON
    clean   = series.dropna().reset_index(drop=True)
    start   = max(len(clean) - n_origins - horizon, max(_HAR_LAGS) + 1)

    abs_err, inside, width = [], [], []
    for end in range(start, len(clean) - horizon + 1):
        fc     = forecast_series(clean.iloc[:end], horizon=horizon, method=method)
        actual = clean.iloc[end:end + horizon].values
        abs_err.extend(np.abs(actual - fc["forecast"].values))
        inside.extend((actual >= fc["lower_bound"].values) & (actual <= fc["upper_bound"].values))
        width.extend(fc["upper_bound"].values - fc["lower_bound"].values)

    if not abs_err:
        return {"mae": np.nan, "coverage": np.nan, "width": np.nan, "n": 0}
    return {
        "mae":      float(np.mean(abs_err)),
        "coverage": float(np.mean(inside)),
        "width":    float(np.mean(width)),
        "n":        len(abs_err),
    }