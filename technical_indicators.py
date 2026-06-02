"""
========================================================================
technical_indicators.py -- Market Pulse Technical Analysis Engine
========================================================================
Computes all technical indicators used by the platform. Operates
exclusively on the historical price DataFrame produced by data_loader
and historical_metrics. Never reads files directly.

Architecture decisions
----------------------
- Pure computation layer. No I/O, no plotting, no dashboard logic.
- All thresholds and window sizes come from config.py.
- All functions are vectorised (pandas/numpy). No Python for-loops
  over rows.
- Two public entry points:
    compute_symbol_indicators(df_full, symbol) -> pd.DataFrame
        Per-symbol time series with all indicators attached as columns.
    compute_market_indicators(hist_df)         -> pd.DataFrame
        Market-level (aggregate) indicator series from hist_df.
- Both return clean DataFrames that dashboard.py can visualise directly
  without any further calculation.
- Supports incremental extension: adding a new indicator means adding
  one helper function and one column assignment -- nothing else changes.

Integration with pipeline
-------------------------
  main.py Stage 2 calls historical_metrics.compute_all() which returns
  hist_df. dashboard.py calls technical_indicators.compute_market_indicators(hist_df)
  to get the indicator overlay for the Technical tab. Symbol-level
  indicators are computed on demand (per symbol) inside dashboard.build().

All config references
---------------------
  ROLLING.SHORT_WINDOW    -- 7
  ROLLING.LONG_WINDOW     -- 30
========================================================================
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import ROLLING, TECHNICAL, INDICATORS

logger = logging.getLogger(__name__)


# ======================================================================
# HELPER: SINGLE-SERIES INDICATORS
# Each function takes a pd.Series and returns a pd.Series.
# They are intentionally generic -- callers pass in whatever series
# they need (price, % change, volatility, sentiment score, etc.)
# ======================================================================

def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """
    Simple Moving Average.

    Parameters
    ----------
    series : numeric pd.Series (e.g. Price or avg_pct)
    window : lookback period in rows

    Returns
    -------
    pd.Series of the same length, NaN for the first (window-1) rows.
    """
    return series.rolling(window=window, min_periods=1).mean().round(4)


def compute_ema(series: pd.Series, window: int) -> pd.Series:
    """
    Exponential Moving Average using pandas ewm with span=window.

    EMA gives more weight to recent observations than SMA, making it
    more responsive to new data -- useful for detecting trend changes
    sooner than SMA.

    Parameters
    ----------
    series : numeric pd.Series
    window : span (analogous to period in SMA)
    """
    return series.ewm(span=window, adjust=False, min_periods=1).mean().round(4)


def compute_volatility(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling standard deviation of a series.

    Used to track how much % Change (or any metric) varies over time.
    Higher values = more uncertainty / wider price swings.

    Parameters
    ----------
    series : numeric pd.Series (typically % Change)
    window : rolling window size
    """
    return series.rolling(window=window, min_periods=1).std().round(4)


def compute_momentum(series: pd.Series, window: int) -> pd.Series:
    """
    Rate-of-change momentum: (current - value_n_periods_ago) / abs(value_n_periods_ago).

    Returns percentage momentum. Positive = upward momentum.
    NaN for the first `window` rows.

    Parameters
    ----------
    series : numeric pd.Series
    window : lookback period
    """
    lagged = series.shift(window)
    mom    = (series - lagged) / lagged.abs().replace(0, np.nan)
    return mom.round(4)


def compute_signal_strength(series: pd.Series, window: int) -> pd.Series:
    """
    Normalised signal strength: z-score of the series over a rolling window.

    Maps each value to how many standard deviations it sits above or
    below the rolling mean. Useful for detecting unusually strong or
    weak readings regardless of absolute scale.

    Returns a Series where:
        > +1.0 = notably strong signal
        < -1.0 = notably weak signal
        near 0 = in-line with recent history
    """
    roll_mean = series.rolling(window=window, min_periods=1).mean()
    roll_std  = series.rolling(window=window, min_periods=1).std().replace(0, np.nan)
    return ((series - roll_mean) / roll_std).round(4)


def classify_trend(
    short_ma: pd.Series,
    long_ma: pd.Series,
    threshold: float = TECHNICAL.TREND_GAP_THRESHOLD,
) -> pd.Series:
    """
    Classify trend direction by comparing two moving averages.

    Logic
    -----
    - If short_ma > long_ma by more than `threshold` * long_ma  -> "Bullish"
    - If short_ma < long_ma by more than `threshold` * long_ma  -> "Bearish"
    - Otherwise                                                   -> "Neutral"

    Parameters
    ----------
    short_ma  : faster moving average (e.g. SMA-7)
    long_ma   : slower moving average (e.g. SMA-30)
    threshold : minimum relative gap to classify as trending (default 0.5%)
    """
    gap = (short_ma - long_ma) / long_ma.abs().replace(0, np.nan)
    return pd.Series(
        np.where(gap >  threshold, "Bullish",
        np.where(gap < -threshold, "Bearish", "Neutral")),
        index=short_ma.index,
    )


def compute_relative_momentum(
    series: pd.Series,
    universe_mean: pd.Series,
) -> pd.Series:
    """
    Relative momentum: how much this series outperforms the universe mean.

    Returns (series - universe_mean), useful for ranking stocks or
    comparing a symbol's performance against the market average.

    Parameters
    ----------
    series        : individual symbol % Change series
    universe_mean : average % Change across all symbols for the same dates
    """
    return (series - universe_mean).round(4)


# ======================================================================
# MARKET-LEVEL INDICATORS
# Computed from hist_df (one row per trading day).
# ======================================================================

def compute_market_indicators(hist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators on the market-level aggregate series.

    Input
    -----
    hist_df : output of historical_metrics.compute_all()["hist_df"]
              Must contain: Date, avg_pct, volatility, breadth_pct

    Output
    ------
    pd.DataFrame with all original columns plus:
        sma_avg_7d        -- 7-day SMA of avg_pct
        sma_avg_30d       -- 30-day SMA of avg_pct
        ema_avg_7d        -- 7-day EMA of avg_pct
        ema_avg_30d       -- 30-day EMA of avg_pct
        mom_avg_7d        -- 7-day momentum of avg_pct
        vol_zscore_7d     -- z-score of volatility (7-day rolling)
        breadth_sma_7d    -- 7-day SMA of breadth_pct
        breadth_ema_7d    -- 7-day EMA of breadth_pct
        breadth_mom_7d    -- 7-day momentum of breadth_pct
        signal_strength   -- normalised signal strength of avg_pct
        trend_7_30        -- trend direction from SMA-7 vs SMA-30
    """
    if hist_df.empty:
        logger.warning("technical_indicators: hist_df is empty -- returning as-is.")
        return hist_df

    df = hist_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    w_short = ROLLING.SHORT_WINDOW
    w_long  = ROLLING.LONG_WINDOW

    # -- Average % Change indicators ---------------------------------
    df["sma_avg_7d"]    = compute_sma(df["avg_pct"], w_short)
    df["sma_avg_30d"]   = compute_sma(df["avg_pct"], w_long)
    df["ema_avg_7d"]    = compute_ema(df["avg_pct"], w_short)
    df["ema_avg_30d"]   = compute_ema(df["avg_pct"], w_long)
    df["mom_avg_7d"]    = compute_momentum(df["avg_pct"], w_short)

    # -- Volatility indicators ----------------------------------------
    df["vol_zscore_7d"] = compute_signal_strength(df["volatility"], w_short)

    # -- Breadth indicators -------------------------------------------
    df["breadth_sma_7d"]  = compute_sma(df["breadth_pct"], w_short)
    df["breadth_ema_7d"]  = compute_ema(df["breadth_pct"], w_short)
    df["breadth_mom_7d"]  = compute_momentum(df["breadth_pct"], w_short)

    # -- Signal strength and trend direction -------------------------
    df["signal_strength"] = compute_signal_strength(df["avg_pct"], w_short)
    df["trend_7_30"]      = classify_trend(
        df["sma_avg_7d"],
        df["sma_avg_30d"],
    )

    logger.info(
        "technical_indicators: market indicators computed for %d days.",
        len(df),
    )
    return df


# ======================================================================
# SYMBOL-LEVEL INDICATORS
# Computed from the full price history for a single symbol.
# ======================================================================

def compute_symbol_indicators(
    df_full: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """
    Compute technical indicators for a single ticker symbol.

    Input
    -----
    df_full : full historical price DataFrame from data_loader
    symbol  : ticker string, e.g. "AAPL"

    Output
    ------
    pd.DataFrame (one row per day the symbol appeared) with columns:
        Date, Symbol, Price, Change, % Change  (original)
        sma_price_7d     -- 7-day SMA of Price
        sma_price_30d    -- 30-day SMA of Price
        ema_price_7d     -- 7-day EMA of Price
        price_momentum   -- 7-day rate-of-change on Price
        pct_volatility   -- 7-day rolling std of % Change
        signal_strength  -- z-score of % Change
        trend_direction  -- Bullish / Bearish / Neutral
        rel_momentum     -- outperformance vs market avg_pct on same days
    """
    df_sym = df_full[df_full["Symbol"] == symbol].copy()
    if df_sym.empty:
        logger.warning("technical_indicators: symbol '%s' not found in dataset.", symbol)
        return pd.DataFrame()

    df_sym = df_sym.sort_values("Date").reset_index(drop=True)
    df_sym["Date"] = pd.to_datetime(df_sym["Date"])

    w_short = ROLLING.SHORT_WINDOW
    w_long  = ROLLING.LONG_WINDOW

    # -- Price indicators --------------------------------------------
    df_sym["sma_price_7d"]   = compute_sma(df_sym["Price"], w_short)
    df_sym["sma_price_30d"]  = compute_sma(df_sym["Price"], w_long)
    df_sym["ema_price_7d"]   = compute_ema(df_sym["Price"], w_short)
    df_sym["price_momentum"] = compute_momentum(df_sym["Price"], w_short)

    # -- % Change indicators -----------------------------------------
    df_sym["pct_volatility"]  = compute_volatility(df_sym["% Change"], w_short)
    df_sym["signal_strength"] = compute_signal_strength(df_sym["% Change"], w_short)

    # -- Trend direction from price SMAs -----------------------------
    df_sym["trend_direction"] = classify_trend(
        df_sym["sma_price_7d"],
        df_sym["sma_price_30d"],
    )

    # -- Relative momentum vs market ---------------------------------
    mkt_avg = (
        df_full.groupby("Date")["% Change"]
        .mean()
        .rename("mkt_avg")
        .reset_index()
    )
    df_sym = df_sym.merge(mkt_avg, on="Date", how="left")
    df_sym["rel_momentum"] = compute_relative_momentum(
        df_sym["% Change"], df_sym["mkt_avg"]
    )
    df_sym = df_sym.drop(columns=["mkt_avg"])

    logger.debug(
        "technical_indicators: computed for %s (%d rows).", symbol, len(df_sym)
    )
    return df_sym


# ======================================================================
# UNIVERSE SUMMARY
# Fast overview of indicator signals across all symbols for today.
# ======================================================================

def compute_universe_signals(df_latest: pd.DataFrame, df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a per-symbol signal summary for the latest trading day.

    Used by the dashboard Technical tab to show a ranked overview of
    all 100 active stocks with their current indicator readings.

    Input
    -----
    df_latest : latest-day price slice (from data_loader.load_latest_day())
    df_full   : full historical price data

    Output
    ------
    pd.DataFrame (one row per symbol) with columns:
        Symbol, Price, pct_change,
        trend_direction, signal_strength, rel_momentum, pct_volatility
    Sorted by signal_strength descending.
    """
    if df_latest.empty or df_full.empty:
        return pd.DataFrame()

    latest_date = pd.to_datetime(df_latest["Date"].max())
    mkt_avg_today = df_latest["% Change"].mean()

    rows = []
    for symbol in df_latest["Symbol"].tolist():
        sym_df = compute_symbol_indicators(df_full, symbol)
        if sym_df.empty:
            continue
        latest_row = sym_df[sym_df["Date"] == latest_date]
        if latest_row.empty:
            continue
        r = latest_row.iloc[0]
        rows.append({
            "Symbol":         symbol,
            "Price":          r["Price"],
            "pct_change":     r["% Change"],
            "trend_direction": r.get("trend_direction", "Neutral"),
            "signal_strength": r.get("signal_strength", 0.0),
            "rel_momentum":    r.get("rel_momentum", 0.0),
            "pct_volatility":  r.get("pct_volatility", 0.0),
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values("signal_strength", ascending=False)
    return result.reset_index(drop=True)


# ======================================================================
# OHLCV INDICATOR FUNCTIONS
# New in historical layer. Operate on OHLCV DataFrames (Open, High,
# Low, Close, Volume) rather than the daily-scrape % Change series.
# All existing helpers (compute_sma, compute_ema, etc.) are reused.
# Config parameters come from config.INDICATORS.
# ======================================================================

def compute_rsi(close: pd.Series, window: int = None) -> pd.Series:
    """
    Relative Strength Index (RSI).

    Standard Wilder smoothing (ewm with alpha=1/window).
    Returns values in [0, 100].
      > 70 : overbought territory
      < 30 : oversold territory

    Parameters
    ----------
    close  : closing price series
    window : lookback period (default: INDICATORS.RSI_WINDOW = 14)
    """
    from config import INDICATORS
    window = window or INDICATORS.RSI_WINDOW

    delta   = close.diff()
    gain    = delta.clip(lower=0)
    loss    = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    return rsi.round(4)


def compute_macd(
    close: pd.Series,
    fast: int = None,
    slow: int = None,
    signal: int = None,
) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence (MACD).

    Returns a DataFrame with three columns:
      macd        : MACD line (fast_ema - slow_ema)
      macd_signal : signal line (EMA of macd)
      macd_hist   : histogram (macd - macd_signal)

    Parameters
    ----------
    close  : closing price series
    fast   : fast EMA window (default: INDICATORS.MACD_FAST = 12)
    slow   : slow EMA window (default: INDICATORS.MACD_SLOW = 26)
    signal : signal EMA window (default: INDICATORS.MACD_SIGNAL = 9)
    """
    from config import INDICATORS
    fast   = fast   or INDICATORS.MACD_FAST
    slow   = slow   or INDICATORS.MACD_SLOW
    signal = signal or INDICATORS.MACD_SIGNAL

    ema_fast    = compute_ema(close, fast)
    ema_slow    = compute_ema(close, slow)
    macd_line   = (ema_fast - ema_slow).round(4)
    signal_line = compute_ema(macd_line, signal)
    histogram   = (macd_line - signal_line).round(4)

    return pd.DataFrame({
        "macd":        macd_line,
        "macd_signal": signal_line,
        "macd_hist":   histogram,
    }, index=close.index)


def compute_bollinger_bands(
    close: pd.Series,
    window: int = None,
    n_std: float = None,
) -> pd.DataFrame:
    """
    Bollinger Bands.

    Returns a DataFrame with four columns:
      bb_middle : SMA of close
      bb_upper  : middle + n_std * rolling std
      bb_lower  : middle - n_std * rolling std
      bb_width  : (upper - lower) / middle — normalised bandwidth

    Parameters
    ----------
    close  : closing price series
    window : rolling window (default: INDICATORS.BB_WINDOW = 20)
    n_std  : standard deviation multiplier (default: INDICATORS.BB_STD = 2.0)
    """
    from config import INDICATORS
    window = window or INDICATORS.BB_WINDOW
    n_std  = n_std  or INDICATORS.BB_STD

    middle = compute_sma(close, window)
    std    = close.rolling(window=window, min_periods=1).std()
    upper  = (middle + n_std * std).round(4)
    lower  = (middle - n_std * std).round(4)
    width   = ((upper - lower) / middle.replace(0, float("nan"))).round(4)
    pct_b   = ((close - lower) / (upper - lower).replace(0, float("nan"))).round(4)

    return pd.DataFrame({
        "bb_middle": middle,
        "bb_upper":  upper,
        "bb_lower":  lower,
        "bb_width":  width,
        "bb_pct_b":  pct_b,
    }, index=close.index)


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = None,
) -> pd.Series:
    """
    Average True Range (ATR).

    True Range = max(high-low, |high-prev_close|, |low-prev_close|).
    ATR = Wilder-smoothed EMA of True Range.

    Used as a volatility measure that accounts for overnight gaps.

    Parameters
    ----------
    high, low, close : OHLCV price series
    window           : smoothing window (default: INDICATORS.ATR_WINDOW = 14)
    """
    from config import INDICATORS
    window = window or INDICATORS.ATR_WINDOW

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return atr.round(4)


def compute_drawdown(close: pd.Series, window: int = None) -> pd.DataFrame:
    """
    Rolling drawdown from the recent peak.

    Returns a DataFrame with two columns:
      drawdown_pct : percentage decline from rolling max (negative values)
      rolling_max  : rolling maximum close over the window

    A drawdown of -0.15 means the price is 15% below its rolling peak.

    Parameters
    ----------
    close  : closing price series
    window : rolling lookback for the peak (default: INDICATORS.DRAWDOWN_WINDOW = 252)
    """
    from config import INDICATORS
    window = window or INDICATORS.DRAWDOWN_WINDOW

    rolling_max  = close.rolling(window=window, min_periods=1).max()
    drawdown_pct = ((close - rolling_max) / rolling_max.replace(0, float("nan"))).round(4)

    return pd.DataFrame({
        "drawdown_pct": drawdown_pct,
        "rolling_max":  rolling_max.round(4),
    }, index=close.index)


def get_regime_windows(close: pd.Series) -> dict:
    """
    Select indicator window sizes based on the symbol's current volatility
    regime relative to its own recent history.

    Strategy
    --------
    - Compute 21-day realised volatility (std of daily returns).
    - Compute the empirical percentile of that reading within the last
      INDICATORS.ADAPTIVE_LOOKBACK (63) days of rolling vol history.
    - Map the percentile to a multiplier:
        >= ADAPTIVE_HIGH_PCT (0.80) → ADAPTIVE_FAST_MULT (0.70)  — shorter windows
        <= ADAPTIVE_LOW_PCT  (0.20) → ADAPTIVE_SLOW_MULT (1.35)  — longer  windows
        otherwise                   → 1.0                         — base windows
    - Apply the multiplier to each base window and enforce per-indicator
      minimums so no window collapses to a meaningless value.
    - Returns a dict also containing 'mult' so callers can log or store
      the regime classification.

    Falls back to base windows if the series is too short for a reliable
    vol percentile estimate.
    """
    from config import INDICATORS

    returns = close.pct_change().dropna()

    if len(returns) < INDICATORS.ADAPTIVE_LOOKBACK:
        return {
            "rsi":         INDICATORS.RSI_WINDOW,
            "bb":          INDICATORS.BB_WINDOW,
            "atr":         INDICATORS.ATR_WINDOW,
            "macd_fast":   INDICATORS.MACD_FAST,
            "macd_slow":   INDICATORS.MACD_SLOW,
            "macd_signal": INDICATORS.MACD_SIGNAL,
            "mult":        1.0,
        }

    recent_vol = float(returns.tail(21).std())
    hist_vol   = returns.rolling(INDICATORS.ADAPTIVE_LOOKBACK).std().dropna()
    pct_rank   = float((hist_vol < recent_vol).mean())

    if pct_rank >= INDICATORS.ADAPTIVE_HIGH_PCT:
        mult = INDICATORS.ADAPTIVE_FAST_MULT
    elif pct_rank <= INDICATORS.ADAPTIVE_LOW_PCT:
        mult = INDICATORS.ADAPTIVE_SLOW_MULT
    else:
        mult = 1.0

    macd_fast   = max(3,  round(INDICATORS.MACD_FAST    * mult))
    macd_slow   = max(8,  round(INDICATORS.MACD_SLOW    * mult))
    macd_signal = max(3,  round(INDICATORS.MACD_SIGNAL  * mult))

    # MACD slow must always exceed fast to remain meaningful
    if macd_slow <= macd_fast:
        macd_slow = macd_fast + 2

    return {
        "rsi":         max(5,  round(INDICATORS.RSI_WINDOW  * mult)),
        "bb":          max(8,  round(INDICATORS.BB_WINDOW   * mult)),
        "atr":         max(5,  round(INDICATORS.ATR_WINDOW  * mult)),
        "macd_fast":   macd_fast,
        "macd_slow":   macd_slow,
        "macd_signal": macd_signal,
        "mult":        round(mult, 3),
    }


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = None,
) -> pd.DataFrame:
    """
    Average Directional Index (ADX) with +DI and -DI.

    ADX measures trend strength regardless of direction:
      ADX > 25 : trending market
      ADX < 20 : ranging / no clear trend

    Returns a DataFrame with columns:
      adx_14    — trend strength (0-100, higher = stronger trend)
      plus_di   — positive directional indicator (upward pressure)
      minus_di  — negative directional indicator (downward pressure)

    Parameters
    ----------
    high, low, close : OHLCV price series
    window           : smoothing window (default: INDICATORS.ATR_WINDOW = 14)
    """
    from config import INDICATORS
    window = window or INDICATORS.ATR_WINDOW

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm   = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=close.index,
    )
    minus_dm  = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=close.index,
    )

    alpha      = 1 / window
    atr_s      = tr.ewm(alpha=alpha, min_periods=window, adjust=False).mean()
    plus_dm_s  = plus_dm.ewm(alpha=alpha, min_periods=window, adjust=False).mean()
    minus_dm_s = minus_dm.ewm(alpha=alpha, min_periods=window, adjust=False).mean()

    plus_di  = (100 * plus_dm_s  / atr_s.replace(0, np.nan)).round(4)
    minus_di = (100 * minus_dm_s / atr_s.replace(0, np.nan)).round(4)

    di_sum   = (plus_di + minus_di).replace(0, np.nan)
    dx       = (100 * (plus_di - minus_di).abs() / di_sum).round(4)
    adx      = dx.ewm(alpha=alpha, min_periods=window, adjust=False).mean().round(4)

    return pd.DataFrame(
        {"adx_14": adx, "plus_di": plus_di, "minus_di": minus_di},
        index=close.index,
    )


def compute_stoch_rsi(
    close: pd.Series,
    rsi_window: int = None,
    stoch_window: int = None,
) -> pd.DataFrame:
    """
    Stochastic RSI — positions RSI within its own recent range.

    Unlike RSI which compares price changes, StochRSI oscillates faster
    and spends more time near extremes, making it more sensitive for
    overbought/oversold detection.

    Returns a DataFrame with columns:
      stochrsi_k — %K line (fast, smoothed with 3-day SMA)
      stochrsi_d — %D line (slow signal, 3-day SMA of %K)

    Both in [0, 1]:
      > 0.8 : overbought
      < 0.2 : oversold

    Parameters
    ----------
    close        : closing price series
    rsi_window   : RSI period (default: INDICATORS.RSI_WINDOW = 14)
    stoch_window : Stochastic lookback on RSI values (default: 14)
    """
    from config import INDICATORS
    rsi_window   = rsi_window   or INDICATORS.RSI_WINDOW
    stoch_window = stoch_window or INDICATORS.RSI_WINDOW

    rsi     = compute_rsi(close, rsi_window)
    rsi_min = rsi.rolling(stoch_window, min_periods=1).min()
    rsi_max = rsi.rolling(stoch_window, min_periods=1).max()
    rsi_rng = (rsi_max - rsi_min).replace(0, np.nan)

    raw_stoch   = ((rsi - rsi_min) / rsi_rng).round(4)
    stochrsi_k  = raw_stoch.rolling(3, min_periods=1).mean().round(4)
    stochrsi_d  = stochrsi_k.rolling(3, min_periods=1).mean().round(4)

    return pd.DataFrame(
        {"stochrsi_k": stochrsi_k, "stochrsi_d": stochrsi_d},
        index=close.index,
    )


def compute_ohlcv_indicators(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """
    Compute the full suite of technical indicators on an OHLCV DataFrame.

    This is the master function for the historical data layer. It calls
    all individual indicator helpers and assembles the results into a
    single wide DataFrame ready for storage and downstream consumption
    (forecasting, anomaly detection, AI features).

    Input
    -----
    df     : OHLCV DataFrame with columns:
             Date, Open, High, Low, Close, Volume
    symbol : ticker label (added as a column for multi-symbol parquets)

    Output
    ------
    pd.DataFrame with original OHLCV columns plus:
        returns_pct     — daily % return on Close
        sma_20          — 20-day SMA of Close
        sma_50          — 50-day SMA of Close
        ema_12          — 12-day EMA of Close (MACD fast)
        ema_26          — 26-day EMA of Close (MACD slow)
        rsi_14          — 14-period RSI
        macd            — MACD line
        macd_signal     — MACD signal line
        macd_hist       — MACD histogram
        bb_middle       — Bollinger middle band
        bb_upper        — Bollinger upper band
        bb_lower        — Bollinger lower band
        bb_width        — normalised Bollinger width
        atr_14          — 14-period ATR
        drawdown_pct    — rolling drawdown from peak
        rolling_max     — rolling max price
        volatility_21   — 21-day rolling std of returns
        momentum_10     — 10-day rate-of-change momentum
        signal_strength — z-score of returns (short window)
        volume_sma_20   — 20-day SMA of Volume (relative volume proxy)
    """
    required = {"Close", "High", "Low", "Open", "Volume"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.warning(
            "compute_ohlcv_indicators: missing columns %s for %s.",
            missing, symbol or "unknown",
        )
        return df

    from config import ROLLING
    out = df.copy()
    out["Date"] = pd.to_datetime(out.index if "Date" not in out.columns else out["Date"])
    if symbol:
        out["Symbol"] = symbol

    close  = out["Close"]
    high   = out["High"]
    low    = out["Low"]
    volume = out["Volume"]

    # -- Regime-adaptive windows ----------------------------------------
    # Select RSI/BB/ATR/MACD windows based on current vol percentile so
    # indicators respond faster in stressed markets and smoother in calm ones.
    windows = get_regime_windows(close)
    out["vol_regime_mult"] = windows["mult"]  # store for diagnostics

    # -- Returns --------------------------------------------------------
    out["returns_pct"] = close.pct_change().mul(100).round(4)

    # -- Moving averages (fixed reference levels from config — not regime-adaptive)
    out["sma_20"] = compute_sma(close, INDICATORS.SMA_FAST_WINDOW)
    out["sma_50"] = compute_sma(close, INDICATORS.SMA_SLOW_WINDOW)
    out["ema_12"] = compute_ema(close, windows["macd_fast"])
    out["ema_26"] = compute_ema(close, windows["macd_slow"])

    # -- RSI (adaptive window) ------------------------------------------
    out["rsi_14"] = compute_rsi(close, window=windows["rsi"])

    # -- MACD (adaptive windows) ----------------------------------------
    macd_df = compute_macd(
        close,
        fast=windows["macd_fast"],
        slow=windows["macd_slow"],
        signal=windows["macd_signal"],
    )
    out["macd"]        = macd_df["macd"]
    out["macd_signal"] = macd_df["macd_signal"]
    out["macd_hist"]   = macd_df["macd_hist"]

    # -- Bollinger Bands (adaptive window) ------------------------------
    bb_df = compute_bollinger_bands(close, window=windows["bb"])
    out["bb_middle"] = bb_df["bb_middle"]
    out["bb_upper"]  = bb_df["bb_upper"]
    out["bb_lower"]  = bb_df["bb_lower"]
    out["bb_width"]  = bb_df["bb_width"]
    out["bb_pct_b"]  = bb_df["bb_pct_b"]

    # -- ATR (adaptive window) ------------------------------------------
    out["atr_14"] = compute_atr(high, low, close, window=windows["atr"])

    # -- Drawdown -------------------------------------------------------
    dd_df = compute_drawdown(close)
    out["drawdown_pct"] = dd_df["drawdown_pct"]
    out["rolling_max"]  = dd_df["rolling_max"]

    # -- Rolling volatility of returns ----------------------------------
    out["volatility_21"] = compute_volatility(out["returns_pct"], 21)

    # -- Momentum (reuse existing helper) -------------------------------
    out["momentum_10"] = compute_momentum(close, 10)

    # -- Signal strength z-score (reuse existing helper) ---------------
    out["signal_strength"] = compute_signal_strength(
        out["returns_pct"], ROLLING.SHORT_WINDOW
    )

    # -- Volume SMA (relative volume proxy) ----------------------------
    out["volume_sma_20"] = compute_sma(volume.astype(float), 20)

    # -- ADX (trend strength) ------------------------------------------
    adx_df = compute_adx(high, low, close)
    out["adx_14"]   = adx_df["adx_14"]
    out["plus_di"]  = adx_df["plus_di"]
    out["minus_di"] = adx_df["minus_di"]

    # -- Stochastic RSI ------------------------------------------------
    stoch_df = compute_stoch_rsi(close)
    out["stochrsi_k"] = stoch_df["stochrsi_k"]
    out["stochrsi_d"] = stoch_df["stochrsi_d"]

    logger.info(
        "compute_ohlcv_indicators: %d rows, %d indicator columns for %s.",
        len(out), len(out.columns), symbol or "unknown",
    )
    return out.reset_index(drop=True)