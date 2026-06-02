"""
========================================================================
historical_metrics.py — Historical Analytics Engine
========================================================================
Computes all multi-day aggregate and rolling metrics used by the
Historical tab in dashboard.py and consumed by Phase 2 intelligence
modules (technical_indicators, forecasting, anomaly_detection).

Architecture
------------
- Pure computation. No I/O apart from the Parquet cache read/write
  which is isolated in _load_cached_hist() and _save_cached_hist().
- compute_hist_df() is the single-responsibility function for the
  per-day aggregate DataFrame (hist_df).
- compute_all() is the convenience wrapper that also produces the
  secondary outputs (freq_movers, cumulative_df, sessions_df, streaks_df).

Performance fix (v2)
--------------------
  The original code called _daily_aggregates() which internally called
  _apply_rolling_intelligence(), then compute_hist_df() called
  _apply_rolling_intelligence() again on the result — computing all
  rolling columns twice on every incremental update. Fixed by ensuring
  _daily_aggregates() returns raw per-day values only, and
  _apply_rolling_intelligence() is called exactly once at the end of
  compute_hist_df(), across the full sorted series.

Incremental update strategy
----------------------------
  On each run, compute_hist_df() checks the cached file against the
  current dataset. If dates and row counts match, the cache is returned
  immediately. If new dates have arrived, only those new dates are
  aggregated and appended before re-running the rolling pass once.

Output DataFrames
-----------------
  hist_df       — one row per trading day, daily and rolling aggregates
  freq_movers   — symbols ranked by top-N gainer appearances
  cumulative_df — per-symbol cumulative % change since first day
  sessions_df   — strongest/weakest individual sessions
  streaks_df    — longest consecutive bullish/bearish runs

Pip dependencies
----------------
  pip install pandas numpy pyarrow
========================================================================
"""

import logging
import os

import numpy as np
import pandas as pd

from config import PATHS, ROLLING, BREADTH

logger = logging.getLogger(__name__)

HIST_PARQUET = PATHS.HIST_PARQUET


# ══════════════════════════════════════════════════════════════════════
# DAILY AGGREGATE METRICS  (raw, no rolling)
# ══════════════════════════════════════════════════════════════════════

def _daily_aggregates(df_h: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-day aggregate metrics from a price DataFrame.

    Returns raw daily values only — no rolling columns. Rolling
    intelligence is applied exactly once by _apply_rolling_intelligence()
    after the full series has been assembled.

    Columns produced
    ----------------
    Date, n_gainers, n_losers, n_flat, avg_pct, volatility,
    breadth_pct, regime, best_stock, best_pct, worst_stock, worst_pct,
    streak_sign
    """
    grp = df_h.groupby("Date")

    agg = grp["Change"].agg(
        n_gainers=lambda x: (x > 0).sum(),
        n_losers =lambda x: (x < 0).sum(),
        n_flat   =lambda x: (x == 0).sum(),
    ).reset_index()

    agg["avg_pct"]     = grp["% Change"].mean().values
    agg["volatility"]  = grp["% Change"].std().values
    agg["skew_pct"]    = grp["% Change"].skew().values.round(4)
    agg["kurt_pct"]    = grp["% Change"].apply(lambda x: x.kurt()).values.round(4)
    agg["breadth_pct"] = (
        agg["n_gainers"] /
        (agg["n_gainers"] + agg["n_losers"] + agg["n_flat"])
        * 100
    ).round(2)

    # Dynamic regime classification from breadth
    agg["regime"] = np.where(
        agg["breadth_pct"] >= BREADTH.BULLISH_PCT, "Bullish",
        np.where(agg["breadth_pct"] <= BREADTH.BEARISH_PCT, "Bearish", "Mixed"),
    )

    # Best and worst stock each day
    idx_best  = grp["% Change"].idxmax()
    idx_worst = grp["% Change"].idxmin()
    agg["best_stock"]  = df_h.loc[idx_best,  "Symbol"].values
    agg["best_pct"]    = df_h.loc[idx_best,  "% Change"].values
    agg["worst_stock"] = df_h.loc[idx_worst, "Symbol"].values
    agg["worst_pct"]   = df_h.loc[idx_worst, "% Change"].values

    agg["streak_sign"] = np.sign(agg["avg_pct"])

    return agg.sort_values("Date").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
# ROLLING INTELLIGENCE PASS
# Called exactly once on the full sorted series — never inside the
# per-day aggregation loop.
# ══════════════════════════════════════════════════════════════════════

def _persistence(series: pd.Series, window: int) -> pd.Series:
    """Rolling positive-observation share as a percentage."""
    return series.gt(0).rolling(window, min_periods=1).mean().mul(100).round(2)


def _vol_regime(vol: pd.Series, baseline: pd.Series) -> pd.Series:
    """Classify volatility as Elevated / Slightly High / Normal / Suppressed."""
    ratio = vol / baseline.replace(0, np.nan)
    return pd.Series(
        np.where(ratio >= ROLLING.VOL_ELEVATED,      "Elevated",
        np.where(ratio >= ROLLING.VOL_SLIGHTLY_HIGH, "Slightly High",
        np.where(ratio >= ROLLING.VOL_NORMAL_LOW,    "Normal", "Suppressed"))),
        index=vol.index,
    )


def _adaptive_regime(
    breadth: pd.Series,
    lookback: int,
    bull_pct: float,
    bear_pct: float,
) -> pd.Series:
    """
    Classify market regime using the rolling percentile rank of breadth_pct.

    Rather than comparing breadth against fixed thresholds (e.g. >=60 = Bullish),
    this compares it against its own recent history. A breadth of 62% during a
    sustained bull market means less than 62% after weeks of bear readings.

    pct_rank = fraction of the last `lookback` breadth values that are <= today's
    breadth — i.e. the empirical CDF at the current reading.

    Parameters
    ----------
    breadth  : daily breadth_pct Series
    lookback : rolling window size (from config.BREADTH.REGIME_LOOKBACK)
    bull_pct : rank >= this → "Bullish" (from config.BREADTH.REGIME_BULL_PCT)
    bear_pct : rank <= this → "Bearish" (from config.BREADTH.REGIME_BEAR_PCT)
    """
    pct_rank = (
        breadth
        .rolling(lookback, min_periods=10)
        .apply(lambda x: float((x <= x[-1]).mean()), raw=True)
    )
    return pd.Series(
        np.where(pct_rank >= bull_pct, "Bullish",
        np.where(pct_rank <= bear_pct, "Bearish", "Mixed")),
        index=breadth.index,
    )


def _apply_rolling_intelligence(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Attach all rolling intelligence columns to the daily aggregate frame.

    Called once on the full sorted series after aggregation — never on
    incremental slices, so rolling windows always see the full history.

    Columns added
    -------------
    rolling_avg_7d, rolling_avg_30d, rolling_vol_7d, rolling_vol_30d,
    sma_avg_7d, sma_avg_30d, ema_avg_7d, ema_avg_30d,
    trend_persistence_7d, breadth_trend_persistence_7d,
    rolling_signal_7d, volatility_regime
    """
    agg = agg.sort_values("Date").reset_index(drop=True)
    sw  = ROLLING.SHORT_WINDOW   # 7
    lw  = ROLLING.LONG_WINDOW    # 30

    # Rolling means of avg_pct (used both as raw rolling and as SMA alias)
    rolling_avg_7d  = agg["avg_pct"].rolling(sw, min_periods=1).mean().round(4)
    rolling_avg_30d = agg["avg_pct"].rolling(lw, min_periods=1).mean().round(4)

    agg["rolling_avg_7d"]  = rolling_avg_7d
    agg["rolling_avg_30d"] = rolling_avg_30d
    agg["rolling_vol_7d"]  = agg["volatility"].rolling(sw, min_periods=1).mean().round(4)
    agg["rolling_vol_30d"] = agg["volatility"].rolling(lw, min_periods=1).mean().round(4)

    # SMA aliases (same as rolling means — named separately for clarity in consumers)
    agg["sma_avg_7d"]  = rolling_avg_7d
    agg["sma_avg_30d"] = rolling_avg_30d

    # EMA of avg_pct
    agg["ema_avg_7d"]  = agg["avg_pct"].ewm(span=sw, adjust=False, min_periods=1).mean().round(4)
    agg["ema_avg_30d"] = agg["avg_pct"].ewm(span=lw, adjust=False, min_periods=1).mean().round(4)

    # Trend and breadth persistence (% of window where value > 0)
    agg["trend_persistence_7d"]          = _persistence(agg["avg_pct"],              sw)
    agg["breadth_trend_persistence_7d"]  = _persistence(agg["breadth_pct"] - 50,     sw)

    # Normalised rolling signal strength (z-score of avg_pct within window)
    roll_std = agg["avg_pct"].rolling(sw, min_periods=2).std().replace(0, np.nan)
    agg["rolling_signal_7d"] = (
        (agg["avg_pct"] - agg["rolling_avg_7d"]) / roll_std
    ).round(4)

    # Volatility regime classification
    agg["volatility_regime"] = _vol_regime(agg["volatility"], agg["rolling_vol_30d"])

    # Volatility percentile rank — fraction of the rolling window below today's vol.
    # raw=True passes a numpy array so the lambda runs in pure numpy (no Series overhead).
    agg["vol_pct_rank"] = (
        agg["volatility"]
        .rolling(lw, min_periods=5)
        .apply(lambda x: float((x <= x[-1]).mean()), raw=True)
        .round(4)
    )

    # ── Tail-risk metrics (all use ROLLING.VAR_QUANTILE = 5th percentile) ──
    # Compute the 5th and 95th percentile once each — reused by var_5 and
    # tail_ratio to avoid computing the same rolling quantile twice.
    _q   = ROLLING.VAR_QUANTILE
    _p05 = agg["avg_pct"].rolling(lw, min_periods=10).quantile(_q)
    _p95 = agg["avg_pct"].rolling(lw, min_periods=10).quantile(1 - _q)

    # VaR_5: 5th-percentile market return over rolling window.
    # Negative value = market loses at least this much on the worst 5% of days.
    agg["var_5"] = _p05.round(4)

    # CVaR_5 (Expected Shortfall): mean return on days that fell below VaR_5.
    # Always <= var_5; captures the severity of tail losses, not just their threshold.
    def _cvar(x: np.ndarray) -> float:
        threshold = np.quantile(x, _q)
        tail = x[x <= threshold]
        return float(tail.mean()) if len(tail) > 0 else float(threshold)

    agg["cvar_5"] = (
        agg["avg_pct"]
        .rolling(lw, min_periods=10)
        .apply(_cvar, raw=True)
        .round(4)
    )

    # Tail ratio: upside 95th percentile / absolute downside 5th percentile.
    # > 1 = fatter upside tail; < 1 = fatter downside tail.
    agg["tail_ratio"] = (
        _p95 / _p05.abs().replace(0, np.nan)
    ).round(4)

    # ── Adaptive regime ────────────────────────────────────────────────
    # Uses rolling breadth percentile rank so the same raw breadth reading
    # is interpreted relative to the recent market environment, not a fixed cutoff.
    agg["regime_adaptive"] = _adaptive_regime(
        agg["breadth_pct"],
        BREADTH.REGIME_LOOKBACK,
        BREADTH.REGIME_BULL_PCT,
        BREADTH.REGIME_BEAR_PCT,
    )

    return agg


# ══════════════════════════════════════════════════════════════════════
# MOST FREQUENT TOP MOVERS
# ══════════════════════════════════════════════════════════════════════

def _frequent_movers(
    df_h: pd.DataFrame,
    n_top: int = None,
    top_n_per_day: int = None,
) -> pd.DataFrame:
    """
    Identify symbols that appear most often in the daily top-N gainers.

    Parameters
    ----------
    df_h         : full price history DataFrame
    n_top        : return this many symbols (default: ROLLING.TOP_MOVERS_N)
    top_n_per_day: count the top N stocks per day (default: ROLLING.TOP_MOVERS_PER_DAY)

    Returns
    -------
    pd.DataFrame — Symbol, Appearances, Pct_Days
    """
    n_top         = n_top         or ROLLING.TOP_MOVERS_N
    top_n_per_day = top_n_per_day or ROLLING.TOP_MOVERS_PER_DAY
    n_days        = df_h["Date"].nunique()

    gainers = df_h[df_h["% Change"] > 0].copy()
    if gainers.empty:
        return pd.DataFrame(columns=["Symbol", "Appearances", "Pct_Days"])

    def _top_syms(grp: pd.DataFrame) -> pd.Series:
        return grp.nlargest(top_n_per_day, "% Change")["Symbol"]

    all_tops = gainers.groupby("Date", group_keys=False).apply(_top_syms)
    counts   = all_tops.value_counts().head(n_top).reset_index()
    counts.columns = ["Symbol", "Appearances"]
    counts["Pct_Days"] = (counts["Appearances"] / n_days * 100).round(1)
    return counts.reset_index(drop=True)


def _frequent_losers(
    df_h: pd.DataFrame,
    n_top: int = None,
    top_n_per_day: int = None,
) -> pd.DataFrame:
    """
    Identify symbols that appear most often in the daily top-N decliners.
    Mirror of _frequent_movers for the downside.
    """
    n_top         = n_top         or ROLLING.TOP_MOVERS_N
    top_n_per_day = top_n_per_day or ROLLING.TOP_MOVERS_PER_DAY
    n_days        = df_h["Date"].nunique()

    losers = df_h[df_h["% Change"] < 0].copy()
    if losers.empty:
        return pd.DataFrame(columns=["Symbol", "Appearances", "Pct_Days"])

    def _bottom_syms(grp: pd.DataFrame) -> pd.Series:
        return grp.nsmallest(top_n_per_day, "% Change")["Symbol"]

    all_bottoms = losers.groupby("Date", group_keys=False).apply(_bottom_syms)
    counts      = all_bottoms.value_counts().head(n_top).reset_index()
    counts.columns = ["Symbol", "Appearances"]
    counts["Pct_Days"] = (counts["Appearances"] / n_days * 100).round(1)
    return counts.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
# CUMULATIVE PERFORMANCE
# ══════════════════════════════════════════════════════════════════════

def _cumulative_performance(df_h: pd.DataFrame, top_n: int = None) -> pd.DataFrame:
    """
    Compute compounded cumulative % change per symbol from first appearance.

    Only symbols present in at least 50 % of trading days are included
    to avoid sparse series distorting the chart.

    Returns
    -------
    Wide-format DataFrame: rows=dates, columns=symbol tickers
    """
    top_n    = top_n or ROLLING.TOP_MOVERS_N
    n_days   = df_h["Date"].nunique()
    min_days = max(1, int(n_days * ROLLING.MIN_APPEARANCE_PCT))

    sym_counts = df_h.groupby("Symbol")["Date"].nunique()
    valid_syms = sym_counts[sym_counts >= min_days].index

    sub = df_h[df_h["Symbol"].isin(valid_syms)][["Date", "Symbol", "% Change"]].copy()
    if sub.empty:
        return pd.DataFrame(columns=["Date"])

    pivot = sub.pivot_table(
        index="Date", columns="Symbol", values="% Change", aggfunc="mean"
    ).sort_index()

    cum = (1 + pivot / 100).cumprod() - 1
    cum = (cum * 100).round(4)

    final_returns = cum.apply(
        lambda s: s.dropna().iloc[-1] if not s.dropna().empty else np.nan
    ).dropna().sort_values(ascending=False)

    if final_returns.empty:
        return pd.DataFrame(columns=["Date"])

    top_syms    = final_returns.head(top_n).index.tolist()
    bottom_syms = final_returns.tail(top_n).index.tolist()
    selected    = list(dict.fromkeys(top_syms + bottom_syms))
    return cum[selected].reset_index()


# ══════════════════════════════════════════════════════════════════════
# STRONGEST / WEAKEST SESSIONS
# ══════════════════════════════════════════════════════════════════════

def _extreme_sessions(hist_agg: pd.DataFrame, n: int = None) -> pd.DataFrame:
    """
    Return the n strongest and n weakest sessions by average % change.

    Parameters
    ----------
    hist_agg : output of _apply_rolling_intelligence()
    n        : sessions per extreme (default: ROLLING.EXTREME_SESSIONS_N)
    """
    n = n or ROLLING.EXTREME_SESSIONS_N

    if hist_agg.empty:
        return hist_agg.assign(direction=pd.Series(dtype="str"))

    total = len(hist_agg)
    if total == 1:
        result = hist_agg.copy()
    else:
        per_side = min(n, total // 2)
        weakest  = hist_agg.nsmallest(per_side, "avg_pct").copy()
        strongest = hist_agg.drop(index=weakest.index).nlargest(per_side, "avg_pct").copy()
        result   = pd.concat([strongest, weakest]).sort_values("avg_pct", ascending=False)

    result["direction"] = np.where(result["avg_pct"] >= 0, "Bullish", "Bearish")
    return result.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
# CONSECUTIVE STREAK ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def _streak_analysis(hist_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Identify consecutive bullish/bearish day-streaks.

    Returns
    -------
    pd.DataFrame — start_date, end_date, length, direction, avg_pct
    Sorted by length descending.
    """
    if hist_agg.empty:
        return pd.DataFrame(columns=[
            "start_date", "end_date", "length", "direction", "avg_pct",
        ])

    df = hist_agg[["Date", "avg_pct", "streak_sign"]].copy()
    df["group"] = (df["streak_sign"] != df["streak_sign"].shift()).cumsum()

    streaks = []
    for _, grp in df.groupby("group"):
        sign = grp["streak_sign"].iloc[0]
        if sign == 0:
            continue
        streaks.append({
            "start_date": grp["Date"].iloc[0],
            "end_date":   grp["Date"].iloc[-1],
            "length":     len(grp),
            "direction":  "Bullish" if sign > 0 else "Bearish",
            "avg_pct":    round(float(grp["avg_pct"].mean()), 4),
        })

    if not streaks:
        return pd.DataFrame(columns=[
            "start_date", "end_date", "length", "direction", "avg_pct",
        ])
    return (
        pd.DataFrame(streaks)
        .sort_values("length", ascending=False)
        .reset_index(drop=True)
    )


# ══════════════════════════════════════════════════════════════════════
# INCREMENTAL CACHE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════

def _load_cached_hist() -> pd.DataFrame:
    """Load cached hist_df from Parquet if it exists and is readable."""
    if os.path.exists(HIST_PARQUET):
        try:
            df = pd.read_parquet(HIST_PARQUET, engine="pyarrow")
            df["Date"] = pd.to_datetime(df["Date"])
            return df
        except Exception as exc:
            logger.warning(
                "historical_metrics: could not read cache (%s). Rebuilding.", exc
            )
    return pd.DataFrame()


def _save_cached_hist(hist: pd.DataFrame) -> None:
    """Write hist_df to Parquet cache. Logs but does not raise on failure."""
    try:
        hist.to_parquet(HIST_PARQUET, index=False, engine="pyarrow")
        logger.info("historical_metrics: cache written → %s.", HIST_PARQUET)
    except Exception as exc:
        logger.warning("historical_metrics: cache write failed (%s).", exc)


# ══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════

def compute_hist_df(df_full: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """
    Return the per-day aggregate metrics DataFrame.

    Incremental strategy
    --------------------
    1. Load the cached hist_df.
    2. If dates and row-counts match the current dataset, return cache.
    3. If new dates exist, aggregate only those dates, append to the
       cached history, then run _apply_rolling_intelligence() once on
       the full combined series.
    4. If force=True, recompute everything from scratch.

    Parameters
    ----------
    df_full : full price history from data_loader.load_price_data()
    force   : recompute all dates even if cache is current

    Returns
    -------
    pd.DataFrame — one row per trading day with all rolling metrics
    """
    df_full = df_full.copy()
    df_full["Date"] = pd.to_datetime(df_full["Date"])
    all_dates = sorted(df_full["Date"].unique())

    if not force:
        cached = _load_cached_hist()
        if not cached.empty:
            # Schema migration: if aggregate columns added after the cache was
            # written are missing, discard the cache and do a full recompute.
            # (These come from _daily_aggregates, so _apply_rolling_intelligence
            #  cannot patch them — only a full recompute can.)
            required_agg_cols = {"skew_pct", "kurt_pct"}
            if not required_agg_cols.issubset(cached.columns):
                logger.info(
                    "historical_metrics: cache missing %s — forcing full recompute.",
                    required_agg_cols - set(cached.columns),
                )
                cached = pd.DataFrame()

        if not cached.empty:
            cached_latest  = cached["Date"].max()
            data_latest    = df_full["Date"].max()
            cached_dates   = set(pd.to_datetime(cached["Date"]).dt.normalize())
            data_dates     = set(pd.to_datetime(df_full["Date"]).dt.normalize())

            # Row-count check: cached totals vs current dataset per day
            cached_counts = (
                cached.set_index(cached["Date"].dt.normalize())
                [["n_gainers", "n_losers", "n_flat"]]
                .sum(axis=1)
                .to_dict()
            )
            data_counts = df_full.groupby(df_full["Date"].dt.normalize()).size().to_dict()

            if (
                cached_latest == data_latest
                and cached_dates == data_dates
                and cached_counts == data_counts
            ):
                # Cache is current — ensure rolling columns are present.
                # vol_pct_rank added by _apply_rolling_intelligence; if missing
                # from an older cache, regenerate it without a full recompute.
                required_cols = {
                    "ema_avg_7d", "trend_persistence_7d",
                    "volatility_regime", "rolling_signal_7d",
                    "vol_pct_rank", "var_5", "cvar_5", "tail_ratio", "regime_adaptive",
                }
                if not required_cols.issubset(cached.columns):
                    cached = _apply_rolling_intelligence(cached)
                    _save_cached_hist(cached)
                logger.info(
                    "historical_metrics: cache current (%s). Skipping recompute.",
                    str(cached_latest.date()),
                )
                return cached

            # Incremental: aggregate only new dates, then roll the full series
            new_dates = [d for d in all_dates if d > cached_latest]
            if new_dates:
                logger.info(
                    "historical_metrics: computing %d new day(s) incrementally.",
                    len(new_dates),
                )
                new_df   = df_full[df_full["Date"].isin(new_dates)]
                new_agg  = _daily_aggregates(new_df)
                combined = pd.concat([cached, new_agg], ignore_index=True)
                combined = _apply_rolling_intelligence(combined)
                _save_cached_hist(combined)
                return combined

    # Full recompute path
    logger.info(
        "historical_metrics: full recompute across %d day(s).", len(all_dates)
    )
    raw  = _daily_aggregates(df_full)
    hist = _apply_rolling_intelligence(raw)
    _save_cached_hist(hist)
    return hist


# ══════════════════════════════════════════════════════════════════════
# OHLCV-EXTENDED HISTORY FOR FORECASTING
# Builds a long-baseline hist_df by prepending years of per-symbol OHLCV
# data to the scraped daily aggregates.  Only used by forecasting — the
# regular hist_df (compute_hist_df) still drives all other tabs.
# ══════════════════════════════════════════════════════════════════════

def _ohlcv_daily_aggregates() -> pd.DataFrame:
    """
    Compute daily cross-sectional market metrics from all cached OHLCV files.

    For each trading day in the OHLCV history:
      - avg_pct    : mean daily % change across all symbols
      - volatility : std dev of daily % changes
      - breadth_pct: % of advancing symbols

    Returns a DataFrame with the same raw schema as _daily_aggregates()
    so it merges cleanly with scraped aggregates in build_extended_hist_df().
    """
    import data_loader as _dl  # local import — avoids circular dependency risk

    symbols = _dl.list_available_ohlcv()
    if not symbols:
        logger.info("historical_metrics: no OHLCV files found — skipping OHLCV aggregation.")
        return pd.DataFrame()

    frames = []
    for sym in symbols:
        ohlcv = _dl.load_ohlcv(sym)
        if ohlcv.empty or "Close" not in ohlcv.columns or len(ohlcv) < 2:
            continue
        o = ohlcv[["Date", "Close"]].copy().sort_values("Date").reset_index(drop=True)
        o["Date"] = pd.to_datetime(o["Date"])
        if o["Date"].dt.tz is not None:
            o["Date"] = o["Date"].dt.tz_convert(None)
        o["Date"]        = o["Date"].dt.normalize()
        o["pct_change"]  = o["Close"].pct_change() * 100
        o["abs_change"]  = o["Close"].diff()
        frames.append(o[["Date", "pct_change", "abs_change"]].dropna())

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])

    grp = combined.groupby("Date")

    n_gainers  = grp["abs_change"].apply(lambda x: int((x > 0).sum()))
    n_losers   = grp["abs_change"].apply(lambda x: int((x < 0).sum()))
    n_flat     = grp["abs_change"].apply(lambda x: int((x == 0).sum()))
    avg_pct    = grp["pct_change"].mean().round(4)
    volatility = grp["pct_change"].std().fillna(0.0).round(4)
    skew_pct   = grp["pct_change"].skew().fillna(0.0).round(4)
    kurt_pct   = grp["pct_change"].apply(lambda x: x.kurt()).fillna(0.0).round(4)
    sym_count  = grp["pct_change"].count()

    agg = pd.DataFrame({
        "Date":       avg_pct.index,
        "n_gainers":  n_gainers.values,
        "n_losers":   n_losers.values,
        "n_flat":     n_flat.values,
        "avg_pct":    avg_pct.values,
        "volatility": volatility.values,
        "skew_pct":   skew_pct.values,
        "kurt_pct":   kurt_pct.values,
    })

    # Drop dates where fewer than 10 symbols have data (early history / gaps)
    agg = agg[sym_count.values >= 10].copy().reset_index(drop=True)

    agg["breadth_pct"] = (
        agg["n_gainers"] /
        (agg["n_gainers"] + agg["n_losers"] + agg["n_flat"])
        * 100
    ).round(2)

    agg["regime"] = np.where(
        agg["breadth_pct"] >= BREADTH.BULLISH_PCT, "Bullish",
        np.where(agg["breadth_pct"] <= BREADTH.BEARISH_PCT, "Bearish", "Mixed"),
    )

    # Placeholder display columns — not used by forecasting or rolling intelligence
    agg["best_stock"]  = ""
    agg["best_pct"]    = 0.0
    agg["worst_stock"] = ""
    agg["worst_pct"]   = 0.0
    agg["streak_sign"] = np.sign(agg["avg_pct"])
    agg["source"]      = "ohlcv"

    logger.info(
        "historical_metrics: OHLCV aggregation — %d days, %d symbols.",
        len(agg), len(symbols),
    )
    return agg.sort_values("Date").reset_index(drop=True)


def build_extended_df_full(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Build an extended df_full by prepending OHLCV-derived per-symbol
    % Change history to the scraped price data.

    Used by signal_engine detectors (detect_breakout, detect_unusual_movers,
    compute_risk_scores, compute_opportunity_ranking) so their z-score
    baselines draw from 2+ years of real price history per symbol instead
    of just the scraped days.

    Strategy
    --------
    For each tracked symbol with OHLCV data:
      - Compute daily % Change from Close prices
      - Keep only dates NOT already in the scraped dataset
      - Concatenate OHLCV history first, then scraped data (scraped wins on overlap)

    Parameters
    ----------
    df_full : scraped price history from data_loader.load_price_data()

    Returns
    -------
    pd.DataFrame — same schema as df_full (Date, Symbol, Company Name,
                   Price, Change, % Change), sorted ascending by Date.
    """
    import data_loader as _dl

    symbols      = df_full["Symbol"].unique().tolist()
    df_full      = df_full.copy()
    df_full["Date"] = pd.to_datetime(df_full["Date"]).dt.normalize()
    scraped_dates = set(df_full["Date"].unique())

    company_map = df_full.groupby("Symbol")["Company Name"].first().to_dict()

    frames = []
    for sym in symbols:
        ohlcv = _dl.load_ohlcv(sym)
        if ohlcv.empty or "Close" not in ohlcv.columns or len(ohlcv) < 2:
            continue

        o = ohlcv[["Date", "Close"]].copy().sort_values("Date").reset_index(drop=True)
        o["Date"] = pd.to_datetime(o["Date"])
        if o["Date"].dt.tz is not None:
            o["Date"] = o["Date"].dt.tz_convert(None)
        o["Date"] = o["Date"].dt.normalize()

        # Only keep pre-scraper dates
        o = o[~o["Date"].isin(scraped_dates)].copy()
        if o.empty:
            continue

        o["pct"]    = o["Close"].pct_change() * 100
        o["chg"]    = o["Close"].diff()
        o           = o.dropna(subset=["pct"])

        o["Symbol"]       = sym
        o["Company Name"] = company_map.get(sym, "")
        o["Price"]        = o["Close"].round(4)
        o["Change"]       = o["chg"].round(4)
        o["% Change"]     = o["pct"].round(4)

        frames.append(o[["Date", "Symbol", "Company Name", "Price", "Change", "% Change"]])

    if not frames:
        logger.info("build_extended_df_full: no OHLCV data available — returning scraped df_full.")
        return df_full

    ohlcv_rows = pd.concat(frames, ignore_index=True)
    extended   = pd.concat([ohlcv_rows, df_full], ignore_index=True)
    extended   = extended.sort_values("Date").reset_index(drop=True)

    logger.info(
        "build_extended_df_full: %d total rows (%d OHLCV-prior + %d scraped) across %d symbols.",
        len(extended), len(ohlcv_rows), len(df_full), len(symbols),
    )
    return extended


def build_extended_hist_df(df_full: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """
    Build an extended hist_df for forecasting by prepending years of OHLCV
    history to the scraped daily aggregates.

    Strategy
    --------
    1. Derive raw daily aggregates from the scraped price data.
    2. Derive raw daily aggregates from all cached OHLCV files.
    3. Remove OHLCV rows that overlap with scraped dates — scraped data
       captures the actual "most active" universe and takes precedence.
    4. Concatenate (OHLCV history first, then scraped) and run
       _apply_rolling_intelligence() once on the full combined series so
       all rolling windows see the complete long-term baseline.

    Parameters
    ----------
    df_full : full price history from data_loader.load_price_data()
    force   : passed through to compute_hist_df (forces cache rebuild)

    Returns
    -------
    pd.DataFrame — same schema as hist_df, ascending by Date.
    Only pass this to forecasting.forecast_market_metrics(); use the
    regular compute_hist_df() for all Historical-tab rendering.
    """
    # Raw scraped aggregates (no rolling columns yet)
    scraped_raw = _daily_aggregates(df_full.copy())
    scraped_raw["Date"]   = pd.to_datetime(scraped_raw["Date"]).dt.normalize()
    scraped_raw["source"] = "scraped"
    scraped_dates         = set(scraped_raw["Date"])

    ohlcv_raw = _ohlcv_daily_aggregates()

    if ohlcv_raw.empty:
        logger.info("build_extended_hist_df: no OHLCV data — falling back to scraped hist_df.")
        return compute_hist_df(df_full, force=force)

    ohlcv_raw["Date"] = pd.to_datetime(ohlcv_raw["Date"]).dt.normalize()

    # Keep only OHLCV dates not already covered by the scraper
    ohlcv_prior = ohlcv_raw[~ohlcv_raw["Date"].isin(scraped_dates)].copy()

    if ohlcv_prior.empty:
        logger.info("build_extended_hist_df: all OHLCV dates overlap scraped data — using scraped.")
        return compute_hist_df(df_full, force=force)

    # Align columns so concat is clean
    all_cols = list(scraped_raw.columns)
    for col in all_cols:
        if col not in ohlcv_prior.columns:
            ohlcv_prior[col] = np.nan

    combined_raw = pd.concat(
        [ohlcv_prior[all_cols], scraped_raw[all_cols]],
        ignore_index=True,
    ).sort_values("Date").reset_index(drop=True)

    # Single rolling pass across the full combined series
    extended = _apply_rolling_intelligence(combined_raw)

    logger.info(
        "build_extended_hist_df: %d total days (%d OHLCV-prior + %d scraped).",
        len(extended), len(ohlcv_prior), len(scraped_raw),
    )
    return extended


def compute_all(df_full: pd.DataFrame, force: bool = False) -> dict:
    """
    Compute all historical analytics in one call.

    Parameters
    ----------
    df_full : full price history from data_loader.load_price_data()
    force   : force full recompute of hist_df cache

    Returns
    -------
    dict with keys:
        hist_df        — pd.DataFrame: per-day aggregates + rolling metrics
        freq_movers    — pd.DataFrame: symbol appearance frequency
        cumulative_df  — pd.DataFrame: compounded cumulative returns
        sessions_df    — pd.DataFrame: strongest/weakest sessions
        streaks_df     — pd.DataFrame: consecutive bullish/bearish streaks
        n_days         — int: total trading days in dataset
        has_history    — bool: True if >= 2 days of data
        latest_vol     — float: latest day volatility
        baseline_vol   — float: rolling historical average volatility
    """
    hist_df       = compute_hist_df(df_full, force=force)
    freq_movers   = _frequent_movers(df_full)
    freq_losers   = _frequent_losers(df_full)
    cumulative_df = _cumulative_performance(df_full)
    sessions_df   = _extreme_sessions(hist_df)
    streaks_df    = _streak_analysis(hist_df)
    n_days        = len(hist_df)
    has_history   = n_days >= 2

    if has_history:
        latest_vol   = float(hist_df["volatility"].iloc[-1])
        baseline_vol = float(hist_df["volatility"].iloc[:-1].mean())
    else:
        latest_vol   = float(df_full["% Change"].std()) if not df_full.empty else 0.0
        baseline_vol = latest_vol

    logger.info(
        "historical_metrics: compute_all complete — %d day(s), latest vol %.4f.",
        n_days, latest_vol,
    )

    return {
        "hist_df":       hist_df,
        "freq_movers":   freq_movers,
        "freq_losers":   freq_losers,
        "cumulative_df": cumulative_df,
        "sessions_df":   sessions_df,
        "streaks_df":    streaks_df,
        "n_days":        n_days,
        "has_history":   has_history,
        "latest_vol":    latest_vol,
        "baseline_vol":  baseline_vol,
    }