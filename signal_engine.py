"""
========================================================================
signal_engine.py — Market Pulse Signal Generation Engine
========================================================================
Generates actionable market signals from price, technical indicator,
volatility, and sentiment data. Pure computation — no I/O, no plotting.

Architecture
------------
- All thresholds come from config.py (SIGNALS block).
- Signals are standardised DataFrames with a canonical schema so new
  signal types can be added without changing dashboard.py.
- Each signal row carries: Symbol, Date, signal_type, direction,
  strength, score, reason.
- compute_all_signals() is the single public entry point.

Signal types
------------
  momentum_breakout   : z-score of today's move vs own short window
  volatility_spike    : absolute move exceeds market volatility threshold
  sentiment_divergence: sentiment and price direction conflict
  unusual_mover       : move is statistically unusual vs full history
  risk_scores         : composite per-symbol risk (not in all_signals)
  opportunities       : composite opportunity ranking (not in all_signals)

Integration fixes (v2)
----------------------
- SIGNAL_SCHEMA: canonical column list extracted as a module constant,
  shared with data_loader and validators for consistent schema checks.
- detect_breakout: replaced per-symbol copy() with a simple filter;
  removed redundant sort_values (df_full is already sorted by data_loader).
- detect_volatility_spikes: the ``threshold`` calculation now guards
  against mkt_std being zero (single-stock or identical prices edge case).
- detect_sentiment_divergence: row.get() on a pd.Series is correct but
  requires explicit handling of NaN; added pd.isna guard before
  applying threshold comparisons.
- compute_opportunity_ranking: row.get() replaced with safe column
  access to avoid silent AttributeError on Series vs dict get.
- compute_all_signals: logging now includes per-type counts in a
  consistent format.
========================================================================
"""

import logging

import numpy as np
import pandas as pd

from config import ROLLING, SIGNALS, TECHNICAL
from technical_indicators import compute_signal_strength

logger = logging.getLogger(__name__)

# Canonical signal output columns — shared with data_loader and validators
SIGNAL_SCHEMA: list[str] = [
    "Symbol", "Date", "signal_type", "direction", "strength", "score", "reason"
]

# Module-level threshold aliases (all values from config — never hardcoded)
_BREAKOUT_Z           = SIGNALS.BREAKOUT_Z
_VOL_SPIKE_RATIO      = SIGNALS.VOL_SPIKE_RATIO
_SENT_DIVERGE_MIN     = SIGNALS.SENT_DIVERGE_MIN
_SENT_DIVERGE_PRICE   = SIGNALS.SENT_DIVERGE_PRICE
_UNUSUAL_MOVER_Z      = SIGNALS.UNUSUAL_MOVER_Z
_OPPORTUNITY_TOP_N    = SIGNALS.OPPORTUNITY_TOP_N

# Minimum history required for symbol-level signal calculations (from config).
# Defined here at module level so it is available to all detectors below.
TECHNICAL_MIN_HISTORY: int = TECHNICAL.MIN_SYMBOL_HISTORY


def _empty_signal_df() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical signal schema."""
    return pd.DataFrame(columns=SIGNAL_SCHEMA)


# ======================================================================
# INDIVIDUAL SIGNAL DETECTORS
# ======================================================================

def detect_breakout(
    df_latest: pd.DataFrame,
    df_full:   pd.DataFrame,
    today: pd.Timestamp = None,
) -> pd.DataFrame:
    """
    Identify momentum breakout signals for today's active stocks.

    A breakout is a stock whose current % Change sits more than
    ``_BREAKOUT_Z`` standard deviations above its own rolling mean
    over the short window. Captures unusual positive (or negative)
    momentum relative to the stock's recent history.

    Parameters
    ----------
    df_latest : latest-day price slice
    df_full   : full historical price data (sorted ascending by Date)

    Returns
    -------
    pd.DataFrame with SIGNAL_SCHEMA columns, sorted by score descending.
    """
    today   = today or pd.to_datetime(df_latest["Date"].max())
    signals = []

    for _, row in df_latest.iterrows():
        sym    = row["Symbol"]
        pct    = row["% Change"]
        sym_df = df_full[df_full["Symbol"] == sym]

        if len(sym_df) < TECHNICAL_MIN_HISTORY:
            continue

        strength = compute_signal_strength(sym_df["% Change"], ROLLING.SHORT_WINDOW)
        latest_z = float(strength.iloc[-1]) if pd.notna(strength.iloc[-1]) else 0.0

        if abs(latest_z) < _BREAKOUT_Z:
            continue

        signals.append({
            "Symbol":      sym,
            "Date":        today,
            "signal_type": "momentum_breakout",
            "direction":   "Bullish" if pct >= 0 else "Bearish",
            "strength":    round(latest_z, 4),
            "score":       round(abs(latest_z), 4),
            "reason": (
                f"{sym} moved {pct:+.2f}% today — "
                f"z-score {latest_z:+.2f} vs {ROLLING.SHORT_WINDOW}-day history"
            ),
        })

    if not signals:
        return _empty_signal_df()

    return (
        pd.DataFrame(signals)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )




def detect_volatility_spikes(
    df_latest:    pd.DataFrame,
    baseline_vol: float,
    today: pd.Timestamp = None,
) -> pd.DataFrame:
    """
    Identify stocks whose absolute % Change exceeds the market volatility
    baseline by a significant margin.

    Threshold = max(market_std * ratio, baseline_vol * ratio) where
    ratio = SIGNALS.VOL_SPIKE_RATIO. This prevents the threshold from
    collapsing to near-zero on calm days.

    Parameters
    ----------
    df_latest    : latest-day price slice
    baseline_vol : rolling average volatility from historical_metrics

    Returns
    -------
    pd.DataFrame with SIGNAL_SCHEMA columns, sorted by score descending.
    """
    if baseline_vol <= 0:
        return _empty_signal_df()

    today    = today or pd.to_datetime(df_latest["Date"].max())
    _std     = df_latest["% Change"].std()
    mkt_std  = float(_std) if pd.notna(_std) and _std > 0 else float(baseline_vol)
    threshold = max(mkt_std, baseline_vol) * _VOL_SPIKE_RATIO
    signals  = []

    for _, row in df_latest.iterrows():
        sym = row["Symbol"]
        pct = row["% Change"]

        if abs(pct) < threshold:
            continue

        # Score: how many times the threshold is exceeded
        score = round(abs(pct) / max(mkt_std, 0.001), 4)
        signals.append({
            "Symbol":      sym,
            "Date":        today,
            "signal_type": "volatility_spike",
            "direction":   "Bullish" if pct >= 0 else "Bearish",
            "strength":    round(abs(pct), 4),
            "score":       score,
            "reason": (
                f"{sym} moved {pct:+.2f}% vs market threshold {threshold:.2f}% "
                f"(ratio: {score:.2f}x)"
            ),
        })

    if not signals:
        return _empty_signal_df()

    return (
        pd.DataFrame(signals)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )


def detect_sentiment_divergence(df_latest: pd.DataFrame, today: pd.Timestamp = None) -> pd.DataFrame:
    """
    Identify stocks where sentiment and price move in opposite directions.

    Divergence types
    ----------------
    Bullish: negative price performance but positive sentiment
             (market may be oversold relative to news)
    Bearish: positive price performance but negative sentiment
             (price may be running ahead of deteriorating fundamentals)

    Fix (v2): added pd.isna() guard before threshold comparisons to
    prevent TypeError when Sentiment_Score is NaN (common when the
    sentiment merge produced no match for a symbol).

    Parameters
    ----------
    df_latest : latest-day price slice, must contain Sentiment_Score column

    Returns
    -------
    pd.DataFrame with SIGNAL_SCHEMA columns, sorted by score descending.
    """
    if "Sentiment_Score" not in df_latest.columns:
        logger.debug(
            "signal_engine: Sentiment_Score not in df_latest — "
            "sentiment divergence detection skipped."
        )
        return _empty_signal_df()

    today   = today or pd.to_datetime(df_latest["Date"].max())
    signals = []

    for _, row in df_latest.iterrows():
        sym  = row["Symbol"]
        pct  = row["% Change"]
        sent = row["Sentiment_Score"]

        # Guard: NaN sentiment → skip
        if pd.isna(sent) or abs(float(sent)) < _SENT_DIVERGE_MIN:
            continue

        sent = float(sent)

        # Bearish divergence: price up but sentiment negative
        if pct > abs(_SENT_DIVERGE_PRICE) and sent < -_SENT_DIVERGE_MIN:
            direction = "Bearish"
            reason    = (
                f"{sym}: price up {pct:+.2f}% but sentiment {sent:+.4f} — "
                "price may be running ahead of news"
            )
        # Bullish divergence: price down but sentiment positive
        elif pct < _SENT_DIVERGE_PRICE and sent > _SENT_DIVERGE_MIN:
            direction = "Bullish"
            reason    = (
                f"{sym}: price down {pct:+.2f}% but sentiment {sent:+.4f} — "
                "possible oversold opportunity"
            )
        else:
            continue

        score = round(abs(sent) * abs(pct) / 100, 4)
        signals.append({
            "Symbol":      sym,
            "Date":        today,
            "signal_type": "sentiment_divergence",
            "direction":   direction,
            "strength":    round(abs(sent), 4),
            "score":       score,
            "reason":      reason,
        })

    if not signals:
        return _empty_signal_df()

    return (
        pd.DataFrame(signals)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )


def detect_unusual_movers(
    df_latest: pd.DataFrame,
    df_full:   pd.DataFrame,
    today: pd.Timestamp = None,
) -> pd.DataFrame:
    """
    Identify stocks whose today's move is statistically unusual vs their
    full historical distribution (not just the recent window).

    Unlike momentum_breakout, which uses a short rolling window, this
    uses the entire available history to provide long-horizon context.

    Returns
    -------
    pd.DataFrame with SIGNAL_SCHEMA columns, sorted by score descending.
    """
    today   = today or pd.to_datetime(df_latest["Date"].max())
    signals = []

    for _, row in df_latest.iterrows():
        sym    = row["Symbol"]
        pct    = row["% Change"]
        hist   = df_full[
            (df_full["Symbol"] == sym) &
            (pd.to_datetime(df_full["Date"]) < today)
        ]["% Change"].dropna()

        if len(hist) < TECHNICAL_MIN_HISTORY + 2:
            continue

        mu    = float(hist.mean())
        sigma = float(hist.std())
        if sigma == 0:
            continue

        z = (pct - mu) / sigma
        if abs(z) < _UNUSUAL_MOVER_Z:
            continue

        signals.append({
            "Symbol":      sym,
            "Date":        today,
            "signal_type": "unusual_mover",
            "direction":   "Bullish" if z > 0 else "Bearish",
            "strength":    round(abs(z), 4),
            "score":       round(abs(z), 4),
            "reason": (
                f"{sym} moved {pct:+.2f}% — z-score {z:+.2f} vs "
                f"full history (mean {mu:+.2f}%, std {sigma:.2f}%)"
            ),
        })

    if not signals:
        return _empty_signal_df()

    return (
        pd.DataFrame(signals)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )


# ======================================================================
# COMPOSITE RANKINGS
# ======================================================================

def compute_risk_scores(
    df_latest:    pd.DataFrame,
    df_full:      pd.DataFrame,
    baseline_vol: float,
    today: pd.Timestamp = None,
) -> pd.DataFrame:
    """
    Compute a composite risk score per stock for today.

    Components (weighted)
    ---------------------
    1. Volatility ratio  (30%): abs(% Change) / baseline_vol
    2. Historical z-score(40%): how unusual today's move is
    3. Price drawdown    (30%): % below rolling max price

    Scores clipped to [0, 100].

    Returns
    -------
    pd.DataFrame — Symbol, Date, risk_score, risk_label,
                   vol_ratio, hist_z, drawdown
    Sorted by risk_score descending.
    """
    today = today or pd.to_datetime(df_latest["Date"].max())
    rows  = []

    for _, row in df_latest.iterrows():
        sym   = row["Symbol"]
        pct   = row["% Change"]
        price = row["Price"]
        sym_df = df_full[
            (df_full["Symbol"] == sym) &
            (pd.to_datetime(df_full["Date"]) < today)
        ]

        vol_ratio = abs(pct) / max(baseline_vol, 0.001)

        hist_pct = sym_df["% Change"].dropna()
        z = (
            abs((pct - float(hist_pct.mean())) / max(float(hist_pct.std()), 0.001))
            if len(hist_pct) >= 3 else 0.0
        )

        hist_price = sym_df["Price"].dropna()
        if not hist_price.empty:
            max_price = float(hist_price.max())
            drawdown  = max(0.0, (max_price - price) / max(max_price, 0.001) * 100)
        else:
            drawdown = 0.0

        raw_score = vol_ratio * 30 + z * 40 + drawdown * 0.3
        score     = min(round(float(raw_score), 2), 100.0)

        rows.append({
            "Symbol":     sym,
            "Date":       today,
            "risk_score": score,
            "risk_label": (
                "High Risk"   if score >= 70 else
                "Medium Risk" if score >= 40 else
                "Low Risk"
            ),
            "vol_ratio":  round(float(vol_ratio), 4),
            "hist_z":     round(float(z),         4),
            "drawdown":   round(float(drawdown),  4),
        })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("risk_score", ascending=False)
        .reset_index(drop=True)
    )


def compute_opportunity_ranking(
    df_latest: pd.DataFrame,
    df_full:   pd.DataFrame,
    today: pd.Timestamp = None,
) -> pd.DataFrame:
    """
    Rank stocks by composite opportunity score for today.

    Components (weighted)
    ---------------------
    1. Relative momentum (40%): outperformance vs market average
    2. Signal strength z (40%): how strong is today's move vs own history
    3. Sentiment score   (20%): positive news backing (if available)

    Fix (v2): explicit column presence check for Sentiment_Score
    with pd.isna guard, replacing the fragile row.get() pattern.

    Returns
    -------
    pd.DataFrame — Symbol, Date, opportunity_score, opportunity_label,
                   rel_momentum, signal_z, sentiment
    Top _OPPORTUNITY_TOP_N rows by |opportunity_score|.
    """
    today   = today or pd.to_datetime(df_latest["Date"].max())
    mkt_avg = float(df_latest["% Change"].mean())
    has_sent = "Sentiment_Score" in df_latest.columns
    rows    = []

    for _, row in df_latest.iterrows():
        sym  = row["Symbol"]
        pct  = row["% Change"]

        rel_mom = pct - mkt_avg

        sym_hist = df_full[
            (df_full["Symbol"] == sym) &
            (pd.to_datetime(df_full["Date"]) < today)
        ]["% Change"].dropna()
        z = (
            (pct - float(sym_hist.mean())) / max(float(sym_hist.std()), 0.001)
            if len(sym_hist) >= 3 else 0.0
        )

        raw_sent = row["Sentiment_Score"] if has_sent else 0.0
        sent     = 0.0 if pd.isna(raw_sent) else float(raw_sent)

        score = round(rel_mom * 0.4 + z * 0.4 + sent * SIGNALS.SENT_SCALE_FACTOR * 0.2, 4)

        rows.append({
            "Symbol":            sym,
            "Date":              today,
            "opportunity_score": score,
            "opportunity_label": (
                "Strong Buy Signal"   if score >  1.0 else
                "Moderate Opportunity"if score >  0.3 else
                "Strong Sell Signal"  if score < -1.0 else
                "Moderate Risk"       if score < -0.3 else
                "Neutral"
            ),
            "rel_momentum": round(float(rel_mom), 4),
            "signal_z":     round(float(z),       4),
            "sentiment":    round(float(sent),    4),
        })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .assign(_abs=lambda x: x["opportunity_score"].abs())
        .sort_values("_abs", ascending=False)
        .drop(columns=["_abs"])
        .head(_OPPORTUNITY_TOP_N)
        .reset_index(drop=True)
    )


# ======================================================================
# MASTER ENTRY POINT
# ======================================================================

def compute_all_signals(
    df_latest:    pd.DataFrame,
    df_full:      pd.DataFrame,
    baseline_vol: float = 1.0,
) -> dict:
    """
    Run all signal detectors and return a structured result dict.

    Parameters
    ----------
    df_latest    : latest-day price slice (may include Sentiment_Score
                   if the dashboard has merged sentiment)
    df_full      : full historical price data
    baseline_vol : rolling average volatility from historical_metrics

    Returns
    -------
    dict with keys:
        breakouts       — pd.DataFrame: momentum breakout signals
        vol_spikes      — pd.DataFrame: volatility spike signals
        sent_divergence — pd.DataFrame: sentiment divergence signals
        unusual_movers  — pd.DataFrame: statistically unusual movers
        risk_scores     — pd.DataFrame: per-symbol composite risk
        opportunities   — pd.DataFrame: per-symbol opportunity ranking
        all_signals     — pd.DataFrame: unified log (breakouts +
                          vol_spikes + sent_divergence + unusual_movers)
    """
    # Compute today once — passed to all detectors to avoid 6 redundant calls
    today      = pd.to_datetime(df_latest["Date"].max())
    breakouts  = detect_breakout(df_latest, df_full, today=today)
    vol_spikes = detect_volatility_spikes(df_latest, baseline_vol, today=today)
    sent_div   = detect_sentiment_divergence(df_latest, today=today)
    unusual    = detect_unusual_movers(df_latest, df_full, today=today)
    risks      = compute_risk_scores(df_latest, df_full, baseline_vol, today=today)
    opps       = compute_opportunity_ranking(df_latest, df_full, today=today)

    signal_frames = [
        df for df in [breakouts, vol_spikes, sent_div, unusual]
        if not df.empty
    ]
    all_signals = (
        pd.concat(signal_frames, ignore_index=True)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
        if signal_frames else pd.DataFrame(columns=SIGNAL_SCHEMA)
    )

    logger.info(
        "signal_engine: %d total signals — "
        "%d breakout, %d vol-spike, %d sentiment-divergence, %d unusual.",
        len(all_signals),
        len(breakouts), len(vol_spikes), len(sent_div), len(unusual),
    )

    return {
        "breakouts":       breakouts,
        "vol_spikes":      vol_spikes,
        "sent_divergence": sent_div,
        "unusual_movers":  unusual,
        "risk_scores":     risks,
        "opportunities":   opps,
        "all_signals":     all_signals,
    }