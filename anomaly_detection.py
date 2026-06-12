"""
========================================================================
anomaly_detection.py — Market Pulse Anomaly Detection Engine
========================================================================
Identifies statistically abnormal market events using z-score methods,
IQR-based outlier detection, and cross-signal consistency checks.

Architecture
------------
- Pure computation. No I/O, no plotting.
- All thresholds come from config.py (ANOMALY block).
- Anomalies are distinct from signals: an anomaly is something the
  market has not seen recently, regardless of direction.
- Output schema is consistent with signal_engine for clean merging.
- compute_all_anomalies() is the single entry point.

Integration fixes (v2)
----------------------
- ANOMALY_SCHEMA: canonical column list extracted as module constant,
  shared with data_loader and validators.
- detect_price_gap_anomalies: removed per-symbol .copy() call (the
  filter+dropna is sufficient; copy was defensive but costly at scale).
- detect_volatility_regime_anomaly: z-score calculation now uses the
  full history minus the last point as the baseline (not the whole
  series), preventing the latest observation from inflating the mean.
- detect_breadth_extremes: same fix — baseline excludes current day.
- detect_cluster_spike: explicit pd.DataFrame(columns=ANOMALY_SCHEMA)
  return instead of bare empty DataFrame with hard-coded column list.
- detect_sentiment_volatility_divergence: replaced bare `from time_series`
  import with module-level import for import-safety; added a guard for
  the case where rolling_correlation returns all-NaN.
- compute_all_anomalies: empty frame check now uses ANOMALY_SCHEMA
  as the canonical column source.
- Logging: consistent format across all detectors.
========================================================================
"""

import logging

import numpy as np
import pandas as pd

from config import ROLLING, ANOMALY
import time_series

logger = logging.getLogger(__name__)

# Canonical anomaly output schema — shared with data_loader and validators
ANOMALY_SCHEMA: list[str] = [
    "Date", "Symbol", "anomaly_type", "severity", "z_score", "value", "reason"
]

# Module-level threshold aliases (all sourced from config)
_PRICE_GAP_Z        = ANOMALY.PRICE_GAP_Z
_VOL_REGIME_Z       = ANOMALY.VOL_REGIME_Z
_BREADTH_EXTREME_Z  = ANOMALY.BREADTH_EXTREME_Z
_CLUSTER_MIN_PCT    = ANOMALY.CLUSTER_MIN_PCT
_CLUSTER_MIN_MOVE   = ANOMALY.CLUSTER_MIN_MOVE
_SENT_VOL_CORR_THR  = ANOMALY.SENT_VOL_CORR_THRESHOLD
_MIN_BASELINE       = ANOMALY.MIN_BASELINE_DAYS


def _empty_anomaly_df() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical anomaly schema."""
    return pd.DataFrame(columns=ANOMALY_SCHEMA)


def _severity(abs_z: float) -> str:
    """
    Map an absolute z-score to a severity label using config thresholds.
    Thresholds come from ANOMALY.SEVERITY_CRITICAL_Z and ANOMALY.SEVERITY_HIGH_Z.
    """
    if abs_z >= ANOMALY.SEVERITY_CRITICAL_Z: return "Critical"
    if abs_z >= ANOMALY.SEVERITY_HIGH_Z:     return "High"
    return "Moderate"


# ======================================================================
# INDIVIDUAL ANOMALY DETECTORS
# ======================================================================

def detect_price_gap_anomalies(
    df_latest: pd.DataFrame,
    df_full:   pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify stocks whose today's % Change is anomalous vs their own
    full historical distribution.

    Uses z-score against the stock's lifetime distribution. A higher
    threshold than signal_engine's unusual_mover ensures only genuinely
    rare events appear here.

    Parameters
    ----------
    df_latest : latest-day price slice
    df_full   : full historical price data

    Returns
    -------
    pd.DataFrame with ANOMALY_SCHEMA columns.
    """
    today = pd.to_datetime(df_latest["Date"].max())
    rows  = []

    for _, row in df_latest.iterrows():
        sym  = row["Symbol"]
        pct  = row["% Change"]
        hist = df_full[
            (df_full["Symbol"] == sym) &
            (pd.to_datetime(df_full["Date"]) < today)
        ]["% Change"].dropna()

        if len(hist) < _MIN_BASELINE:
            continue

        mu    = float(hist.mean())
        sigma = float(hist.std())
        if sigma == 0:
            continue

        z = (pct - mu) / sigma
        if abs(z) < _PRICE_GAP_Z:
            continue

        rows.append({
            "Symbol":       sym,
            "Date":         today,
            "anomaly_type": "price_gap",
            "severity":     _severity(abs(z)),
            "z_score":      round(float(z), 4),
            "value":        round(float(pct), 4),
            "reason": (
                f"{sym} moved {pct:+.2f}% — z-score {z:+.2f} "
                f"(mu={mu:+.2f}%, sigma={sigma:.2f}%, n={len(hist)} days)"
            ),
        })

    if not rows:
        return _empty_anomaly_df()

    return (
        pd.DataFrame(rows)
        .sort_values("z_score", key=abs, ascending=False)
        .reset_index(drop=True)
    )


def detect_volatility_regime_anomaly(hist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect if the current market volatility regime is historically abnormal.

    Fix (v2): baseline mean and std are computed from all rows except the
    last one, so the latest observation does not contaminate its own
    reference distribution.

    Parameters
    ----------
    hist_df : output of historical_metrics.compute_all()["hist_df"]

    Returns
    -------
    pd.DataFrame with one row if an anomaly exists, empty otherwise.
    """
    if hist_df.empty or "volatility" not in hist_df.columns or len(hist_df) < _MIN_BASELINE:
        return _empty_anomaly_df()

    df     = hist_df.sort_values("Date")
    vol    = df["volatility"].dropna()
    today  = pd.to_datetime(df["Date"].iloc[-1])
    latest = float(vol.iloc[-1])

    # Baseline: exclude the current observation (fix vs original)
    baseline = vol.iloc[:-1]
    mu       = float(baseline.mean())
    sigma    = float(baseline.std())

    if sigma == 0:
        return _empty_anomaly_df()

    z = (latest - mu) / sigma
    if abs(z) < _VOL_REGIME_Z:
        return _empty_anomaly_df()

    direction = "above" if z > 0 else "below"
    return pd.DataFrame([{
        "Date":         today,
        "Symbol":       pd.NA,
        "anomaly_type": "vol_regime",
        "severity":     _severity(abs(z)),
        "z_score":      round(float(z), 4),
        "value":        round(float(latest), 4),
        "reason": (
            f"Market volatility {latest:.4f}% is {abs(z):.2f} std devs {direction} "
            f"its historical mean (mu={mu:.4f}%, sigma={sigma:.4f}%)"
        ),
    }])


def detect_breadth_extremes(hist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect if today's market breadth is historically extreme.

    Breadth extremes can indicate exhaustion (very high) or capitulation
    (very low). Baseline excludes the current day (same fix as
    detect_volatility_regime_anomaly).

    Parameters
    ----------
    hist_df : output of historical_metrics.compute_all()["hist_df"]

    Returns
    -------
    pd.DataFrame with one row if an anomaly exists, empty otherwise.
    """
    if hist_df.empty or "breadth_pct" not in hist_df.columns or len(hist_df) < _MIN_BASELINE:
        return _empty_anomaly_df()

    df      = hist_df.sort_values("Date")
    breadth = df["breadth_pct"].dropna()
    today   = pd.to_datetime(df["Date"].iloc[-1])
    latest  = float(breadth.iloc[-1])

    baseline = breadth.iloc[:-1]
    mu       = float(baseline.mean())
    sigma    = float(baseline.std())

    if sigma == 0:
        return _empty_anomaly_df()

    z = (latest - mu) / sigma
    if abs(z) < _BREADTH_EXTREME_Z:
        return _empty_anomaly_df()

    direction = "overbought" if z > 0 else "oversold"
    return pd.DataFrame([{
        "Date":         today,
        "Symbol":       pd.NA,
        "anomaly_type": "breadth_extreme",
        "severity":     _severity(abs(z)),
        "z_score":      round(float(z), 4),
        "value":        round(float(latest), 4),
        "reason": (
            f"Market breadth {latest:.1f}% is historically {direction} "
            f"(z={z:+.2f}, mu={mu:.1f}%, sigma={sigma:.1f}%)"
        ),
    }])


def detect_cluster_spike(df_latest: pd.DataFrame) -> pd.DataFrame:
    """
    Detect coordinated cluster moves where a large fraction of stocks
    move strongly in the same direction simultaneously.

    These events indicate macro-driven sessions rather than stock-specific
    news and deserve explicit flagging.

    Parameters
    ----------
    df_latest : latest-day price slice

    Returns
    -------
    pd.DataFrame with at most two rows (one per direction), empty if no
    cluster spike detected.
    """
    today = pd.to_datetime(df_latest["Date"].max())
    n     = len(df_latest)
    if n == 0:
        return _empty_anomaly_df()

    big_up   = df_latest[df_latest["% Change"] >=  _CLUSTER_MIN_MOVE]
    big_down = df_latest[df_latest["% Change"] <= -_CLUSTER_MIN_MOVE]

    rows = []
    for direction, group, label in [
        ("Bullish", big_up,   "upside"),
        ("Bearish", big_down, "downside"),
    ]:
        frac = len(group) / n
        if frac < _CLUSTER_MIN_PCT:
            continue

        avg_move = float(group["% Change"].mean())
        rows.append({
            "Date":         today,
            "Symbol":       pd.NA,
            "anomaly_type": "cluster_spike",
            "severity":     (
                "Critical" if frac >= 0.60 else
                "High"     if frac >= 0.45 else
                "Moderate"
            ),
            "z_score":      round(frac, 4),   # participation fraction [0, 1]
            "value":        round(avg_move, 4),
            "reason": (
                f"{len(group)} of {n} stocks ({frac:.0%}) showed strong {label} moves "
                f"(avg {avg_move:+.2f}%) — coordinated macro-driven session"
            ),
        })

    if not rows:
        return _empty_anomaly_df()

    return pd.DataFrame(rows).reset_index(drop=True)


def detect_sentiment_volatility_divergence(
    hist_df:       pd.DataFrame,
    daily_sent_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect sessions where sentiment and price volatility move in conflict.

    A strong divergence between sentiment trend and price volatility
    suggests the market is reacting in a direction that news does not
    support — a potential mean-reversion setup.

    Fix (v2): time_series imported at module level (not inline) for
    import safety; added guard for all-NaN rolling correlation result.

    Parameters
    ----------
    hist_df       : historical price metrics (must have volatility, Date)
    daily_sent_df : daily sentiment averages (must have avg_score, Date)

    Returns
    -------
    pd.DataFrame with one row if divergence detected, empty otherwise.
    """
    if hist_df.empty or daily_sent_df.empty:
        return _empty_anomaly_df()

    if "avg_score" not in daily_sent_df.columns:
        return _empty_anomaly_df()

    vol_s = hist_df.set_index(pd.to_datetime(hist_df["Date"]))["volatility"]
    sent_s = (
        daily_sent_df
        .assign(Date=pd.to_datetime(daily_sent_df["Date"]))
        .set_index("Date")["avg_score"]
    )
    common = vol_s.index.intersection(sent_s.index)

    min_required = ROLLING.SHORT_WINDOW + 1
    if len(common) < min_required:
        return _empty_anomaly_df()

    roll_corr = time_series.rolling_correlation(
        vol_s.loc[common],
        sent_s.loc[common],
        ROLLING.SHORT_WINDOW,
    )

    valid_corr = roll_corr.dropna()
    if valid_corr.empty:
        return _empty_anomaly_df()

    latest_corr = float(valid_corr.iloc[-1])
    today       = common[-1]

    if latest_corr > _SENT_VOL_CORR_THR:
        return _empty_anomaly_df()

    severity = "High" if latest_corr < -0.75 else "Moderate"
    return pd.DataFrame([{
        "Date":         today,
        "Symbol":       pd.NA,
        "anomaly_type": "sent_vol_divergence",
        "severity":     severity,
        "z_score":      round(float(latest_corr), 4),
        "value":        round(float(latest_corr), 4),
        "reason": (
            f"Sentiment-volatility rolling correlation: {latest_corr:+.3f} — "
            f"sentiment and price volatility are moving in conflict "
            f"over the last {ROLLING.SHORT_WINDOW} days"
        ),
    }])


# ======================================================================
# MASTER ENTRY POINT
# ======================================================================

def compute_all_anomalies(
    df_latest:     pd.DataFrame,
    df_full:       pd.DataFrame,
    hist_df:       pd.DataFrame,
    daily_sent_df: pd.DataFrame = None,
) -> dict:
    """
    Run all anomaly detectors and return a structured result dict.

    Parameters
    ----------
    df_latest     : latest-day price slice
    df_full       : full historical price data
    hist_df       : per-day aggregate metrics from historical_metrics
    daily_sent_df : daily average sentiment (optional)

    Returns
    -------
    dict with keys:
        price_gaps      — pd.DataFrame
        vol_regime      — pd.DataFrame
        breadth_extreme — pd.DataFrame
        cluster_spikes  — pd.DataFrame
        sent_vol_div    — pd.DataFrame
        all_anomalies   — pd.DataFrame: concatenated all types,
                          sorted by |z_score| descending
        n_critical      — int: count of Critical-severity anomalies
        n_high          — int: count of High-severity anomalies
    """
    if daily_sent_df is None:
        daily_sent_df = pd.DataFrame()

    price_gaps  = detect_price_gap_anomalies(df_latest, df_full)
    vol_regime  = detect_volatility_regime_anomaly(hist_df)
    breadth_ext = detect_breadth_extremes(hist_df)
    clusters    = detect_cluster_spike(df_latest)
    sent_vol    = detect_sentiment_volatility_divergence(hist_df, daily_sent_df)

    frames = [
        df for df in [price_gaps, vol_regime, breadth_ext, clusters, sent_vol]
        if not df.empty
    ]
    all_anomalies = (
        pd.concat(frames, ignore_index=True)
        .sort_values("z_score", key=abs, ascending=False)
        .reset_index(drop=True)
        if frames else pd.DataFrame(columns=ANOMALY_SCHEMA)
    )

    n_critical = int((all_anomalies["severity"] == "Critical").sum()) if not all_anomalies.empty else 0
    n_high     = int((all_anomalies["severity"] == "High").sum())     if not all_anomalies.empty else 0

    logger.info(
        "anomaly_detection: %d anomalies detected — %d critical, %d high.",
        len(all_anomalies), n_critical, n_high,
    )

    return {
        "price_gaps":      price_gaps,
        "vol_regime":      vol_regime,
        "breadth_extreme": breadth_ext,
        "cluster_spikes":  clusters,
        "sent_vol_div":    sent_vol,
        "all_anomalies":   all_anomalies,
        "n_critical":      n_critical,
        "n_high":          n_high,
    }