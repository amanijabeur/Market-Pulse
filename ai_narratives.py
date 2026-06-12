"""
========================================================================
ai_narratives.py — Market Pulse AI Narrative Engine
========================================================================
Generates professional, data-driven market commentary by translating
computed metrics, signals, and anomalies into structured natural language.

Architecture
------------
- PURELY data-driven. Every sentence derives from actual computed values.
  Nothing is hardcoded. No fake commentary, no random phrases.
- All narrative functions accept structured DataFrames/dicts and return
  plain strings (or a dict of strings for the briefing).
- generate_daily_briefing() is the master entry point called by
  dashboard.build() for the AI Briefing tab.
- Thresholds for language choices match those used by signal_engine and
  anomaly_detection so the narrative is consistent with charted signals.
- Language is institutional: direct, specific, avoids hedging where
  data is clear, and acknowledges uncertainty where it genuinely exists.

Integration fixes (v2)
----------------------
- generate_session_summary: added safe guard when hist_df is empty or
  rolling_avg_7d is missing; no IndexError on first-day runs.
- generate_volatility_commentary: z_score calculation now uses a
  non-zero denominator guard (baseline * 0.3, min 1e-6).
- generate_sentiment_commentary: sent_summary.get() now provides
  explicit float defaults to avoid KeyError when keys are missing.
- generate_signal_commentary: checks for "direction" column before
  calling .sum() to avoid AttributeError on empty all_signals DataFrame.
- generate_forecast_commentary: removed the redundant f5 assignment
  that computed the same value twice; cleaned up the iloc[-1] call.
- generate_daily_briefing: n_anomalies now sums n_critical + n_high
  from the anomalies_dict (consistent with anomaly_detection output).
- All language helpers now accept float inputs safely (no int assumption).

Public API
----------
  generate_session_summary(...)     -> str
  generate_volatility_commentary(...)-> str
  generate_sentiment_commentary(...) -> str
  generate_signal_commentary(...)    -> str
  generate_anomaly_commentary(...)   -> str
  generate_risk_commentary(...)      -> str
  generate_forecast_commentary(...)  -> str
  generate_daily_briefing(...)       -> dict
========================================================================
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import ROLLING, BREADTH, NARRATIVE, SENTIMENT as _SENT_CFG
from sentiment import BULLISH_THRESHOLD, BEARISH_THRESHOLD

logger = logging.getLogger(__name__)


# ======================================================================
# LANGUAGE HELPERS
# Deterministic phrase selection driven by numeric thresholds.
# All qualitative labels derive from config-consistent thresholds.
# ======================================================================

def _breadth_phrase(breadth_pct: float) -> str:
    """Map breadth % to an institutional session description."""
    if breadth_pct >= BREADTH.BROAD_ADVANCE:  return "a broad-based advance"
    if breadth_pct >= BREADTH.MOD_ADVANCE:    return "a moderately positive session"
    if breadth_pct >= BREADTH.MIXED:          return "a mixed market session"
    if breadth_pct >= BREADTH.MOD_DECLINE:    return "a moderately negative session"
    return "a broad-based decline"


def _vol_phrase(z_score: float) -> str:
    """Map volatility z-score to a descriptive phrase (thresholds from NARRATIVE config)."""
    if z_score >  NARRATIVE.VOL_SEVERE_Z:     return "severely elevated relative to recent history"
    if z_score >  NARRATIVE.VOL_ELEVATED_Z:   return "elevated above recent norms"
    if z_score >  NARRATIVE.VOL_SLIGHT_Z:     return "slightly above average"
    if z_score > NARRATIVE.VOL_CALM_Z:        return "in line with recent history"
    if z_score > NARRATIVE.VOL_SUPPRESSED_Z:  return "below recent averages — a calm session"
    return "suppressed — unusually quiet conditions"


def _momentum_phrase(avg_pct: float) -> str:
    """Map average % change to a momentum descriptor (thresholds from NARRATIVE config)."""
    if avg_pct >  NARRATIVE.MOM_STRONG_POS: return "strong positive momentum across the universe"
    if avg_pct >  NARRATIVE.MOM_POS:        return "positive momentum"
    if avg_pct >  0.0:                       return "marginal positive drift"
    if avg_pct > NARRATIVE.MOM_NEG:         return "marginal negative drift"
    if avg_pct > NARRATIVE.MOM_STRONG_NEG:  return "negative momentum"
    return "strong negative momentum across the universe"


def _sentiment_phrase(mean_score: float, pct_bullish: float) -> str:
    """Map sentiment metrics to a market tone descriptor (thresholds from config)."""
    if pct_bullish >= BREADTH.BULLISH_PCT     and mean_score >= BULLISH_THRESHOLD: return "risk-on — broad bullish news tone"
    if pct_bullish <= _SENT_CFG.RISK_OFF_BULL_PCT and mean_score <= BEARISH_THRESHOLD: return "risk-off — broad bearish news tone"
    if mean_score >= BULLISH_THRESHOLD:  return "cautiously bullish news tone"
    if mean_score <= BEARISH_THRESHOLD:  return "cautiously bearish news tone"
    return "a neutral news environment"


def _trend_phrase(trend_label: str) -> str:
    """Map SMA trend label to an adjectival descriptor."""
    return {
        "Bullish": "upward trending",
        "Bearish": "downward trending",
        "Neutral": "range-bound",
    }.get(trend_label, "indeterminate")


def _severity_phrase(severity: str) -> str:
    """Map severity label to a grammatically suitable article + adjective."""
    return {
        "Critical": "a critical",
        "High":     "a high-severity",
        "Moderate": "a moderate",
    }.get(severity, "an")


# ======================================================================
# NARRATIVE GENERATORS
# ======================================================================

def generate_session_summary(
    df_latest: pd.DataFrame,
    hist_df:   pd.DataFrame,
    n_gain:    int,
    n_loss:    int,
    n_flat:    int,
    avg_chg:   float,
    top_sym:   str,
    top_pct:   float,
    bot_sym:   str,
    bot_pct:   float,
) -> str:
    """
    Generate a concise session summary paragraph.

    Fix (v2): rolling_avg_7d extraction now checks for column presence
    and sufficient rows before slicing, avoiding IndexError on day-one runs.
    """
    n_total     = max(n_gain + n_loss + n_flat, 1)
    breadth_pct = n_gain / n_total * 100
    breadth_str = _breadth_phrase(breadth_pct)
    momentum    = _momentum_phrase(avg_chg)

    context_str = ""
    if (
        not hist_df.empty
        and "rolling_avg_7d" in hist_df.columns
        and len(hist_df) >= 2
    ):
        rolling_avg = float(hist_df["rolling_avg_7d"].iloc[-1])
        diff        = avg_chg - rolling_avg
        if   diff >  1.0:
            context_str = (
                f" This represents a notable improvement over the recent "
                f"7-day average of {rolling_avg:+.2f}%."
            )
        elif diff < -1.0:
            context_str = (
                f" This marks a deterioration from the recent "
                f"7-day average of {rolling_avg:+.2f}%."
            )
        else:
            context_str = (
                f" This is broadly consistent with the recent "
                f"7-day average of {rolling_avg:+.2f}%."
            )

    return (
        f"The session registered {breadth_str} with {n_gain} of {n_total} stocks advancing "
        f"and {n_loss} declining. The average move across the universe was {avg_chg:+.2f}%, "
        f"reflecting {momentum}.{context_str} "
        f"The session was led by {top_sym} ({top_pct:+.2f}%), while {bot_sym} posted "
        f"the sharpest decline at {bot_pct:+.2f}%."
    )


def generate_volatility_commentary(
    hist_df:      pd.DataFrame,
    baseline_vol: float,
    latest_vol:   float,
) -> str:
    """
    Generate volatility regime commentary.

    Fix (v2): denominator in z-score calculation uses max(baseline * 0.3, 1e-6)
    to prevent division-by-zero on perfectly flat baselines.
    """
    if baseline_vol <= 0 or np.isnan(baseline_vol):
        return (
            f"Session volatility recorded at {latest_vol:.4f}% standard deviation. "
            "Insufficient history to contextualise against a baseline."
        )

    if not hist_df.empty and "volatility" in hist_df.columns and len(hist_df) >= 5:
        vol_hist = hist_df["volatility"].dropna().iloc[:-1]
        sigma = float(vol_hist.std()) if len(vol_hist) >= 4 else 0.0
        denom = max(sigma, 1e-6)
    else:
        denom = max(baseline_vol * 0.3, 1e-6)
    z_score = (latest_vol - baseline_vol) / denom
    vol_str = _vol_phrase(z_score)

    trend_str = ""
    if not hist_df.empty and "volatility" in hist_df.columns:
        recent = hist_df["volatility"].dropna()
        if len(recent) >= 3:
            prev, curr = float(recent.iloc[-2]), float(recent.iloc[-1])
            if   curr > prev * 1.1: trend_str = " The trend is accelerating."
            elif curr < prev * 0.9: trend_str = " The trend is decelerating."
            else:                   trend_str = " The trend is stable."

    return (
        f"Intraday volatility is {vol_str} "
        f"(session std dev {latest_vol:.4f}% vs baseline {baseline_vol:.4f}%)."
        f"{trend_str}"
    )


def generate_sentiment_commentary(
    sent_summary:  dict,
    daily_sent_df: pd.DataFrame,
) -> str:
    """
    Generate sentiment intelligence commentary.

    Fix (v2): explicit .get() with float defaults on all summary keys
    to prevent KeyError when scraper has not yet run for today.
    """
    if not sent_summary or sent_summary.get("n_total", 0) == 0:
        return "Sentiment data is not available for this session."

    mean_score  = float(sent_summary.get("mean_score",  0.0))
    pct_bullish = float(sent_summary.get("pct_bullish", 0.0))
    pct_bearish = float(sent_summary.get("pct_bearish", 0.0))
    n_bullish   = int(sent_summary.get("n_bullish",   0))
    n_bearish   = int(sent_summary.get("n_bearish",   0))
    regime      = str(sent_summary.get("regime", "Mixed"))
    tone        = _sentiment_phrase(mean_score, pct_bullish)

    trend_str = ""
    if not daily_sent_df.empty and "avg_score" in daily_sent_df.columns and len(daily_sent_df) >= 2:
        prev  = float(daily_sent_df["avg_score"].iloc[-2])
        curr  = float(daily_sent_df["avg_score"].iloc[-1])
        delta = curr - prev
        if   delta >  0.03: trend_str = " Sentiment is improving from the prior session."
        elif delta < -0.03: trend_str = " Sentiment has deteriorated from the prior session."
        else:               trend_str = " Sentiment is broadly unchanged from the prior session."

    return (
        f"News sentiment across the 100-stock universe reflects {tone}. "
        f"{n_bullish} stocks carry a bullish news signal ({pct_bullish:.1f}%) "
        f"against {n_bearish} with a bearish signal ({pct_bearish:.1f}%). "
        f"The mean sentiment score stands at {mean_score:+.4f}, "
        f"consistent with a {regime} market regime."
        f"{trend_str}"
    )


def generate_signal_commentary(signals_dict: dict) -> str:
    """
    Generate a concise signal intelligence summary.

    Fix (v2): checks "direction" column exists before calling .sum()
    to avoid AttributeError on an empty all_signals DataFrame.
    """
    all_sigs   = signals_dict.get("all_signals",      pd.DataFrame())
    breakouts  = signals_dict.get("breakouts",         pd.DataFrame())
    vol_spikes = signals_dict.get("vol_spikes",        pd.DataFrame())
    sent_div   = signals_dict.get("sent_divergence",   pd.DataFrame())

    if all_sigs is None or all_sigs.empty:
        return "No material signals detected in the current session."

    has_dir = "direction" in all_sigs.columns
    n_total = len(all_sigs)
    n_bull  = int((all_sigs["direction"] == "Bullish").sum()) if has_dir else 0
    n_bear  = int((all_sigs["direction"] == "Bearish").sum()) if has_dir else 0

    parts = [
        f"The signal engine identified {n_total} active signal(s) "
        f"({n_bull} bullish, {n_bear} bearish)."
    ]

    if breakouts is not None and not breakouts.empty:
        top = breakouts.iloc[0]
        parts.append(
            f"The most significant momentum breakout is {top['Symbol']} "
            f"(signal strength z-score: {float(top['strength']):+.2f})."
        )

    if vol_spikes is not None and not vol_spikes.empty:
        top = vol_spikes.iloc[0]
        parts.append(
            f"Volatility spikes detected in {len(vol_spikes)} stock(s), "
            f"led by {top['Symbol']} at {float(top['strength']):.2f}%."
        )

    if sent_div is not None and not sent_div.empty and "direction" in sent_div.columns:
        n_bull_div = int((sent_div["direction"] == "Bullish").sum())
        n_bear_div = int((sent_div["direction"] == "Bearish").sum())
        parts.append(
            f"Sentiment divergence detected in {len(sent_div)} stock(s) "
            f"({n_bull_div} bullish, {n_bear_div} bearish)."
        )

    return " ".join(parts)


def generate_anomaly_commentary(anomalies_dict: dict) -> str:
    """
    Generate anomaly intelligence commentary.

    Covers the count, severity distribution, and leading anomaly reason.
    """
    n_critical = anomalies_dict.get("n_critical", 0)
    n_high     = anomalies_dict.get("n_high",     0)
    all_anom   = anomalies_dict.get("all_anomalies", pd.DataFrame())

    if all_anom is None or all_anom.empty:
        return (
            "No statistical anomalies detected in the current session. "
            "Market conditions are within normal historical parameters."
        )

    if   n_critical > 0: sev_str = f"{n_critical} critical-severity event(s) require immediate attention."
    elif n_high     > 0: sev_str = f"{n_high} high-severity anomaly(ies) noted."
    else:                sev_str = "All detected anomalies are of moderate severity."

    top    = all_anom.iloc[0]
    reason = str(top.get("reason", "")) if hasattr(top, "get") else ""

    return (
        f"The anomaly detection engine identified {len(all_anom)} statistical anomaly(ies). "
        f"{sev_str} "
        f"Highest-priority event: {reason}"
    )


def generate_risk_commentary(risk_scores_df: pd.DataFrame) -> str:
    """
    Generate risk intelligence commentary from the risk score table.
    """
    if risk_scores_df is None or risk_scores_df.empty:
        return "Risk scoring data is unavailable."

    n_high   = int((risk_scores_df["risk_label"] == "High Risk").sum())
    n_medium = int((risk_scores_df["risk_label"] == "Medium Risk").sum())
    n_low    = int((risk_scores_df["risk_label"] == "Low Risk").sum())
    n_total  = len(risk_scores_df)
    top_risk = risk_scores_df.iloc[0]

    if n_high == 0:
        risk_str = "The universe is in a generally contained risk environment with no high-risk classifications."
    elif n_high <= 5:
        risk_str = (
            f"{n_high} stock(s) flagged as high-risk out of {n_total} analysed. "
            "Risk is concentrated rather than systemic."
        )
    else:
        risk_str = (
            f"{n_high} of {n_total} stocks classified as high-risk. "
            "Elevated risk is broad-based and warrants caution across the universe."
        )

    return (
        f"{risk_str} "
        f"The highest composite risk score belongs to {top_risk['Symbol']} "
        f"(score: {float(top_risk['risk_score']):.1f}/100). "
        f"Distribution: {n_high} high, {n_medium} medium, {n_low} low risk."
    )


def generate_forecast_commentary(forecast_dict: dict) -> str:
    """
    Generate a brief forward-looking commentary from forecast outputs.

    Fix (v2): removed the redundant f5 re-assignment; now uses a single
    clean iloc reference. Empty guard on each sub-frame before access.
    """
    if not forecast_dict:
        return "Forecast data unavailable."

    parts = []

    avg_fcast = forecast_dict.get("avg_pct")
    if avg_fcast is not None and not avg_fcast.empty and "forecast" in avg_fcast.columns:
        f1 = float(avg_fcast["forecast"].iloc[0])
        fn = float(avg_fcast["forecast"].iloc[-1])
        direction = "positive" if f1 >= 0 else "negative"
        parts.append(
            f"The {len(avg_fcast)}-day average % change projection end-point is {fn:+.2f}% "
            f"with a near-term bias toward {direction} territory ({f1:+.2f}% next session)."
        )

    vol_fcast = forecast_dict.get("volatility")
    if vol_fcast is not None and not vol_fcast.empty and "forecast" in vol_fcast.columns:
        fv = float(vol_fcast["forecast"].iloc[0])
        parts.append(f"Projected volatility for the next session: {fv:.4f}%.")

    breadth_fcast = forecast_dict.get("breadth_pct")
    if breadth_fcast is not None and not breadth_fcast.empty and "forecast" in breadth_fcast.columns:
        fb = float(breadth_fcast["forecast"].iloc[0])
        parts.append(f"Breadth forecast for the next session: {fb:.1f}% advancing.")

    if not parts:
        return (
            "Forecast projections are not yet available — "
            "insufficient historical data for extrapolation."
        )

    return (
        " ".join(parts) +
        " Note: projections are statistical extrapolations based on recent trends "
        "and carry inherent uncertainty."
    )


# ======================================================================
# MASTER DAILY BRIEFING
# ======================================================================

def generate_daily_briefing(
    df_latest:      pd.DataFrame,
    hist_df:        pd.DataFrame,
    sent_summary:   dict,
    daily_sent_df:  pd.DataFrame,
    signals_dict:   dict,
    anomalies_dict: dict,
    forecast_dict:  dict,
    n_gain:         int,
    n_loss:         int,
    n_flat:         int,
    avg_chg:        float,
    top_sym:        str,
    top_pct:        float,
    bot_sym:        str,
    bot_pct:        float,
    latest_vol:     float,
    baseline_vol:   float,
) -> dict:
    """
    Generate the complete daily AI market briefing.

    Composes all narrative sections by delegating to the individual
    generators above. dashboard.py renders each section independently
    in the AI Briefing tab.

    Returns
    -------
    dict with keys:
        session_summary   : concise breadth + top/bottom mover summary
        volatility        : regime classification with trend context
        sentiment         : sentiment score distribution and regime
        signals           : signal count and top events
        anomalies         : anomaly count and highest-priority event
        risk              : risk score distribution
        forecast          : statistical projection summary
        executive_summary : 2-sentence top-level synthesis
    """
    session   = generate_session_summary(
        df_latest, hist_df,
        n_gain, n_loss, n_flat, avg_chg,
        top_sym, top_pct, bot_sym, bot_pct,
    )
    vol       = generate_volatility_commentary(hist_df, baseline_vol, latest_vol)
    sent      = generate_sentiment_commentary(sent_summary, daily_sent_df)
    signals   = generate_signal_commentary(signals_dict)
    anomalies = generate_anomaly_commentary(anomalies_dict)
    risk_df   = signals_dict.get("risk_scores", pd.DataFrame())
    risk      = generate_risk_commentary(risk_df)
    forecast  = generate_forecast_commentary(forecast_dict)

    # Executive summary: synthesise key points in two sentences.
    breadth_pct = n_gain / max(n_gain + n_loss + n_flat, 1) * 100
    regime_str  = (
        "bullish" if breadth_pct >= BREADTH.BULLISH_PCT else
        "bearish" if breadth_pct <= BREADTH.BEARISH_PCT else
        "mixed"
    )
    n_signals   = len(signals_dict.get("all_signals", pd.DataFrame()))
    # n_critical + n_high gives the count that warrants monitoring
    n_anomalies = (
        anomalies_dict.get("n_critical", 0) +
        anomalies_dict.get("n_high",     0)
    )
    sentiment_tone = _sentiment_phrase(
        float(sent_summary.get("mean_score",  0.0)),
        float(sent_summary.get("pct_bullish", 50.0)),
    )

    exec_summary = (
        f"The session reflected a {regime_str} market environment with "
        f"{breadth_pct:.0f}% of active stocks advancing and a universe average "
        f"move of {avg_chg:+.2f}%; news sentiment registered {sentiment_tone}. "
        f"The intelligence layer identified {n_signals} active signal(s) and "
        f"{n_anomalies} high-priority anomaly(ies) requiring monitoring."
    )

    logger.info("ai_narratives: daily briefing generated successfully.")

    return {
        "session_summary":   session,
        "volatility":        vol,
        "sentiment":         sent,
        "signals":           signals,
        "anomalies":         anomalies,
        "risk":              risk,
        "forecast":          forecast,
        "executive_summary": exec_summary,
    }