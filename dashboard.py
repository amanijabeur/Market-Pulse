"""
========================================================================
dashboard.py — Market Pulse: Historical Analytics & Market Intelligence
========================================================================
Generates a self-contained interactive HTML dashboard from the
most_active_stocks_dataset.xlsx file produced by scraper.py.

Architecture
------------
  eda.py           → stats_results.py   (latest-day statistics)
  scraper.py       → .xlsx              (growing historical dataset)
  sentiment.py     → DataFrame          (live VADER sentiment per symbol)
  dashboard.py     → market_pulse_dashboard.html

Tab structure
-------------
  Overview         — latest-day snapshot (unchanged)
  Top Movers       — latest-day gainers/losers/dollar moves (unchanged)
  Analytics        — latest-day statistical analytics (unchanged)
  Screener         — sortable full stock table (unchanged)
  Story            — annotated statistical narrative (unchanged)
  Sentiment        — NEW: market sentiment intelligence layer
  Historical       — NEW: multi-day trend analysis

Feature additions (v2)
----------------------
  Feature 1 — sentiment.py integration: VADER scores, Bullish/Bearish/
              Neutral classification, top movers by sentiment,
              sentiment vs price performance scatter, rolling 7-day
              sentiment trend, news intelligence section.

  Feature 2 — Historical Analytics: daily market breadth trend,
              average daily % change with regime shading, most
              frequent top movers across days, daily volatility over
              time (rolling std), sentiment vs market performance
              dual-axis chart, historical breadth heatmap.

  Feature 3 — Dynamic interpretations throughout: all KPI labels,
              narrative text, colour choices, and chart annotations
              are computed at runtime from current data — no hardcoded
              strings describing market conditions.

Dynamic interpretation design
------------------------------
  All market-condition language is derived from real thresholds:
    breadth  > 60%  → "broad advance"
    breadth  < 40%  → "narrow market / distribution"
    volatility > 2σ → "elevated volatility regime"
    sentiment mean > 0.05 → Risk-On, etc.
  Callers should re-run eda.py then dashboard.py after each daily
  scrape; both the stats and all narrative copy will refresh.

Pip dependencies
----------------
  pip install plotly pandas numpy openpyxl vaderSentiment yfinance
========================================================================
"""

import os
import html as html_utils
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from config import PATHS, ROLLING, BREADTH
from dashboard_components.chart_helpers import ChartTheme
from dashboard_components.technical_tab import build_technical_divs, render_technical_tab
from dashboard_components.forecast_tab import (
    build_forecast_divs,
    forecast_dict_from_cache,
    render_forecast_tab,
)
from dashboard_components.signal_tab import (
    build_signal_divs,
    signals_dict_from_cache,
    render_signal_tab,
)
from dashboard_components.anomaly_tab import (
    anomalies_dict_from_cache,
    build_anomaly_divs,
    render_anomaly_tab,
)
from dashboard_components.ai_briefing_tab import render_ai_briefing_tab

# ── Data layer ────────────────────────────────────────────────────────
import data_loader
import historical_metrics as hm
import ai_narratives
import time_series as ts_mod

# ── Sentiment module ──────────────────────────────────────────────────
import sentiment as sentiment_module
from sentiment import (
    SENT_SHEET,
    BULLISH_THRESHOLD,
    BEARISH_THRESHOLD,
    empty_sentiment_df,
    normalise_dates,
    daily_sentiment_avg,
)

# ── Centralised statistics are imported inside build() so that
# orchestrator can reload them after eda.main() runs in the same process.

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build():
    """
    Build the full Market Pulse dashboard HTML file.

    All data loading, chart generation, and HTML assembly runs inside
    this function so dashboard.py is safe to import without side effects.
    Called by main.py orchestrator or directly via __main__ guard below.
    """
    # Re-import stats_results inside the function so the orchestrator can
    # run eda.main() first and dashboard.build() second in the same
    # Python process — without the function boundary the module-level
    # import would cache the pre-EDA values and never refresh.
    import importlib
    import stats_results as _sr
    importlib.reload(_sr)

    from stats_results import (
        PRICE_MEAN, PRICE_MEDIAN, PRICE_STD, PRICE_SKEW,
        N_TOTAL, N_GAIN, N_LOSS, N_FLAT, AVG_CHG,
        TOP_SYM, TOP_PCT, BOT_SYM, BOT_PCT, TOP_NAME, BOT_NAME,
        SP_CORR, SP_P, SP_STRENGTH, SP_DIRECTION,
        SPEARMAN_MATRIX, U_STAT, MW_P, MW_REJECT,
        Q1, Q3, IQR_VAL, UPPER_FENCE, N_OUTLIERS,
        OUTLIER_SYMS, OUTLIER_PRICES,
        PRICE_MAX, PRICE_MIN,
    )
    # DATA LOADING  (via data_loader — Parquet-first, Excel fallback)
    # All file reading, deduplication, and dtype enforcement happens inside
    # data_loader. dashboard.py receives clean DataFrames and does no I/O.
    # ══════════════════════════════════════════════════════════════════════

    df_full = data_loader.load_price_data()
    df      = data_loader.load_latest_day()

    # Force datetime dtype on the Date column regardless of what the
    # in-process cache returns. When main.py runs the scraper and
    # dashboard in the same Python process, the cache may return Date
    # as strings rather than datetime objects, causing .date() to fail.
    df_full["Date"] = pd.to_datetime(df_full["Date"])
    df["Date"]      = pd.to_datetime(df["Date"])

    # Derived columns — latest day
    df["Status"]     = np.where(df["Change"] > 0, "Gainer",
                       np.where(df["Change"] < 0, "Loser", "Flat"))
    df["Abs Change"] = df["Change"].abs()
    df["Price Tier"] = pd.cut(
        df["Price"],
        bins=[0, 10, 25, 50, 100, 200, 9999],
        labels=["<$10", "$10-25", "$25-50", "$50-100", "$100-200", ">$200"],
    )

    gainers = df[df["Status"] == "Gainer"]
    losers  = df[df["Status"] == "Loser"]

    # ══════════════════════════════════════════════════════════════════════
    # DYNAMIC INTERPRETATION HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _breadth_label(pct: float) -> str:
        if pct >= BREADTH.BROAD_ADVANCE: return "broad advance"
        if pct >= BREADTH.MOD_ADVANCE:   return "moderate advance"
        if pct >= BREADTH.MIXED:         return "mixed session"
        if pct >= BREADTH.MOD_DECLINE:   return "moderate decline"
        return "broad decline"

    def _volatility_label(std: float, baseline: float) -> str:
        """Classify session volatility relative to a rolling baseline."""
        if baseline == 0 or np.isnan(baseline):
            return "indeterminate volatility"
        ratio = std / baseline
        if ratio >= ROLLING.VOL_ELEVATED:      return "elevated volatility"
        if ratio >= ROLLING.VOL_SLIGHTLY_HIGH: return "slightly elevated volatility"
        if ratio >= ROLLING.VOL_NORMAL_LOW:    return "normal volatility"
        return "suppressed volatility"

    def _corr_label(rho: float, p: float) -> str:
        strength = "negligible" if abs(rho) < 0.1 else \
                   "weak"       if abs(rho) < 0.3 else \
                   "moderate"   if abs(rho) < 0.5 else "strong"
        sig = "statistically significant" if p < 0.05 else "not statistically significant"
        direction = "positive" if rho >= 0 else "negative"
        return f"{strength} {direction} ({sig}, rho={rho:+.3f})"

    def _sentiment_regime_label(mean_score: float, pct_bullish: float, pct_bearish: float) -> str:
        if pct_bullish >= BREADTH.BULLISH_PCT and mean_score >= BULLISH_THRESHOLD:
            return "Risk-On — broad bullish news flow"
        if pct_bearish >= BREADTH.BULLISH_PCT and mean_score <= BEARISH_THRESHOLD:
            return "Risk-Off — broad bearish news flow"
        if mean_score >= BULLISH_THRESHOLD:
            return "Cautiously Bullish — marginal positive tone"
        if mean_score <= BEARISH_THRESHOLD:
            return "Cautiously Bearish — marginal negative tone"
        return "Neutral — no dominant news direction"

    # ── Latest-day derived scalars ─────────────────────────────────────────
    latest_date    = df_full["Date"].max().date()
    latest_label   = latest_date.strftime("%b %d, %Y")
    breadth_pct    = N_GAIN / N_TOTAL * 100 if N_TOTAL else 0
    decline_pct    = N_LOSS / N_TOTAL * 100 if N_TOTAL else 0
    flat_pct       = N_FLAT / N_TOTAL * 100 if N_TOTAL else 0
    net_breadth    = N_GAIN - N_LOSS
    advance_width  = min(max(breadth_pct, 0), 100)
    low_price_count= int((df["Price"] < 25).sum())
    session_label  = "Bullish" if N_GAIN > N_LOSS else "Bearish" if N_LOSS > N_GAIN else "Mixed"
    session_sentence = ("tilted bullish"  if N_GAIN > N_LOSS else
                        "tilted bearish"  if N_LOSS > N_GAIN else "was mixed")
    top_row        = df.loc[df["% Change"].idxmax()]
    bot_row        = df.loc[df["% Change"].idxmin()]
    big_move_row   = df.loc[df["Change"].abs().idxmax()]
    sp_note        = "significant" if SP_P < 0.05 else "not statistically significant"
    sp_action      = "suggests a relationship" if SP_P < 0.05 else "does not show a reliable relationship"
    sp_summary     = f"Price {sp_action} with % change in this run"
    mw_note        = "significant price difference" if MW_REJECT else "no significant price difference"
    mw_summary     = ("gainers and losers had different price distributions" if MW_REJECT else
                      "gainers and losers were not separated by price")
    skew_label     = ("right-skewed" if PRICE_SKEW > 1 else
                      "left-skewed"  if PRICE_SKEW < -1 else "roughly balanced")
    momentum_word  = "risk-on" if N_GAIN > N_LOSS else "risk-off" if N_LOSS > N_GAIN else "balanced"
    volatility_span   = df["% Change"].max() - df["% Change"].min()
    high_move_count   = int((df["% Change"].abs() >= 5).sum())
    outlier_preview   = ", ".join(OUTLIER_SYMS[:5]) if OUTLIER_SYMS else "None"
    breadth_narrative = _breadth_label(breadth_pct)
    market_takeaway   = (
        f"{session_label} {momentum_word} tape: "
        f"{N_GAIN} advancers vs {N_LOSS} decliners, "
        f"led by {TOP_SYM} at {TOP_PCT:+.1f}%."
    )
    method_takeaway   = (
        f"Price vs % change is {SP_STRENGTH} {SP_DIRECTION} "
        f"and {sp_note}; {mw_summary}."
    )

    # ══════════════════════════════════════════════════════════════════════
    # SENTIMENT ANALYSIS  (Feature 1)
    #
    # Mirrors the scraper pattern exactly:
    #   - Load existing sentiment history from the "sentiment_history" sheet
    #     in the same Excel file (if it exists).
    #   - If today's date is already stored, skip the yfinance fetch entirely
    #     and reuse the stored scores — fast and avoids redundant API calls.
    #   - If today is not stored, run VADER for all 100 symbols, append the
    #     new rows, and write back to the sheet.
    #   - Deduplication on Date+Symbol is applied before every save, matching
    #     the same backstop used for the price data sheet.
    #
    # Result: sentiment_history grows by 100 rows per trading day alongside
    # the price dataset, enabling genuine rolling trend charts over time.
    # ══════════════════════════════════════════════════════════════════════

    # SENT_SHEET is imported from sentiment.py — no local definition needed.
    today_str  = str(latest_date)  # "YYYY-MM-DD" — same format as scraper

    # ── Load existing sentiment history via data_loader ───────────────────
    sent_history = data_loader.load_sentiment_history()

    # ── Decide whether to fetch or reuse ─────────────────────────────────
    if today_str in sent_history["Date"].values:
        logger.info("Sentiment already stored for %s — loading from cache.", today_str)
        sent_df = sent_history[sent_history["Date"] == today_str].copy().reset_index(drop=True)
    else:
        logger.info("Running sentiment analysis for %d symbols ...", len(df["Symbol"].tolist()))
        sent_df = sentiment_module.run(df["Symbol"].tolist(), max_headlines=10)

        combined_sent = pd.concat([sent_history, sent_df], ignore_index=True)
        data_loader.save_sentiment_history(combined_sent)
        sent_history = combined_sent  # already in memory — no need to reload
        logger.info(
            "Sentiment saved — %d total rows across %d days.",
            len(combined_sent),
            combined_sent["Date"].nunique(),
        )

    # ── Merge today's sentiment scores onto the latest-day price data ─────
    df = df.merge(
        sent_df[["Symbol", "Sentiment_Score", "Sentiment_Label", "Top_Headline"]],
        on="Symbol",
        how="left",
    )
    df["Sentiment_Label"] = df["Sentiment_Label"].fillna("No Data")
    df["Top_Headline"]    = df["Top_Headline"].fillna("No headline available")

    # ── Pre-compute daily sentiment averages once — reused by two charts ──
    # daily_sentiment_avg() lives in sentiment.py; calling it here means
    # neither the rolling trend chart nor the dual-axis chart needs to
    # repeat the groupby or the pd.to_numeric coerce.
    daily_sent = daily_sentiment_avg(sent_history)

    # High-level sentiment breadth
    sent_summary     = sentiment_module.breadth_summary(sent_df)
    pct_bull         = sent_summary["pct_bullish"]
    pct_bear         = sent_summary["pct_bearish"]
    pct_neut         = sent_summary["pct_neutral"]
    pct_no_data      = sent_summary.get("pct_no_data", 0.0)
    mean_sent        = sent_summary["mean_score"]
    sent_regime      = _sentiment_regime_label(mean_sent, pct_bull, pct_bear)
    n_bull_stocks    = sent_summary["n_bullish"]
    n_bear_stocks    = sent_summary["n_bearish"]
    n_neut_stocks    = sent_summary["n_neutral"]

    # Top movers by sentiment
    top_bull_df = sentiment_module.top_bullish(sent_df, n=5)
    top_bear_df = sentiment_module.top_bearish(sent_df, n=5)

    # ══════════════════════════════════════════════════════════════════════
    # HISTORICAL ANALYTICS  (Feature 3 data prep)
    # ══════════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════════
    # HISTORICAL ANALYTICS  (via historical_metrics module)
    # Incremental: only new days are computed; all else is served from
    # historical_metrics.parquet cache. Heavy Python loops removed.
    # ══════════════════════════════════════════════════════════════════════

    _hm          = hm.compute_all(df_full)
    hist_df      = _hm["hist_df"]
    freq_movers  = _hm["freq_movers"]
    cumulative_df= _hm["cumulative_df"]
    sessions_df  = _hm["sessions_df"]
    streaks_df   = _hm["streaks_df"]
    n_days_total = _hm["n_days"]
    has_history  = _hm["has_history"]
    latest_vol   = _hm["latest_vol"]
    baseline_vol = _hm["baseline_vol"]

    technical_df  = data_loader.load_technical_data()
    forecast_df   = data_loader.load_forecast_data()
    signal_df     = data_loader.load_signal_data()
    anomaly_df    = data_loader.load_anomaly_data()

    # Extended hist_df (OHLCV + scraped) used only for the forecast chart's
    # historical context — gives 30 days of real data before the projection.
    extended_hist_df = hm.build_extended_hist_df(df_full)

    # ── Time Series Analysis — uses extended_hist_df (2y OHLCV + scraped)
    # so cycles, crossovers, and decomposition see the full history baseline.
    # Falls back to scraped-only hist_df if OHLCV hasn't been fetched yet.
    _ts_hist   = extended_hist_df if not extended_hist_df.empty else hist_df
    _ts_series = _ts_hist["avg_pct"] if not _ts_hist.empty else pd.Series(dtype=float)
    ts_cycles   = ts_mod.compute_market_cycles(_ts_hist)
    ts_trend    = ts_mod.detect_trend_change(_ts_series)
    ts_decomp   = ts_mod.decompose(_ts_series, window=ROLLING.SHORT_WINDOW)
    ts_breadth  = ts_mod.compute_breadth_trend(_ts_hist)
    ts_stats    = ts_mod.rolling_stats(_ts_series, window=ROLLING.SHORT_WINDOW)

    # Align time series results with _ts_hist dates
    if not _ts_hist.empty and not ts_trend.empty:
        ts_trend = ts_trend.copy()
        ts_trend["Date"] = _ts_hist["Date"].values
    if not ts_breadth.empty:
        ts_breadth["Date"] = pd.to_datetime(ts_breadth["Date"])

    # KPIs for time series tab header
    if not ts_cycles.empty:
        _cur_cyc    = ts_cycles.iloc[-1]
        ts_cyc_dir  = _cur_cyc["direction"]
        ts_cyc_days = int(_cur_cyc["length_days"])
        ts_cyc_ret  = float(_cur_cyc["avg_return"])
    else:
        ts_cyc_dir, ts_cyc_days, ts_cyc_ret = "N/A", 0, 0.0

    if not ts_trend.empty:
        _crosses = ts_trend[ts_trend["crossover"] != ""]
        if not _crosses.empty:
            _last_cross     = _crosses.iloc[-1]
            ts_last_cross   = _last_cross["crossover"]
            ts_last_cross_d = pd.Timestamp(_last_cross["Date"]).strftime("%b %d, %Y")
        else:
            ts_last_cross, ts_last_cross_d = "None yet", "—"
    else:
        ts_last_cross, ts_last_cross_d = "None yet", "—"

    ts_cur_breadth_trend = ts_breadth["breadth_trend"].iloc[-1] if not ts_breadth.empty else "N/A"
    ts_cur_residual      = float(ts_decomp["residual"].iloc[-1]) if not _ts_hist.empty and len(ts_decomp.get("residual", [])) > 0 else 0.0

    # Historical OHLCV availability (zero if not yet fetched via --historical-fetch)
    n_ohlcv_symbols = len(data_loader.list_available_ohlcv())

    # Dynamic volatility label — uses helpers already defined above
    vol_label = (
        _volatility_label(latest_vol, baseline_vol)
        if has_history
        else "single-day data — no rolling baseline yet"
    )

    # ══════════════════════════════════════════════════════════════════════
    # COLOUR PALETTE (unchanged from v1)
    # ══════════════════════════════════════════════════════════════════════

    G   = "#00FFB2"   # electric mint  — gainers / bullish
    L   = "#FF3366"   # hot pink-red   — losers / bearish
    FL  = "#FFD600"   # vivid yellow   — flat / neutral
    AC  = "#00B4FF"   # electric blue  — accent / analytics
    PU  = "#BF5FFF"   # vivid purple
    OR  = "#FF7700"   # vivid orange
    BG  = "#06080F"   # near-black background
    SF  = "#0C0F1A"   # dark navy surface
    BD  = "#1A2035"   # border
    TX  = "#F0F4FF"   # cool white text
    MU  = "#5A6A90"   # muted blue-grey

    TIER_COLORS  = [G, AC, PU, OR, FL, L]
    SENT_COLORS  = {"Bullish": G, "Neutral": FL, "Bearish": L, "No Data": MU}

    GR  = "rgba(26,32,53,0.9)"
    FN  = dict(family="JetBrains Mono, Courier New, monospace", color=TX, size=12)
    HL  = dict(bgcolor="#0C0F1A", bordercolor=BD, font=dict(color=TX, size=12))
    theme = ChartTheme(
        G=G, L=L, FL=FL, AC=AC, PU=PU, OR=OR,
        BG=BG, SF=SF, BD=BD, TX=TX, MU=MU, GR=GR,
        FN=FN, HL=HL,
    )


    # ── Plotly layout helpers (unchanged from v1) ──────────────────────────
    def base(h=400):
        return dict(
            paper_bgcolor=BG, plot_bgcolor=SF, font=FN, height=h,
            margin=dict(l=60, r=40, t=55, b=55), hoverlabel=HL,
        )

    def ax(title="", color=G, zeroline=False):
        d = dict(
            showgrid=True, gridcolor=GR, gridwidth=1, linecolor=BD,
            tickfont=dict(color=MU, size=11),
            title_font=dict(color=MU, size=12),
            zeroline=zeroline,
        )
        if title:
            d["title_text"] = title
        if zeroline:
            d["zerolinecolor"] = "rgba(255,255,255,0.06)"
        return d

    def ttl(t, s=""):
        sub = f"<br><span style='font-size:11px;color:{MU}'>{s}</span>" if s else ""
        return dict(
            text=f"<b>{t}</b>{sub}",
            font=dict(size=14, color=TX),
            x=0.01, xanchor="left",
        )

    def to_div(fig, div_id):
        return fig.to_html(
            full_html=False, include_plotlyjs=False, div_id=div_id,
            config=dict(responsive=True, displaylogo=False,
                        modeBarButtonsToRemove=["lasso2d", "select2d"]),
        )


    divs = {}

    # ══════════════════════════════════════════════════════════════════════
    # EXISTING CHARTS — OVERVIEW TAB (unchanged)
    # ══════════════════════════════════════════════════════════════════════

    # Price tier bucket order
    bo = ["<$10", "$10-25", "$25-50", "$50-100", "$100-200", ">$200"]
    bv = df["Price Tier"].value_counts().reindex(bo).fillna(0).astype(int)

    # Scatter: Price vs % Change
    fig = go.Figure()
    for grp, color, name in [(gainers, G, "Gainer"), (losers, L, "Loser")]:
        cd = grp[["Company Name", "Price", "% Change"]].values.tolist()
        fig.add_trace(go.Scatter(
            x=grp["Price"].tolist(), y=grp["% Change"].tolist(),
            mode="markers", name=name,
            marker=dict(color=color, size=10, opacity=0.87,
                        line=dict(width=0.8, color="rgba(255,255,255,0.1)")),
            text=grp["Symbol"].tolist(), customdata=cd,
            hovertemplate="<b>%{text}</b>  %{customdata[0]}<br>"
                          "Price: $%{customdata[1]:.2f}  ·  %{customdata[2]:+.2f}%<extra></extra>",
        ))
    fig.update_layout(**base(390),
        title=ttl("Price vs % Change", "Each dot = one active stock · hover for details"),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TX, size=11),
                    orientation="h", x=0, y=-0.13))
    fig.update_xaxes(**ax("Stock Price ($)"))
    fig.update_yaxes(**ax("% Change", zeroline=True))
    divs["ov_scatter"] = to_div(fig, "ov_scatter")

    # Donut: Market Breadth
    fig = go.Figure(go.Pie(
        values=[N_GAIN, N_LOSS, N_FLAT],
        labels=["Gainers", "Losers", "Flat"],
        hole=0.62,
        marker=dict(colors=[G, L, FL], line=dict(color=BG, width=3)),
        textfont=dict(color=TX, size=12), showlegend=True,
        hovertemplate="<b>%{label}</b>: %{value} stocks<extra></extra>",
    ))
    fig.update_layout(**base(390),
        title=ttl("Market Breadth", "Session direction"),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TX, size=11),
                    orientation="h", x=0.05, y=-0.08),
        annotations=[dict(
            text=f"<b>{breadth_pct:.0f}%</b><br>{session_label}",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=15, color=G),
        )],
    )
    divs["ov_donut"] = to_div(fig, "ov_donut")

    # Bar: Price Distribution by Tier
    fig = go.Figure(go.Bar(
        x=bo, y=bv.values.tolist(),
        marker=dict(color=TIER_COLORS, opacity=0.9,
                    line=dict(color="rgba(255,255,255,0.05)", width=1)),
        text=bv.values.tolist(), textposition="outside",
        textfont=dict(color=TX, size=12),
        hovertemplate="<b>%{x}</b><br>%{y} stocks<extra></extra>",
    ))
    fig.update_layout(**base(300),
        title=ttl("Price Distribution",
                  f"{low_price_count} of {N_TOTAL} stocks trade under $25"))
    fig.update_xaxes(**ax("Price Tier"))
    fig.update_yaxes(**ax("Count"))
    divs["ov_hist"] = to_div(fig, "ov_hist")

    # ══════════════════════════════════════════════════════════════════════
    # TOP MOVERS TAB (unchanged)
    # ══════════════════════════════════════════════════════════════════════

    top10  = df.nlargest(10, "% Change").sort_values("% Change", ascending=True)
    bot10  = df.nsmallest(10, "% Change").sort_values("% Change", ascending=False)
    top15  = df.nlargest(15, "Abs Change").sort_values("Change")

    # Gainers bar
    gc = px.colors.sample_colorscale(
        [[0, "#003d2e"], [0.5, AC], [1, G]], [i / 9 for i in range(10)]
    )
    cd = top10[["Company Name", "Price"]].values.tolist()
    fig = go.Figure(go.Bar(
        x=top10["% Change"].tolist(), y=top10["Symbol"].tolist(), orientation="h",
        marker=dict(color=gc, line=dict(color="rgba(0,255,178,0.15)", width=1)),
        text=[f"+{v:.2f}%" for v in top10["% Change"]],
        textposition="outside", textfont=dict(color=G, size=11),
        customdata=cd,
        hovertemplate="<b>%{y}</b>  %{customdata[0]}<br>$%{customdata[1]:.2f}  ·  <b>+%{x:.2f}%</b><extra></extra>",
    ))
    fig.update_layout(**base(370),
        title=ttl("Top 10 Gainers", "Ranked by % change today"))
    fig.update_xaxes(**ax("% Change"), range=[0, top10["% Change"].max() * 1.3])
    fig.update_yaxes(showgrid=False,
                     tickfont=dict(color=G, size=12, family="JetBrains Mono"),
                     linecolor=BD)
    divs["mv_gainers"] = to_div(fig, "mv_gainers")

    # Losers bar
    cd = bot10[["Company Name", "Price", "Change"]].values.tolist()
    fig = go.Figure(go.Bar(
        x=bot10["% Change"].abs().tolist(), y=bot10["Symbol"].tolist(), orientation="h",
        marker=dict(color=["rgba(255,51,102,0.8)"] * 10,
                    line=dict(color="rgba(255,51,102,0.2)", width=1)),
        text=[f"{v:.2f}%" for v in bot10["% Change"]],
        textposition="outside", textfont=dict(color=L, size=11),
        customdata=cd,
        hovertemplate="<b>%{y}</b>  %{customdata[0]}<br>$%{customdata[1]:.2f}  ·  $%{customdata[2]:.2f}<extra></extra>",
    ))
    fig.update_layout(**base(370),
        title=ttl("Top 10 Losers", "Ranked by % decline today"))
    fig.update_xaxes(**ax("Absolute % Change"),
                     range=[0, bot10["% Change"].abs().max() * 1.3])
    fig.update_yaxes(showgrid=False,
                     tickfont=dict(color=L, size=12, family="JetBrains Mono"),
                     linecolor=BD)
    divs["mv_losers"] = to_div(fig, "mv_losers")

    # Dollar moves
    cd = top15[["Company Name", "Price"]].values.tolist()
    fig = go.Figure(go.Bar(
        x=top15["Symbol"].tolist(), y=top15["Change"].tolist(),
        marker=dict(color=[G if v >= 0 else L for v in top15["Change"]],
                    opacity=0.88,
                    line=dict(color="rgba(255,255,255,0.04)", width=1)),
        text=[f"{'+'if v >= 0 else ''}${v:.2f}" for v in top15["Change"]],
        textposition="outside", textfont=dict(size=10, color=TX),
        customdata=cd,
        hovertemplate="<b>%{x}</b>  %{customdata[0]}<br>$%{customdata[1]:.2f}  ·  $%{y:+.2f}<extra></extra>",
    ))
    fig.update_layout(**base(330),
        title=ttl("Top 15 Biggest Dollar Moves", "Raw $ movement per stock"))
    fig.update_xaxes(showgrid=False,
                     tickfont=dict(color=MU, size=11, family="JetBrains Mono"),
                     linecolor=BD)
    fig.update_yaxes(**ax("Dollar Change ($)", zeroline=True))
    divs["mv_dollar"] = to_div(fig, "mv_dollar")

    # ══════════════════════════════════════════════════════════════════════
    # ANALYTICS TAB (unchanged)
    # ══════════════════════════════════════════════════════════════════════

    # Bucket bar
    fig = go.Figure(go.Bar(
        x=bo, y=bv.values.tolist(),
        marker=dict(color=TIER_COLORS, opacity=0.9,
                    line=dict(color="rgba(255,255,255,0.04)", width=1)),
        text=bv.values.tolist(), textposition="outside",
        textfont=dict(color=TX, size=12),
        hovertemplate="<b>%{x}</b><br>%{y} stocks<extra></extra>",
    ))
    fig.update_layout(**base(320), title=ttl("Stock Count by Price Tier"))
    fig.update_xaxes(**ax())
    fig.update_yaxes(**ax("Count"))
    divs["an_bucket"] = to_div(fig, "an_bucket")

    # Box per price tier
    fig = go.Figure()
    for i, tier in enumerate(bo):
        sub = df[df["Price Tier"] == tier]["% Change"].dropna().tolist()
        if not sub:
            continue
        r = int(TIER_COLORS[i][1:3], 16)
        g2 = int(TIER_COLORS[i][3:5], 16)
        b = int(TIER_COLORS[i][5:7], 16)
        fig.add_trace(go.Box(
            y=sub, name=tier, boxpoints="outliers",
            marker=dict(color=TIER_COLORS[i], size=5),
            line=dict(color=TIER_COLORS[i]),
            fillcolor=f"rgba({r},{g2},{b},0.18)",
            hovertemplate=f"<b>{tier}</b><br>%{{y:.2f}}%<extra></extra>",
        ))
    fig.update_layout(**base(320),
        title=ttl("% Change by Price Band",
                  "Does stock price affect daily volatility?"))
    fig.update_xaxes(showgrid=False,
                     tickfont=dict(color=MU, size=11), linecolor=BD)
    fig.update_yaxes(**ax("% Change", zeroline=True))
    divs["an_box"] = to_div(fig, "an_box")

    # Spearman heatmap
    corr_vals = SPEARMAN_MATRIX
    corr_cols  = ["Price", "Change", "% Change"]
    fig = go.Figure(go.Heatmap(
        z=corr_vals, x=corr_cols, y=corr_cols,
        colorscale=[[0, L], [0.35, SF], [0.5, "#1A2035"], [0.65, SF], [1, G]],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr_vals],
        texttemplate="%{text}", textfont=dict(color=TX, size=16),
        hovertemplate="%{y} x %{x}<br>rho = %{z:.2f}<extra></extra>",
        showscale=True,
        colorbar=dict(tickfont=dict(color=MU, size=10), thickness=14, len=0.85,
                      tickcolor=MU, outlinecolor=BD),
    ))
    fig.update_layout(**base(320),
        title=ttl("Spearman Correlation Matrix", "Rank-based correlation"))
    fig.update_xaxes(showgrid=False, tickfont=dict(color=TX, size=12), linecolor=BD)
    fig.update_yaxes(showgrid=False, tickfont=dict(color=TX, size=12), linecolor=BD)
    divs["an_corr"] = to_div(fig, "an_corr")

    # Violin: Gainer vs Loser % Change
    fig = go.Figure()
    for status, color in [("Gainer", G), ("Loser", L)]:
        sub = df[df["Status"] == status]["% Change"].tolist()
        r = int(color[1:3], 16)
        g2 = int(color[3:5], 16)
        b = int(color[5:7], 16)
        fig.add_trace(go.Violin(
            y=sub, name=status,
            line_color=color, fillcolor=f"rgba({r},{g2},{b},0.22)",
            box_visible=True, meanline_visible=True,
            meanline=dict(color=color, width=2), points="outliers",
            marker=dict(color=color, size=4, opacity=0.7),
            hovertemplate=f"<b>{status}</b><br>%{{y:.2f}}%<extra></extra>",
        ))
    fig.update_layout(**base(320),
        title=ttl("% Change Violin — Gainers vs Losers",
                  "Density shape + quartile box"),
        showlegend=True, violinmode="overlay",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TX, size=11),
                    x=0, y=-0.1, orientation="h"))
    fig.update_xaxes(showgrid=False, linecolor=BD, tickfont=dict(color=MU, size=11))
    fig.update_yaxes(**ax("% Change", zeroline=True))
    divs["an_violin"] = to_div(fig, "an_violin")

    # ══════════════════════════════════════════════════════════════════════
    # SCREENER TAB (unchanged)
    # ══════════════════════════════════════════════════════════════════════

    sdf = df.sort_values("% Change", ascending=False).reset_index(drop=True)
    mx  = sdf["% Change"].abs().max()

    def _cbg(v):
        a = min(abs(v) / mx, 1) * 0.45 + 0.07 if mx else 0.07
        if v > 0:  return f"rgba(0,255,178,{a:.2f})"
        elif v < 0: return f"rgba(255,51,102,{a:.2f})"
        return "rgba(255,214,0,0.12)"

    rf  = ["rgba(12,15,26,1)" if i % 2 == 0 else "rgba(16,20,34,1)" for i in range(len(sdf))]
    pb  = [_cbg(v) for v in sdf["% Change"].tolist()]
    cb  = [_cbg(v) for v in sdf["Change"].tolist()]
    pf  = [G if v > 0 else (L if v < 0 else FL) for v in sdf["% Change"].tolist()]
    cf  = [G if v > 0 else (L if v < 0 else FL) for v in sdf["Change"].tolist()]
    sf2 = [{"Gainer": G, "Loser": L, "Flat": FL}.get(s, TX) for s in sdf["Status"].tolist()]

    fig = go.Figure(go.Table(
        columnwidth=[28, 55, 220, 75, 80, 82, 60],
        header=dict(
            values=["<b>#</b>", "<b>SYM</b>", "<b>COMPANY</b>",
                    "<b>PRICE</b>", "<b>$ CHG</b>", "<b>% CHG</b>", "<b>STATUS</b>"],
            fill_color="#080B16",
            font=dict(color=MU, size=10, family="JetBrains Mono"),
            line_color=BD,
            align=["center", "left", "left", "right", "right", "right", "center"],
            height=36,
        ),
        cells=dict(
            values=[
                list(range(1, len(sdf) + 1)),
                sdf["Symbol"].tolist(),
                sdf["Company Name"].tolist(),
                [f"${v:.2f}" for v in sdf["Price"].tolist()],
                [f"{'+' if v >= 0 else ''}${v:.2f}" for v in sdf["Change"].tolist()],
                [f"{'+' if v > 0 else ''}{v:.2f}%" for v in sdf["% Change"].tolist()],
                sdf["Status"].tolist(),
            ],
            fill_color=[rf, rf, rf, rf, cb, pb, rf],
            font=dict(
                color=[[MU] * len(sdf), [AC] * len(sdf), [TX] * len(sdf),
                       [TX] * len(sdf), cf, pf, sf2],
                size=11, family="JetBrains Mono",
            ),
            line_color=BD,
            align=["center", "left", "left", "right", "right", "right", "center"],
            height=27,
        ),
    ))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=SF, font=FN, height=2940,
        margin=dict(l=20, r=20, t=80, b=20), hoverlabel=HL,
        title=ttl("All 100 Active Stocks",
                  "Sorted by % change · Color = move magnitude"),
    )
    divs["screener"] = to_div(fig, "screener")

    # ══════════════════════════════════════════════════════════════════════
    # STORY TAB (unchanged)
    # ══════════════════════════════════════════════════════════════════════

    upper = UPPER_FENCE

    # Annotated scatter
    fig = go.Figure()
    for grp, color, name in [(gainers, G, "Gainer"), (losers, L, "Loser")]:
        cd = grp[["Company Name", "% Change"]].values.tolist()
        fig.add_trace(go.Scatter(
            x=grp["Price"].tolist(), y=grp["% Change"].tolist(),
            mode="markers", name=name,
            marker=dict(color=color, size=10, opacity=0.82,
                        line=dict(width=0.7, color="rgba(255,255,255,0.07)")),
            text=grp["Symbol"].tolist(), customdata=cd,
            hovertemplate="<b>%{text}</b>  %{customdata[0]}<br>$%{x:.2f} · %{customdata[1]:+.2f}%<extra></extra>",
        ))

    anno = []
    annotated_symbols = set()
    for row, col, yshift in [
        (top_row, G, -38), (bot_row, L, 42),
        (big_move_row, G if big_move_row["Change"] >= 0 else L, 28),
    ]:
        symbol = str(row["Symbol"])
        if symbol in annotated_symbols:
            continue
        annotated_symbols.add(symbol)
        label = f"{html_utils.escape(str(row['Symbol']))} {row['% Change']:+.1f}%"
        anno.append((row["Price"], row["% Change"], label, 65, yshift, col))

    fig.update_layout(**base(430),
        title=ttl("Annotated Price Map",
                  "Where every active stock landed today"),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TX, size=11),
                    orientation="h", x=0, y=-0.1),
        annotations=[
            dict(x=px_, y=py_, text=f"<b>{lbl}</b>", showarrow=True,
                 ax=ax_, ay=ay_, arrowcolor=col, arrowwidth=1.5, arrowsize=0.8,
                 font=dict(color=col, size=10))
            for px_, py_, lbl, ax_, ay_, col in anno
        ],
    )
    fig.update_xaxes(**ax("Stock Price ($)"))
    fig.update_yaxes(**ax("% Change", zeroline=True))
    divs["st_scatter"] = to_div(fig, "st_scatter")

    # Skewness histogram
    fig = go.Figure(go.Histogram(
        x=df["Price"].tolist(), nbinsx=25,
        marker=dict(color=AC, opacity=0.8, line=dict(color=BG, width=0.5)),
        hovertemplate="~$%{x:.0f}<br>%{y} stocks<extra></extra>",
    ))
    fig.add_vline(x=PRICE_MEDIAN, line_dash="dot", line_color=G, line_width=2,
        annotation=dict(text=f"Median ${PRICE_MEDIAN:.0f}", font_color=G,
                        font_size=11, bgcolor="rgba(0,0,0,0.5)",
                        yref="paper", y=0.95, xanchor="left"))
    fig.add_vline(x=PRICE_MEAN, line_dash="dash", line_color=FL, line_width=2,
        annotation=dict(text=f"Mean ${PRICE_MEAN:.0f}", font_color=FL,
                        font_size=11, bgcolor="rgba(0,0,0,0.5)",
                        yref="paper", y=0.75, xanchor="left"))
    fig.update_layout(**base(340),
        title=ttl("Price Skewness Exposed",
                  f"Mean ${PRICE_MEAN:.0f} vs Median ${PRICE_MEDIAN:.0f} — mega-caps distort the average"))
    fig.update_xaxes(**ax("Stock Price ($)"))
    fig.update_yaxes(**ax("Count"))
    divs["st_hist"] = to_div(fig, "st_hist")

    # IQR outlier detection
    _out  = df[df["Price"] > UPPER_FENCE].sort_values("Price", ascending=False).reset_index(drop=True)
    _norm = df[df["Price"] <= UPPER_FENCE].reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(_norm))), y=_norm["Price"].tolist(),
        mode="markers", name="Normal",
        marker=dict(color=AC, size=7, opacity=0.65),
        text=_norm["Symbol"].tolist(),
        hovertemplate="<b>%{text}</b><br>$%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(_out))), y=_out["Price"].tolist(),
        mode="markers+text", name=f"Outlier (n={N_OUTLIERS})",
        marker=dict(color=FL, size=14, symbol="diamond",
                    line=dict(width=2, color=L)),
        text=_out["Symbol"].tolist(), textposition="top right",
        textfont=dict(color=FL, size=10),
        hovertemplate="<b>%{text}</b><br>$%{y:.2f} — OUTLIER<extra></extra>",
    ))
    fig.add_hline(y=UPPER_FENCE, line_dash="dash", line_color=L, line_width=1.5,
        annotation=dict(text=f"IQR Upper Fence: ${UPPER_FENCE:.0f}",
                        font_color=L, font_size=11, bgcolor="rgba(0,0,0,0.5)",
                        xanchor="right"))
    fig.update_layout(**base(360),
        title=ttl("IQR Outlier Detection",
                  f"Upper fence ${UPPER_FENCE:.0f} · {N_OUTLIERS} outliers"),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TX, size=11),
                    x=0, y=-0.1, orientation="h"))
    fig.update_xaxes(**ax("Stock Index"))
    fig.update_yaxes(**ax("Stock Price ($)"))
    divs["st_iqr"] = to_div(fig, "st_iqr")

    # Mann-Whitney box
    fig = go.Figure()
    for status, color in [("Gainer", G), ("Loser", L)]:
        sub = df[df["Status"] == status]["Price"].tolist()
        r = int(color[1:3], 16)
        g2 = int(color[3:5], 16)
        b = int(color[5:7], 16)
        fig.add_trace(go.Box(
            y=sub, name=status, boxpoints="outliers",
            marker=dict(color=color, size=5), line=dict(color=color),
            fillcolor=f"rgba({r},{g2},{b},0.18)",
            hovertemplate=f"<b>{status}</b><br>$%{{y:.2f}}<extra></extra>",
        ))
    mw_txt = f"p = {MW_P:.4f} - {mw_note}"
    fig.update_layout(**base(340),
        title=ttl("Are Gainers Priced Differently?",
                  f"Mann-Whitney p={MW_P:.3f} · {mw_txt}"),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TX, size=11)))
    fig.update_xaxes(showgrid=False, linecolor=BD, tickfont=dict(color=MU, size=11))
    fig.update_yaxes(**ax("Stock Price ($)"))
    divs["st_box"] = to_div(fig, "st_box")

    # Session extremes
    top5g = df.nlargest(5, "% Change").sort_values("% Change")
    top5l = df.nsmallest(5, "% Change").sort_values("% Change", ascending=False)
    comb  = pd.concat([top5l, top5g])
    extreme_spread = df["% Change"].max() - df["% Change"].min()
    fig = go.Figure(go.Bar(
        x=comb["Symbol"].tolist(), y=comb["% Change"].tolist(),
        marker=dict(color=[L if v < 0 else G for v in comb["% Change"].tolist()],
                    opacity=0.9,
                    line=dict(color="rgba(255,255,255,0.04)", width=1)),
        text=[f"{'+' if v > 0 else ''}{v:.1f}%" for v in comb["% Change"].tolist()],
        textposition="outside", textfont=dict(size=10, color=TX),
        hovertemplate="<b>%{x}</b><br>%{y:+.2f}%<extra></extra>",
    ))
    fig.update_layout(**base(340),
        title=ttl("Top 5 Declines vs Top 5 Surges",
                  f"Session extremes — {extreme_spread:.1f}pp spread"))
    fig.update_xaxes(showgrid=False,
                     tickfont=dict(color=MU, size=11, family="JetBrains Mono"),
                     linecolor=BD)
    fig.update_yaxes(**ax("% Change", zeroline=True))
    divs["st_extremes"] = to_div(fig, "st_extremes")

    # Spearman rank scatter
    fig = go.Figure(go.Scatter(
        x=df["Price"].rank().tolist(), y=df["% Change"].rank().tolist(),
        mode="markers",
        marker=dict(
            color=df["% Change"].tolist(),
            colorscale=[[0, L], [0.4, MU], [0.6, MU], [1, G]],
            size=8, opacity=0.8, showscale=True,
            colorbar=dict(title=dict(text="% Chg", font=dict(color=MU, size=10)),
                          tickfont=dict(color=MU, size=9), thickness=12, len=0.75,
                          outlinecolor=BD),
        ),
        text=df["Symbol"].tolist(),
        hovertemplate="<b>%{text}</b><br>Price rank: %{x:.0f}<br>%%Chg rank: %{y:.0f}<extra></extra>",
    ))
    fig.update_layout(**base(340),
        title=ttl("Spearman Rank: Price vs % Change",
                  f"rho = {SP_CORR:.3f}  |  p = {SP_P:.4f}  |  {sp_summary}"),
        annotations=[dict(
            x=0.98, y=0.04, xref="paper", yref="paper",
            text=f"<b>rho = {SP_CORR:.3f}</b>  p = {SP_P:.4f}",
            showarrow=False, font=dict(color=FL, size=12),
            bgcolor="rgba(0,0,0,0.6)", bordercolor=BD, align="right",
        )],
    )
    fig.update_xaxes(**ax("Price Rank"))
    fig.update_yaxes(**ax("% Change Rank"))
    divs["st_spearman"] = to_div(fig, "st_spearman")

    # ══════════════════════════════════════════════════════════════════════
    # SENTIMENT TAB — Feature 1 + 2
    # All chart subtitles and annotations are built from live computed
    # values — no hardcoded market descriptions.
    # ══════════════════════════════════════════════════════════════════════

    # --- 1. Market Sentiment Overview: Donut ----------------------------
    bull_donut_color  = G if pct_bull > pct_bear else MU
    donut_center_text = f"<b>{pct_bull:.0f}%</b><br>Bullish"
    fig = go.Figure(go.Pie(
        values=[pct_bull, pct_bear, pct_neut, pct_no_data],
        labels=["Bullish", "Bearish", "Neutral", "No Data"],
        hole=0.62,
        marker=dict(colors=[G, L, FL, MU], line=dict(color=BG, width=3)),
        textfont=dict(color=TX, size=12),
        hovertemplate="<b>%{label}</b>: %{value:.1f}%  (%{customdata} stocks)<extra></extra>",
        customdata=[n_bull_stocks, n_bear_stocks, n_neut_stocks],
    ))
    fig.update_layout(**base(380),
        title=ttl("Market Sentiment Overview",
                  f"Regime: {sent_regime} · mean sentiment score: {mean_sent:+.4f}"),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TX, size=11),
                    orientation="h", x=0.05, y=-0.08),
        annotations=[dict(
            text=donut_center_text,
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=15, color=bull_donut_color),
        )],
    )
    divs["sent_donut"] = to_div(fig, "sent_donut")

    # --- 2. Top Bullish Stocks bar --------------------------------------
    if not top_bull_df.empty:
        bull_sorted = top_bull_df.sort_values("Sentiment_Score", ascending=True)
        fig = go.Figure(go.Bar(
            x=bull_sorted["Sentiment_Score"].tolist(),
            y=bull_sorted["Symbol"].tolist(),
            orientation="h",
            marker=dict(
                color=bull_sorted["Sentiment_Score"].tolist(),
                colorscale=[[0, "#004d33"], [1, G]],
                line=dict(color="rgba(0,255,178,0.15)", width=1),
            ),
            text=[f"{v:+.4f}" for v in bull_sorted["Sentiment_Score"].tolist()],
            textposition="outside", textfont=dict(color=G, size=11),
            customdata=bull_sorted[["Top_Headline", "Headlines_Used"]].values.tolist(),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Sentiment: %{x:+.4f}<br>"
                "Headlines: %{customdata[1]}<br>"
                "%{customdata[0]}<extra></extra>"
            ),
        ))
        fig.update_layout(**base(330),
            title=ttl("Top Bullish Stocks",
                      "Highest sentiment scores — most positive news tone"))
        fig.update_xaxes(**ax("Compound Score"), range=[0, 1.0])
        fig.update_yaxes(showgrid=False,
                         tickfont=dict(color=G, size=12, family="JetBrains Mono"),
                         linecolor=BD)
        divs["sent_bull"] = to_div(fig, "sent_bull")
    else:
        divs["sent_bull"] = "<p style='color:#5A6A90;padding:20px'>No bullish signals detected in current data.</p>"

    # --- 3. Top Bearish Stocks bar -------------------------------------
    if not top_bear_df.empty:
        bear_sorted = top_bear_df.sort_values("Sentiment_Score", ascending=False)
        fig = go.Figure(go.Bar(
            x=bear_sorted["Sentiment_Score"].abs().tolist(),
            y=bear_sorted["Symbol"].tolist(),
            orientation="h",
            marker=dict(
                color=bear_sorted["Sentiment_Score"].abs().tolist(),
                colorscale=[[0, "#4d0011"], [1, L]],
                line=dict(color="rgba(255,51,102,0.2)", width=1),
            ),
            text=[f"{v:+.4f}" for v in bear_sorted["Sentiment_Score"].tolist()],
            textposition="outside", textfont=dict(color=L, size=11),
            customdata=bear_sorted[["Sentiment_Score", "Top_Headline", "Headlines_Used"]].values.tolist(),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Sentiment: %{customdata[0]:+.4f}<br>"
                "%{customdata[1]}<br>"
                "Headlines: %{customdata[2]}<extra></extra>"
            ),
        ))
        fig.update_layout(**base(330),
            title=ttl("Top Bearish Stocks",
                      "Most negative sentiment scores — weakest news sentiment"))
        fig.update_xaxes(**ax("Absolute Compound Score"), range=[0, 1.0])
        fig.update_yaxes(showgrid=False,
                         tickfont=dict(color=L, size=12, family="JetBrains Mono"),
                         linecolor=BD)
        divs["sent_bear"] = to_div(fig, "sent_bear")
    else:
        divs["sent_bear"] = "<p style='color:#5A6A90;padding:20px'>No bearish signals detected in current data.</p>"

    # --- 4. Sentiment vs Price Performance scatter ----------------------
    # Colour each point by its sentiment label; marker size = abs % Change
    sent_scatter_df = df.dropna(subset=["Sentiment_Score", "% Change"]).copy()
    sent_scatter_df["marker_size"] = sent_scatter_df["% Change"].abs().clip(lower=2) * 1.5
    sent_scatter_df["pt_color"]    = sent_scatter_df["Sentiment_Label"].map(SENT_COLORS).fillna(MU)

    fig = go.Figure()
    for label, color in SENT_COLORS.items():
        sub = sent_scatter_df[sent_scatter_df["Sentiment_Label"] == label]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["Sentiment_Score"].tolist(),
            y=sub["% Change"].tolist(),
            mode="markers", name=label,
            marker=dict(
                color=color, size=sub["marker_size"].tolist(),
                opacity=0.80, line=dict(width=0.6, color="rgba(255,255,255,0.1)"),
            ),
            text=sub["Symbol"].tolist(),
            customdata=sub[["Company Name", "Sentiment_Score", "% Change"]].values.tolist(),
            hovertemplate=(
                "<b>%{text}</b>  %{customdata[0]}<br>"
                "Sentiment: %{customdata[1]:+.4f}<br>"
                "% Change: %{customdata[2]:+.2f}%<extra></extra>"
            ),
        ))

    # Compute and annotate Pearson correlation of sentiment vs price change
    _s = sent_scatter_df[["Sentiment_Score", "% Change"]].dropna()
    if len(_s) >= 3:
        sent_vs_price_corr = float(np.corrcoef(_s["Sentiment_Score"], _s["% Change"])[0, 1])
        corr_annotation = f"Pearson r = {sent_vs_price_corr:+.3f}"
    else:
        sent_vs_price_corr = 0.0
        corr_annotation = "Insufficient data for correlation"

    fig.update_layout(**base(420),
        title=ttl("Sentiment vs Price Performance",
                  f"x = sentiment score · y = % Change · bubble size = magnitude · {corr_annotation}"),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TX, size=11),
                    orientation="h", x=0, y=-0.13),
        annotations=[dict(
            x=0.98, y=0.04, xref="paper", yref="paper",
            text=f"<b>{corr_annotation}</b>",
            showarrow=False, font=dict(color=FL, size=12),
            bgcolor="rgba(0,0,0,0.6)", bordercolor=BD,
        )],
    )
    fig.update_xaxes(**ax("Sentiment Score", zeroline=True),
                     range=[-1.05, 1.05])
    fig.update_yaxes(**ax("% Change", zeroline=True))
    divs["sent_vs_price"] = to_div(fig, "sent_vs_price")

    # --- 5. Rolling Sentiment Trend (real historical data) --------------
    # daily_sent was computed once above using daily_sentiment_avg() from
    # sentiment.py — no groupby or pd.to_numeric needed here.
    if not daily_sent.empty:
        roll_colors = [
            G if v >= BULLISH_THRESHOLD else (L if v <= BEARISH_THRESHOLD else FL)
            for v in daily_sent["rolling_7d"].tolist()
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily_sent["Date"].tolist(),
            y=daily_sent["avg_score"].tolist(),
            name="Daily Avg Score",
            marker=dict(
                color=[G if v >= BULLISH_THRESHOLD else (L if v <= BEARISH_THRESHOLD else FL)
                       for v in daily_sent["avg_score"].tolist()],
                opacity=0.5,
            ),
            hovertemplate="<b>%{x}</b><br>Daily avg: %{y:+.4f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=daily_sent["Date"].tolist(),
            y=daily_sent["rolling_7d"].tolist(),
            mode="lines+markers",
            name="7-Day Rolling Avg",
            line=dict(color=AC, width=2),
            marker=dict(color=roll_colors, size=7,
                        line=dict(width=1, color="rgba(255,255,255,0.1)")),
            hovertemplate="<b>%{x}</b><br>7D rolling avg: %{y:+.4f}<extra></extra>",
        ))
        fig.add_hline(y=BULLISH_THRESHOLD, line_dash="dot", line_color=G, line_width=1,
            annotation=dict(text="Bullish threshold", font_color=G,
                            font_size=10, xanchor="left"))
        fig.add_hline(y=BEARISH_THRESHOLD, line_dash="dot", line_color=L, line_width=1,
            annotation=dict(text="Bearish threshold", font_color=L,
                            font_size=10, xanchor="left"))
        n_sent_days = daily_sent["Date"].nunique()
        fig.update_layout(**base(360),
            title=ttl("7-Day Rolling Sentiment Trend",
                      f"Real data across {n_sent_days} stored day(s) — grows with each daily run"),
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TX, size=11),
                        orientation="h", x=0, y=-0.13))
        fig.update_xaxes(**ax("Date"))
        fig.update_yaxes(**ax("Sentiment Score", zeroline=True), range=[-1, 1])
        divs["sent_trend"] = to_div(fig, "sent_trend")
    else:
        divs["sent_trend"] = (
            "<p style='color:#5A6A90;padding:20px;font-family:monospace'>"
            "Rolling sentiment trend will appear here after the first dashboard run."
            "</p>"
        )

    # --- 6. News Intelligence Section ----------------------------------
    # Build sorted list of all headlines with their scores for display.
    news_rows_html = ""
    news_src = sent_df.sort_values("Sentiment_Score", ascending=False)
    for _, row in news_src.iterrows():
        score    = row["Sentiment_Score"]
        label    = row["Sentiment_Label"]
        color    = SENT_COLORS.get(label, MU)
        headline = html_utils.escape(str(row["Top_Headline"]))
        sym      = html_utils.escape(str(row["Symbol"]))
        n_used   = int(row["Headlines_Used"])
        news_rows_html += (
            f'<div class="news-row">'
            f'<div class="news-sym" style="color:{color}">{sym}</div>'
            f'<div class="news-hl">{headline}</div>'
            f'<div class="news-score" style="color:{color}">{score:+.4f}</div>'
            f'<div class="news-label" style="color:{color}">{label}</div>'
            f'<div class="news-n">{n_used} headlines</div>'
            f'</div>'
        )

    # ══════════════════════════════════════════════════════════════════════
    # HISTORICAL TAB — Feature 3
    # ══════════════════════════════════════════════════════════════════════

    date_strs = [str(d.date()) for d in hist_df["Date"].tolist()]
    hist_colors = [G if r == "Bullish" else (L if r == "Bearish" else FL)
                   for r in hist_df["regime"].tolist()]

    # --- 1. Daily Market Breadth Trend ----------------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=date_strs, y=hist_df["n_gainers"].tolist(),
        mode="lines+markers", name="Gainers",
        line=dict(color=G, width=2),
        marker=dict(color=G, size=6),
        hovertemplate="<b>%{x}</b><br>Gainers: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=date_strs, y=hist_df["n_losers"].tolist(),
        mode="lines+markers", name="Losers",
        line=dict(color=L, width=2),
        marker=dict(color=L, size=6),
        hovertemplate="<b>%{x}</b><br>Losers: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=date_strs, y=hist_df["n_flat"].tolist(),
        mode="lines+markers", name="Flat",
        line=dict(color=FL, width=1.5, dash="dot"),
        marker=dict(color=FL, size=5),
        hovertemplate="<b>%{x}</b><br>Flat: %{y}<extra></extra>",
    ))
    breadth_note = (
        f"Latest session: {hist_df['n_gainers'].iloc[-1]} gainers, "
        f"{hist_df['n_losers'].iloc[-1]} losers — "
        f"{_breadth_label(hist_df['breadth_pct'].iloc[-1])}"
        if not hist_df.empty else "No data"
    )
    fig.update_layout(**base(360),
        title=ttl("Daily Market Breadth Trend", breadth_note),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TX, size=11),
                    orientation="h", x=0, y=-0.13))
    fig.update_xaxes(**ax("Date"))
    fig.update_yaxes(**ax("Number of Stocks"), range=[0, N_TOTAL + 5])
    divs["hist_breadth"] = to_div(fig, "hist_breadth")

    # --- 2. Daily Average % Change with Regime Shading -----------------
    avg_values = hist_df["avg_pct"].tolist()
    avg_colors = [G if v >= 0 else L for v in avg_values]

    fig = go.Figure(go.Bar(
        x=date_strs, y=avg_values,
        marker=dict(color=avg_colors, opacity=0.85,
                    line=dict(color="rgba(255,255,255,0.04)", width=1)),
        text=[f"{'+' if v >= 0 else ''}{v:.2f}%" for v in avg_values],
        textposition="outside", textfont=dict(color=TX, size=10),
        hovertemplate="<b>%{x}</b><br>Avg change: %{y:+.2f}%<extra></extra>",
    ))
    # Rolling 7-day average overlay
    fig.add_trace(go.Scatter(
        x=date_strs, y=hist_df["rolling_avg_7d"].tolist(),
        mode="lines", name="7-Day Rolling Avg",
        line=dict(color=AC, width=2, dash="dash"),
        hovertemplate="<b>%{x}</b><br>7D avg: %{y:+.2f}%<extra></extra>",
    ))
    _latest_avg = avg_values[-1] if avg_values else 0
    _rolling_avg = hist_df["rolling_avg_7d"].iloc[-1] if not hist_df.empty else 0
    avg_note = (
        f"Latest: {_latest_avg:+.2f}% · 7D rolling avg: {_rolling_avg:+.2f}%"
    )
    fig.update_layout(**base(360),
        title=ttl("Average Daily Market % Change", avg_note),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TX, size=11),
                    orientation="h", x=0, y=-0.13))
    fig.update_xaxes(**ax("Date"))
    fig.update_yaxes(**ax("Avg % Change", zeroline=True))
    divs["hist_avg"] = to_div(fig, "hist_avg")

    # --- 3. Most Frequent Top Movers ------------------------------------
    if not freq_movers.empty:
        fm_sorted = freq_movers.sort_values("Appearances", ascending=True)
        _max_days  = n_days_total
        fig = go.Figure(go.Bar(
            x=fm_sorted["Appearances"].tolist(),
            y=fm_sorted["Symbol"].tolist(),
            orientation="h",
            marker=dict(
                color=fm_sorted["Appearances"].tolist(),
                colorscale=[[0, "#1a2035"], [0.5, AC], [1, G]],
                line=dict(color="rgba(0,180,255,0.2)", width=1),
            ),
            text=[f"{v}d / {_max_days}d" for v in fm_sorted["Appearances"].tolist()],
            textposition="outside", textfont=dict(color=AC, size=11),
            hovertemplate="<b>%{y}</b><br>Top-10 gainer appearances: %{x} days<extra></extra>",
        ))
        fig.update_layout(**base(360),
            title=ttl("Most Frequent Top Movers",
                      f"Symbols appearing most in daily top-10 gainers across {_max_days} scraped days"))
        fig.update_xaxes(**ax("Appearances in Top 10"), range=[0, _max_days + 1])
        fig.update_yaxes(showgrid=False,
                         tickfont=dict(color=AC, size=12, family="JetBrains Mono"),
                         linecolor=BD)
        divs["hist_movers"] = to_div(fig, "hist_movers")
    else:
        divs["hist_movers"] = (
            "<p style='color:#5A6A90;padding:20px;font-family:monospace'>"
            "Insufficient data for frequent mover analysis. Requires multiple scraped days."
            "</p>"
        )

    # --- 4. Volatility Over Time ----------------------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=date_strs, y=hist_df["volatility"].tolist(),
        mode="lines+markers", name="Daily Std Dev",
        line=dict(color=OR, width=2),
        marker=dict(
            color=[G if v <= baseline_vol else (L if v > baseline_vol * ROLLING.VOL_ELEVATED else FL)
                   for v in hist_df["volatility"].tolist()],
            size=7,
        ),
        hovertemplate="<b>%{x}</b><br>Volatility: %{y:.4f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=date_strs, y=hist_df["rolling_vol_7d"].tolist(),
        mode="lines", name="7-Day Rolling Avg",
        line=dict(color=PU, width=2, dash="dash"),
        hovertemplate="<b>%{x}</b><br>7D avg: %{y:.4f}%<extra></extra>",
    ))
    if not np.isnan(baseline_vol) and baseline_vol > 0:
        fig.add_hline(y=baseline_vol, line_dash="dot", line_color=MU, line_width=1,
            annotation=dict(text=f"Historical avg: {baseline_vol:.2f}%",
                            font_color=MU, font_size=10, xanchor="left"))
        fig.add_hline(y=baseline_vol * ROLLING.VOL_ELEVATED, line_dash="dot", line_color=L, line_width=1,
            annotation=dict(text="Elevated threshold (1.5x avg)",
                            font_color=L, font_size=10, xanchor="left"))
    fig.update_layout(**base(380),
        title=ttl("Volatility Over Time",
                  f"Latest: {latest_vol:.2f}% std dev · Regime: {vol_label}"),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TX, size=11),
                    orientation="h", x=0, y=-0.13))
    fig.update_xaxes(**ax("Date"))
    fig.update_yaxes(**ax("% Change Std Dev"))
    divs["hist_vol"] = to_div(fig, "hist_vol")

    # --- 5. Historical Breadth Heatmap ---------------------------------
    # Rows = metric (Breadth %, Avg Change, Volatility); Columns = dates.
    # Values normalised per row for colour clarity.
    if has_history:
        hm_labels = ["Breadth %", "Avg % Chg", "Volatility"]
        hm_data = [
            hist_df["breadth_pct"].tolist(),
            hist_df["avg_pct"].tolist(),
            hist_df["volatility"].tolist(),
        ]
        fig = go.Figure(go.Heatmap(
            z=hm_data,
            x=date_strs,
            y=hm_labels,
            colorscale=[[0, L], [0.5, "#1A2035"], [1, G]],
            hovertemplate="<b>%{y}</b> on %{x}<br>Value: %{z:.2f}<extra></extra>",
            showscale=True,
            colorbar=dict(tickfont=dict(color=MU, size=10), thickness=14,
                          tickcolor=MU, outlinecolor=BD),
        ))
        fig.update_layout(**base(280),
            title=ttl("Historical Market Metrics Heatmap",
                      "Green = positive / high breadth · Red = negative / high loss"))
        fig.update_xaxes(showgrid=False, tickfont=dict(color=MU, size=10), linecolor=BD)
        fig.update_yaxes(showgrid=False, tickfont=dict(color=TX, size=12), linecolor=BD)
        divs["hist_heatmap"] = to_div(fig, "hist_heatmap")
    else:
        divs["hist_heatmap"] = (
            "<p style='color:#5A6A90;padding:20px;font-family:monospace'>"
            "Heatmap requires at least 2 days of data."
            "</p>"
        )

    # --- 6. Sentiment vs Market Performance dual-axis -------------------
    # daily_sent already computed above — just join against hist_df here.
    if has_history and not daily_sent.empty:
        # hist_df dates are Timestamps — normalise to string for the join
        hist_df_str = hist_df.copy()
        hist_df_str["Date"] = hist_df_str["Date"].dt.date.astype(str)
        joined = hist_df_str.merge(daily_sent[["Date", "avg_score"]], on="Date", how="inner")

        if not joined.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=joined["Date"].tolist(),
                y=joined["avg_score"].tolist(),
                mode="lines+markers", name="Avg Sentiment Score",
                line=dict(color=PU, width=2),
                marker=dict(color=PU, size=6),
                yaxis="y1",
                hovertemplate="<b>%{x}</b><br>Sentiment: %{y:+.4f}<extra></extra>",
            ))
            fig.add_trace(go.Bar(
                x=joined["Date"].tolist(),
                y=joined["avg_pct"].tolist(),
                name="Avg % Change",
                marker=dict(
                    color=[G if v >= 0 else L for v in joined["avg_pct"].tolist()],
                    opacity=0.55,
                ),
                yaxis="y2",
                hovertemplate="<b>%{x}</b><br>Avg % Chg: %{y:+.2f}%<extra></extra>",
            ))
            fig.update_layout(**base(400),
                title=ttl("Sentiment vs Market Performance",
                          "Dual-axis: sentiment score (line) vs avg daily % change (bars) — real data"),
                showlegend=True,
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TX, size=11),
                            orientation="h", x=0, y=-0.13),
                yaxis=dict(title_text="Avg Sentiment Score", **ax(""),
                           side="left", range=[-1, 1]),
                yaxis2=dict(title_text="Avg % Change", **ax(""),
                            overlaying="y", side="right"),
            )
            divs["hist_sent_perf"] = to_div(fig, "hist_sent_perf")
        else:
            divs["hist_sent_perf"] = (
                "<p style='color:#5A6A90;padding:20px;font-family:monospace'>"
                "No overlapping dates between price and sentiment history yet. "
                "Run the dashboard again tomorrow to populate this chart."
                "</p>"
            )
    else:
        divs["hist_sent_perf"] = (
            "<p style='color:#5A6A90;padding:20px;font-family:monospace'>"
            "Chart populates with 2+ days of scraped and sentiment data."
            "</p>"
        )

    # --- 7. Cumulative Stock Performance --------------------------------
    if not cumulative_df.empty and len(cumulative_df) > 1:
        date_col = cumulative_df["Date"].astype(str).tolist()
        sym_cols = [c for c in cumulative_df.columns if c != "Date"]
        fig = go.Figure()
        for sym in sym_cols:
            final_val = cumulative_df[sym].iloc[-1]
            color     = G if final_val >= 0 else L
            fig.add_trace(go.Scatter(
                x=date_col,
                y=cumulative_df[sym].round(2).tolist(),
                mode="lines",
                name=sym,
                line=dict(width=1.8, color=color),
                opacity=0.82,
                hovertemplate=f"<b>{sym}</b><br>%{{x}}<br>Cumulative: %{{y:+.2f}}%<extra></extra>",
            ))
        fig.add_hline(y=0, line_dash="dot", line_color=MU, line_width=1)
        fig.update_layout(**base(420),
            title=ttl("Cumulative Stock Performance",
                      "Compounded daily returns from first day in dataset — green=net gain, red=net loss"),
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MU, size=10),
                        orientation="v", x=1.01, y=1))
        fig.update_xaxes(**ax("Date"))
        fig.update_yaxes(**ax("Cumulative % Return", zeroline=True))
        divs["hist_cumulative"] = to_div(fig, "hist_cumulative")
    else:
        divs["hist_cumulative"] = (
            "<p style='color:#5A6A90;padding:20px;font-family:monospace'>"
            "Cumulative performance chart requires at least 2 days of data."
            "</p>"
        )

    # --- 8. Strongest / Weakest Sessions --------------------------------
    if not sessions_df.empty:
        sess_sorted = sessions_df.sort_values("avg_pct")
        sess_colors = [G if d == "Bullish" else L for d in sess_sorted["direction"].tolist()]
        date_labels = pd.to_datetime(sess_sorted["Date"]).dt.strftime("%b %d").tolist()
        cd = sess_sorted[["best_stock", "best_pct", "worst_stock", "worst_pct"]].values.tolist()
        fig = go.Figure(go.Bar(
            x=date_labels,
            y=sess_sorted["avg_pct"].tolist(),
            marker=dict(color=sess_colors, opacity=0.88,
                        line=dict(color="rgba(255,255,255,0.04)", width=1)),
            text=[f"{'+' if v >= 0 else ''}{v:.2f}%" for v in sess_sorted["avg_pct"].tolist()],
            textposition="outside", textfont=dict(color=TX, size=10),
            customdata=cd,
            hovertemplate=(
                "<b>%{x}</b><br>Avg move: %{y:+.2f}%<br>"
                "Best: %{customdata[0]} %{customdata[1]:+.1f}%<br>"
                "Worst: %{customdata[2]} %{customdata[3]:+.1f}%<extra></extra>"
            ),
        ))
        fig.update_layout(**base(360),
            title=ttl("Strongest and Weakest Sessions",
                      f"Top {len(sessions_df)//2} best and worst trading days by average % change"))
        fig.update_xaxes(showgrid=False,
                         tickfont=dict(color=MU, size=11, family="JetBrains Mono"), linecolor=BD)
        fig.update_yaxes(**ax("Avg % Change", zeroline=True))
        divs["hist_sessions"] = to_div(fig, "hist_sessions")
    else:
        divs["hist_sessions"] = (
            "<p style='color:#5A6A90;padding:20px;font-family:monospace'>"
            "Sessions chart populates with 2+ days of data."
            "</p>"
        )

    # --- 9. 30-Day Rolling Metrics (avg change + volatility) ------------
    if has_history and "rolling_avg_30d" in hist_df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=date_strs, y=hist_df["rolling_avg_30d"].tolist(),
            mode="lines", name="30D Avg Change",
            line=dict(color=G, width=2),
            hovertemplate="<b>%{x}</b><br>30D avg: %{y:+.2f}%<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=date_strs, y=hist_df["rolling_vol_30d"].tolist(),
            mode="lines", name="30D Avg Volatility",
            line=dict(color=OR, width=2, dash="dash"),
            yaxis="y2",
            hovertemplate="<b>%{x}</b><br>30D vol: %{y:.2f}%<extra></extra>",
        ))
        fig.update_layout(**base(380),
            title=ttl("30-Day Rolling Market Metrics",
                      "Green = rolling avg daily change · Orange = rolling volatility (right axis)"),
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TX, size=11),
                        orientation="h", x=0, y=-0.13),
            yaxis=dict(title_text="Avg % Change", **ax(""), side="left"),
            yaxis2=dict(title_text="Volatility (Std Dev)", **ax(""),
                        overlaying="y", side="right"),
        )
        divs["hist_rolling30"] = to_div(fig, "hist_rolling30")
    else:
        divs["hist_rolling30"] = (
            "<p style='color:#5A6A90;padding:20px;font-family:monospace'>"
            "30-day rolling chart requires at least 2 days of data."
            "</p>"
        )

    # --- 10. Consecutive Streak Table -----------------------------------
    if not streaks_df.empty:
        streak_rows_html = ""
        for _, row in streaks_df.head(10).iterrows():
            color     = G if row["direction"] == "Bullish" else L
            start     = str(row["start_date"])[:10]
            end       = str(row["end_date"])[:10]
            streak_rows_html += (
                f'<div class="news-row">'
                f'<div class="news-sym" style="color:{color}">{row["direction"]}</div>'
                f'<div class="news-hl">{start} — {end}</div>'
                f'<div class="news-score" style="color:{color}">{int(row["length"])}d</div>'
                f'<div class="news-label" style="color:{color}">{row["avg_pct"]:+.2f}%</div>'
                f'<div class="news-n">avg/day</div>'
                f'</div>'
            )
        divs["hist_streaks"] = (
            f'<div class="news-panel" style="margin-bottom:0">'
            f'<div class="news-panel-title">Direction · Date Range · Length · Avg Daily Change</div>'
            f'{streak_rows_html}'
            f'</div>'
        )
    else:
        divs["hist_streaks"] = (
            "<p style='color:#5A6A90;padding:20px;font-family:monospace'>"
            "Streak analysis populates with 2+ days of data."
            "</p>"
        )

    # --- 11. 2-Year Cumulative Price Return from OHLCV -------------------
    _avail_ohlcv = data_loader.list_available_ohlcv()
    if _avail_ohlcv:
        # Rank symbols by today's volume; fall back to alphabetical
        if "Volume" in df.columns and "Symbol" in df.columns:
            _ranked = (
                df[df["Symbol"].isin(_avail_ohlcv)]
                .sort_values("Volume", ascending=False)["Symbol"]
                .tolist()
            )
            _plot_syms = (_ranked or _avail_ohlcv)[:15]
        else:
            _plot_syms = _avail_ohlcv[:15]

        _cum_fig = go.Figure()
        _loaded = 0
        for _sym in _plot_syms:
            _ohlcv = data_loader.load_ohlcv(_sym)
            if _ohlcv.empty or "Close" not in _ohlcv.columns:
                continue
            _ohlcv = _ohlcv.sort_values("Date").dropna(subset=["Close"])
            if len(_ohlcv) < 20:
                continue
            _dates = [str(d.date()) if hasattr(d, "date") else str(d)
                      for d in _ohlcv["Date"]]
            _cum_ret = ((_ohlcv["Close"] / _ohlcv["Close"].iloc[0]) - 1) * 100
            _cum_fig.add_trace(go.Scatter(
                x=_dates, y=_cum_ret.round(2).tolist(),
                mode="lines", name=_sym, line=dict(width=1.5),
                hovertemplate=f"<b>{_sym}</b>  %{{x}}<br>Return: %{{y:+.2f}}%<extra></extra>",
            ))
            _loaded += 1

        if _loaded:
            _cum_fig.add_hline(y=0, line_dash="dot", line_color=MU, line_width=1)
            _cum_fig.update_layout(**base(460),
                title=ttl("2-Year Cumulative Price Return",
                          f"Normalised to 0% at first bar · {_loaded} symbols · from price history"),
                showlegend=True,
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TX, size=10),
                            orientation="v", x=1.01, y=1, xanchor="left"),
            )
            _cum_fig.update_xaxes(**ax("Date"))
            _cum_fig.update_yaxes(**ax("Cumulative Return (%)", zeroline=True))
            divs["hist_ohlcv_perf"] = to_div(_cum_fig, "hist_ohlcv_perf")
        else:
            divs["hist_ohlcv_perf"] = (
                "<p style='color:#5A6A90;padding:20px;font-family:monospace'>"
                "OHLCV files found but no usable price data loaded."
                "</p>"
            )
    else:
        divs["hist_ohlcv_perf"] = (
            "<p style='color:#5A6A90;padding:20px;font-family:monospace'>"
            "No price history yet — run the pipeline to fetch 2-year price data."
            "</p>"
        )

    # -- Serialize per-symbol features for JS deep-dive charts -----------
    import json as _json
    _feat_cols = ["Date", "Close", "sma_20", "sma_50", "rsi_14",
                  "macd", "macd_signal", "macd_hist", "bb_upper", "bb_lower"]
    _sym_chart_data: dict = {}
    for _sym in (_avail_ohlcv or []):
        _feat = data_loader.load_features(_sym)
        if _feat.empty or "Close" not in _feat.columns:
            continue
        _feat = _feat.sort_values("Date").dropna(subset=["Close"])
        if len(_feat) < 20:
            continue
        _sub = _feat[[c for c in _feat_cols if c in _feat.columns]].copy()
        _sub["Date"] = _sub["Date"].dt.strftime("%Y-%m-%d")
        _raw = _sub.to_dict(orient="list")
        _sym_chart_data[_sym] = {
            col: [None if isinstance(v, float) and v != v else v for v in vals]
            for col, vals in _raw.items()
        }
    _sym_chart_json  = _json.dumps(_sym_chart_data)
    _sym_list        = sorted(_sym_chart_data.keys())
    _sym_opts_html   = "".join(f'<option value="{s}">{s}</option>' for s in _sym_list)
    _sym_sel_style   = (
        "background:#0d1829;color:#c9d1d9;border:1px solid #1e2d3d;"
        "padding:6px 14px;border-radius:6px;font-family:JetBrains Mono;"
        "font-size:13px;cursor:pointer;min-width:170px"
    )

    # Phase 2 component rendering: dashboard.py coordinates cached data,
    # while tab modules own chart/table HTML for maintainability.
    tech_divs, technical_render_df = build_technical_divs(technical_df, hist_df, theme)
    divs.update(tech_divs)
    divs.update(build_forecast_divs(forecast_df, theme, hist_df=extended_hist_df, sent_daily=daily_sent))
    divs.update(build_signal_divs(signal_df, theme))
    divs.update(build_anomaly_divs(anomaly_df, theme))

    # ── Time Series Charts ────────────────────────────────────────────────
    _ts_dates    = _ts_hist["Date"].tolist() if not _ts_hist.empty else []
    _ts_avg      = _ts_hist["avg_pct"].tolist() if not _ts_hist.empty else []
    _no_ts_html  = f"<p style='color:{MU};padding:24px'>Not enough history yet — run the pipeline for at least 7 days.</p>"

    # Chart 1: Trend Decomposition — original bars + trend + smoothed lines
    if not hist_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=_ts_dates, y=_ts_avg, name="Avg % Change",
            marker_color=[G if v >= 0 else L for v in _ts_avg], opacity=0.5,
        ))
        fig.add_trace(go.Scatter(
            x=_ts_dates, y=ts_decomp["trend"].tolist(), mode="lines",
            name="Trend (7d centred)", line=dict(color=AC, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=_ts_dates, y=ts_decomp["smoothed"].tolist(), mode="lines",
            name="Smoothed (7d trailing)", line=dict(color=PU, width=1.5, dash="dot"),
        ))
        fig.update_layout(**base(390), title=ttl("Trend Decomposition",
            "Centred rolling mean extracts the underlying trend from daily noise"))
        fig.update_xaxes(**ax("Date"))
        fig.update_yaxes(**ax("Avg % Change", zeroline=True))
        divs["ts_decomp"] = to_div(fig, "ts_decomp")
    else:
        divs["ts_decomp"] = _no_ts_html

    # Chart 2: Residual — deviation from trend
    if not hist_df.empty:
        _resid = ts_decomp["residual"].tolist()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=_ts_dates, y=_resid, name="Residual",
            marker_color=[G if v >= 0 else L for v in _resid],
        ))
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.12)")
        fig.update_layout(**base(300), title=ttl("Residual (Detrended)",
            "Daily % change minus the 7-day trend — pure noise / mean-reversion component"))
        fig.update_xaxes(**ax("Date"))
        fig.update_yaxes(**ax("Residual %", zeroline=True))
        divs["ts_residual"] = to_div(fig, "ts_residual")
    else:
        divs["ts_residual"] = _no_ts_html

    # Chart 3: MA Crossover Signals
    if not hist_df.empty and not ts_trend.empty:
        _short_ma = ts_trend["short_ma"].tolist()
        _long_ma  = ts_trend["long_ma"].tolist()
        _td       = ts_trend["Date"].tolist()
        _bull_m   = ts_trend[ts_trend["crossover"] == "Bullish"]
        _bear_m   = ts_trend[ts_trend["crossover"] == "Bearish"]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=_ts_dates, y=_ts_avg, name="Avg % Change",
            marker_color=[G if v >= 0 else L for v in _ts_avg], opacity=0.35,
        ))
        fig.add_trace(go.Scatter(
            x=_td, y=_short_ma, mode="lines",
            name=f"Short MA ({ROLLING.SHORT_WINDOW}d)", line=dict(color=AC, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=_td, y=_long_ma, mode="lines",
            name=f"Long MA ({ROLLING.LONG_WINDOW}d)", line=dict(color=OR, width=2),
        ))
        if not _bull_m.empty:
            fig.add_trace(go.Scatter(
                x=_bull_m["Date"].tolist(), y=_bull_m["short_ma"].tolist(),
                mode="markers", name="Bullish Cross",
                marker=dict(color=G, size=12, symbol="triangle-up"),
            ))
        if not _bear_m.empty:
            fig.add_trace(go.Scatter(
                x=_bear_m["Date"].tolist(), y=_bear_m["short_ma"].tolist(),
                mode="markers", name="Bearish Cross",
                marker=dict(color=L, size=12, symbol="triangle-down"),
            ))
        fig.update_layout(**base(410), title=ttl("MA Crossover Signals",
            f"{ROLLING.SHORT_WINDOW}d vs {ROLLING.LONG_WINDOW}d — triangles mark crossover events"))
        fig.update_xaxes(**ax("Date"))
        fig.update_yaxes(**ax("Avg % Change", zeroline=True))
        divs["ts_trend"] = to_div(fig, "ts_trend")
    else:
        divs["ts_trend"] = _no_ts_html

    # Chart 4: Breadth Trend — SMA/EMA + net breadth bars
    if not ts_breadth.empty:
        _bt_dates = ts_breadth["Date"].tolist()
        _bt_net   = ts_breadth["net_breadth"].tolist()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=_bt_dates, y=_bt_net, name="Net Breadth (G − L)",
            marker_color=[G if v >= 0 else L for v in _bt_net],
            opacity=0.45, yaxis="y2",
        ))
        fig.add_trace(go.Scatter(
            x=_bt_dates, y=ts_breadth["breadth_sma_7d"].tolist(),
            mode="lines", name="Breadth SMA 7d", line=dict(color=AC, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=_bt_dates, y=ts_breadth["breadth_ema_7d"].tolist(),
            mode="lines", name="Breadth EMA 7d",
            line=dict(color=PU, width=1.5, dash="dot"),
        ))
        fig.add_hline(y=50, line_color="rgba(255,255,255,0.1)", line_dash="dot")
        fig.update_layout(
            **base(390),
            title=ttl("Breadth Trend", "% of advancing stocks · SMA and EMA smoothing · net gainers/losers bars"),
            yaxis2=dict(
                overlaying="y", side="right", showgrid=False,
                tickfont=dict(color=MU, size=11),
                title_text="Net Breadth",
                title_font=dict(color=MU, size=12),
            ),
        )
        fig.update_xaxes(**ax("Date"))
        fig.update_yaxes(**ax("Breadth %"))
        divs["ts_breadth_trend"] = to_div(fig, "ts_breadth_trend")
    else:
        divs["ts_breadth_trend"] = _no_ts_html

    # Chart 5: Market Cycles — bull/bear run durations
    if not ts_cycles.empty:
        _cyc_colors = [G if d == "Bull" else L for d in ts_cycles["direction"].tolist()]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=ts_cycles["start_date"].astype(str).tolist(),
            y=ts_cycles["length_days"].tolist(),
            name="Cycle Length (days)",
            marker_color=_cyc_colors,
            customdata=ts_cycles[["direction","avg_return","peak_return","end_date"]].values.tolist(),
            hovertemplate=(
                "<b>%{customdata[0]}</b> cycle<br>"
                "Start: %{x} → End: %{customdata[3]}<br>"
                "Duration: %{y} days<br>"
                "Avg Return: %{customdata[1]:+.3f}%<br>"
                "Peak Return: %{customdata[2]:+.3f}%<extra></extra>"
            ),
        ))
        fig.update_layout(**base(360), title=ttl("Market Cycles — Bull & Bear Runs",
            "Each bar = one contiguous run where the 7-day rolling avg stays positive (bull) or negative (bear)"),
            bargap=0.18)
        fig.update_xaxes(**ax("Cycle Start"))
        fig.update_yaxes(**ax("Duration (days)"))
        divs["ts_cycles"] = to_div(fig, "ts_cycles")
    else:
        divs["ts_cycles"] = _no_ts_html

    # Chart 6: Rolling Statistics — mean ± std band
    if not hist_df.empty and not ts_stats.empty:
        _rs_mean = ts_stats["mean"].tolist()
        _rs_std  = ts_stats["std"].fillna(0).tolist()
        _upper   = [m + s for m, s in zip(_rs_mean, _rs_std)]
        _lower   = [m - s for m, s in zip(_rs_mean, _rs_std)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=_ts_dates + _ts_dates[::-1], y=_upper + _lower[::-1],
            fill="toself", fillcolor="rgba(0,180,255,0.07)",
            line=dict(color="rgba(0,0,0,0)"), name="±1 Std Dev",
        ))
        fig.add_trace(go.Scatter(
            x=_ts_dates, y=_rs_mean, mode="lines", name="Rolling Mean (7d)",
            line=dict(color=AC, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=_ts_dates, y=_ts_avg, mode="lines", name="Daily Avg % Change",
            line=dict(color=FL, width=1, dash="dot"), opacity=0.65,
        ))
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.1)")
        fig.update_layout(**base(380), title=ttl("Rolling Statistics (7d)",
            "7-day rolling mean with ±1 standard deviation confidence band"))
        fig.update_xaxes(**ax("Date"))
        fig.update_yaxes(**ax("Avg % Change", zeroline=True))
        divs["ts_rolling_stats"] = to_div(fig, "ts_rolling_stats")
    else:
        divs["ts_rolling_stats"] = _no_ts_html

    forecast_dict = forecast_dict_from_cache(forecast_df)
    signals_dict = signals_dict_from_cache(signal_df)
    anomalies_dict = anomalies_dict_from_cache(anomaly_df)
    sent_summary = {"mean_score": mean_sent, "pct_bullish": pct_bull, "pct_bearish": pct_bear}
    briefing = ai_narratives.generate_daily_briefing(
        df, hist_df, sent_summary, daily_sent, signals_dict, anomalies_dict, forecast_dict,
        N_GAIN, N_LOSS, N_FLAT, AVG_CHG, TOP_SYM, TOP_PCT, BOT_SYM, BOT_PCT,
        latest_vol, baseline_vol,
    )
    phase2_technical_html = render_technical_tab(
        technical_render_df, hist_df, divs, theme, _sym_opts_html, _sym_sel_style
    )
    phase2_forecast_html = render_forecast_tab(divs, briefing)
    phase2_signal_html = render_signal_tab(signal_df, divs, theme)
    phase2_briefing_html = render_ai_briefing_tab(briefing)
    phase2_anomaly_html = render_anomaly_tab(anomaly_df, divs, theme)

    # ══════════════════════════════════════════════════════════════════════
    # HTML MINI-ROW HELPERS (unchanged from v1)
    # ══════════════════════════════════════════════════════════════════════

    def mover_rows(frame, color):
        rows = []
        for _, row in frame.iterrows():
            rows.append(
                f'<div class="mini-row">'
                f'<div><b>{html_utils.escape(str(row["Symbol"]))}</b>'
                f'<span>{html_utils.escape(str(row["Company Name"]))}</span></div>'
                f'<strong style="color:{color}">{row["% Change"]:+.2f}%</strong>'
                f'</div>'
            )
        return "".join(rows)


    def sentiment_mini_rows(frame, color):
        """Build mini-row HTML for sentiment top bullish/bearish panels."""
        rows = []
        for _, row in frame.iterrows():
            headline = html_utils.escape(str(row["Top_Headline"])[:80]) + ("..." if len(str(row["Top_Headline"])) > 80 else "")
            rows.append(
                f'<div class="mini-row">'
                f'<div><b style="color:{color}">{html_utils.escape(str(row["Symbol"]))}</b>'
                f'<span>{headline}</span></div>'
                f'<strong style="color:{color}">{row["Sentiment_Score"]:+.4f}</strong>'
                f'</div>'
            )
        return "".join(rows)


    top_mini_html   = mover_rows(df.nlargest(5, "% Change"), G)
    bot_mini_html   = mover_rows(df.nsmallest(5, "% Change"), L)
    bull_mini_html  = sentiment_mini_rows(top_bull_df, G)  if not top_bull_df.empty  else "<p style='color:#5A6A90'>No bullish signals</p>"
    bear_mini_html  = sentiment_mini_rows(top_bear_df, L)  if not top_bear_df.empty  else "<p style='color:#5A6A90'>No bearish signals</p>"
    stock_options   = "".join(
        f'<option value="{html_utils.escape(str(s))}"></option>'
        for s in df["Symbol"].tolist()
    )

    logger.info("All %d divs built. Assembling HTML...", len(divs))

    # ══════════════════════════════════════════════════════════════════════
    # HTML ASSEMBLY
    # ══════════════════════════════════════════════════════════════════════

    TABS = [
        ("overview",   "Overview"),
        ("movers",     "Top Movers"),
        ("analytics",  "Analytics"),
        ("screener",   "Screener"),
        ("story",      "Story"),
        ("sentiment",  "Sentiment"),
        ("historical", "Historical"),
        ("timeseries", "Time Series"),
        ("technical",  "Technical Intelligence"),
        ("forecasting","Forecasting"),
        ("signals",    "Signal Intelligence"),
        ("briefing",   "AI Briefing"),
        ("risk",       "Risk / Anomaly"),
    ]

    OUT = PATHS.DASHBOARD_HTML

    CSS = f"""
    *{{margin:0;padding:0;box-sizing:border-box}}
    html{{scroll-behavior:smooth}}
    body{{background:radial-gradient(circle at 15% -10%,rgba(0,180,255,0.13),transparent 34%),
         radial-gradient(circle at 85% 0%,rgba(0,255,178,0.11),transparent 32%),{BG};
         color:{TX};font-family:'Space Grotesk',sans-serif;min-height:100vh}}
    body::before{{content:'';position:fixed;inset:0;pointer-events:none;
      background:linear-gradient(rgba(255,255,255,0.025) 1px,transparent 1px),
                 linear-gradient(90deg,rgba(255,255,255,0.02) 1px,transparent 1px);
      background-size:42px 42px;
      mask-image:linear-gradient(to bottom,rgba(0,0,0,.7),transparent 70%)}}
    .hdr{{background:rgba(6,8,15,0.84);border-bottom:1px solid rgba(90,106,144,0.28);
      padding:16px 36px;display:flex;align-items:center;justify-content:space-between;
      position:sticky;top:0;z-index:100;backdrop-filter:blur(18px)}}
    .logo{{display:flex;align-items:center;gap:14px}}
    .logo-icon{{width:42px;height:42px;background:linear-gradient(135deg,{G},{AC});
      border-radius:12px;display:flex;align-items:center;justify-content:center;
      font-size:20px;color:{BG};box-shadow:0 0 26px rgba(0,255,178,0.28)}}
    .logo-name{{font-size:20px;font-weight:700;letter-spacing:-0.2px;
      background:linear-gradient(90deg,{G},{AC});
      -webkit-background-clip:text;-webkit-text-fill-color:transparent}}
    .logo-sub{{font-size:10px;color:{MU};letter-spacing:2.8px;text-transform:uppercase;
      font-family:'JetBrains Mono',monospace;margin-top:3px}}
    .hdr-r{{display:flex;align-items:center;gap:20px}}
    .hstat{{text-align:right}}
    .hval{{font-size:16px;font-weight:700;font-family:'JetBrains Mono',monospace}}
    .hlbl{{font-size:9px;color:{MU};text-transform:uppercase;letter-spacing:1.5px}}
    .live{{display:flex;align-items:center;gap:7px;background:rgba(0,255,178,0.08);
      border:1px solid rgba(0,255,178,0.28);border-radius:999px;padding:7px 16px;
      font-size:10px;font-family:'JetBrains Mono',monospace;color:{G};letter-spacing:1px}}
    .dot{{width:7px;height:7px;background:{G};border-radius:50%;animation:pulse 1.8s infinite;
      box-shadow:0 0 10px {G}}}
    @keyframes pulse{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.25;transform:scale(.65)}}}}
    .sidebar{{position:fixed;left:0;top:75px;bottom:0;width:224px;
      background:rgba(6,8,16,0.97);border-right:1px solid rgba(90,106,144,0.18);
      backdrop-filter:blur(20px);z-index:89;overflow-y:auto;overflow-x:hidden;
      transition:width .22s cubic-bezier(.4,0,.2,1);padding:10px 0 40px;display:flex;flex-direction:column}}
    .sidebar.collapsed{{width:60px}}
    .sidebar::-webkit-scrollbar{{width:0}}
    .sb-toggle{{display:flex;align-items:center;gap:10px;padding:12px 18px;cursor:pointer;
      background:none;border:none;color:{MU};width:100%;text-align:left;margin-bottom:6px;
      transition:color .18s}}
    .sb-toggle:hover{{color:{TX}}}
    .sb-toggle-icon{{display:flex;flex-direction:column;gap:4px;flex-shrink:0}}
    .sb-toggle-icon span{{display:block;width:18px;height:2px;background:currentColor;
      border-radius:2px;transition:all .22s}}
    .sb-toggle-label{{font-size:10px;font-family:'JetBrains Mono',monospace;
      letter-spacing:2px;text-transform:uppercase;white-space:nowrap;
      transition:opacity .18s,width .22s;overflow:hidden}}
    .sidebar.collapsed .sb-toggle-label{{opacity:0;width:0}}
    .sb-group{{margin-bottom:4px}}
    .sb-group-lbl{{font-size:9px;font-family:'JetBrains Mono',monospace;letter-spacing:2.2px;
      text-transform:uppercase;color:rgba(90,106,144,0.6);padding:10px 18px 4px;
      white-space:nowrap;overflow:hidden;transition:opacity .18s}}
    .sidebar.collapsed .sb-group-lbl{{opacity:0}}
    .sb-item{{display:flex;align-items:center;gap:12px;padding:10px 18px;cursor:pointer;
      border-left:2px solid transparent;transition:all .18s;color:{MU};
      white-space:nowrap;position:relative}}
    .sb-item:hover{{color:{TX};background:rgba(255,255,255,0.04);border-left-color:rgba(0,255,178,0.3)}}
    .sb-item.active{{color:{G};background:rgba(0,255,178,0.07);border-left-color:{G}}}
    .sb-item.active .sb-icon{{color:{G};text-shadow:0 0 12px rgba(0,255,178,0.5)}}
    .sb-icon{{font-size:15px;width:22px;text-align:center;flex-shrink:0;font-family:'JetBrains Mono',monospace}}
    .sb-lbl{{font-size:12px;font-weight:600;letter-spacing:.1px;
      transition:opacity .18s,width .22s;overflow:hidden}}
    .sidebar.collapsed .sb-lbl{{opacity:0;width:0}}
    .sidebar.collapsed .sb-item{{justify-content:center;padding:11px 0}}
    .sidebar.collapsed .sb-item::after{{content:attr(data-tip);position:absolute;
      left:66px;top:50%;transform:translateY(-50%);background:{SF};
      border:1px solid rgba(90,106,144,0.3);border-radius:8px;padding:5px 11px;
      font-size:11px;color:{TX};font-family:'JetBrains Mono',monospace;white-space:nowrap;
      opacity:0;pointer-events:none;transition:opacity .15s;z-index:300;
      box-shadow:0 4px 18px rgba(0,0,0,.4)}}
    .sidebar.collapsed .sb-item:hover::after{{opacity:1}}
    .sb-divider{{height:1px;background:rgba(90,106,144,0.14);margin:6px 18px}}
    .content{{padding:24px 28px 40px;margin-left:224px;transition:margin-left .22s cubic-bezier(.4,0,.2,1);min-height:100vh}}
    .content.sb-collapsed{{margin-left:60px}}
    .hero{{display:grid;grid-template-columns:1.25fr .75fr;gap:20px;margin-bottom:20px}}
    .hero-main,.hero-side,.tool-strip,.mini-panel,.card,.kpi,.callout,.news-panel{{
      background:linear-gradient(180deg,rgba(18,23,39,0.94),rgba(12,15,26,0.94));
      border:1px solid rgba(90,106,144,0.26);box-shadow:0 18px 48px rgba(0,0,0,0.23)}}
    .hero-main{{border-radius:18px;padding:26px;position:relative;overflow:hidden}}
    .hero-main::after{{content:'';position:absolute;right:-120px;top:-160px;width:340px;height:340px;
      background:radial-gradient(circle,rgba(0,255,178,0.16),transparent 62%);pointer-events:none}}
    .eyebrow{{font-family:'JetBrains Mono',monospace;color:{AC};font-size:11px;
      letter-spacing:2px;text-transform:uppercase;margin-bottom:10px}}
    .hero-title{{font-size:38px;line-height:1.02;font-weight:700;letter-spacing:-0.6px;max-width:760px}}
    .hero-copy{{color:#B9C3DE;font-size:14px;line-height:1.7;margin-top:14px;max-width:760px}}
    .hero-meta{{display:flex;flex-wrap:wrap;gap:10px;margin-top:20px}}
    .pill{{border:1px solid rgba(90,106,144,0.32);background:rgba(255,255,255,0.035);
      border-radius:999px;padding:8px 12px;font-family:'JetBrains Mono',monospace;
      font-size:11px;color:#C8D2EA}}
    .breadth-meter{{margin-top:22px}}
    .meter-label{{display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;
      font-size:11px;color:{MU};margin-bottom:8px}}
    .meter-track{{height:12px;border-radius:999px;background:rgba(255,51,102,0.25);overflow:hidden;
      border:1px solid rgba(255,255,255,0.06)}}
    .meter-fill{{height:100%;width:{advance_width:.1f}%;
      background:linear-gradient(90deg,{G},{AC});border-radius:999px}}
    .hero-side{{border-radius:18px;padding:20px;display:grid;gap:14px}}
    .signal{{display:flex;align-items:center;justify-content:space-between;
      border-bottom:1px solid rgba(90,106,144,0.2);padding-bottom:12px}}
    .signal:last-child{{border-bottom:0;padding-bottom:0}}
    .signal span{{font-size:12px;color:{MU};text-transform:uppercase;letter-spacing:1.3px;
      font-family:'JetBrains Mono',monospace}}
    .signal strong{{font-family:'JetBrains Mono',monospace;font-size:18px}}
    .tool-strip{{display:flex;align-items:center;justify-content:space-between;gap:16px;
      border-radius:16px;padding:14px 16px;margin-bottom:20px}}
    .search-wrap{{display:flex;align-items:center;gap:10px;flex:1;min-width:240px}}
    .search-wrap input{{width:100%;background:rgba(6,8,15,0.8);
      border:1px solid rgba(90,106,144,0.35);border-radius:12px;color:{TX};
      padding:11px 13px;font-family:'JetBrains Mono',monospace;font-size:12px;outline:none}}
    .search-wrap input:focus{{border-color:{AC};box-shadow:0 0 0 3px rgba(0,180,255,0.12)}}
    .actions{{display:flex;gap:10px;flex-wrap:wrap}}
    .btn{{border:1px solid rgba(90,106,144,0.35);background:rgba(255,255,255,0.045);
      color:{TX};border-radius:12px;padding:10px 13px;font-size:12px;font-weight:600;
      cursor:pointer;transition:all .2s}}
    .btn:hover{{border-color:{G};color:{G};transform:translateY(-1px)}}
    .panel{{display:none;animation:fi .28s ease}}
    .panel.active{{display:block}}
    @keyframes fi{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:translateY(0)}}}}
    .kpi-grid{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:14px;margin-bottom:20px}}
    .kpi{{border-radius:16px;padding:18px;position:relative;overflow:hidden;transition:border-color .2s,transform .2s}}
    .kpi:hover{{border-color:rgba(0,255,178,0.42);transform:translateY(-2px)}}
    .kpi::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;
      background:linear-gradient(90deg,{G},{AC});opacity:.55}}
    .kpi-lbl{{font-size:9px;color:{MU};text-transform:uppercase;letter-spacing:1.8px;
      margin-bottom:10px;font-family:'JetBrains Mono',monospace}}
    .kpi-val{{font-size:28px;font-weight:700;font-family:'JetBrains Mono',monospace;
      letter-spacing:-1px;line-height:1}}
    .kpi-sub{{font-size:11px;color:{MU};margin-top:7px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
    .row2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}
    .row1{{margin-bottom:20px}}
    .card{{border-radius:16px;overflow:hidden;transition:border-color .2s,transform .2s}}
    .card:hover{{border-color:rgba(0,255,178,0.24)}}
    .mini-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}
    .mini-panel{{border-radius:16px;padding:18px}}
    .mini-title{{font-size:12px;color:{MU};letter-spacing:1.4px;text-transform:uppercase;
      font-family:'JetBrains Mono',monospace;margin-bottom:12px}}
    .mini-row{{display:flex;align-items:center;justify-content:space-between;gap:14px;
      padding:10px 0;border-bottom:1px solid rgba(90,106,144,0.16)}}
    .mini-row:last-child{{border-bottom:0}}
    .mini-row b{{font-family:'JetBrains Mono',monospace;font-size:13px;color:{TX};display:block}}
    .mini-row span{{display:block;font-size:11px;color:{MU};max-width:380px;white-space:nowrap;
      overflow:hidden;text-overflow:ellipsis;margin-top:2px}}
    .mini-row strong{{font-family:'JetBrains Mono',monospace;font-size:13px}}
    .callouts{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px}}
    .callout{{border-radius:16px;padding:22px}}
    .callout.pos{{border-color:rgba(0,255,178,0.3);
      background:linear-gradient(135deg,rgba(0,255,178,0.08),rgba(12,15,26,0.94))}}
    .callout.neg{{border-color:rgba(255,51,102,0.3);
      background:linear-gradient(135deg,rgba(255,51,102,0.08),rgba(12,15,26,0.94))}}
    .callout.warn{{border-color:rgba(0,180,255,0.3);
      background:linear-gradient(135deg,rgba(0,180,255,0.08),rgba(12,15,26,0.94))}}
    .callout.sent{{border-color:rgba(191,95,255,0.3);
      background:linear-gradient(135deg,rgba(191,95,255,0.08),rgba(12,15,26,0.94))}}
    .ci{{font-family:'JetBrains Mono',monospace;font-size:14px;color:{MU};margin-bottom:10px}}
    .cn{{font-size:30px;font-weight:700;font-family:'JetBrains Mono',monospace;
      margin-bottom:4px;letter-spacing:-1px}}
    .ct{{font-size:13px;font-weight:700;margin-bottom:7px}}
    .cb{{font-size:12px;color:#AEB9D3;line-height:1.7}}
    .slbl{{font-size:10px;font-weight:600;letter-spacing:2px;text-transform:uppercase;
      color:{MU};margin-bottom:14px;margin-top:4px;display:flex;align-items:center;gap:10px}}
    .slbl::after{{content:'';flex:1;height:1px;background:{BD}}}
    .sent-kpi{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-bottom:20px}}
    .news-panel{{border-radius:16px;padding:18px;margin-bottom:20px}}
    .news-panel-title{{font-size:12px;color:{MU};letter-spacing:1.4px;text-transform:uppercase;
      font-family:'JetBrains Mono',monospace;margin-bottom:14px}}
    .news-row{{display:grid;grid-template-columns:60px 1fr 90px 80px 90px;
      align-items:center;gap:12px;padding:10px 0;
      border-bottom:1px solid rgba(90,106,144,0.14);font-family:'JetBrains Mono',monospace}}
    .news-row:last-child{{border-bottom:0}}
    .news-sym{{font-size:13px;font-weight:700}}
    .news-hl{{font-size:11px;color:#AEB9D3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
    .news-score{{font-size:12px;text-align:right}}
    .news-label{{font-size:11px;text-align:center}}
    .news-n{{font-size:10px;color:{MU};text-align:right}}
    .intel-row{{grid-template-columns:minmax(72px,.55fr) minmax(130px,.8fr) minmax(0,2.8fr) minmax(90px,.45fr);
      align-items:start;column-gap:18px;row-gap:6px;padding:18px 0}}
    .intel-scope,.intel-type,.intel-value{{font-size:12px;font-weight:700;line-height:1.35;
      overflow-wrap:anywhere}}
    .intel-type{{color:{TX}}}
    .intel-reason{{min-width:0;color:{TX};font-size:13px;line-height:1.45;white-space:normal;
      overflow-wrap:anywhere}}
    .intel-value{{text-align:right}}
    .toast{{position:fixed;right:24px;bottom:24px;background:{SF};
      border:1px solid rgba(0,255,178,0.32);border-radius:12px;padding:12px 14px;
      color:{G};font-family:'JetBrains Mono',monospace;font-size:12px;
      box-shadow:0 14px 38px rgba(0,0,0,.3);opacity:0;transform:translateY(10px);
      transition:all .2s;z-index:200}}
    .toast.show{{opacity:1;transform:translateY(0)}}
    ::-webkit-scrollbar{{width:6px;height:6px}}
    ::-webkit-scrollbar-track{{background:{BG}}}
    ::-webkit-scrollbar-thumb{{background:{BD};border-radius:6px}}
    @media (max-width:1180px){{
      .hero,.row2,.mini-grid,.callouts,.sent-kpi{{grid-template-columns:1fr}}
      .kpi-grid{{grid-template-columns:repeat(3,1fr)}}
      .news-row{{grid-template-columns:50px 1fr 70px}}
      .intel-row{{grid-template-columns:70px 110px minmax(0,1fr) 70px}}
      .news-label,.news-n{{display:none}}
      .sidebar{{width:60px}}.sidebar .sb-lbl,.sidebar .sb-group-lbl,.sidebar .sb-toggle-label{{opacity:0;width:0}}
      .sidebar .sb-item{{justify-content:center;padding:11px 0}}
      .content{{margin-left:60px}}
    }}
    @media (max-width:760px){{
      .hdr{{padding:14px 18px;align-items:flex-start;gap:14px;flex-direction:column}}
      .hdr-r{{width:100%;justify-content:space-between}}
      .sidebar{{width:60px;top:132px}}
      .content{{padding:18px;margin-left:60px}}
      .hero-title{{font-size:28px}}
      .kpi-grid{{grid-template-columns:1fr 1fr}}
      .tool-strip{{align-items:stretch;flex-direction:column}}
      .intel-row{{grid-template-columns:1fr;gap:6px}}
      .intel-value{{text-align:left}}
    }}
    """

    html = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Market Pulse — 100 Most Active Stocks</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;600;700&display=swap" rel="stylesheet">
    <style>{CSS}</style>
    </head>
    <body>
    <header class="hdr">
      <div class="logo">
        <div class="logo-icon">M</div>
        <div>
          <div class="logo-name">MARKET PULSE</div>
          <div class="logo-sub">100 Most Active · Yahoo Finance</div>
        </div>
      </div>
      <div class="hdr-r">
        <div class="hstat"><div class="hval" style="color:{G}">{N_GAIN} &uarr;</div><div class="hlbl">Advancing</div></div>
        <div class="hstat"><div class="hval" style="color:{L}">{N_LOSS} &darr;</div><div class="hlbl">Declining</div></div>
        <div class="hstat"><div class="hval" style="color:{PU}">{pct_bull:.0f}% Bull</div><div class="hlbl">Sentiment</div></div>
        {f'<div class="hstat"><div class="hval" style="color:{AC}">{n_ohlcv_symbols}</div><div class="hlbl">Price History</div></div>' if n_ohlcv_symbols else ""}
        <div class="live"><div class="dot"></div>SCRAPED LIVE</div>
      </div>
    </header>
    """

    SIDEBAR_GROUPS = [
        ("Today's Market", [
            ("overview",    "Overview",      "⊞"),
            ("movers",      "Top Movers",    "↕"),
            ("analytics",   "Analytics",     "≈"),
            ("screener",    "Screener",      "≡"),
            ("story",       "Story",         "¶"),
        ]),
        ("Intelligence", [
            ("sentiment",   "Sentiment",     "◉"),
            ("signals",     "Signals",       "◈"),
            ("briefing",    "AI Briefing",   "◆"),
            ("risk",        "Risk",          "⚠"),
        ]),
        ("History & Analysis", [
            ("historical",  "Historical",    "▦"),
            ("timeseries",  "Time Series",   "∿"),
            ("technical",   "Technical",     "⊕"),
            ("forecasting", "Forecasting",   "◇"),
        ]),
    ]

    html += '<nav class="sidebar" id="sidebar">\n'
    html += '''  <button class="sb-toggle" onclick="toggleSidebar()">
    <div class="sb-toggle-icon"><span></span><span></span><span></span></div>
    <span class="sb-toggle-label">COLLAPSE</span>
  </button>\n'''

    first_key = SIDEBAR_GROUPS[0][1][0][0]
    for g_idx, (group_name, items) in enumerate(SIDEBAR_GROUPS):
        html += f'  <div class="sb-group">\n'
        html += f'    <div class="sb-group-lbl">{group_name}</div>\n'
        for key, lbl, icon in items:
            act = "active" if key == first_key else ""
            html += (
                f'    <div class="sb-item {act}" onclick="showTab(\'{key}\')" '
                f'id="tb-{key}" data-tip="{lbl}">'
                f'<span class="sb-icon">{icon}</span>'
                f'<span class="sb-lbl">{lbl}</span></div>\n'
            )
        html += '  </div>\n'
        if g_idx < len(SIDEBAR_GROUPS) - 1:
            html += '  <div class="sb-divider"></div>\n'

    html += f"""</nav>
    <main class="content" id="mainContent">
      <section class="hero">
        <div class="hero-main">
          <div class="eyebrow">Daily market read — {latest_label}</div>
          <div class="hero-title">{market_takeaway}</div>
          <p class="hero-copy">
            {method_takeaway} Session shows {breadth_narrative} with {vol_label}.
            The dashboard updates after each new scrape and EDA run — all numbers,
            charts, and narrative copy refresh automatically with the latest data.
          </p>
          <div class="hero-meta">
            <div class="pill">Net breadth: {net_breadth:+d}</div>
            <div class="pill">Avg move: {AVG_CHG:+.2f}%</div>
            <div class="pill">High movers: {high_move_count}</div>
            <div class="pill">Outliers: {N_OUTLIERS}</div>
            <div class="pill">Sentiment: {sent_regime}</div>
            <div class="pill">Days tracked: {n_days_total}</div>
            {f'<div class="pill" style="display:{"block" if n_ohlcv_symbols else "none"}">OHLCV: {n_ohlcv_symbols} symbols</div>'}
          </div>
          <div class="breadth-meter">
            <div class="meter-label">
              <span>Decliners {decline_pct:.0f}%</span>
              <span>Advancers {breadth_pct:.0f}%</span>
            </div>
            <div class="meter-track"><div class="meter-fill"></div></div>
          </div>
        </div>
        <aside class="hero-side">
          <div class="signal"><span>Top gainer</span><strong style="color:{G}">{TOP_SYM} {TOP_PCT:+.1f}%</strong></div>
          <div class="signal"><span>Top loser</span><strong style="color:{L}">{BOT_SYM} {BOT_PCT:+.1f}%</strong></div>
          <div class="signal"><span>Spearman rho</span><strong style="color:{AC}">{SP_CORR:+.3f}</strong></div>
          <div class="signal"><span>Price range</span><strong>${PRICE_MIN:.2f} — ${PRICE_MAX:.2f}</strong></div>
          <div class="signal"><span>Sent. regime</span><strong style="color:{PU}">{sent_regime.split(' — ')[0]}</strong></div>
          <div class="signal"><span>Volatility</span><strong style="color:{OR}">{vol_label.split(' ')[0].title()}</strong></div>
        </aside>
      </section>

      <section class="tool-strip">
        <label class="search-wrap" for="tickerSearch">
          <span>Find ticker</span>
          <input id="tickerSearch" list="stockSymbols"
                 placeholder="Type a symbol, then press Enter" autocomplete="off">
          <datalist id="stockSymbols">{stock_options}</datalist>
        </label>
        <div class="actions">
          <button class="btn" onclick="showTab('screener')">Open screener</button>
          <button class="btn" onclick="showTab('sentiment')">Sentiment Intel</button>
          <button class="btn" onclick="showTab('historical')">Historical View</button>
          <button class="btn" onclick="showTab('story')">Read analysis</button>
          <button class="btn" onclick="window.print()">Print / PDF</button>
        </div>
      </section>

    <!-- ═══ OVERVIEW ═══════════════════════════════════════════════════ -->
    <div class="panel active" id="panel-overview">
      <div class="kpi-grid">
        <div class="kpi"><div class="kpi-lbl">Avg Stock Price</div><div class="kpi-val">${PRICE_MEAN:.2f}</div><div class="kpi-sub">Median: ${PRICE_MEDIAN:.2f}</div></div>
        <div class="kpi"><div class="kpi-lbl">Advancing Today</div><div class="kpi-val" style="color:{G}">{N_GAIN} &uarr;</div><div class="kpi-sub">{breadth_pct:.0f}% — {breadth_narrative}</div></div>
        <div class="kpi"><div class="kpi-lbl">Declining Today</div><div class="kpi-val" style="color:{L}">{N_LOSS} &darr;</div><div class="kpi-sub">{session_label} session</div></div>
        <div class="kpi"><div class="kpi-lbl">Top Gainer</div><div class="kpi-val" style="color:{G}">{TOP_PCT:+.1f}%</div><div class="kpi-sub">{TOP_SYM} — {html_utils.escape(TOP_NAME)}</div></div>
        <div class="kpi"><div class="kpi-lbl">Top Loser</div><div class="kpi-val" style="color:{L}">{BOT_PCT:+.1f}%</div><div class="kpi-sub">{BOT_SYM} — {html_utils.escape(BOT_NAME)}</div></div>
        <div class="kpi"><div class="kpi-lbl">Price Skewness</div><div class="kpi-val" style="color:{FL}">{PRICE_SKEW:+.2f}</div><div class="kpi-sub">{skew_label}</div></div>
      </div>
      <div class="mini-grid">
        <div class="mini-panel"><div class="mini-title">Top 5 gainers</div>{top_mini_html}</div>
        <div class="mini-panel"><div class="mini-title">Top 5 decliners</div>{bot_mini_html}</div>
      </div>
      <div class="row2">
        <div class="card">{divs["ov_scatter"]}</div>
        <div class="card">{divs["ov_donut"]}</div>
      </div>
      <div class="row1"><div class="card">{divs["ov_hist"]}</div></div>
    </div>

    <!-- ═══ TOP MOVERS ══════════════════════════════════════════════════ -->
    <div class="panel" id="panel-movers">
      <div class="row2">
        <div class="card">{divs["mv_gainers"]}</div>
        <div class="card">{divs["mv_losers"]}</div>
      </div>
      <div class="row1"><div class="card">{divs["mv_dollar"]}</div></div>
    </div>

    <!-- ═══ ANALYTICS ═══════════════════════════════════════════════════ -->
    <div class="panel" id="panel-analytics">
      <div class="row2">
        <div class="card">{divs["an_bucket"]}</div>
        <div class="card">{divs["an_box"]}</div>
      </div>
      <div class="row2">
        <div class="card">{divs["an_corr"]}</div>
        <div class="card">{divs["an_violin"]}</div>
      </div>
    </div>

    <!-- ═══ SCREENER ════════════════════════════════════════════════════ -->
    <div class="panel" id="panel-screener">
      <div class="card">{divs["screener"]}</div>
    </div>

    <!-- ═══ STORY ═══════════════════════════════════════════════════════ -->
    <div class="panel" id="panel-story">
      <div class="callouts">
        <div class="callout pos">
          <div class="ci">BREADTH</div>
          <div class="cn" style="color:{G}">{breadth_pct:.0f}%</div>
          <div class="ct">{session_label} Breadth</div>
          <div class="cb">
            {N_GAIN} of {N_TOTAL} most active stocks closed higher.
            The session {session_sentence} based on the latest scrape.
            Breadth classification: {breadth_narrative}.
          </div>
        </div>
        <div class="callout neg">
          <div class="ci">WORST MOVE</div>
          <div class="cn" style="color:{L}">{BOT_PCT:+.1f}%</div>
          <div class="ct">{BOT_SYM} Biggest Decliner</div>
          <div class="cb">
            {html_utils.escape(BOT_NAME)} had the sharpest percentage decline
            in the latest dataset. Volatility span: {volatility_span:.1f}pp
            peak-to-trough.
          </div>
        </div>
        <div class="callout warn">
          <div class="ci">CORRELATION</div>
          <div class="cn" style="color:{AC}">rho={SP_CORR:.3f}</div>
          <div class="ct">Price vs % Change</div>
          <div class="cb">
            Spearman correlation is {_corr_label(SP_CORR, SP_P)}.
            {mw_summary.capitalize()}.
          </div>
        </div>
      </div>
      <div class="slbl">Chapter 1 — Annotated Price Map</div>
      <div class="row1"><div class="card">{divs["st_scatter"]}</div></div>
      <div class="slbl">Chapter 2 — Distribution and Outliers</div>
      <div class="row2">
        <div class="card">{divs["st_hist"]}</div>
        <div class="card">{divs["st_iqr"]}</div>
      </div>
      <div class="slbl">Chapter 3 — Hypothesis Testing</div>
      <div class="row2">
        <div class="card">{divs["st_box"]}</div>
        <div class="card">{divs["st_spearman"]}</div>
      </div>
      <div class="slbl">Chapter 4 — Session Extremes</div>
      <div class="row1"><div class="card">{divs["st_extremes"]}</div></div>
    </div>

    <!-- ═══ SENTIMENT ═══════════════════════════════════════════════════ -->
    <div class="panel" id="panel-sentiment">
      <div class="sent-kpi">
        <div class="kpi">
          <div class="kpi-lbl">Sentiment Regime</div>
          <div class="kpi-val" style="color:{PU};font-size:18px">{sent_regime.split(' — ')[0]}</div>
          <div class="kpi-sub">{sent_regime.split(' — ')[1] if ' — ' in sent_regime else sent_regime}</div>
        </div>
        <div class="kpi">
          <div class="kpi-lbl">Bullish Stocks</div>
          <div class="kpi-val" style="color:{G}">{n_bull_stocks}</div>
          <div class="kpi-sub">{pct_bull:.1f}% of universe</div>
        </div>
        <div class="kpi">
          <div class="kpi-lbl">Bearish Stocks</div>
          <div class="kpi-val" style="color:{L}">{n_bear_stocks}</div>
          <div class="kpi-sub">{pct_bear:.1f}% of universe</div>
        </div>
        <div class="kpi">
          <div class="kpi-lbl">Mean Sentiment Score</div>
          <div class="kpi-val" style="color:{'#00FFB2' if mean_sent >= BULLISH_THRESHOLD else '#FF3366' if mean_sent <= BEARISH_THRESHOLD else '#FFD600'}">{mean_sent:+.4f}</div>
          <div class="kpi-sub">NLP sentiment avg across all symbols</div>
        </div>
      </div>
      <div class="row2">
        <div class="card">{divs["sent_donut"]}</div>
        <div class="card">{divs["sent_vs_price"]}</div>
      </div>
      <div class="row2">
        <div class="card">{divs["sent_bull"]}</div>
        <div class="card">{divs["sent_bear"]}</div>
      </div>
      <div class="row1"><div class="card">{divs["sent_trend"]}</div></div>
      <div class="mini-grid">
        <div class="mini-panel">
          <div class="mini-title">Top Bullish — highest sentiment scores</div>
          {bull_mini_html}
        </div>
        <div class="mini-panel">
          <div class="mini-title">Top Bearish — lowest sentiment scores</div>
          {bear_mini_html}
        </div>
      </div>
      <div class="slbl">News Intelligence — All Symbols Ranked by Sentiment</div>
      <div class="news-panel">
        <div class="news-panel-title">
          Symbol · Top Representative Headline · Sentiment Score · Classification · Headlines Scored
        </div>
        {news_rows_html}
      </div>
    </div>

    <!-- ═══ HISTORICAL ══════════════════════════════════════════════════ -->
    <div class="panel" id="panel-historical">
      <div class="kpi-grid">
        <div class="kpi">
          <div class="kpi-lbl">Days Tracked</div>
          <div class="kpi-val">{n_days_total}</div>
          <div class="kpi-sub">Trading days in dataset</div>
        </div>
        <div class="kpi">
          <div class="kpi-lbl">Latest Breadth</div>
          <div class="kpi-val" style="color:{G if hist_df['breadth_pct'].iloc[-1] >= 50 else L}">{hist_df['breadth_pct'].iloc[-1]:.0f}%</div>
          <div class="kpi-sub">{_breadth_label(hist_df['breadth_pct'].iloc[-1])}</div>
        </div>
        <div class="kpi">
          <div class="kpi-lbl">Latest Avg Move</div>
          <div class="kpi-val" style="color:{G if hist_df['avg_pct'].iloc[-1] >= 0 else L}">{hist_df['avg_pct'].iloc[-1]:+.2f}%</div>
          <div class="kpi-sub">Mean % change — latest session</div>
        </div>
        <div class="kpi">
          <div class="kpi-lbl">Latest Volatility</div>
          <div class="kpi-val" style="color:{OR}">{latest_vol:.2f}%</div>
          <div class="kpi-sub">{vol_label}</div>
        </div>
        <div class="kpi">
          <div class="kpi-lbl">Most Active Mover</div>
          <div class="kpi-val" style="color:{AC}">{freq_movers['Symbol'].iloc[0] if not freq_movers.empty else 'N/A'}</div>
          <div class="kpi-sub">{f"{int(freq_movers['Appearances'].iloc[0])} appearances ({freq_movers['Pct_Days'].iloc[0]:.0f}% of days)" if not freq_movers.empty else 'N/A'}</div>
        </div>
        <div class="kpi">
          <div class="kpi-lbl">Regime</div>
          <div class="kpi-val" style="color:{G if hist_df['regime'].iloc[-1] == 'Bullish' else L if hist_df['regime'].iloc[-1] == 'Bearish' else FL};font-size:18px">{hist_df['regime'].iloc[-1]}</div>
          <div class="kpi-sub">Latest-day market regime</div>
        </div>
      </div>
      <div class="slbl">Section 1 — Market Breadth Over Time</div>
      <div class="row1"><div class="card">{divs["hist_breadth"]}</div></div>
      <div class="slbl">Section 2 — Average Daily Market Change</div>
      <div class="row1"><div class="card">{divs["hist_avg"]}</div></div>
      <div class="slbl">Section 3 — Volatility Over Time</div>
      <div class="row1"><div class="card">{divs["hist_vol"]}</div></div>
      <div class="slbl">Section 4 — 30-Day Rolling Market Metrics</div>
      <div class="row1"><div class="card">{divs["hist_rolling30"]}</div></div>
      <div class="slbl">Section 5 — Most Frequent Top Movers</div>
      <div class="row1"><div class="card">{divs["hist_movers"]}</div></div>
      <div class="slbl">Section 6 — Cumulative Stock Performance</div>
      <div class="row1"><div class="card">{divs["hist_cumulative"]}</div></div>
      <div class="slbl">Section 7 — Strongest and Weakest Sessions</div>
      <div class="row1"><div class="card">{divs["hist_sessions"]}</div></div>
      <div class="slbl">Section 8 — Consecutive Bullish / Bearish Streaks</div>
      <div class="row1">{divs["hist_streaks"]}</div>
      <div class="slbl">Section 9 — Historical Metrics Heatmap</div>
      <div class="row1"><div class="card">{divs["hist_heatmap"]}</div></div>
      <div class="slbl">Section 10 — Sentiment vs Market Performance</div>
      <div class="row1"><div class="card">{divs["hist_sent_perf"]}</div></div>
      <div class="slbl">Section 11 — 2-Year Cumulative Price Return (Price History)</div>
      <div class="row1"><div class="card">{divs["hist_ohlcv_perf"]}</div></div>
    </div>

    <!-- ═══ TIME SERIES ════════════════════════════════════════════════ -->
    <div class="panel" id="panel-timeseries">
      <div class="kpi-grid">
        <div class="kpi">
          <div class="kpi-lbl">Current Cycle</div>
          <div class="kpi-val" style="color:{'#00FFB2' if ts_cyc_dir == 'Bull' else '#FF3366' if ts_cyc_dir == 'Bear' else '#5A6A90'}">{ts_cyc_dir}</div>
          <div class="kpi-sub">{ts_cyc_days} trading days · avg {ts_cyc_ret:+.3f}%/day</div>
        </div>
        <div class="kpi">
          <div class="kpi-lbl">Last MA Crossover</div>
          <div class="kpi-val" style="color:{'#00FFB2' if ts_last_cross == 'Bullish' else '#FF3366' if ts_last_cross == 'Bearish' else '#5A6A90'};font-size:18px">{ts_last_cross}</div>
          <div class="kpi-sub">{ts_last_cross_d}</div>
        </div>
        <div class="kpi">
          <div class="kpi-lbl">Breadth Trend</div>
          <div class="kpi-val" style="color:{'#00FFB2' if ts_cur_breadth_trend == 'Expanding' else '#FF3366' if ts_cur_breadth_trend == 'Contracting' else '#FFD600'};font-size:18px">{ts_cur_breadth_trend}</div>
          <div class="kpi-sub">Market participation direction</div>
        </div>
        <div class="kpi">
          <div class="kpi-lbl">Latest Residual</div>
          <div class="kpi-val" style="color:{'#00FFB2' if ts_cur_residual >= 0 else '#FF3366'}">{ts_cur_residual:+.3f}%</div>
          <div class="kpi-sub">Deviation above/below trend</div>
        </div>
      </div>
      <div class="slbl">Section 1 — Trend Decomposition</div>
      <div class="row1"><div class="card">{divs["ts_decomp"]}</div></div>
      <div class="slbl">Section 2 — Residual (Detrended Series)</div>
      <div class="row1"><div class="card">{divs["ts_residual"]}</div></div>
      <div class="slbl">Section 3 — Rolling Statistics (Mean ± Std Band)</div>
      <div class="row1"><div class="card">{divs["ts_rolling_stats"]}</div></div>
      <div class="slbl">Section 4 — MA Crossover Signals</div>
      <div class="row1"><div class="card">{divs["ts_trend"]}</div></div>
      <div class="slbl">Section 5 — Breadth Trend</div>
      <div class="row1"><div class="card">{divs["ts_breadth_trend"]}</div></div>
      <div class="slbl">Section 6 — Bull &amp; Bear Market Cycles</div>
      <div class="row1"><div class="card">{divs["ts_cycles"]}</div></div>
    </div>

    {phase2_technical_html}
    {phase2_forecast_html}
    {phase2_signal_html}
    {phase2_briefing_html}
    {phase2_anomaly_html}

    </main>
    <div class="toast" id="toast">Ticker highlighted in the screener</div>

    <script>
    function showToast(message){{
      const toast = document.getElementById('toast');
      toast.textContent = message;
      toast.classList.add('show');
      window.clearTimeout(window.__toastTimer);
      window.__toastTimer = window.setTimeout(() => toast.classList.remove('show'), 2200);
    }}
    function showTab(key){{
      document.querySelectorAll('.sb-item').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
      document.getElementById('tb-' + key).classList.add('active');
      document.getElementById('panel-' + key).classList.add('active');
      window.dispatchEvent(new Event('resize'));
      window.scrollTo({{top: 0, behavior: 'smooth'}});
    }}
    function toggleSidebar(){{
      const sb = document.getElementById('sidebar');
      const mc = document.getElementById('mainContent');
      const collapsed = sb.classList.toggle('collapsed');
      if(mc) mc.classList.toggle('sb-collapsed', collapsed);
      try{{ localStorage.setItem('sb_collapsed', collapsed ? '1' : '0'); }}catch(e){{}}
    }}
    (function(){{
      try{{
        if(localStorage.getItem('sb_collapsed') === '1'){{
          const sb = document.getElementById('sidebar');
          const mc = document.getElementById('mainContent');
          if(sb){{ sb.classList.add('collapsed'); }}
          if(mc){{ mc.classList.add('sb-collapsed'); }}
        }}
      }}catch(e){{}}
    }})();
    const search = document.getElementById('tickerSearch');
    if (search) {{
      search.addEventListener('keydown', event => {{
        if (event.key === 'Enter' && search.value.trim()) {{
          showTab('screener');
          showToast(search.value.trim().toUpperCase() + ' is available in the screener table');
        }}
      }});
    }}
    const _symData = __SYMBOL_DATA__;
    function renderSymbolChart(containerId, sym) {{
      const el = document.getElementById(containerId);
      if (!sym || !_symData[sym]) {{ if(el) el.style.minHeight='0'; return; }}
      el.style.minHeight = '560px';
      const d = _symData[sym], dates = d.Date;
      const bg='#0d1117', grid='#1e2d3d', axc='#8b949e';
      const axBase = {{type:'date',gridcolor:grid,linecolor:grid,color:axc,
                       showgrid:true,tickfont:{{color:axc,size:10}}}};
      const traces = [
        {{x:dates,y:d.Close,type:'scatter',mode:'lines',name:'Close',
          line:{{color:'#00c8ff',width:2}},xaxis:'x',yaxis:'y'}},
        {{x:dates,y:d.bb_upper,type:'scatter',mode:'lines',name:'BB Upper',
          line:{{color:'rgba(255,152,0,0.5)',width:1,dash:'dot'}},xaxis:'x',yaxis:'y'}},
        {{x:dates,y:d.bb_lower,type:'scatter',mode:'lines',name:'BB Lower',
          line:{{color:'rgba(255,152,0,0.5)',width:1,dash:'dot'}},
          fill:'tonexty',fillcolor:'rgba(255,152,0,0.04)',xaxis:'x',yaxis:'y'}},
        {{x:dates,y:d.sma_20,type:'scatter',mode:'lines',name:'SMA 20',
          line:{{color:'rgba(156,39,176,0.8)',width:1.5}},xaxis:'x',yaxis:'y'}},
        {{x:dates,y:d.sma_50,type:'scatter',mode:'lines',name:'SMA 50',
          line:{{color:'rgba(255,193,7,0.8)',width:1.5}},xaxis:'x',yaxis:'y'}},
        {{x:dates,y:d.rsi_14,type:'scatter',mode:'lines',name:'RSI 14',
          line:{{color:'#e040fb',width:1.5}},xaxis:'x2',yaxis:'y2'}},
        {{x:[dates[0],dates[dates.length-1]],y:[70,70],type:'scatter',mode:'lines',
          line:{{color:'rgba(244,67,54,0.35)',width:1,dash:'dot'}},
          xaxis:'x2',yaxis:'y2',showlegend:false}},
        {{x:[dates[0],dates[dates.length-1]],y:[30,30],type:'scatter',mode:'lines',
          line:{{color:'rgba(76,175,80,0.35)',width:1,dash:'dot'}},
          xaxis:'x2',yaxis:'y2',showlegend:false}},
        {{x:dates,y:d.macd,type:'scatter',mode:'lines',name:'MACD',
          line:{{color:'#00e5ff',width:1.5}},xaxis:'x3',yaxis:'y3'}},
        {{x:dates,y:d.macd_signal,type:'scatter',mode:'lines',name:'Signal',
          line:{{color:'#ff6e40',width:1.5}},xaxis:'x3',yaxis:'y3'}},
        {{x:dates,y:(d.macd_hist||[]),type:'bar',name:'Histogram',
          marker:{{color:(d.macd_hist||[]).map(v=>v>=0
            ?'rgba(76,175,80,0.55)':'rgba(244,67,54,0.55)')}},
          xaxis:'x3',yaxis:'y3'}},
      ];
      const layout = {{
        paper_bgcolor:bg,plot_bgcolor:bg,
        font:{{color:'#c9d1d9',size:11,family:'JetBrains Mono'}},
        height:560,margin:{{l:55,r:20,t:44,b:20}},
        title:{{text:sym+' — 2-Year Technical Chart',font:{{color:'#c9d1d9',size:14}}}},
        showlegend:true,
        legend:{{bgcolor:'rgba(0,0,0,0)',font:{{color:'#c9d1d9',size:10}},
                 orientation:'h',x:0,y:-0.03}},
        xaxis: {{...axBase,domain:[0,1],anchor:'y'}},
        xaxis2:{{...axBase,domain:[0,1],anchor:'y2',matches:'x'}},
        xaxis3:{{...axBase,domain:[0,1],anchor:'y3',matches:'x'}},
        yaxis: {{title:{{text:'Price ($)'}},gridcolor:grid,linecolor:grid,
                 color:axc,domain:[0.46,1.0]}},
        yaxis2:{{title:{{text:'RSI'}},gridcolor:grid,linecolor:grid,
                 color:axc,domain:[0.26,0.43],range:[0,100]}},
        yaxis3:{{title:{{text:'MACD'}},gridcolor:grid,linecolor:grid,
                 color:axc,domain:[0.0,0.23]}},
      }};
      Plotly.react(containerId, traces, layout, {{responsive:true,displaylogo:false}});
    }}
    window.addEventListener('load', function() {{
      const screenerEl = document.getElementById('screener');
      if (!screenerEl || !screenerEl.on) return;
      screenerEl.on('plotly_click', function(data) {{
        if (!data.points || !data.points.length) return;
        const pt = data.points[0];
        if (pt.rowIndex < 1) return;
        const sym = String((pt.data.cells.values[1] || [])[pt.rowIndex - 1] || '').trim();
        if (!sym || !_symData[sym]) return;
        showTab('technical');
        const sel = document.getElementById('tech-sym-select');
        if (sel) sel.value = sym;
        renderSymbolChart('tech-sym-chart', sym);
        setTimeout(function() {{
          const el = document.getElementById('tech-sym-chart');
          if (el) el.scrollIntoView({{behavior:'smooth', block:'start'}});
        }}, 120);
      }});
    }});
    </script>
    </body>
    </html>
    """

    html = html.replace('__SYMBOL_DATA__', _sym_chart_json)

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)

    sz = os.path.getsize(OUT) / 1e6
    logger.info("Saved -> %s  (%.1f MB)  |  %d Plotly divs", OUT, sz, len(divs))
    print(f"Saved -> {OUT}  ({sz:.1f} MB)  |  {len(divs)} divs")

if __name__ == "__main__":
    build()