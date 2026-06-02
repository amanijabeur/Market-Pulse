"""
========================================================================
dashboard_components/technical_tab.py — Technical Intelligence Tab
========================================================================
Renders the Technical Intelligence dashboard tab from cached Parquet
data. All indicator computation lives in technical_indicators.py;
this module is presentation-only.

Public API
----------
  build_technical_divs(technical_df, hist_df, theme) -> (dict, pd.DataFrame)
  render_technical_tab(technical_df, hist_df, divs, theme) -> str
========================================================================
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .chart_helpers import (
    ChartTheme, axis_layout, base_layout, chart_title,
    date_labels, empty_state, float_list, to_div,
)
from .shared_components import kpi


def build_technical_divs(
    technical_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    theme: ChartTheme,
) -> tuple[dict, pd.DataFrame]:
    """
    Build Plotly chart divs for the Technical Intelligence tab.

    Uses technical_df if available (richer indicator set from the
    technical_indicators cache), falling back to hist_df if the cache
    has not been populated yet. Both carry sma_avg_7d / ema_avg_7d.

    Returns
    -------
    (divs, source_df)
        divs      : dict mapping div-id strings to HTML strings
        source_df : the DataFrame actually used (for KPI extraction)
    """
    source = (
        technical_df
        if technical_df is not None and not technical_df.empty
        else hist_df
    )

    if source is None or source.empty:
        placeholder = empty_state(theme, "Technical cache not available yet.")
        return {"tech_trend": placeholder, "tech_signal": placeholder}, pd.DataFrame()

    tech = source.copy()
    tech["Date"] = date_labels(pd.to_datetime(tech["Date"], errors="coerce"))

    # Determine data-quality subtitle
    note = (
        "Cached SMA/EMA indicators; overlap expected until more history accumulates"
        if len(tech) < 7
        else "Cached SMA/EMA market indicators"
    )
    dates = tech["Date"].tolist()

    # ── Trend Stack: SMA 7D / EMA 7D / SMA 30D ──────────────────────
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=dates,
        y=float_list(tech.get("sma_avg_7d", tech["avg_pct"])),
        mode="lines+markers",
        name="SMA 7D",
        line=dict(color=theme.G, width=3),
        marker=dict(size=8, symbol="circle"),
        hovertemplate="<b>%{x}</b><br>SMA 7D: %{y:+.4f}%<extra></extra>",
    ))
    fig_trend.add_trace(go.Scatter(
        x=dates,
        y=float_list(tech.get("ema_avg_7d", tech["avg_pct"])),
        mode="lines+markers",
        name="EMA 7D",
        line=dict(color=theme.AC, width=2, dash="dash"),
        marker=dict(size=7, symbol="diamond"),
        hovertemplate="<b>%{x}</b><br>EMA 7D: %{y:+.4f}%<extra></extra>",
    ))
    if "sma_avg_30d" in tech.columns:
        fig_trend.add_trace(go.Scatter(
            x=dates,
            y=float_list(tech["sma_avg_30d"]),
            mode="lines+markers",
            name="SMA 30D",
            line=dict(color=theme.PU, width=2, dash="dot"),
            marker=dict(size=7, symbol="square"),
            hovertemplate="<b>%{x}</b><br>SMA 30D: %{y:+.4f}%<extra></extra>",
        ))
    fig_trend.update_layout(
        **base_layout(theme, 380),
        title=chart_title(theme, "Technical Trend Stack", note),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=theme.TX, size=11),
            orientation="h", x=0, y=-0.12,
        ),
    )
    fig_trend.update_xaxes(**axis_layout(theme, "Date"), type="category")
    fig_trend.update_yaxes(**axis_layout(theme, "Average % Change", zeroline=True))

    # ── Signal Strength Bar ──────────────────────────────────────────
    sig_series = pd.to_numeric(
        tech.get("signal_strength", pd.Series([0] * len(tech))),
        errors="coerce",
    ).fillna(0)

    fig_signal = go.Figure(go.Bar(
        x=dates,
        y=float_list(sig_series),
        marker=dict(
            color=[theme.G if v >= 0 else theme.L for v in sig_series],
            opacity=0.85,
        ),
        hovertemplate="<b>%{x}</b><br>Signal z-score: %{y:+.4f}<extra></extra>",
    ))
    fig_signal.update_layout(
        **base_layout(theme, 320),
        title=chart_title(theme, "Rolling Signal Strength", "Z-score of market average move"),
    )
    fig_signal.update_xaxes(**axis_layout(theme, "Date"), type="category")
    fig_signal.update_yaxes(**axis_layout(theme, "Signal z-score", zeroline=True))

    return {
        "tech_trend":  to_div(fig_trend,  "tech_trend"),
        "tech_signal": to_div(fig_signal, "tech_signal"),
    }, tech


def render_technical_tab(
    technical_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    divs: dict,
    theme: ChartTheme,
    sym_opts_html: str = "",
    sym_sel_style: str = "",
) -> str:
    """
    Render the Technical Intelligence tab HTML.

    KPI values are extracted from the most-recent row of the cached
    technical DataFrame. Falls back to hist_df values if the technical
    cache is absent.

    Parameters
    ----------
    technical_df : Phase 2 technical cache (may be empty)
    hist_df      : historical aggregate metrics (always available)
    divs         : pre-built chart divs from build_technical_divs()
    theme        : active ChartTheme
    """
    trend      = "N/A"
    signal     = 0.0
    vol_regime = "N/A"

    source = technical_df if technical_df is not None and not technical_df.empty else hist_df
    if source is not None and not source.empty:
        if "trend_7_30" in source.columns:
            trend = str(source["trend_7_30"].iloc[-1])
        if "signal_strength" in source.columns:
            raw = source["signal_strength"].iloc[-1]
            signal = float(raw) if pd.notna(raw) else 0.0

    if hist_df is not None and not hist_df.empty and "volatility_regime" in hist_df.columns:
        vol_regime = str(hist_df["volatility_regime"].iloc[-1])

    trend_color  = theme.G if trend == "Bullish" else (theme.L if trend == "Bearish" else theme.AC)
    signal_color = theme.G if signal >= 0 else theme.L

    return f"""
    <!-- Phase 2: Technical Intelligence -->
    <div class="panel" id="panel-technical">
      <div class="kpi-grid">
        {kpi("Latest Trend",    trend,                "SMA 7D vs SMA 30D",           trend_color,  "18px")}
        {kpi("Signal Strength", f"{signal:+.2f}",     "Latest rolling z-score",       signal_color        )}
        {kpi("Vol Regime",      vol_regime,            "Rolling volatility class",      theme.OR,    "18px")}
      </div>
      <div class="row2">
        <div class="card">{divs.get("tech_trend",  "")}</div>
        <div class="card">{divs.get("tech_signal", "")}</div>
      </div>
      <div class="slbl">Symbol Deep-Dive — 2-Year Technical Chart</div>
      <div style="padding:8px 0 4px">
        <select id="tech-sym-select" onchange="renderSymbolChart('tech-sym-chart',this.value)"
                style="{sym_sel_style}">
          <option value="">— Select a symbol —</option>{sym_opts_html}
        </select>
      </div>
      <div id="tech-sym-chart" style="min-height:0"></div>
    </div>
    """

