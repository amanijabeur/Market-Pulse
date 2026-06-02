"""
========================================================================
dashboard_components/forecast_tab.py — Forecasting Tab
========================================================================
Renders the Forecasting dashboard tab from the cached forecast_metrics
Parquet file. All forecast computation lives in forecasting.py.

Public API
----------
  build_forecast_divs(forecast_df, theme) -> dict
  forecast_dict_from_cache(forecast_df)   -> dict
  render_forecast_tab(divs, briefing)     -> str
========================================================================
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .chart_helpers import (
    ChartTheme, axis_layout, base_layout, chart_title,
    date_labels, empty_state, float_list, to_div,
)
from .shared_components import callout


# Metric display spec: (cache key, panel label, theme colour attribute, row, col)
_METRIC_SPECS: list[tuple[str, str, str, int, int]] = [
    ("avg_pct",    "Avg % Change", "G",  1, 1),
    ("volatility", "Volatility",   "OR", 1, 2),
    ("breadth_pct","Breadth %",    "AC", 2, 1),
    ("sentiment",  "Sentiment",    "PU", 2, 2),
]


_HISTORY_DAYS = 14   # days of actual data shown before the forecast line


def _hist_series(
    metric:     str,
    hist_df:    pd.DataFrame | None,
    sent_daily: pd.DataFrame | None,
) -> tuple[list, list]:
    """Return (x_dates as Timestamps, y_values) for the last _HISTORY_DAYS of actual data."""
    col = "avg_score" if metric == "sentiment" else metric
    df  = sent_daily  if metric == "sentiment" else hist_df

    if df is None or df.empty or col not in df.columns:
        return [], []

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", col]).sort_values("Date").tail(_HISTORY_DAYS)
    return df["Date"].tolist(), float_list(df[col])


def build_forecast_divs(
    forecast_df: pd.DataFrame,
    theme:       ChartTheme,
    hist_df:     pd.DataFrame | None = None,
    sent_daily:  pd.DataFrame | None = None,
) -> dict:
    """
    Build the Forecasting tab chart divs from the cached forecast rows.

    Shows the last 30 days of actual data before the forecast so the
    trend direction is visible in context, with a dashed separator at
    the today boundary and a shaded confidence band on the forecast.

    Parameters
    ----------
    forecast_df : long-format forecast cache from data_loader.load_forecast_data()
    theme       : active ChartTheme
    hist_df     : historical daily aggregates for actual-data context (optional)
    sent_daily  : daily sentiment averages for sentiment context (optional)

    Returns
    -------
    dict with key "forecast_paths" -> HTML div string
    """
    if forecast_df is None or forecast_df.empty:
        return {"forecast_paths": empty_state(theme, "Forecast cache not available yet.")}

    forecasts = forecast_df.copy()
    forecasts["Date"] = date_labels(pd.to_datetime(forecasts["Date"], errors="coerce"))

    fig = make_subplots(
        rows=2, cols=2,
        shared_xaxes=False,
        subplot_titles=[spec[1] for spec in _METRIC_SPECS],
        vertical_spacing=0.16,
        horizontal_spacing=0.08,
    )

    for metric, label, color_attr, row_i, col_i in _METRIC_SPECS:
        color = getattr(theme, color_attr)
        sub   = forecasts[forecasts["metric"] == metric].sort_values("Date")
        if sub.empty:
            continue

        # Use raw Timestamps so Plotly can auto-space tick labels
        x_fcast    = pd.to_datetime(sub["Date"]).tolist()
        y_forecast = float_list(sub["forecast"])
        y_upper    = float_list(sub["upper_bound"])
        y_lower    = float_list(sub["lower_bound"])

        # ── Historical actual data ────────────────────────────────────
        x_hist, y_hist = _hist_series(metric, hist_df, sent_daily)
        if x_hist and y_hist:
            fig.add_trace(go.Scatter(
                x=x_hist,
                y=y_hist,
                mode="lines",
                name=f"{label} actual",
                line=dict(color=color, width=1.5, dash="dot"),
                opacity=0.5,
                hovertemplate="%{x|%b %d}<br>Actual: %{y:.3f}<extra></extra>",
                showlegend=False,
            ), row=row_i, col=col_i)

        # ── Today separator ───────────────────────────────────────────
        if x_fcast:
            boundary = x_fcast[0]
            fig.add_vline(
                x=boundary.timestamp() * 1000,   # Plotly expects ms for date axes
                line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dash"),
                row=row_i, col=col_i,
            )

        # ── Confidence band ───────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=x_fcast + x_fcast[::-1],
            y=y_upper + y_lower[::-1],
            fill="toself",
            fillcolor="rgba(90,106,144,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        ), row=row_i, col=col_i)

        # ── Forecast line ─────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=x_fcast,
            y=y_forecast,
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2.5),
            marker=dict(size=7, symbol="circle"),
            hovertemplate="%{x|%b %d}<br>Forecast: %{y:.3f}<extra></extra>",
        ), row=row_i, col=col_i)

    layout = base_layout(theme, 600)
    layout["margin"] = dict(l=55, r=30, t=90, b=60)
    fig.update_layout(
        **layout,
        title=chart_title(
            theme,
            "Forward Forecasts",
            f"Past {_HISTORY_DAYS} days (dotted) → 5-day projection (solid) · 95% confidence band shaded",
        ),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=theme.TX, size=11),
            orientation="h", x=0.5, xanchor="center", y=-0.06,
        ),
    )
    fig.update_annotations(font=dict(size=12, color=theme.TX), yshift=-4)
    fig.update_xaxes(
        type="date",
        tickformat="%b %d",
        dtick=7 * 24 * 60 * 60 * 1000,   # weekly ticks (ms)
        tickangle=0,
        tickfont=dict(color=theme.MU, size=10),
        linecolor=theme.BD,
        showgrid=False,
    )
    fig.update_yaxes(**axis_layout(theme, "Value", zeroline=True))

    return {"forecast_paths": to_div(fig, "forecast_paths")}


def forecast_dict_from_cache(forecast_df: pd.DataFrame) -> dict:
    """
    Convert the long forecast cache to the dict ai_narratives expects.

    Splits the long-format cache by metric, drops the "metric" column
    from each sub-frame, and returns a plain dict keyed by metric name.

    Parameters
    ----------
    forecast_df : long-format forecast cache

    Returns
    -------
    dict {metric_name -> pd.DataFrame} or empty dict if cache absent
    """
    if forecast_df is None or forecast_df.empty or "metric" not in forecast_df.columns:
        return {}
    return {
        metric: frame.drop(columns=["metric"], errors="ignore").reset_index(drop=True)
        for metric, frame in forecast_df.groupby("metric")
    }


def render_forecast_tab(divs: dict, briefing: dict) -> str:
    """
    Render the Forecasting tab HTML.

    Parameters
    ----------
    divs     : pre-built chart divs from build_forecast_divs()
    briefing : AI narrative dict from ai_narratives.generate_daily_briefing()
    """
    forecast_body = briefing.get("forecast", "")
    return f"""
    <!-- Phase 2: Forecasting -->
    <div class="panel" id="panel-forecasting">
      <div class="row1"><div class="card">{divs.get("forecast_paths", "")}</div></div>
      <div class="callouts">
        {callout("warn", "FORECAST", "Projection Read", forecast_body)}
      </div>
    </div>
    """