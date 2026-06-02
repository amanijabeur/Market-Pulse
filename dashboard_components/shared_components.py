"""
========================================================================
dashboard_components/shared_components.py — Reusable HTML Fragments
========================================================================
Every repeated HTML generation pattern used across dashboard.py and tab
modules lives here. Callers import the function they need; the HTML
template is defined once.

Design rules
------------
- All user-facing strings are HTML-escaped before insertion.
- No analytics, no data loading, no Plotly imports.
- Functions that previously lived as local closures inside dashboard.py
  (mover_rows, sentiment_mini_rows) are now importable from here.
- intel_rows fixed: former use of DataFrame.get() (a Series method)
  replaced with column-safe access via DataFrame[col] with a default.

Public API
----------
  esc(value)                         -> str
  kpi(label, value, sub, ...)        -> str
  callout(kind, eyebrow, title, body)-> str
  news_panel(title, body_html)       -> str
  narrative_row(label, body)         -> str
  intel_rows(frame, value_col, ...)  -> str
  mover_rows(frame, color, theme)    -> str
  sentiment_mini_rows(frame, color)  -> str
  streak_rows(frame, theme)          -> str
========================================================================
"""

from __future__ import annotations

import html
from typing import Optional

import pandas as pd

from .chart_helpers import ChartTheme, empty_state


def esc(value) -> str:
    """HTML-escape a value for safe insertion into the static dashboard."""
    return html.escape(str(value))


# ======================================================================
# KPI CARD
# ======================================================================

def kpi(
    label: str,
    value,
    sub: str,
    color: Optional[str] = None,
    font_size: Optional[str] = None,
) -> str:
    """
    Render a KPI card matching the existing dashboard structure.

    Parameters
    ----------
    label     : metric label above the value
    value     : the large central number/string
    sub       : small subtitle below the value
    color     : optional CSS colour override for the value
    font_size : optional CSS font-size override for the value
    """
    style_parts = []
    if color:
        style_parts.append(f"color:{color}")
    if font_size:
        style_parts.append(f"font-size:{font_size}")
    style_attr = f' style="{";".join(style_parts)}"' if style_parts else ""

    return (
        '<div class="kpi">'
        f'<div class="kpi-lbl">{esc(label)}</div>'
        f'<div class="kpi-val"{style_attr}>{esc(value)}</div>'
        f'<div class="kpi-sub">{esc(sub)}</div>'
        '</div>'
    )


# ======================================================================
# CALLOUT BLOCK
# ======================================================================

def callout(kind: str, eyebrow: str, title: str, body: str) -> str:
    """
    Render a narrative callout block (pos / warn / neg variants).

    Parameters
    ----------
    kind   : CSS modifier class — "pos", "warn", or "neg"
    eyebrow: small category label above the title
    title  : bold callout title
    body   : narrative body text
    """
    return (
        f'<div class="callout {esc(kind)}">'
        f'<div class="ci">{esc(eyebrow)}</div>'
        f'<div class="ct">{esc(title)}</div>'
        f'<div class="cb">{esc(body)}</div>'
        '</div>'
    )


# ======================================================================
# NEWS PANEL SHELL
# ======================================================================

def news_panel(title: str, body_html: str) -> str:
    """
    Wrap rows in the dashboard news-panel chrome.

    Parameters
    ----------
    title     : panel header text (escaped)
    body_html : pre-rendered inner HTML (trusted — not re-escaped)
    """
    return (
        '<div class="news-panel">'
        f'<div class="news-panel-title">{esc(title)}</div>'
        f'{body_html}'
        '</div>'
    )


# ======================================================================
# NARRATIVE ROW
# ======================================================================

def narrative_row(label: str, body: str) -> str:
    """
    Render a compact narrative row for the AI Briefing section.

    Parameters
    ----------
    label : short category tag (e.g. "SESSION", "VOL", "SENT")
    body  : narrative body text
    """
    return (
        '<div class="news-row">'
        f'<div class="news-sym">{esc(label)}</div>'
        f'<div class="news-head"><span>{esc(body)}</span></div>'
        '</div>'
    )


# ======================================================================
# INTELLIGENCE ROWS (signals / anomalies)
# ======================================================================

def intel_rows(
    frame: pd.DataFrame,
    value_col: str,
    label_col: str,
    theme: ChartTheme,
    limit: int = 12,
) -> str:
    """
    Render signal or anomaly intelligence rows.

    Fix vs original: DataFrame.get() is a Series/dict method, not a
    DataFrame method. Replaced with safe column access using `in` check
    plus .iloc access to avoid AttributeError on non-empty DataFrames.

    Parameters
    ----------
    frame     : signal or anomaly DataFrame
    value_col : column whose value drives the colour (e.g. "score", "severity")
    label_col : column shown as the type label (e.g. "signal_type", "anomaly_type")
    theme     : active ChartTheme
    limit     : maximum rows rendered
    """
    if frame is None or frame.empty:
        return empty_state(theme, "No rows available.")

    def _get(row: pd.Series, col: str, default="") -> str:
        return str(row[col]) if col in row.index else str(default)

    rows = []
    for _, row in frame.head(limit).iterrows():
        scope     = _get(row, "Symbol", "MARKET")
        label     = _get(row, label_col).replace("_", " ").title()
        reason    = _get(row, "reason")[:150]
        raw_val   = row[value_col] if value_col in row.index else 0
        direction = _get(row, "direction").lower()

        if direction == "bullish":
            color = theme.G
        elif direction == "bearish":
            color = theme.L
        else:
            color = theme.OR

        rows.append(
            '<div class="news-row intel-row">'
            f'<div class="intel-scope">{esc(scope)}</div>'
            f'<div class="intel-type">{esc(label)}</div>'
            f'<div class="intel-reason">{esc(reason)}</div>'
            f'<div class="intel-value" style="color:{color}">{esc(raw_val)}</div>'
            '</div>'
        )
    return "".join(rows)


# ======================================================================
# MOVER MINI ROWS
# Replaces the former local closure mover_rows() in dashboard.build().
# ======================================================================

def mover_rows(frame: pd.DataFrame, color: str) -> str:
    """
    Build mini-row HTML for top gainer / loser side panels.

    Replaces the former local closure inside dashboard.build()::

        def mover_rows(frame, color):
            for _, row in frame.iterrows():
                ...

    Parameters
    ----------
    frame : DataFrame with Symbol, Company Name, % Change columns
    color : CSS colour string for the percentage value
    """
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            '<div class="mini-row">'
            f'<div><b>{esc(str(row["Symbol"]))}</b>'
            f'<span>{esc(str(row["Company Name"]))}</span></div>'
            f'<strong style="color:{color}">{row["% Change"]:+.2f}%</strong>'
            '</div>'
        )
    return "".join(rows)


# ======================================================================
# SENTIMENT MINI ROWS
# Replaces the former local closure sentiment_mini_rows() in dashboard.build().
# ======================================================================

def sentiment_mini_rows(frame: pd.DataFrame, color: str) -> str:
    """
    Build mini-row HTML for top bullish/bearish sentiment side panels.

    Replaces the former local closure inside dashboard.build()::

        def sentiment_mini_rows(frame, color):
            ...

    Parameters
    ----------
    frame : DataFrame with Symbol, Top_Headline, Sentiment_Score columns
    color : CSS colour for symbol and score
    """
    rows = []
    for _, row in frame.iterrows():
        headline_raw = str(row.get("Top_Headline", ""))
        headline     = esc(headline_raw[:80]) + ("..." if len(headline_raw) > 80 else "")
        rows.append(
            '<div class="mini-row">'
            f'<div><b style="color:{color}">{esc(str(row["Symbol"]))}</b>'
            f'<span>{headline}</span></div>'
            f'<strong style="color:{color}">{float(row.get("Sentiment_Score", 0)):+.4f}</strong>'
            '</div>'
        )
    return "".join(rows)


# ======================================================================
# STREAK ROWS
# Replaces the inline streak HTML loop in dashboard.build().
# ======================================================================

def streak_rows(frame: pd.DataFrame, theme: ChartTheme, limit: int = 10) -> str:
    """
    Render consecutive-streak rows for the Historical tab.

    Replaces the former inline for-loop in dashboard.build()::

        for _, row in streaks_df.head(10).iterrows():
            color = G if row["direction"] == "Bullish" else L
            ...

    Parameters
    ----------
    frame : DataFrame with direction, start_date, end_date, length, avg_pct
    theme : active ChartTheme
    limit : maximum rows to render
    """
    if frame is None or frame.empty:
        return empty_state(theme, "Streak analysis populates with 2+ days of data.")

    rows = []
    for _, row in frame.head(limit).iterrows():
        color = theme.G if row.get("direction") == "Bullish" else theme.L
        start = str(row.get("start_date", ""))[:10]
        end   = str(row.get("end_date",   ""))[:10]
        avg   = float(row.get("avg_pct", 0))
        rows.append(
            '<div class="news-row">'
            f'<div class="news-sym" style="color:{color}">{esc(str(row.get("direction", "")))}</div>'
            f'<div class="news-hl">{esc(start)} — {esc(end)}</div>'
            f'<div class="news-score" style="color:{color}">{int(row.get("length", 0))}d</div>'
            f'<div class="news-label" style="color:{color}">{avg:+.2f}%</div>'
            f'<div class="news-n">avg/day</div>'
            '</div>'
        )
    return "".join(rows)