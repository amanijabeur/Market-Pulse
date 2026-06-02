"""
========================================================================
dashboard_components/chart_helpers.py — Plotly Rendering Helpers
========================================================================
Single source of truth for every reusable Plotly layout, axis, title,
and serialisation helper used across dashboard.py and all tab modules.

Design rules
------------
- Pure presentation layer. No data loading, no analytics, no imports
  from the pipeline.
- All helpers accept a ChartTheme instance so they are fully theme-
  agnostic and safe to unit-test in isolation.
- dashboard.py's former local closures (base, ax, ttl, to_div) are the
  direct equivalents of base_layout, axis_layout, chart_title, to_div
  defined here. The closures have been removed from dashboard.py and
  replaced with calls to these functions.
- date_labels and float_list normalise data before handing it to
  Plotly so every chart gets the same safe, rounded inputs.
- color_for_value is extracted here so screener and historical tabs
  all use the same cell-colouring rule.

Integration
-----------
  from dashboard_components.chart_helpers import (
      ChartTheme, base_layout, axis_layout, chart_title,
      to_div, date_labels, float_list, empty_state,
      color_for_value, direction_color,
  )
========================================================================
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd


# ======================================================================
# THEME CONTAINER
# ======================================================================

@dataclass(frozen=True)
class ChartTheme:
    """
    Immutable palette and Plotly defaults shared by all component modules.

    Attributes
    ----------
    G   : electric mint   — gainers / bullish
    L   : hot pink-red    — losers  / bearish
    FL  : vivid yellow    — flat / neutral
    AC  : electric blue   — accent / analytics
    PU  : vivid purple
    OR  : vivid orange
    BG  : near-black background
    SF  : dark navy surface
    BD  : border colour
    TX  : cool-white text
    MU  : muted blue-grey (labels, subtitles)
    GR  : grid-line colour (rgba string)
    FN  : Plotly font dict
    HL  : Plotly hoverlabel dict
    """
    G:  str
    L:  str
    FL: str
    AC: str
    PU: str
    OR: str
    BG: str
    SF: str
    BD: str
    TX: str
    MU: str
    GR: str
    FN: dict
    HL: dict


# ======================================================================
# LAYOUT BUILDERS
# These are the canonical replacements for the former local closures
# base(), ax(), ttl() that lived inside dashboard.build().
# ======================================================================

def base_layout(theme: ChartTheme, height: int = 400) -> dict:
    """
    Return the standard dark Plotly layout shared across all tabs.

    Replaces the former dashboard.build() local closure ``base(h=400)``.

    Parameters
    ----------
    theme  : active ChartTheme
    height : chart pixel height
    """
    return dict(
        paper_bgcolor=theme.BG,
        plot_bgcolor=theme.SF,
        font=theme.FN,
        height=height,
        margin=dict(l=60, r=40, t=55, b=55),
        hoverlabel=theme.HL,
    )


def axis_layout(
    theme: ChartTheme,
    title: str = "",
    zeroline: bool = False,
    color: Optional[str] = None,
) -> dict:
    """
    Return the standard axis dict for update_xaxes / update_yaxes.

    Replaces the former dashboard.build() local closure ``ax()``.

    Parameters
    ----------
    theme    : active ChartTheme
    title    : axis label (empty = no label)
    zeroline : draw the zero reference line
    color    : override tick colour (defaults to theme.MU)
    """
    tick_color = color or theme.MU
    d: dict = dict(
        showgrid=True,
        gridcolor=theme.GR,
        gridwidth=1,
        linecolor=theme.BD,
        tickfont=dict(color=tick_color, size=11),
        title_font=dict(color=tick_color, size=12),
        zeroline=zeroline,
    )
    if title:
        d["title_text"] = title
    if zeroline:
        d["zerolinecolor"] = "rgba(255,255,255,0.06)"
    return d


def chart_title(theme: ChartTheme, title: str, subtitle: str = "") -> dict:
    """
    Build the dashboard's compact Plotly title object.

    Replaces the former dashboard.build() local closure ``ttl()``.

    Parameters
    ----------
    theme    : active ChartTheme
    title    : bold chart title
    subtitle : optional greyed-out sub-line
    """
    sub = (
        f"<br><span style='font-size:11px;color:{theme.MU}'>{subtitle}</span>"
        if subtitle else ""
    )
    return dict(
        text=f"<b>{title}</b>{sub}",
        font=dict(size=14, color=theme.TX),
        x=0.01,
        xanchor="left",
    )


def to_div(fig, div_id: str) -> str:
    """
    Render a Plotly figure to a responsive self-contained dashboard div.

    Replaces the former dashboard.build() local closure ``to_div()``.
    Uses a stable config so all charts have consistent toolbar behaviour.

    Parameters
    ----------
    fig    : plotly.graph_objects.Figure
    div_id : stable DOM id for CSS / JS targeting
    """
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
        config=dict(
            responsive=True,
            displaylogo=False,
            modeBarButtonsToRemove=["lasso2d", "select2d"],
        ),
    )


# ======================================================================
# DATA NORMALISATION HELPERS
# Centralise safe conversion of pandas/numpy values to plain Python
# types that Plotly's JS serialiser handles without surprises.
# ======================================================================

def date_labels(series: pd.Series, fmt: str = "%Y-%m-%d") -> list[str]:
    """
    Normalise datetime-like values to stable string chart labels.

    Parameters
    ----------
    series : pd.Series of date/datetime values
    fmt    : strftime format (default: ISO "YYYY-MM-DD")
    """
    return pd.to_datetime(series, errors="coerce").dt.strftime(fmt).tolist()


def float_list(values: Iterable, digits: int = 4) -> list[float]:
    """
    Convert pandas/numpy numeric values to plain Python floats.

    Plotly's JSON serialiser can choke on numpy scalars. This helper
    coerces errors to NaN, rounds, and casts to native float.

    Parameters
    ----------
    values : any iterable of numeric-ish values
    digits : decimal places (default 4)
    """
    return (
        pd.to_numeric(pd.Series(list(values)), errors="coerce")
        .round(digits)
        .astype(float)
        .tolist()
    )


# ======================================================================
# COLOURING HELPERS
# ======================================================================

def color_for_value(
    value: float,
    theme: ChartTheme,
    max_abs: float = 1.0,
    alpha_min: float = 0.07,
    alpha_max: float = 0.52,
) -> str:
    """
    Return an rgba colour string scaled to the magnitude of ``value``.

    Positive  -> green  (theme.G)
    Negative  -> red    (theme.L)
    Zero      -> faint yellow

    Alpha is linearly interpolated between alpha_min and alpha_max so
    small moves are faint and large moves are vivid.

    Replaces the former local ``_cbg(v)`` closure in dashboard.py::

        def _cbg(v):
            a = min(abs(v) / mx, 1) * 0.45 + 0.07 if mx else 0.07
            ...

    Parameters
    ----------
    value     : numeric value (e.g. % change)
    theme     : active ChartTheme
    max_abs   : denominator for alpha scale (e.g. df["% Change"].abs().max())
    alpha_min : alpha when value is near zero
    alpha_max : alpha when |value| == max_abs
    """
    if value == 0:
        return "rgba(255,214,0,0.12)"

    denom = max_abs if (max_abs and max_abs > 0) else 1.0
    alpha = min(abs(value) / denom, 1.0) * (alpha_max - alpha_min) + alpha_min

    hex_color = theme.G if value > 0 else theme.L
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def direction_color(value: float, theme: ChartTheme) -> str:
    """
    Return theme.G for positive, theme.L for negative, theme.FL for zero.

    Used for per-cell text colouring in tables, bars, and badges.
    """
    if value > 0:
        return theme.G
    if value < 0:
        return theme.L
    return theme.FL


# ======================================================================
# EMPTY STATE
# ======================================================================

def empty_state(theme: ChartTheme, message: str) -> str:
    """
    Return a consistent empty-state HTML block for optional sections.

    Shown when a Phase 2 cache is absent or contains no rows so the
    dashboard degrades gracefully instead of crashing.
    """
    safe = html.escape(message)
    return (
        f"<p style='color:{theme.MU};"
        f"padding:20px;"
        f"font-family:JetBrains Mono,Courier New,monospace'>"
        f"{safe}</p>"
    )