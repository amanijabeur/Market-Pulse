"""
========================================================================
dashboard_components/signal_tab.py — Signal Intelligence Tab
========================================================================
Renders the Signal Intelligence dashboard tab from the cached signal
Parquet file. All signal computation lives in signal_engine.py.

Public API
----------
  signal_summary(signal_df)             -> dict
  build_signal_divs(signal_df, theme)   -> dict
  signals_dict_from_cache(signal_df)    -> dict
  render_signal_tab(signal_df, divs, theme) -> str
========================================================================
"""

from __future__ import annotations

import pandas as pd

from .chart_helpers import ChartTheme
from .shared_components import intel_rows, kpi, news_panel


def signal_summary(signal_df: pd.DataFrame) -> dict:
    """
    Return lightweight signal counts for tab KPIs and AI narratives.

    Parameters
    ----------
    signal_df : signal cache from data_loader.load_signal_data()

    Returns
    -------
    dict with keys: active, bullish, bearish
    """
    if signal_df is None or signal_df.empty:
        return {"active": 0, "bullish": 0, "bearish": 0}

    direction = signal_df["direction"] if "direction" in signal_df.columns else pd.Series(dtype=str)
    return {
        "active":  len(signal_df),
        "bullish": int((direction == "Bullish").sum()),
        "bearish": int((direction == "Bearish").sum()),
    }


def build_signal_divs(signal_df: pd.DataFrame, theme: ChartTheme) -> dict:
    """
    Build Signal Intelligence table divs from the cached signal rows.

    Rows are sorted by score descending so the strongest signals
    appear first. Falls back gracefully if the cache is absent or empty.

    Parameters
    ----------
    signal_df : signal cache from data_loader.load_signal_data()
    theme     : active ChartTheme

    Returns
    -------
    dict with key "signal_table" -> HTML string
    """
    if signal_df is not None and not signal_df.empty and "score" in signal_df.columns:
        frame = signal_df.sort_values("score", ascending=False)
    else:
        frame = signal_df

    return {
        "signal_table": news_panel(
            "Symbol · Signal · Rationale · Score",
            intel_rows(frame, "score", "signal_type", theme),
        )
    }


def signals_dict_from_cache(signal_df: pd.DataFrame) -> dict:
    """
    Build the minimal ai_narratives signal input from the cached signal rows.

    Parameters
    ----------
    signal_df : signal cache from data_loader.load_signal_data()

    Returns
    -------
    dict with keys: all_signals (pd.DataFrame), risk_scores (pd.DataFrame)
    """
    return {
        "all_signals": signal_df if signal_df is not None else pd.DataFrame(),
        "risk_scores": pd.DataFrame(),
    }


def render_signal_tab(
    signal_df: pd.DataFrame,
    divs: dict,
    theme: ChartTheme,
) -> str:
    """
    Render the Signal Intelligence tab HTML.

    Parameters
    ----------
    signal_df : signal cache (used for KPI computation)
    divs      : pre-built table divs from build_signal_divs()
    theme     : active ChartTheme
    """
    summary = signal_summary(signal_df)
    return f"""
    <!-- Phase 2: Signal Intelligence -->
    <div class="panel" id="panel-signals">
      <div class="kpi-grid">
        {kpi("Active Signals", summary["active"],  "Signals detected today",         theme.AC)}
        {kpi("Bullish",        summary["bullish"], "Positive direction signals",      theme.G )}
        {kpi("Bearish",        summary["bearish"], "Negative direction signals",      theme.L )}
      </div>
      <div class="row1">{divs.get("signal_table", "")}</div>
    </div>
    """