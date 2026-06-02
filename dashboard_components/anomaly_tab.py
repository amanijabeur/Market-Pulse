"""
========================================================================
dashboard_components/anomaly_tab.py — Risk / Anomaly Tab
========================================================================
Renders the Risk / Anomaly dashboard tab from the cached anomaly
Parquet file. All anomaly computation lives in anomaly_detection.py.

Public API
----------
  anomaly_summary(anomaly_df)                          -> dict
  build_anomaly_divs(anomaly_df, theme)                -> dict
  anomalies_dict_from_cache(anomaly_df)                -> dict
  render_anomaly_tab(anomaly_df, divs, theme)          -> str
========================================================================
"""

from __future__ import annotations

import pandas as pd

from .chart_helpers import ChartTheme
from .shared_components import intel_rows, kpi, news_panel


def anomaly_summary(anomaly_df: pd.DataFrame) -> dict:
    """
    Return lightweight anomaly counts for KPIs and AI narratives.

    Parameters
    ----------
    anomaly_df : anomaly cache from data_loader.load_anomaly_data()

    Returns
    -------
    dict with keys: count, n_critical, n_high
    """
    if anomaly_df is None or anomaly_df.empty:
        return {"count": 0, "n_critical": 0, "n_high": 0}

    severity = anomaly_df["severity"] if "severity" in anomaly_df.columns else pd.Series(dtype=str)
    return {
        "count":      len(anomaly_df),
        "n_critical": int((severity == "Critical").sum()),
        "n_high":     int((severity == "High").sum()),
    }


def build_anomaly_divs(anomaly_df: pd.DataFrame, theme: ChartTheme) -> dict:
    """
    Build Risk / Anomaly table divs from the cached anomaly rows.

    Rows are sorted by z_score magnitude descending so the most
    statistically significant anomalies appear first.

    Parameters
    ----------
    anomaly_df : anomaly cache from data_loader.load_anomaly_data()
    theme      : active ChartTheme

    Returns
    -------
    dict with key "anomaly_table" -> HTML string
    """
    if (
        anomaly_df is not None
        and not anomaly_df.empty
        and "z_score" in anomaly_df.columns
    ):
        frame = anomaly_df.sort_values("z_score", key=abs, ascending=False)
    else:
        frame = anomaly_df

    return {
        "anomaly_table": news_panel(
            "Scope · Anomaly · Rationale · Severity",
            intel_rows(frame, "severity", "anomaly_type", theme),
        )
    }


def anomalies_dict_from_cache(anomaly_df: pd.DataFrame) -> dict:
    """
    Build the ai_narratives anomaly input from the cached anomaly rows.

    Parameters
    ----------
    anomaly_df : anomaly cache from data_loader.load_anomaly_data()

    Returns
    -------
    dict with keys: all_anomalies (pd.DataFrame), n_critical (int), n_high (int)
    """
    summary = anomaly_summary(anomaly_df)
    return {
        "all_anomalies": anomaly_df if anomaly_df is not None else pd.DataFrame(),
        "n_critical":    summary["n_critical"],
        "n_high":        summary["n_high"],
    }


def render_anomaly_tab(
    anomaly_df: pd.DataFrame,
    divs: dict,
    theme: ChartTheme,
) -> str:
    """
    Render the Risk / Anomaly tab HTML.

    Parameters
    ----------
    anomaly_df : anomaly cache (used for KPI computation)
    divs       : pre-built table divs from build_anomaly_divs()
    theme      : active ChartTheme
    """
    summary = anomaly_summary(anomaly_df)
    return f"""
    <!-- Phase 2: Risk / Anomaly -->
    <div class="panel" id="panel-risk">
      <div class="kpi-grid">
        {kpi("Anomalies", summary["count"],      "Detected abnormal events", theme.OR)}
        {kpi("Critical",  summary["n_critical"], "Highest severity",         theme.L )}
        {kpi("High",      summary["n_high"],     "High severity",            theme.OR)}
      </div>
      <div class="row1">{divs.get("anomaly_table", "")}</div>
    </div>
    """