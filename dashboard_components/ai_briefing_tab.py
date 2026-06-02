"""
========================================================================
dashboard_components/ai_briefing_tab.py — AI Briefing Tab
========================================================================
Renders the AI Briefing dashboard tab from the structured briefing dict
produced by ai_narratives.generate_daily_briefing(). Narrative
generation lives entirely in ai_narratives.py; this module is
presentation-only.

Public API
----------
  render_ai_briefing_tab(briefing) -> str
========================================================================
"""

from __future__ import annotations

from .shared_components import callout, narrative_row, news_panel


def render_ai_briefing_tab(briefing: dict) -> str:
    """
    Render the AI Briefing tab HTML from the generated narrative dict.

    The briefing dict is produced by ai_narratives.generate_daily_briefing()
    and contains one string per section. Missing keys produce empty strings
    gracefully — the tab will render but those panels will be blank.

    Parameters
    ----------
    briefing : dict of section_name -> narrative_string

    Sections rendered
    -----------------
    executive_summary : top-level callout (positive)
    signals           : signal read callout (warning)
    anomalies         : risk read callout (negative)
    session_summary   : narrative row
    volatility        : narrative row
    sentiment         : narrative row
    forecast          : narrative row (if present)
    """
    narrative_rows = (
        narrative_row("SESSION",  briefing.get("session_summary", "")) +
        narrative_row("VOL",      briefing.get("volatility",      "")) +
        narrative_row("SENT",     briefing.get("sentiment",       ""))
    )
    # Include forecast row if the briefing contains it
    if briefing.get("forecast"):
        narrative_rows += narrative_row("FORECAST", briefing["forecast"])

    return f"""
    <!-- Phase 2: AI Briefing -->
    <div class="panel" id="panel-briefing">
      <div class="callouts">
        {callout("pos",  "EXECUTIVE", "Daily Synthesis", briefing.get("executive_summary", ""))}
        {callout("warn", "SIGNALS",   "Signal Read",     briefing.get("signals",           ""))}
        {callout("neg",  "RISK",      "Risk Read",       briefing.get("anomalies",         ""))}
      </div>
      {news_panel("Narrative Sections", narrative_rows)}
    </div>
    """
