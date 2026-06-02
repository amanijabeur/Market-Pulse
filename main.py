"""
========================================================================
main.py -- Market Pulse Analytics Platform Orchestrator
========================================================================
Single entry point for the entire pipeline. Runs every stage in the
correct order, validates data at each step, tracks timing, and
produces a structured summary on completion.

Pipeline
--------
  Stage 1 -- Scrape + EDA + Sentiment   (scraper.main)
  Stage 2 -- Historical Metrics          (historical_metrics.compute_all)
  Stage 3 -- Dashboard Build             (dashboard.build)

Usage
-----
  python main.py                    # full pipeline (normal daily run)
  python main.py --skip-scraper     # skip scraping, reuse today's data
  python main.py --dashboard-only   # rebuild dashboard HTML only
  python main.py --force-rebuild    # force Parquet + metrics rebuild
  python main.py --no-browser       # do not open browser on completion

Architecture notes
------------------
- Imports are deferred inside stage functions. A missing dependency in
  a later module never prevents an earlier stage from running.
- dashboard.py wraps all execution in build() so it is safe to import.
- stats_results.py is reloaded inside dashboard.build() via importlib
  so EDA output is always current even in the same Python process.
- All file I/O is centralised in data_loader.py (Parquet-first).
- All constants come from config.py.
- Pre-flight validation runs before any stage via validators.py.

Pip dependencies
----------------
  pip install requests beautifulsoup4 pandas numpy scipy openpyxl
              pyarrow plotly vaderSentiment yfinance colorama
========================================================================
"""

import argparse
import logging
import logging.handlers
import os
import sys
import time
import webbrowser
from datetime import datetime
from typing import Callable

import pandas as pd

# -- Config is the first import -- everything else may depend on it -----
from config import PATHS, LOGGING_CFG

# ======================================================================
# COLOUR SETUP
# Degrades gracefully on terminals that do not support ANSI codes.
# ======================================================================

try:
    import colorama
    colorama.init(autoreset=True)
    GREEN  = colorama.Fore.GREEN
    RED    = colorama.Fore.RED
    YELLOW = colorama.Fore.YELLOW
    CYAN   = colorama.Fore.CYAN
    WHITE  = colorama.Fore.WHITE
    BOLD   = colorama.Style.BRIGHT
    DIM    = colorama.Style.DIM
    RESET  = colorama.Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = CYAN = WHITE = BOLD = DIM = RESET = ""


# ======================================================================
# LOGGING
# File handler: rotating, DEBUG level, full detail.
# Console handler: INFO level, clean and readable.
# ======================================================================

os.makedirs(PATHS.LOGS_DIR, exist_ok=True)

_log_filename = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
_log_path     = os.path.join(PATHS.LOGS_DIR, _log_filename)

_file_handler    = logging.handlers.RotatingFileHandler(
    _log_path,
    maxBytes    = LOGGING_CFG.MAX_BYTES,
    backupCount = LOGGING_CFG.BACKUP_COUNT,
    encoding    = "utf-8",
)
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(getattr(logging, LOGGING_CFG.CONSOLE_LEVEL))

_formatter = logging.Formatter(
    fmt     = LOGGING_CFG.FORMAT,
    datefmt = LOGGING_CFG.DATE_FMT,
)
_file_handler.setFormatter(_formatter)
_console_handler.setFormatter(_formatter)

logging.basicConfig(
    level    = logging.DEBUG,
    handlers = [_file_handler, _console_handler],
)

logger = logging.getLogger("main")


# ======================================================================
# BANNER
# ======================================================================

# ======================================================================
# SHARED PIPELINE CONTEXT
# Stages write outputs here so stage_ai_narratives doesn't recompute
# what stages 3-7 already computed. Keyed by string for debuggability.
# ======================================================================

_run_ctx: dict = {}


_BANNER = (
    f"\n{BOLD}{CYAN}"
    "  +----------------------------------------------------------+\n"
    "  |       MARKET PULSE  --  ANALYTICS PLATFORM               |\n"
    "  |       100 Most Active Stocks  |  Yahoo Finance           |\n"
    "  +----------------------------------------------------------+"
    f"{RESET}"
)


# ======================================================================
# STAGE RUNNER
# Wraps every pipeline stage with timing, logging, and error handling.
# Returns (success: bool, elapsed: float) so main() can build a summary.
# ======================================================================

def run_stage(
    name: str,
    fn: Callable,
    critical: bool = True,
) -> tuple:
    """
    Execute one pipeline stage safely.

    Parameters
    ----------
    name     : label shown in console and logs
    fn       : zero-argument callable that runs the stage
    critical : if True, failure calls sys.exit(1)

    Returns
    -------
    (success: bool, elapsed_seconds: float)
    """
    _divider = "-" * 62
    print(f"\n{BOLD}{CYAN}{_divider}{RESET}")
    print(f"{BOLD}  {name}{RESET}")
    print(f"{CYAN}{_divider}{RESET}")
    logger.info("Stage started: %s", name)

    t_start = time.perf_counter()

    try:
        fn()
        elapsed = time.perf_counter() - t_start
        _pass(name, elapsed)
        return True, elapsed

    except SystemExit as exc:
        elapsed = time.perf_counter() - t_start
        # Exit code 0 / None means the stage completed cleanly
        # (e.g. scraper detected today already scraped and returned early)
        if exc.code in (0, None):
            _pass(name, elapsed)
            return True, elapsed
        _fail(name, elapsed, f"exited with code {exc.code}")
        if critical:
            _abort(name)
        return False, elapsed

    except Exception as exc:
        elapsed = time.perf_counter() - t_start
        _fail(name, elapsed, f"{type(exc).__name__}: {exc}")
        logger.exception("Unhandled exception in stage: %s", name)
        if critical:
            _abort(name)
        return False, elapsed


# -- Stage result helpers -----------------------------------------------

def _pass(name: str, elapsed: float) -> None:
    print(f"\n{GREEN}  [PASS]  {name}  ({elapsed:.1f}s){RESET}")
    logger.info("Stage passed: %s  (%.1fs)", name, elapsed)


def _fail(name: str, elapsed: float, reason: str) -> None:
    print(f"\n{RED}  [FAIL]  {name}  ({elapsed:.1f}s){RESET}")
    print(f"{RED}          Reason: {reason}{RESET}")
    logger.error("Stage failed: %s -- %s  (%.1fs)", name, reason, elapsed)


def _abort(name: str) -> None:
    print(f"\n{RED}{BOLD}  PIPELINE ABORTED -- critical failure in: {name}{RESET}")
    print(f"{DIM}  Check log: {_log_path}{RESET}")
    logger.critical("Pipeline aborted after failure in: %s", name)
    sys.exit(1)


# ======================================================================
# STAGE DEFINITIONS
# Each stage is a thin wrapper that defers the import until the stage
# actually runs. This means a broken module never blocks earlier stages.
# ======================================================================

def stage_scrape() -> None:
    """
    Stage 1: Scrape + EDA + Sentiment.

    scraper.main() handles all three internally:
      1. Scrapes Yahoo Finance and saves price data (Excel + Parquet)
      2. Fetches NLP sentiment for all 100 symbols and saves it
      3. Calls eda.main() to regenerate stats_results.py
    """
    import scraper
    scraper.main()


def stage_eda_standalone() -> None:
    """
    Stage 1b: EDA only (used when --skip-scraper is set).

    Regenerates stats_results.py from the existing dataset without
    hitting Yahoo Finance.
    """
    import eda
    eda.main()


def _latest_price_with_sentiment(force: bool = False) -> tuple:
    """Load current price data and merge latest available sentiment. Cached in _run_ctx."""
    if not force and "df_full" in _run_ctx:
        return _run_ctx["df_full"], _run_ctx["df_latest"]

    import data_loader

    df_full = data_loader.load_price_data()
    df_latest = data_loader.load_latest_day()
    sent = data_loader.load_sentiment_history()

    if not sent.empty:
        sent = sent.copy()
        sent["Date"] = pd.to_datetime(sent["Date"], errors="coerce")
        latest_date = pd.to_datetime(df_latest["Date"].max()).normalize()
        sent_today = sent[sent["Date"].dt.normalize() == latest_date]
        if sent_today.empty:
            sent_today = sent.sort_values("Date").groupby("Symbol", as_index=False).tail(1)
        df_latest = df_latest.merge(
            sent_today.drop(columns=["Date"], errors="ignore"),
            on="Symbol",
            how="left",
        )

    _run_ctx["df_full"]   = df_full
    _run_ctx["df_latest"] = df_latest
    return df_full, df_latest


def _ensure_daily_sent() -> "pd.DataFrame":
    """
    Load and cache the extended daily sentiment average in _run_ctx.
    Blends NLP sentiment history with VIX-derived scores for dates not yet covered,
    giving forecasting and anomaly detection a multi-year sentiment baseline.
    """
    if "sent_daily" in _run_ctx:
        return _run_ctx["sent_daily"]
    import data_loader
    from sentiment import extended_daily_sentiment_avg
    sent_daily = extended_daily_sentiment_avg(data_loader.load_sentiment_history())
    _run_ctx["sent_daily"] = sent_daily
    return sent_daily


def stage_sentiment_refresh() -> None:
    """Stage 2: ensure sentiment history is available for intelligence modules."""
    import data_loader
    from validators import validate_sentiment_data

    sent = data_loader.load_sentiment_history()
    result = validate_sentiment_data(sent)
    print(f"    Sentiment rows       : {len(sent)}")
    print(f"    Sentiment warnings   : {len(result.warnings)}")


def stage_historical_metrics(force: bool = False) -> None:
    """
    Stage 2: Incremental historical analytics.

    Loads the full price history and computes rolling metrics only for
    dates not yet cached. Results are written to historical_metrics.parquet.
    Caches result in _run_ctx for use by stages 4-8.
    """
    import data_loader
    import historical_metrics as hm

    df_full = data_loader.load_price_data(force_rebuild=force)
    result  = hm.compute_all(df_full, force=force)
    _run_ctx["hm_result"]    = result
    _run_ctx["baseline_vol"] = result["baseline_vol"]

    n_days = result["n_days"]
    print(f"    Trading days tracked  : {n_days}")
    print(f"    Multi-day history     : {result['has_history']}")
    print(f"    Latest volatility     : {result['latest_vol']:.4f}%")
    logger.info(
        "Historical metrics done -- %d day(s), vol=%.4f",
        n_days, result["latest_vol"],
    )


def stage_technical_indicators() -> None:
    """Stage 4: compute and cache market-level technical intelligence."""
    import data_loader
    import technical_indicators as ti

    # Use cached df_full and hist_df from _run_ctx if available
    df_full = _run_ctx.get("df_full") if "df_full" in _run_ctx else data_loader.load_price_data()
    hist_df = (_run_ctx.get("hm_result") or {}).get("hist_df")
    if hist_df is None:
        import historical_metrics as hm
        hist_df = hm.compute_hist_df(df_full)
    tech_df = ti.compute_market_indicators(hist_df)
    data_loader.save_technical_data(tech_df)

    latest_trend = tech_df["trend_7_30"].iloc[-1] if not tech_df.empty else "N/A"
    print(f"    Technical rows       : {len(tech_df)}")
    print(f"    Latest trend         : {latest_trend}")


def stage_forecasting() -> None:
    """Stage 5: forecast market metrics and persist long-form forecast cache."""
    import data_loader
    import historical_metrics as hm
    import forecasting
    from config import FORECASTING

    df_full = _run_ctx.get("df_full") if "df_full" in _run_ctx else data_loader.load_price_data()
    hist_df = (_run_ctx.get("hm_result") or {}).get("hist_df")
    if hist_df is None:
        hist_df = hm.compute_hist_df(df_full)
    sent_daily = pd.DataFrame()
    try:
        from sentiment import daily_sentiment_avg
        sent_daily = _ensure_daily_sent()
    except Exception as exc:
        logger.warning("Sentiment forecast skipped: %s", exc)

    # Use OHLCV-extended history so forecasts have a multi-year baseline
    extended_hist = hm.build_extended_hist_df(df_full)
    _run_ctx["extended_hist"] = extended_hist
    forecasts = forecasting.forecast_market_metrics(
        extended_hist,
        horizon=FORECASTING.DEFAULT_HORIZON,
        method=FORECASTING.DEFAULT_METHOD,
    )
    forecasts["sentiment"] = forecasting.forecast_sentiment_trend(
        sent_daily,
        horizon=FORECASTING.DEFAULT_HORIZON,
        method=FORECASTING.DEFAULT_METHOD,
    )

    frames = []
    for metric, frame in forecasts.items():
        if frame is not None and not frame.empty:
            out = frame.copy()
            out.insert(1, "metric", metric)
            frames.append(out)
    forecast_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    data_loader.save_forecast_data(forecast_df)
    print(f"    Forecast rows        : {len(forecast_df)}")
    print(f"    Forecast horizon     : {FORECASTING.DEFAULT_HORIZON} business days")


def stage_signal_engine() -> None:
    """Stage 6: generate signal intelligence and cache unified signals."""
    import data_loader
    import signal_engine
    import historical_metrics as hm

    df_full, df_latest = _latest_price_with_sentiment()
    baseline_vol = _run_ctx.get("baseline_vol") or 1.0

    # Extend df_full with OHLCV history so z-score baselines have 2+ years
    # of per-symbol data instead of just scraped days.
    extended_df_full = hm.build_extended_df_full(df_full)
    _run_ctx["extended_df_full"] = extended_df_full

    signals = signal_engine.compute_all_signals(df_latest, extended_df_full, baseline_vol)
    all_signals = signals.get("all_signals", pd.DataFrame())
    data_loader.save_signal_data(all_signals)
    _run_ctx["signals"] = signals
    print(f"    Signals generated    : {len(all_signals)}")
    print(f"    Risk rows            : {len(signals.get('risk_scores', pd.DataFrame()))}")


def stage_anomaly_detection() -> None:
    """Stage 7: detect and cache current risk/anomaly conditions."""
    import data_loader
    import historical_metrics as hm
    import anomaly_detection
    from sentiment import daily_sentiment_avg

    df_full, df_latest = _latest_price_with_sentiment()
    hist_df = hm.compute_hist_df(df_full)
    daily_sent = _ensure_daily_sent()

    # Use extended df_full for per-symbol price gap z-scores
    extended_df_full = _run_ctx.get("extended_df_full")
    if extended_df_full is None:
        extended_df_full = hm.build_extended_df_full(df_full)
    anomalies = anomaly_detection.compute_all_anomalies(
        df_latest, extended_df_full, hist_df, daily_sent
    )
    all_anomalies = anomalies.get("all_anomalies", pd.DataFrame())
    data_loader.save_anomaly_data(all_anomalies)
    _run_ctx["anomalies"]  = anomalies
    _run_ctx["sent_daily"] = daily_sent
    print(f"    Anomalies detected   : {len(all_anomalies)}")
    print(f"    Critical anomalies   : {anomalies.get('n_critical', 0)}")


def stage_ai_narratives() -> None:
    """
    Stage 8: generate AI-style deterministic narrative summaries.

    Reads all intelligence outputs from _run_ctx instead of recomputing
    them -- eliminating the redundant work done by stages 3-7.
    """
    import ai_narratives
    import data_loader
    import forecasting
    from config import FORECASTING

    df_full, df_latest = _latest_price_with_sentiment()

    # Read from _run_ctx if stages 3-7 already ran; else fall back to recompute
    import historical_metrics as hm
    hist = _run_ctx.get("hm_result") or hm.compute_all(df_full)
    hist_df = hist["hist_df"]

    signals   = _run_ctx.get("signals")
    anomalies = _run_ctx.get("anomalies")
    daily_sent = _run_ctx.get("sent_daily")

    if signals is None:
        import signal_engine
        signals = signal_engine.compute_all_signals(df_latest, df_full, hist["baseline_vol"])
    if anomalies is None:
        import anomaly_detection
        from sentiment import daily_sentiment_avg
        if daily_sent is None:
            daily_sent = _ensure_daily_sent()
        anomalies = anomaly_detection.compute_all_anomalies(df_latest, df_full, hist_df, daily_sent)
    if daily_sent is None:
        daily_sent = _ensure_daily_sent()

    extended_hist = _run_ctx.get("extended_hist")
    if extended_hist is None:
        extended_hist = hm.build_extended_hist_df(df_full)
    forecasts = forecasting.forecast_market_metrics(
        extended_hist,
        horizon=FORECASTING.DEFAULT_HORIZON,
        method=FORECASTING.DEFAULT_METHOD,
    )
    forecasts["sentiment"] = forecasting.forecast_sentiment_trend(daily_sent)

    sent_summary = {
        "mean_score": float(df_latest.get("Sentiment_Score", pd.Series(dtype=float)).mean() or 0),
        "pct_bullish": float((df_latest.get("Sentiment_Label", pd.Series(dtype=str)) == "Bullish").mean() * 100) if "Sentiment_Label" in df_latest else 0,
        "pct_bearish": float((df_latest.get("Sentiment_Label", pd.Series(dtype=str)) == "Bearish").mean() * 100) if "Sentiment_Label" in df_latest else 0,
    }
    # Use extended_hist for the briefing so rolling averages and volatility
    # baselines are drawn from the full 500-day history, not just scraped days.
    ext_latest_vol  = float(extended_hist["volatility"].iloc[-1]) if not extended_hist.empty else hist["latest_vol"]
    ext_baseline_vol = float(extended_hist["volatility"].iloc[:-1].mean()) if len(extended_hist) > 1 else hist["baseline_vol"]

    briefing = ai_narratives.generate_daily_briefing(
        df_latest, extended_hist, sent_summary, daily_sent, signals, anomalies, forecasts,
        int((df_latest["Change"] > 0).sum()),
        int((df_latest["Change"] < 0).sum()),
        int((df_latest["Change"] == 0).sum()),
        float(df_latest["% Change"].mean()),
        str(df_latest.nlargest(1, "% Change")["Symbol"].iloc[0]),
        float(df_latest["% Change"].max()),
        str(df_latest.nsmallest(1, "% Change")["Symbol"].iloc[0]),
        float(df_latest["% Change"].min()),
        ext_latest_vol,
        ext_baseline_vol,
    )
    print(f"    Briefing sections    : {len(briefing)}")
    print(f"    Executive summary    : {briefing['executive_summary'][:110]}...")


def stage_historical_fetch(fred_api_key: str = "") -> None:
    """
    Optional Stage 0: Fetch OHLCV + FRED macro data, compute indicators.

    Runs before stage 1 by default (skip with --no-historical-fetch). Uses the
    current day's 100 scraped symbols as the fetch list (falls back to
    config.HIST_FETCH.DEFAULT_TICKERS on the very first run).

    Non-critical: a failure here never blocks the daily scrape pipeline.
    """
    import data_loader as dl
    from pipelines.fetch_historical import run_historical_fetch
    from pipelines.indicators import run_indicators_batch

    # Use already-scraped symbols if available; else use defaults
    try:
        _ctx_full = _run_ctx.get("df_full") if "df_full" in _run_ctx else dl.load_price_data()
        symbols = _ctx_full["Symbol"].unique().tolist()
    except Exception:
        from config import HIST_FETCH
        symbols = list(HIST_FETCH.DEFAULT_TICKERS)

    result = run_historical_fetch(
        symbols=symbols,
        fetch_macro=True,
        fred_api_key=fred_api_key,
    )
    print(f"    OHLCV fetched        : {len(result['ohlcv_fetched'])} symbols")
    print(f"    OHLCV failed         : {len(result['ohlcv_failed'])} symbols")
    print(f"    OHLCV skipped        : {len(result.get('ohlcv_skipped', []))} symbols (consecutive failures)")
    print(f"    Macro series fetched : {len(result['macro_fetched'])}")

    # Compute indicators for all symbols that have OHLCV
    ind_result = run_indicators_batch()
    print(f"    Indicators computed  : {len(ind_result['succeeded'])} symbols")
    print(f"    Indicator rows total : {ind_result['n_rows']}")


def stage_auto_backfill() -> None:
    """
    Auto-detect and silently fill two classes of gaps:

    1. Missing trading days in the price dataset (PC was off / scraper
       missed a day). Uses yfinance closing prices — same as backfill_date.py
       but without a redundant EDA pass (main.py handles EDA itself).

    2. Sparse sentiment history. If sentiment covers fewer days than the
       price dataset (minus a 7-day grace period), auto-runs the Finnhub
       backfill so forecasting and anomaly detection have a full baseline.
       Requires FINNHUB_API_KEY in .env — skipped silently if absent.
    """
    import data_loader
    import backfill_date

    df_full = data_loader.load_price_data()
    if df_full.empty:
        return

    # ── 1. Missing price days ────────────────────────────────────────
    today_et   = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    first_date = df_full["Date"].min()
    # Only scan the last 30 days to avoid false positives from before tracking started
    window_start = max(first_date, today_et - pd.Timedelta(days=30))
    last_date    = df_full["Date"].max()

    expected_days = pd.bdate_range(start=window_start, end=last_date)
    actual_dates  = set(df_full["Date"].dt.normalize().unique())
    missing_price = [d for d in expected_days if pd.Timestamp(d) not in actual_dates]

    if missing_price:
        print(f"\n  Auto-backfill: {len(missing_price)} missing trading day(s) detected — filling now...")
        for d in missing_price:
            backfill_date.backfill(d.date().isoformat(), run_eda=False)
        data_loader.invalidate_cache()
        logger.info("Auto-backfill: filled %d missing price day(s).", len(missing_price))

    # ── 2. Sparse sentiment ──────────────────────────────────────────
    try:
        from finnhub_sentiment import FINNHUB_API_KEY, backfill as fh_backfill
    except Exception:
        return

    if not FINNHUB_API_KEY:
        return

    sent = data_loader.load_sentiment_history()
    df_for_sent = data_loader.load_price_data()   # reload in case price was just updated

    price_days = df_for_sent["Date"].dt.normalize().nunique() if not df_for_sent.empty else 0
    sent_days  = pd.to_datetime(sent["Date"]).dt.normalize().nunique() if not sent.empty else 0

    if price_days - sent_days <= 7:
        return   # gap is within acceptable tolerance

    from_date = df_for_sent["Date"].min().date().isoformat()
    symbols   = df_for_sent["Symbol"].unique().tolist()

    print(f"\n  Auto-backfill sentiment: {price_days - sent_days} days missing — fetching from Finnhub...")
    new_sent = fh_backfill(symbols, from_date, FINNHUB_API_KEY)

    if new_sent.empty:
        return

    if not sent.empty:
        existing_keys = set(zip(sent["Date"].astype(str), sent["Symbol"].astype(str)))
        new_sent = new_sent[
            ~new_sent.apply(
                lambda r: (str(r["Date"]), str(r["Symbol"])) in existing_keys, axis=1
            )
        ].copy()

    if not new_sent.empty:
        combined = pd.concat([sent, new_sent], ignore_index=True)
        data_loader.save_sentiment_history(combined)
        data_loader.invalidate_cache()
        logger.info("Auto-backfill: added %d sentiment rows.", len(new_sent))
        print(f"  Sentiment: +{len(new_sent)} rows added automatically.")


def stage_dashboard() -> None:
    """
    Stage 3: Build the interactive HTML dashboard.

    dashboard.build() reloads stats_results.py via importlib so values
    written by EDA in Stage 1 are always current, even in the same
    Python process.
    """
    import dashboard
    dashboard.build()


# ======================================================================
# PIPELINE SUMMARY
# ======================================================================

def print_summary(results: list, total_elapsed: float) -> None:
    """
    Print a structured summary table after all stages complete.

    Parameters
    ----------
    results       : list of dicts {name, success, elapsed}
    total_elapsed : total wall-clock seconds
    """
    print(f"\n{BOLD}{CYAN}{'=' * 62}")
    print("  PIPELINE SUMMARY")
    print(f"{'=' * 62}{RESET}")

    all_passed = all(r["success"] for r in results)

    for r in results:
        icon    = f"{GREEN}PASS{RESET}" if r["success"] else f"{RED}FAIL{RESET}"
        elapsed = f"{r['elapsed']:>6.1f}s"
        print(f"  {icon}  {elapsed}  {r['name']}")

    print(f"\n{BOLD}  Total runtime : {total_elapsed:.1f}s{RESET}")

    if all_passed:
        print(f"{GREEN}{BOLD}  All stages passed.{RESET}")
        logger.info("Pipeline completed successfully in %.1fs.", total_elapsed)
    else:
        failed = [r["name"] for r in results if not r["success"]]
        print(f"{RED}{BOLD}  Failed stages: {', '.join(failed)}{RESET}")
        print(f"{DIM}  Full log: {_log_path}{RESET}")
        logger.warning("Pipeline finished with failures: %s", failed)

    print(f"\n{CYAN}{'=' * 62}{RESET}\n")


# ======================================================================
# ARGUMENT PARSER
# ======================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "python main.py",
        description = "Market Pulse Analytics Pipeline",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = (
            "Examples:\n"
            "  python main.py                    # full pipeline (run daily)\n"
            "  python main.py --skip-scraper     # reuse today's data, rebuild dashboard\n"
            "  python main.py --dashboard-only   # rebuild HTML only\n"
            "  python main.py --force-rebuild    # force Parquet + metrics rebuild\n"
            "  python main.py --no-browser       # skip auto-opening browser\n"
        ),
    )
    parser.add_argument(
        "--skip-scraper",
        action = "store_true",
        help   = "Skip Yahoo Finance scraping. Reuse existing data.",
    )
    parser.add_argument(
        "--dashboard-only",
        action = "store_true",
        help   = "Skip all data stages. Rebuild dashboard HTML only.",
    )
    parser.add_argument(
        "--force-rebuild",
        action = "store_true",
        help   = "Force full Parquet and historical metrics cache rebuild.",
    )
    parser.add_argument(
        "--no-browser",
        action = "store_true",
        help   = "Do not open the dashboard in a browser on completion.",
    )
    parser.add_argument(
        "--no-historical-fetch",
        action="store_true",
        help="Skip OHLCV + FRED macro fetch (runs by default).",
    )
    parser.add_argument(
        "--fred-key",
        type=str, default="",
        metavar="KEY",
        help="FRED API key for macro data (or set FRED_API_KEY env variable).",
    )
    return parser


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    args       = _build_parser().parse_args()
    t_pipeline = time.perf_counter()
    results    = []

    # -- Banner + startup info ---------------------------------------
    print(_BANNER)
    mode = (
        "dashboard-only" if args.dashboard_only else
        "skip-scraper"   if args.skip_scraper   else
        "full pipeline"
    )
    print(f"\n  Started  : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print(f"  Mode     : {BOLD}{mode}{RESET}")
    print(f"  Log      : {_log_path}")
    logger.info("Pipeline started | mode=%s | args=%s", mode, vars(args))

    # -- Pre-flight validation ---------------------------------------
    from validators import run_preflight
    preflight_ok = run_preflight(
        skip_scraper   = args.skip_scraper,
        dashboard_only = args.dashboard_only,
    )
    if not preflight_ok:
        logger.critical("Pre-flight validation failed. Aborting.")
        sys.exit(1)

    # -- Stage 1: Scrape / EDA / Sentiment ---------------------------
    # -- Auto-backfill: fill missing price days + sparse sentiment ------
    if not args.dashboard_only:
        success, elapsed = run_stage(
            "Auto-backfill -- Gap Detection & Fill",
            stage_auto_backfill,
            critical=False,
        )
        results.append({"name": "Auto-backfill", "success": success, "elapsed": elapsed})

    # -- Stage 0: Historical Data Fetch (runs by default, skip with --no-historical-fetch)
    if not getattr(args, 'no_historical_fetch', False) and not args.dashboard_only:
        ok, t = run_stage(
            "Stage 0 -- Historical OHLCV + Macro Fetch",
            lambda: stage_historical_fetch(
                fred_api_key=getattr(args, 'fred_key', '')
            ),
            critical=False,
        )
        results.append({"name": "Historical Fetch", "success": ok, "elapsed": t})

    if not args.dashboard_only:
        if not args.skip_scraper:
            success, elapsed = run_stage(
                "Stage 1 -- Scrape / EDA / Sentiment",
                stage_scrape,
                critical = True,
            )
            results.append({
                "name":    "Scrape + EDA + Sentiment",
                "success": success,
                "elapsed": elapsed,
            })
        else:
            print(f"\n{YELLOW}  Scraper skipped -- running EDA to refresh stats.{RESET}")
            logger.info("Scraper skipped. Running EDA independently.")
            success, elapsed = run_stage(
                "Stage 1b -- EDA (standalone)",
                stage_eda_standalone,
                critical = True,
            )
            results.append({
                "name":    "EDA (standalone)",
                "success": success,
                "elapsed": elapsed,
            })

    # -- Stage 2: Sentiment validation / refresh ----------------------
    if not args.dashboard_only:
        success, elapsed = run_stage(
            "Stage 2 -- Sentiment Intelligence Input",
            stage_sentiment_refresh,
            critical = False,
        )
        results.append({
            "name":    "Sentiment Input",
            "success": success,
            "elapsed": elapsed,
        })

    # -- Stage 3: Historical Metrics ---------------------------------
    if not args.dashboard_only:
        _force = args.force_rebuild

        success, elapsed = run_stage(
            f"Stage 3 -- Historical Metrics {'(force rebuild)' if _force else '(incremental)'}",
            lambda: stage_historical_metrics(force=_force),
            critical = False,   # non-critical: dashboard can use cached data
        )
        results.append({
            "name":    "Historical Metrics",
            "success": success,
            "elapsed": elapsed,
        })

    # -- Stages 4-8: Intelligence pipeline ---------------------------
    if not args.dashboard_only:
        intelligence_stages = [
            ("Stage 4 -- Technical Indicators", "Technical Indicators", stage_technical_indicators),
            ("Stage 5 -- Forecasting", "Forecasting", stage_forecasting),
            ("Stage 6 -- Signal Engine", "Signal Engine", stage_signal_engine),
            ("Stage 7 -- Anomaly Detection", "Anomaly Detection", stage_anomaly_detection),
            ("Stage 8 -- AI Narratives", "AI Narratives", stage_ai_narratives),
        ]
        for stage_label, result_label, stage_fn in intelligence_stages:
            success, elapsed = run_stage(stage_label, stage_fn, critical=False)
            results.append({
                "name":    result_label,
                "success": success,
                "elapsed": elapsed,
            })

    # -- Stage 9: Dashboard Build ------------------------------------
    success, elapsed = run_stage(
        "Stage 9 -- Dashboard Build",
        stage_dashboard,
        critical = True,
    )
    results.append({
        "name":    "Dashboard Build",
        "success": success,
        "elapsed": elapsed,
    })

    # -- Open browser ------------------------------------------------
    if success and not args.no_browser:
        dashboard_path = os.path.abspath(PATHS.DASHBOARD_HTML)
        if os.path.exists(dashboard_path):
            webbrowser.open(f"file:///{dashboard_path}")
            print(f"\n{GREEN}  Dashboard opened in your browser.{RESET}")
            logger.info("Opened dashboard: %s", dashboard_path)
        else:
            print(f"\n{YELLOW}  Dashboard HTML not found at: {dashboard_path}{RESET}")

    # -- Summary -----------------------------------------------------
    total_elapsed = time.perf_counter() - t_pipeline
    print_summary(results, total_elapsed)


# ======================================================================
# FUTURE HOOKS (uncomment to activate)
# ======================================================================
#
# def stage_notify() -> None:
#     """Send Slack / email alert with today's top movers + sentiment."""
#
# def stage_ml_signals() -> None:
#     """Run ML momentum scoring and append signals to the dataset."""
#
# def stage_export_pdf() -> None:
#     """Export dashboard HTML to a PDF archive."""
#
# def stage_upload_s3() -> None:
#     """Push HTML and Parquet files to an S3 bucket for sharing."""
#
# ======================================================================

if __name__ == "__main__":
    main()