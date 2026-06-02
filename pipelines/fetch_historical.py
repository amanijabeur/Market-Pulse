"""
========================================================================
pipelines/fetch_historical.py — Historical Data Fetcher
========================================================================
Fetches OHLCV price data (yfinance) and macroeconomic series (FRED) and
persists them to the local parquet store managed by data_loader.py.

Architecture
------------
- Pure I/O layer. No analytics, no indicators, no plotting.
- All paths come from config.HISTORICAL_PATHS.
- All persistence goes through data_loader.save_ohlcv() and
  data_loader.save_macro() so the in-process cache is always consistent.
- Incremental by default: if a symbol already has local data, only the
  missing tail is fetched and appended. Full re-fetch if force=True.
- Failure of one ticker never aborts the rest (non-critical pattern
  matching the existing scraper.py and signal_engine.py philosophy).

Integration with existing system
---------------------------------
- main.py calls run_historical_fetch() as an optional Stage 0 that
  runs before the daily scrape stages.
- The ticker list defaults to the current day's 100 scraped symbols
  so OHLCV history is always aligned with what the screener shows.
- data_loader.list_available_ohlcv() lets dashboard and analytics
  modules know which symbols have OHLCV without loading files.

Public API
----------
  fetch_ohlcv(symbol, period, interval, force) -> pd.DataFrame
  fetch_fred(series_alias, years_back, force)  -> pd.DataFrame
  run_historical_fetch(symbols, force)         -> dict
========================================================================
"""

import datetime
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd

from config import HIST_FETCH, FRED_SERIES, FRED_CFG
import data_loader

# Sentinel file: records the last US market date on which OHLCV was fully updated.
# If it matches today's market date, the entire fetch stage is skipped.
_SENTINEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".ohlcv_last_fetch")

# Failed-ticker tracker: symbols that fail repeatedly are skipped automatically.
_FAILED_TICKERS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "failed_tickers.json",
)
_MAX_CONSECUTIVE_FAILURES = 3


def _load_failed_tickers() -> dict:
    import json
    try:
        if os.path.exists(_FAILED_TICKERS_PATH):
            with open(_FAILED_TICKERS_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_failed_tickers(data: dict) -> None:
    import json
    os.makedirs(os.path.dirname(_FAILED_TICKERS_PATH), exist_ok=True)
    try:
        with open(_FAILED_TICKERS_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.warning("Could not save failed_tickers.json: %s", exc)


def _record_fetch_result(sym: str, success: bool, tracker: dict) -> dict:
    """Increment failure count on miss, clear it on success."""
    if success:
        tracker.pop(sym, None)
    else:
        entry = tracker.get(sym, {"consecutive_failures": 0})
        entry["consecutive_failures"] = entry.get("consecutive_failures", 0) + 1
        entry["last_failure"] = pd.Timestamp.today().strftime("%Y-%m-%d")
        tracker[sym] = entry
    return tracker


# ── Algorithmic US market holiday computation ─────────────────────────
# Replaces a hardcoded list with rules that work for any year automatically.
# Covers all 10 standard NYSE holidays using US federal "observed" rules
# (Saturday → prior Friday, Sunday → following Monday).
# One-off closures (national days of mourning etc.) are not modelled here
# as they can't be computed in advance.

import calendar as _cal

_HOLIDAY_CACHE: dict = {}   # {year: frozenset of ISO date strings}


def _us_market_holidays(year: int) -> frozenset:
    """Compute NYSE market holidays for any given year."""

    def _observed(d: datetime.date) -> datetime.date:
        if d.weekday() == 5:                          # Saturday → Friday
            return d - datetime.timedelta(days=1)
        if d.weekday() == 6:                          # Sunday → Monday
            return d + datetime.timedelta(days=1)
        return d

    def _nth_weekday(y: int, month: int, weekday: int, n: int) -> datetime.date:
        """Return the nth occurrence of weekday (0=Mon) in the given month."""
        count = 0
        for day in range(1, 32):
            try:
                d = datetime.date(y, month, day)
            except ValueError:
                break
            if d.weekday() == weekday:
                count += 1
                if count == n:
                    return d

    def _last_weekday(y: int, month: int, weekday: int) -> datetime.date:
        """Return the last occurrence of weekday (0=Mon) in the given month."""
        last = _cal.monthrange(y, month)[1]
        for day in range(last, 0, -1):
            d = datetime.date(y, month, day)
            if d.weekday() == weekday:
                return d

    def _easter(y: int) -> datetime.date:
        """Compute Easter Sunday via the Anonymous Gregorian algorithm."""
        a = y % 19
        b, c = y // 100, y % 100
        d, e = b // 4, b % 4
        f    = (b + 8) // 25
        g    = (b - f + 1) // 3
        h    = (19 * a + b - d - g + 15) % 30
        i, k = c // 4, c % 4
        l    = (32 + 2 * e + 2 * i - h - k) % 7
        m    = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day   = ((h + l - 7 * m + 114) % 31) + 1
        return datetime.date(y, month, day)

    MO, TH = 0, 3   # weekday constants

    holidays = {
        _observed(datetime.date(year, 1,  1)),             # New Year's Day
        _nth_weekday(year, 1,  MO, 3),                     # MLK Day (3rd Mon Jan)
        _nth_weekday(year, 2,  MO, 3),                     # Presidents' Day (3rd Mon Feb)
        _easter(year) - datetime.timedelta(days=2),         # Good Friday
        _last_weekday(year, 5, MO),                        # Memorial Day (last Mon May)
        _observed(datetime.date(year, 6,  19)),             # Juneteenth
        _observed(datetime.date(year, 7,  4)),              # Independence Day
        _nth_weekday(year, 9,  MO, 1),                     # Labor Day (1st Mon Sep)
        _nth_weekday(year, 11, TH, 4),                     # Thanksgiving (4th Thu Nov)
        _observed(datetime.date(year, 12, 25)),             # Christmas
    }
    return frozenset(d.isoformat() for d in holidays if d is not None)


def is_market_holiday(d: datetime.date) -> bool:
    """Return True if d is an NYSE market holiday. Results are cached per year."""
    if d.year not in _HOLIDAY_CACHE:
        _HOLIDAY_CACHE[d.year] = _us_market_holidays(d.year)
    return d.isoformat() in _HOLIDAY_CACHE[d.year]


def _last_closed_trading_day() -> str:
    """
    Return the most recent US trading day whose session has CLOSED (YYYY-MM-DD).

    Before 4 PM ET on a weekday the market is still open (or hasn't opened),
    so the last *closed* session is the previous trading day. After 4 PM ET
    today's session counts. Weekends and NYSE holidays are skipped.
    Holidays are computed algorithmically for any year — no hardcoded list.
    """
    from zoneinfo import ZoneInfo
    now = pd.Timestamp.now(tz=ZoneInfo("America/New_York"))
    d   = now.date()
    if now.hour < 16:
        d -= datetime.timedelta(days=1)
    while d.weekday() >= 5 or is_market_holiday(d):
        d -= datetime.timedelta(days=1)
    return d.isoformat()


def _already_fetched_today() -> bool:
    """Return True if OHLCV was already fully fetched for the last closed trading day."""
    if not os.path.exists(_SENTINEL):
        return False
    try:
        with open(_SENTINEL) as f:
            return f.read().strip() == _last_closed_trading_day()
    except Exception:
        return False


def _mark_fetched_today() -> None:
    """Write the last closed trading day to the sentinel file."""
    try:
        with open(_SENTINEL, "w") as f:
            f.write(_last_closed_trading_day())
    except Exception as exc:
        logger.warning("Could not write OHLCV sentinel: %s", exc)

logger = logging.getLogger(__name__)

# Required OHLCV columns from yfinance (column names vary by version)
_OHLCV_RENAME: dict[str, str] = {
    "Open": "Open", "High": "High", "Low": "Low",
    "Close": "Close", "Volume": "Volume",
    # yfinance sometimes returns "Adj Close" — map to Close if needed
}


def _safe_import_yfinance():
    """Import yfinance with a clear error if not installed."""
    try:
        import yfinance as yf
        return yf
    except ImportError:
        raise ImportError(
            "yfinance is required for historical fetching. "
            "Install with: pip install yfinance"
        )


def _safe_import_fredapi():
    """Import fredapi with a clear error if not installed."""
    try:
        from fredapi import Fred
        return Fred
    except ImportError:
        raise ImportError(
            "fredapi is required for macro data fetching. "
            "Install with: pip install fredapi"
        )


def fetch_ohlcv(
    symbol: str,
    period: str = None,
    interval: str = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Fetch OHLCV history for a single ticker via yfinance.

    Incremental strategy
    --------------------
    1. Load existing local data.
    2. If the local data covers through yesterday and force=False,
       return the cached data immediately — no network call.
    3. If local data exists but is stale, fetch only the missing tail
       (start = last_date + 1 day).
    4. If no local data exists, fetch the full period.
    5. Merge, deduplicate, save.

    Parameters
    ----------
    symbol   : ticker string, e.g. "AAPL"
    period   : yfinance period (default: HIST_FETCH.DEFAULT_PERIOD = "2y")
    interval : bar interval (default: HIST_FETCH.DEFAULT_INTERVAL = "1d")
    force    : always re-fetch the full period

    Returns
    -------
    pd.DataFrame with columns: Date, Open, High, Low, Close, Volume
    Empty DataFrame if the fetch fails.
    """
    yf      = _safe_import_yfinance()
    period   = period   or HIST_FETCH.DEFAULT_PERIOD
    interval = interval or HIST_FETCH.DEFAULT_INTERVAL
    sym_up   = symbol.upper()

    # -- Check existing cache -----------------------------------------
    existing = data_loader.load_ohlcv(sym_up)
    # Strip tz from existing data — older parquet files may carry tz-aware dates
    if not existing.empty and existing["Date"].dt.tz is not None:
        existing = existing.copy()
        existing["Date"] = existing["Date"].dt.tz_convert(None)

    if not force and not existing.empty:
        last_date = pd.to_datetime(existing["Date"].max()).normalize()
        last_closed = pd.Timestamp(_last_closed_trading_day()).normalize()
        if last_date >= last_closed:
            logger.info("fetch_historical: %s OHLCV is current. Skipping fetch.", sym_up)
            return existing

    # -- Check for unadjusted corporate actions in cached data ---------
    # If a split or reverse-split is detected, discard the cache and force
    # a full re-fetch so yfinance returns retroactively adjusted prices.
    if not force and not existing.empty:
        from pipelines.preprocess import detect_splits
        if detect_splits(existing):
            logger.warning(
                "fetch_historical: %s — unadjusted split in cache; forcing full re-fetch.",
                sym_up,
            )
            force    = True
            existing = pd.DataFrame()

    # -- Determine fetch window ---------------------------------------
    start_date: Optional[str] = None
    if not force and not existing.empty:
        last_date  = pd.to_datetime(existing["Date"].max())
        start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info("fetch_historical: incremental fetch for %s from %s.", sym_up, start_date)
    else:
        logger.info("fetch_historical: full fetch for %s (period=%s).", sym_up, period)

    # -- Fetch with retry ---------------------------------------------
    for attempt in range(1, HIST_FETCH.RETRY_ATTEMPTS + 1):
        try:
            ticker = yf.Ticker(sym_up)
            if start_date:
                raw = ticker.history(start=start_date, interval=interval, auto_adjust=True)
            else:
                raw = ticker.history(period=period, interval=interval, auto_adjust=True)

            if raw is None or raw.empty:
                logger.warning(
                    "fetch_historical: %s returned no data (attempt %d/%d).",
                    sym_up, attempt, HIST_FETCH.RETRY_ATTEMPTS,
                )
                if attempt < HIST_FETCH.RETRY_ATTEMPTS:
                    time.sleep(HIST_FETCH.RETRY_DELAY_SEC)
                continue

            # Normalise: reset index so Date becomes a column
            df = raw.reset_index()
            df = df.rename(columns={"index": "Date", "Datetime": "Date"})
            _dates = pd.to_datetime(df["Date"], errors="coerce")
            if _dates.dt.tz is not None:
                _dates = _dates.dt.tz_convert(None)
            df["Date"] = _dates.dt.normalize()

            # Keep only standard OHLCV columns
            keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"]
                    if c in df.columns]
            if "Adj Close" in df.columns and "Close" not in df.columns:
                df = df.rename(columns={"Adj Close": "Close"})
                keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
            df = df[[c for c in keep if c in df.columns]]

            # Merge with existing
            if not existing.empty and not force:
                df = pd.concat([existing, df], ignore_index=True)

            df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
            data_loader.save_ohlcv(df, sym_up)
            logger.info(
                "fetch_historical: %s — %d total rows saved.", sym_up, len(df)
            )
            return df

        except Exception as exc:
            logger.warning(
                "fetch_historical: %s fetch error (attempt %d/%d): %s",
                sym_up, attempt, HIST_FETCH.RETRY_ATTEMPTS, exc,
            )
            if attempt < HIST_FETCH.RETRY_ATTEMPTS:
                time.sleep(HIST_FETCH.RETRY_DELAY_SEC)

    logger.error("fetch_historical: all retry attempts failed for %s.", sym_up)
    return existing if not existing.empty else pd.DataFrame()


def fetch_fred(
    series_alias: str,
    years_back: int = None,
    api_key: str = "",
    force: bool = False,
) -> pd.DataFrame:
    """
    Fetch a FRED macroeconomic series via fredapi.

    The series_alias must be a key in config.FRED_SERIES.
    The actual FRED series ID is looked up from that mapping.

    Parameters
    ----------
    series_alias : local alias (e.g. "fed_funds_rate", "vix")
    years_back   : how many years of history to fetch
                   (default: FRED_CFG.DEFAULT_PERIOD_YEARS = 5)
    api_key      : FRED API key (free at fred.stlouisfed.org)
                   If empty, attempts to read FRED_API_KEY env variable.
    force        : re-fetch even if local data exists

    Returns
    -------
    pd.DataFrame with columns: Date, value, series_id
    """
    import os as _os
    Fred      = _safe_import_fredapi()
    years_back = years_back or FRED_CFG.DEFAULT_PERIOD_YEARS

    if series_alias not in FRED_SERIES:
        logger.warning(
            "fetch_historical: unknown FRED alias '%s'. "
            "Available: %s", series_alias, list(FRED_SERIES.keys()),
        )
        return pd.DataFrame(columns=["Date", "value", "series_id"])

    series_id = FRED_SERIES[series_alias]

    # -- Check existing cache -----------------------------------------
    existing = data_loader.load_macro(series_alias)
    if not force and not existing.empty:
        last_date = pd.to_datetime(existing["Date"].max()).normalize()
        today     = pd.Timestamp.today().normalize()
        if last_date >= today - pd.Timedelta(days=7):
            logger.info(
                "fetch_historical: macro '%s' is current. Skipping.", series_alias
            )
            return existing

    # -- Resolve API key -----------------------------------------------
    key = api_key or _os.environ.get("FRED_API_KEY", "")
    if not key:
        logger.warning(
            "fetch_historical: FRED_API_KEY not set. "
            "Set the environment variable or pass api_key to fetch_fred()."
        )
        return existing if not existing.empty else pd.DataFrame(
            columns=["Date", "value", "series_id"]
        )

    # -- Fetch ---------------------------------------------------------
    start = (pd.Timestamp.today() - pd.DateOffset(years=years_back)).strftime("%Y-%m-%d")
    try:
        fred = Fred(api_key=key)
        raw  = fred.get_series(series_id, observation_start=start)
        if raw is None or raw.empty:
            logger.warning("fetch_historical: FRED '%s' returned no data.", series_id)
            return existing if not existing.empty else pd.DataFrame(
                columns=["Date", "value", "series_id"]
            )

        df = pd.DataFrame({
            "Date":      pd.to_datetime(raw.index),
            "value":     raw.values.astype(float),
            "series_id": series_id,
        }).dropna(subset=["value"]).sort_values("Date").reset_index(drop=True)

        data_loader.save_macro(df, series_alias)
        time.sleep(FRED_CFG.FETCH_DELAY_SEC)
        logger.info(
            "fetch_historical: FRED '%s' (%s) — %d rows saved.",
            series_alias, series_id, len(df),
        )
        return df

    except Exception as exc:
        logger.error(
            "fetch_historical: FRED fetch failed for '%s': %s", series_alias, exc
        )
        return existing if not existing.empty else pd.DataFrame(
            columns=["Date", "value", "series_id"]
        )


def run_historical_fetch(
    symbols: list[str] = None,
    fetch_macro: bool = True,
    fred_api_key: str = "",
    force: bool = False,
) -> dict:
    """
    Orchestrate a full historical data fetch run.

    Fetches OHLCV for all requested symbols and (optionally) all
    configured FRED macro series. Rate-limited between requests.
    A failure on any single symbol/series does not abort the run.

    Parameters
    ----------
    symbols      : list of ticker strings to fetch
                   (defaults to HIST_FETCH.DEFAULT_TICKERS)
    fetch_macro  : whether to also fetch FRED macro series
    fred_api_key : FRED API key (falls back to FRED_API_KEY env var)
    force        : re-fetch all data even if local cache is current

    Returns
    -------
    dict with keys:
        ohlcv_fetched  : list of successfully fetched symbols
        ohlcv_failed   : list of symbols that failed
        macro_fetched  : list of successfully fetched macro series
        macro_failed   : list of macro series that failed
        n_ohlcv_rows   : total OHLCV rows now in local store
        n_macro_rows   : total macro rows now in local store
    """
    symbols = symbols or list(HIST_FETCH.DEFAULT_TICKERS)

    # Skip the entire fetch if it already ran on today's market date
    if not force and _already_fetched_today():
        logger.info("fetch_historical: OHLCV already current for %s — skipping.", _last_closed_trading_day())
        return {
            "ohlcv_fetched": [], "ohlcv_failed": [],
            "macro_fetched": [], "macro_failed": [],
            "n_ohlcv_rows": 0,   "n_macro_rows": 0,
        }

    # Filter out symbols that have failed too many consecutive times
    failed_tracker = _load_failed_tickers()
    skipped: list[str] = []
    if not force:
        active_symbols, skipped = [], []
        for s in symbols:
            if failed_tracker.get(s.upper(), {}).get("consecutive_failures", 0) >= _MAX_CONSECUTIVE_FAILURES:
                skipped.append(s.upper())
            else:
                active_symbols.append(s)
        if skipped:
            logger.warning(
                "fetch_historical: skipping %d symbol(s) with %d+ consecutive failures: %s",
                len(skipped), _MAX_CONSECUTIVE_FAILURES, skipped,
            )
    else:
        active_symbols = symbols

    ohlcv_fetched: list[str] = []
    ohlcv_failed:  list[str] = []
    n_ohlcv_rows   = 0

    logger.info(
        "fetch_historical: starting OHLCV fetch for %d symbols.", len(active_symbols)
    )

    def _fetch_one(sym: str) -> tuple[str, pd.DataFrame]:
        return sym.upper(), fetch_ohlcv(sym, force=force)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_one, sym): sym for sym in active_symbols}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                sym_up, df = future.result()
                success = not df.empty
                failed_tracker = _record_fetch_result(sym_up, success, failed_tracker)
                if success:
                    ohlcv_fetched.append(sym_up)
                    n_ohlcv_rows += len(df)
                else:
                    ohlcv_failed.append(sym_up)
            except Exception as exc:
                logger.warning("fetch_historical: %s failed — %s", sym, exc)
                sym_up = sym.upper()
                failed_tracker = _record_fetch_result(sym_up, False, failed_tracker)
                ohlcv_failed.append(sym_up)

    _save_failed_tickers(failed_tracker)

    macro_fetched: list[str] = []
    macro_failed:  list[str] = []
    n_macro_rows   = 0

    if fetch_macro:
        logger.info(
            "fetch_historical: fetching %d FRED series.", len(FRED_SERIES)
        )
        for alias in FRED_SERIES:
            try:
                df = fetch_fred(alias, api_key=fred_api_key, force=force)
                if not df.empty:
                    macro_fetched.append(alias)
                    n_macro_rows += len(df)
                else:
                    macro_failed.append(alias)
            except Exception as exc:
                logger.warning("fetch_historical: FRED '%s' failed — %s", alias, exc)
                macro_failed.append(alias)

    _mark_fetched_today()
    logger.info(
        "fetch_historical complete: %d/%d OHLCV, %d/%d macro, %d skipped.",
        len(ohlcv_fetched), len(active_symbols),
        len(macro_fetched), len(FRED_SERIES),
        len(skipped),
    )

    return {
        "ohlcv_fetched":  ohlcv_fetched,
        "ohlcv_failed":   ohlcv_failed,
        "ohlcv_skipped":  skipped,
        "macro_fetched":  macro_fetched,
        "macro_failed":   macro_failed,
        "n_ohlcv_rows":   n_ohlcv_rows,
        "n_macro_rows":   n_macro_rows,
    }
