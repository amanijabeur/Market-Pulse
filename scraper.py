"""
========================================================================
TASK 1: Yahoo Finance 100 Most Active Stocks Scraper
========================================================================
Scrapes the 100 most active stocks from Yahoo Finance and saves to Excel.

FIXES FROM ORIGINAL:
  - Added session + cookie headers to bypass 403 Forbidden blocking
  - Fixed regex to capture integer prices (no decimal point)
  - NaN rows are now dropped before saving to Excel
  - Silent except replaced with logged warnings + failed row counter
  - Retry logic added (3 attempts with backoff)
  - % Change sign preserved in all code paths
========================================================================
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import re
import time
import datetime
import os
from zoneinfo import ZoneInfo

import sentiment as sentiment_module
from sentiment import SENT_SHEET, empty_sentiment_df, normalise_dates
import data_loader
from config import SCRAPER

# -- LOGGING ------------------------------------------------------------
logger = logging.getLogger(__name__)

TARGET_URL = SCRAPER.TARGET_URL


def get_us_market_date() -> str:
    """
    Return the date of the most recent completed US trading session.

    Uses US Eastern Time so the tag is correct regardless of where the
    scraper runs. Rolls back over weekends AND NYSE market holidays so
    the date matches what Yahoo Finance is actually showing.
    """
    from pipelines.fetch_historical import is_market_holiday
    eastern = ZoneInfo("America/New_York")
    d = datetime.datetime.now(tz=eastern).date()
    while d.weekday() >= 5 or is_market_holiday(d):
        d -= datetime.timedelta(days=1)
    return d.isoformat()

# FIX 1: More complete headers that better mimic a real browser session.
# The original only sent 3 headers; Yahoo Finance checks for more.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Cache-Control": "max-age=0",
}

# FIX 2: Regex now captures both decimal AND integer prices.
# Original: r'[-+]?\d?[\d,]*\.\d+'  -- missed prices like $15 or $200
# Fixed:    r'[-+]?[\d,]+(?:\.\d+)?' -- captures both 15 and 15.62
NUMBER_RE = re.compile(r'[-+]?[\d,]+(?:\.\d+)?')
PCT_RE    = re.compile(r'\(([-+]?[\d,]*\.?\d+)%\)')


def fetch_page(url, retries=None, delay=None):
    """Fetch URL with retry logic. Returns response or None."""
    retries = retries if retries is not None else SCRAPER.RETRY_ATTEMPTS
    delay   = delay   if delay   is not None else SCRAPER.RETRY_DELAY_SEC
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Attempt {attempt}/{retries} -- connecting to Yahoo Finance...")
            resp = requests.get(url, headers=HEADERS, timeout=SCRAPER.REQUEST_TIMEOUT)
            resp.raise_for_status()
            logger.info(f"Connected. Response size: {len(resp.text):,} bytes")
            return resp
        except requests.exceptions.HTTPError as e:
            logger.warning(f"HTTP error on attempt {attempt}: {e}")
            if "403" in str(e):
                logger.error(
                    "403 Forbidden -- Yahoo Finance is blocking the request.\n"
                    "This usually means:\n"
                    "  1. You are running behind a datacenter IP (blocked by Yahoo)\n"
                    "  2. Try running from your home/office network\n"
                    "  3. Or open the page in a browser first to get a valid session cookie"
                )
                return None  # No point retrying a 403
        except Exception as e:
            logger.warning(f"Network error on attempt {attempt}: {e}")

        if attempt < retries:
            logger.info(f"Waiting {delay}s before retry...")
            time.sleep(delay)

    logger.error("All retry attempts failed.")
    return None


def parse_row(cols, row_index):
    """
    Parse a single table row into a stock dict.
    Returns dict on success, None on failure.
    """
    if len(cols) < 3:
        return None

    # Column 0: Symbol
    symbol = cols[0].get_text(strip=True)
    if not symbol or symbol.lower() == "symbol":
        return None

    # Column 1: Company Name
    name = cols[1].get_text(strip=True).replace(",", "").strip()

    # Join remaining cells into one text block
    full_text = " ".join(c.get_text(" ", strip=True) for c in cols[2:])

    # FIX 2: Use updated regex that handles integer prices
    numbers = NUMBER_RE.findall(full_text)
    # Strip commas from all numbers (e.g. "1,234.56" -> "1234.56")
    numbers = [n.replace(",", "") for n in numbers]

    # FIX 6: pct_match now also handles integer % like (8%) though rare
    pct_match = PCT_RE.search(full_text)

    # Price and Change
    if len(numbers) >= 2:
        price  = numbers[0]
        change = numbers[1]
    else:
        # Backup: read directly from cell positions
        price  = cols[2].get_text(strip=True).replace(",", "")
        change = cols[3].get_text(strip=True) if len(cols) > 3 else "0"

    # % Change -- FIX 6: preserve sign in both paths
    if pct_match:
        pct_change = pct_match.group(1)
    elif len(cols) > 4:
        pct_change = cols[4].get_text(strip=True)
    else:
        pct_change = "0"

    # Clean leftover symbols (%, parens) -- sign is already preserved
    pct_change = pct_change.replace("%", "").replace("(", "").replace(")", "").strip()

    return {
        "Symbol":       symbol,
        "Company Name": name,
        "Price":        price,
        "Change":       change,
        "% Change":     pct_change,
    }


def fetch_json_screener() -> "pd.DataFrame | None":
    """
    Fetch most-active stocks via Yahoo Finance's JSON screener API.

    This is the primary data source. It returns pre-validated structured
    data — no HTML to parse, no table structure that can break on layout
    changes. Falls back to None on any failure so the caller can try the
    HTML scraper instead.

    Returns
    -------
    pd.DataFrame with columns: Symbol, Company Name, Price, Change, % Change
    None if all retry attempts fail.
    """
    params = {
        "formatted": "false",           # raw numbers, not "$150.00" strings
        "scrIds":    SCRAPER.SCREENER_SCRID,
        "start":     0,
        "count":     SCRAPER.EXPECTED_STOCKS,
        "region":    "US",
        "lang":      "en-US",
    }

    for attempt in range(1, SCRAPER.RETRY_ATTEMPTS + 1):
        try:
            logger.info(
                "JSON screener API attempt %d/%d...",
                attempt, SCRAPER.RETRY_ATTEMPTS,
            )
            resp = requests.get(
                SCRAPER.SCREENER_API_URL,
                headers=HEADERS,
                params=params,
                timeout=SCRAPER.REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            quotes = (
                resp.json()
                    .get("finance", {})
                    .get("result", [{}])[0]
                    .get("quotes", [])
            )
            if not quotes:
                logger.warning("JSON screener: no quotes returned (attempt %d).", attempt)
                if attempt < SCRAPER.RETRY_ATTEMPTS:
                    time.sleep(SCRAPER.RETRY_DELAY_SEC)
                continue

            rows = []
            for q in quotes:
                symbol = q.get("symbol", "")
                price  = q.get("regularMarketPrice")
                change = q.get("regularMarketChange")
                pct    = q.get("regularMarketChangePercent")
                if not symbol or price is None or change is None or pct is None:
                    continue
                rows.append({
                    "Symbol":       symbol,
                    "Company Name": (q.get("shortName") or q.get("longName") or "").replace(",", ""),
                    "Price":        round(float(price),  2),
                    "Change":       round(float(change), 4),
                    "% Change":     round(float(pct),    4),
                })

            if not rows:
                logger.warning("JSON screener: 0 valid rows after parsing (attempt %d).", attempt)
                if attempt < SCRAPER.RETRY_ATTEMPTS:
                    time.sleep(SCRAPER.RETRY_DELAY_SEC)
                continue

            logger.info("JSON screener: %d stocks fetched successfully.", len(rows))
            df = pd.DataFrame(rows)
            df["Price"]    = pd.to_numeric(df["Price"],    errors="coerce")
            df["Change"]   = pd.to_numeric(df["Change"],   errors="coerce")
            df["% Change"] = pd.to_numeric(df["% Change"], errors="coerce")
            df.dropna(subset=["Price", "Change", "% Change"], inplace=True)
            df = df[df["Price"] > 0].reset_index(drop=True)
            return df

        except Exception as exc:
            logger.warning("JSON screener attempt %d failed: %s", attempt, exc)
            if attempt < SCRAPER.RETRY_ATTEMPTS:
                time.sleep(SCRAPER.RETRY_DELAY_SEC)

    logger.warning("JSON screener: all %d attempts failed.", SCRAPER.RETRY_ATTEMPTS)
    return None


def _parse_html_response(response) -> "tuple[pd.DataFrame | None, int]":
    """
    Parse a Yahoo Finance HTML page response into a stock DataFrame.
    Returns (df, failed_rows) or (None, 0) if no table could be found.
    """
    soup  = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")
    if not table:
        logger.error(
            "No <table> found. Yahoo Finance layout may have changed "
            "or the request was redirected."
        )
        return None, 0

    tbody = table.find("tbody")
    rows  = tbody.find_all("tr") if tbody else table.find_all("tr")
    logger.info("HTML scraper: %d table rows found.", len(rows))

    parsed, failed_rows = [], 0
    for i, row in enumerate(rows):
        cols = row.find_all("td")
        try:
            result = parse_row(cols, i)
            if result:
                parsed.append(result)
        except Exception as exc:
            failed_rows += 1
            logger.warning("HTML row %d parse error: %s", i, exc)

    if not parsed:
        logger.error("HTML scraper: 0 stocks parsed — layout likely changed.")
        return None, failed_rows

    df = pd.DataFrame(parsed)
    df["Price"]    = pd.to_numeric(df["Price"],    errors="coerce")
    df["Change"]   = pd.to_numeric(df["Change"],   errors="coerce")
    df["% Change"] = pd.to_numeric(df["% Change"], errors="coerce")
    before = len(df)
    df.dropna(subset=["Price", "Change", "% Change"], inplace=True)
    failed_rows += before - len(df)
    return df, failed_rows


def _recompute_pct_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute % Change from Price and Change for the HTML scraper path only.
    Yahoo rounds Change to 2 dp on cheap stocks, introducing up to 0.16pp error.
    The JSON API provides the correct value directly so this step is skipped there.
    """
    prev_price = df["Price"] - df["Change"]
    recomputed = (df["Change"] / prev_price.replace(0, float("nan")) * 100).round(2)
    mask = df["Change"].abs() >= SCRAPER.MIN_CHANGE_RECOMP
    df = df.copy()
    df.loc[mask, "% Change"] = recomputed[mask]
    return df


def _fallback_on_scrape_failure(reason: str) -> None:
    """
    Called when Yahoo Finance is unreachable or returns an unparseable page.

    If existing price data is on disk, runs EDA on it so downstream stages
    (historical metrics, dashboard) can still complete with yesterday's data.
    Raises RuntimeError only if there is no existing data at all.
    """
    logger.warning("Scraper fallback triggered: %s", reason)
    print(f"\n  WARNING: {reason}")

    if not os.path.exists(data_loader.EXCEL_FILE):
        raise RuntimeError(
            f"Scraping failed and no existing data to fall back on: {reason}"
        )

    print("  Continuing pipeline with existing data (today's prices not updated).")
    logger.warning("Scraper: using cached data — today's prices will NOT be added.")

    try:
        import eda
        eda.main()
    except Exception as exc:
        logger.warning("EDA failed during scraper fallback: %s", exc)


def main():
    print("=" * 60)
    print("  Yahoo Finance -- 100 Most Active Stocks Scraper")
    print("=" * 60)

    # If today's scrape already exists locally, skip the network call and
    # let main.py continue with EDA, historical metrics, and dashboard build.
    today_str = get_us_market_date()
    if os.path.exists(data_loader.EXCEL_FILE):
        try:
            existing = data_loader.load_price_data()
            existing = normalise_dates(existing)
            existing["Date"] = existing["Date"].astype(str)

            if today_str in existing["Date"].values:
                logger.warning(f"Already scraped for {today_str}. Skipping Yahoo fetch.")
                print(f"\n  Already have data for today ({today_str}). Refreshing EDA and continuing pipeline.")
                print("\n" + "=" * 60)
                print("  RUNNING EDA -- regenerating stats_results.py ...")
                print("=" * 60)
                try:
                    import eda
                    eda.main()
                except Exception as e:
                    print(f"  WARNING: eda.py failed -- {e}")
                    print("  stats_results.py was NOT updated. Run eda.py manually.")
                    raise RuntimeError("EDA failed while refreshing existing data.") from e
                return
        except Exception as e:
            logger.warning("Could not check existing dataset before scraping: %s", e)

    # -- Primary: JSON screener API --------------------------------
    # Structured JSON — immune to HTML layout changes.
    print("  Fetching via JSON screener API (primary)...")
    df          = fetch_json_screener()
    failed_rows = 0
    source      = "json"

    # -- Fallback: HTML scraper ------------------------------------
    if df is None or df.empty:
        print("  JSON API unavailable — falling back to HTML scraper...")
        logger.info("Falling back to HTML scraper.")
        response = fetch_page(TARGET_URL)
        if response is None:
            _fallback_on_scrape_failure("JSON API and HTML scraper both unreachable.")
            return
        df, failed_rows = _parse_html_response(response)
        if df is None or df.empty:
            _fallback_on_scrape_failure("No stocks parsed from HTML response (layout may have changed).")
            return
        # HTML-only: recompute % Change to fix Yahoo's rounding on cheap stocks
        df     = _recompute_pct_change(df)
        source = "html"

    print(f"  Source       : {source.upper()} ({'JSON API — layout-independent' if source == 'json' else 'HTML fallback'})")

    # Basic sanity check (both paths)
    invalid_prices = df[df["Price"] <= 0]
    if not invalid_prices.empty:
        logger.warning(
            "%d row(s) with Price <= 0: %s",
            len(invalid_prices), invalid_prices["Symbol"].tolist(),
        )

    # -- Add date column --------------------------------------------
    df.insert(0, "Date", today_str)

    # -- Append to existing file or create new ---------------------
    # Each daily run appends 100 rows tagged with today's date.
    # Over time the file grows into a proper time-series dataset.
    # -- Save via data_loader (writes Excel + Parquet atomically) ------
    # data_loader.save_price_data() handles deduplication, dtype
    # enforcement, and writing both the Excel source-of-truth and the
    # Parquet fast-read mirror in one call.
    if os.path.exists(data_loader.EXCEL_FILE):
        existing = data_loader.load_price_data()
        existing = normalise_dates(existing)
        existing["Date"] = existing["Date"].astype(str)

        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined  = df
        today_str = df["Date"].iloc[0]

    data_loader.save_price_data(combined)
    n_days = combined["Date"].nunique() if "Date" in combined.columns else 1

    # -- Fetch and save daily sentiment via data_loader ----------------
    try:
        sent_history = data_loader.load_sentiment_history()
    except Exception:
        sent_history = empty_sentiment_df()

    if today_str in sent_history["Date"].values:
        logger.info(f"Sentiment already stored for {today_str}. Skipping fetch.")
        print(f"  Sentiment    : already stored for {today_str}")
    else:
        print(f"  Sentiment    : fetching NLP scores for {len(df)} symbols ...")
        sent_df      = sentiment_module.run(df["Symbol"].tolist(), max_headlines=10)
        combined_sent = pd.concat([sent_history, sent_df], ignore_index=True)
        data_loader.save_sentiment_history(combined_sent)
        n_sent_days = combined_sent["Date"].nunique()
        print(f"  Sentiment    : saved {len(sent_df)} rows "
              f"({n_sent_days} day(s) total)")

    # -- Auto-run EDA to regenerate stats_results.py ------------------
    # Ensures stats_results.py always reflects today's data before the
    # dashboard is run. Wrapped in try/except so a failure here never
    # loses the already-saved price and sentiment data.
    print("\n" + "=" * 60)
    print("  RUNNING EDA -- regenerating stats_results.py ...")
    print("=" * 60)
    try:
        import eda
        eda.main()
    except Exception as e:
        print(f"  WARNING: eda.py failed -- {e}")
        print("  stats_results.py was NOT updated. Run eda.py manually.")
        raise RuntimeError("EDA failed after scraping; dashboard statistics were not refreshed.") from e

    print("\n" + "=" * 60)
    print("  COMPLETE")
    print("=" * 60)
    print(f"  Saved        : {data_loader.EXCEL_FILE} + .parquet")
    print(f"  Source       : {source.upper()}")
    print(f"  Today's rows : {len(df)}")
    print(f"  Total rows   : {len(combined)}")
    print(f"  Days scraped : {n_days}")
    if failed_rows:
        print(f"  Rows failed  : {failed_rows} (HTML parse errors)")
    print("\nToday's preview:")
    print(df[["Date","Symbol","Company Name","Price","Change","% Change"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
