"""
backfill_date.py — Inject a missing trading day into the price dataset.

Fetches closing prices for all tracked symbols via yfinance and computes
Price / Change / % Change, then appends the rows to the Excel+Parquet
store and re-runs EDA so the rest of the pipeline sees a complete dataset.

Usage
-----
    python backfill_date.py 2025-05-30      # backfill a specific date
    python backfill_date.py                 # defaults to yesterday
"""

import sys
import datetime
import logging

import pandas as pd
import yfinance as yf

import data_loader
import eda

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def backfill(target_date_str: str, run_eda: bool = True) -> None:
    target_date = datetime.date.fromisoformat(target_date_str)

    # Load existing data — gives us the symbol list and company names
    df_existing = data_loader.load_price_data()
    if df_existing.empty:
        logger.error("No existing price data found. Run the scraper first.")
        return

    existing_dates = set(df_existing["Date"].dt.date.unique())
    if target_date in existing_dates:
        logger.info("Data for %s already exists — nothing to inject.", target_date_str)
        return

    symbols = df_existing["Symbol"].unique().tolist()
    company_names = df_existing.groupby("Symbol")["Company Name"].first().to_dict()
    logger.info("Symbols tracked: %d", len(symbols))

    # Fetch a 10-day window so we always capture at least one prior close
    fetch_start = (target_date - datetime.timedelta(days=10)).isoformat()
    fetch_end   = (target_date + datetime.timedelta(days=1)).isoformat()  # exclusive

    logger.info("Downloading yfinance data %s → %s ...", fetch_start, fetch_end)
    raw = yf.download(
        symbols,
        start=fetch_start,
        end=fetch_end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        logger.error(
            "yfinance returned no data for the requested window. "
            "%s may be a weekend or market holiday.", target_date_str
        )
        return

    # yfinance returns MultiIndex columns when >1 symbol; normalise to Close only
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        # Single symbol edge-case
        close = raw[["Close"]].rename(columns={"Close": symbols[0]})

    close.index = pd.to_datetime(close.index).date

    if target_date not in close.index:
        logger.error(
            "No market data found for %s — it is likely a weekend or holiday.",
            target_date_str,
        )
        return

    prior_dates = sorted(d for d in close.index if d < target_date)
    if not prior_dates:
        logger.error("No prior trading day found in the fetched window.")
        return

    prior_date   = prior_dates[-1]
    today_close  = close.loc[target_date]
    prior_close  = close.loc[prior_date]

    logger.info("Computing % Change vs prior close (%s) ...", prior_date)

    rows = []
    skipped = 0
    for sym in symbols:
        if sym not in close.columns:
            skipped += 1
            continue

        price = today_close.get(sym)
        prev  = prior_close.get(sym)

        if pd.isna(price) or pd.isna(prev) or prev == 0:
            skipped += 1
            continue

        price      = float(price)
        prev       = float(prev)
        change     = round(price - prev, 4)
        pct_change = round((change / prev) * 100, 2)

        rows.append({
            "Date":         target_date_str,
            "Symbol":       sym,
            "Company Name": company_names.get(sym, ""),
            "Price":        round(price, 4),
            "Change":       change,
            "% Change":     pct_change,
        })

    if not rows:
        logger.error("No valid rows could be built for %s.", target_date_str)
        return

    logger.info("Built %d rows (%d symbols skipped/missing).", len(rows), skipped)

    df_new      = pd.DataFrame(rows)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    data_loader.save_price_data(df_combined)

    n_days = df_combined["Date"].nunique()
    logger.info(
        "Saved. Dataset now has %d rows across %d days.",
        len(df_combined), n_days,
    )

    if run_eda:
        logger.info("Re-running EDA ...")
        try:
            eda.main()
            logger.info("EDA complete.")
        except Exception as exc:
            logger.warning("EDA failed: %s — run eda.py manually before rebuilding the dashboard.", exc)

    print(f"\n  Done. Injected {len(rows)} rows for {target_date_str}.")
    print(f"  Dataset now covers {n_days} day(s).")
    print("  Run dashboard.py (or main.py) to rebuild the dashboard.\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        date_arg = sys.argv[1]
    else:
        date_arg = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
        logger.info("No date supplied — defaulting to yesterday (%s).", date_arg)

    backfill(date_arg)
