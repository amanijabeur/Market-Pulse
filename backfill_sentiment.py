"""
backfill_sentiment.py — Seed sentiment history with Finnhub historical data.

Fetches news for all tracked symbols via Finnhub's free API, scores each
headline with the shared NLP model, and appends the results to sentiment_history parquet.
Already-stored (Date, Symbol) pairs are skipped to avoid duplicates.

Get a free Finnhub API key at https://finnhub.io — no credit card needed.

Usage
-----
    python backfill_sentiment.py --key YOUR_KEY --from 2024-01-01
    python backfill_sentiment.py --key YOUR_KEY --from 2024-06-01 --to 2024-12-31

At 60 calls/minute (free tier) and 124 symbols, a full backfill takes
roughly 2-3 minutes.  Run once; your daily scraper handles new data.
"""

import argparse
import logging
import sys

import pandas as pd

import data_loader
import finnhub_sentiment
from finnhub_sentiment import FINNHUB_API_KEY

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical sentiment from Finnhub.")
    parser.add_argument("--key",  required=False, metavar="API_KEY",    help="Finnhub API key (default: FINNHUB_API_KEY from .env)")
    parser.add_argument("--from", required=True,  dest="from_date",     help="Start date YYYY-MM-DD")
    parser.add_argument("--to",   required=False, dest="to_date",       help="End date YYYY-MM-DD (default: today)", default=None)
    args = parser.parse_args()

    # Load existing symbol list from the price dataset
    df_price = data_loader.load_price_data()
    if df_price.empty:
        logger.error("No price data found. Run the scraper at least once first.")
        sys.exit(1)

    api_key = args.key or FINNHUB_API_KEY
    if not api_key:
        logger.error("No Finnhub API key found. Pass --key or set FINNHUB_API_KEY in .env")
        sys.exit(1)

    symbols = df_price["Symbol"].unique().tolist()
    logger.info("Symbols to backfill: %d", len(symbols))
    logger.info("Date range: %s → %s", args.from_date, args.to_date or "today")

    # Fetch from Finnhub
    new_df = finnhub_sentiment.backfill(symbols, args.from_date, api_key, args.to_date)

    if new_df.empty:
        logger.error("No data returned. Check your API key and date range.")
        sys.exit(1)

    # Merge with existing history — deduplicate on (Date, Symbol)
    existing = data_loader.load_sentiment_history()

    if not existing.empty:
        existing_keys = set(zip(existing["Date"].astype(str), existing["Symbol"].astype(str)))
        new_df = new_df[
            ~new_df.apply(lambda r: (str(r["Date"]), str(r["Symbol"])) in existing_keys, axis=1)
        ].copy()
        logger.info("New rows after dedup: %d", len(new_df))

    if new_df.empty:
        logger.info("All fetched rows already exist in history. Nothing to add.")
        return

    combined = pd.concat([existing, new_df], ignore_index=True)
    data_loader.save_sentiment_history(combined)

    print(f"\n  Done.")
    print(f"  New rows added    : {len(new_df)}")
    print(f"  Total history     : {len(combined)} rows across {combined['Date'].nunique()} days")
    print(f"  Symbols covered   : {combined['Symbol'].nunique()}")
    print("\n  Run main.py --skip-scraper to rebuild the dashboard with extended sentiment.\n")


if __name__ == "__main__":
    main()
