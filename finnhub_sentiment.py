"""
finnhub_sentiment.py — Historical sentiment via Finnhub + VIX proxy
====================================================================
Two complementary sources for historical sentiment data:

  1. Finnhub company-news API (free tier, 60 calls/min)
     Fetches real news articles per symbol, scores with the shared NLP
     model (FinBERT when available, VADER fallback — same as sentiment.py).
     Output schema matches sentiment.py so rows drop straight into
     the existing sentiment_history parquet.

  2. VIX (^VIX via yfinance)
     Market-wide fear index converted to a [-1, +1] sentiment score.
     Used to extend the daily_sentiment_avg baseline before NLP
     sentiment history exists.  High VIX = bearish, low VIX = bullish.

Public API
----------
  fetch_symbol_sentiment(symbol, from_date, to_date, api_key) -> pd.DataFrame
  backfill(symbols, from_date, api_key, to_date)             -> pd.DataFrame
  vix_to_score(vix)                                          -> float
  fetch_vix_history(period)                                  -> pd.DataFrame

Get a free Finnhub API key at: https://finnhub.io  (no credit card needed)
====================================================================
"""

import datetime
import logging
import os
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from config import ROLLING, FINNHUB_CFG, SENTIMENT
from sentiment import BULLISH_THRESHOLD, BEARISH_THRESHOLD, _score_headlines


def _load_api_key() -> str:
    """
    Return the Finnhub API key.
    Reads FINNHUB_API_KEY from the environment, falling back to the
    .env file in the project directory if the env var is not set.
    """
    key = os.environ.get("FINNHUB_API_KEY", "")
    if not key:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("FINNHUB_API_KEY="):
                        key = line.split("=", 1)[1].strip()
                        break
    return key


FINNHUB_API_KEY: str = _load_api_key()

logger = logging.getLogger(__name__)

FINNHUB_BASE  = FINNHUB_CFG.BASE_URL
_CALL_DELAY   = FINNHUB_CFG.CALL_DELAY_SEC
_SENT_COLUMNS = SENTIMENT.COLUMNS


# ══════════════════════════════════════════════════════════════════════
# VIX — MARKET-WIDE SENTIMENT PROXY
# ══════════════════════════════════════════════════════════════════════

def vix_to_score(vix: float) -> float:
    """
    Convert a VIX reading to a sentiment score (-1 to +1).

    Formula: -tanh((vix - 20) / 10)
      VIX 10 → +0.76  (very calm, bullish)
      VIX 15 → +0.46  (calm, bullish)
      VIX 20 →  0.00  (neutral baseline)
      VIX 25 → -0.46  (elevated, bearish)
      VIX 30 → -0.76  (high fear, bearish)
      VIX 40 → -0.96  (extreme fear)
    """
    return float(-np.tanh((float(vix) - 20.0) / 10.0))


def fetch_vix_history(period: str = "2y") -> pd.DataFrame:
    """
    Fetch VIX daily history from yfinance and convert to daily sentiment scores.

    Parameters
    ----------
    period : yfinance period string (default "2y")

    Returns
    -------
    pd.DataFrame with columns:
        Date (str YYYY-MM-DD), avg_score (float), rolling_7d (float), source (str)
    Same schema as sentiment.daily_sentiment_avg() output so it blends
    cleanly in extended_daily_sentiment_avg().
    """
    try:
        raw = yf.download("^VIX", period=period, auto_adjust=True, progress=False)
    except Exception as exc:
        logger.warning("VIX download failed: %s", exc)
        return pd.DataFrame(columns=["Date", "avg_score", "rolling_7d", "source"])

    if raw.empty or "Close" not in raw.columns:
        logger.warning("VIX: no data returned.")
        return pd.DataFrame(columns=["Date", "avg_score", "rolling_7d", "source"])

    close = raw["Close"].copy()
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()

    close.index = pd.to_datetime(close.index)
    if close.index.tz is not None:
        close.index = close.index.tz_convert(None)
    close.index = close.index.normalize()

    df = pd.DataFrame({"vix": close.values}, index=close.index)
    df = df.dropna().sort_index()
    df["avg_score"] = df["vix"].apply(vix_to_score).round(4)
    df["Date"]      = df.index.date.astype(str)
    df["rolling_7d"] = df["avg_score"].rolling(ROLLING.SHORT_WINDOW, min_periods=1).mean().round(4)
    df["source"]    = "vix"

    logger.info("VIX history: %d days loaded (%s → %s).",
                len(df), df["Date"].iloc[0], df["Date"].iloc[-1])
    return df[["Date", "avg_score", "rolling_7d", "source"]].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
# FINNHUB — HISTORICAL NEWS SENTIMENT
# ══════════════════════════════════════════════════════════════════════

def _fetch_company_news(
    symbol:    str,
    from_date: str,
    to_date:   str,
    api_key:   str,
) -> list:
    """Fetch news articles from Finnhub for one symbol in a date range."""
    try:
        resp = requests.get(
            f"{FINNHUB_BASE}/company-news",
            params={"symbol": symbol, "from": from_date, "to": to_date, "token": api_key},
            timeout=FINNHUB_CFG.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json() if isinstance(resp.json(), list) else []
    except Exception as exc:
        logger.warning("Finnhub fetch failed for %s: %s", symbol, exc)
        return []


def _score_articles(articles: list) -> tuple:
    """Score article headlines via the shared NLP model. Returns (avg_score, top_headline, count)."""
    headlines = [
        str(a.get("headline", "")).strip()
        for a in articles
        if a.get("headline") and len(str(a.get("headline", "")).strip()) > 3
    ]

    if not headlines:
        return np.nan, "", 0

    scores  = _score_headlines(headlines)
    avg     = float(np.mean(scores))
    best_i  = int(np.argmin(np.abs(np.array(scores) - avg)))
    return avg, headlines[best_i], len(scores)


def fetch_symbol_sentiment(
    symbol:    str,
    from_date: str,
    to_date:   str,
    api_key:   str,
) -> pd.DataFrame:
    """
    Fetch and score all Finnhub news for one symbol in a date range.

    Parameters
    ----------
    symbol    : ticker string e.g. "AAPL"
    from_date : start date "YYYY-MM-DD"
    to_date   : end date   "YYYY-MM-DD"
    api_key   : Finnhub API key (free at finnhub.io)

    Returns
    -------
    pd.DataFrame with _SENT_COLUMNS schema (one row per day with news).
    Empty DataFrame if no articles found.
    """
    articles = _fetch_company_news(symbol, from_date, to_date, api_key)
    time.sleep(_CALL_DELAY)

    if not articles:
        return pd.DataFrame(columns=_SENT_COLUMNS)

    # Group articles by date
    by_date: dict[str, list] = {}
    for a in articles:
        try:
            date_str = datetime.datetime.fromtimestamp(a["datetime"]).date().isoformat()
        except Exception:
            continue
        by_date.setdefault(date_str, []).append(a)

    rows = []
    for date_str, day_articles in by_date.items():
        avg_score, top_headline, count = _score_articles(day_articles)
        if np.isnan(avg_score) or count == 0:
            continue
        label = ("Bullish" if avg_score >= BULLISH_THRESHOLD else
                 "Bearish" if avg_score <= BEARISH_THRESHOLD else "Neutral")
        rows.append({
            "Date":            date_str,
            "Symbol":          symbol,
            "Sentiment_Score": round(avg_score, 4),
            "Sentiment_Label": label,
            "Headlines_Used":  count,
            "Top_Headline":    top_headline,
        })

    if not rows:
        return pd.DataFrame(columns=_SENT_COLUMNS)

    return pd.DataFrame(rows)[_SENT_COLUMNS]


def backfill(
    symbols:   list,
    from_date: str,
    api_key:   str,
    to_date:   str = None,
) -> pd.DataFrame:
    """
    Fetch historical Finnhub sentiment for all symbols from from_date to to_date.

    Parameters
    ----------
    symbols   : list of ticker strings
    from_date : start date "YYYY-MM-DD"
    api_key   : Finnhub API key
    to_date   : end date (default: today)

    Returns
    -------
    pd.DataFrame with _SENT_COLUMNS schema, ready to merge with
    sentiment_history via data_loader.save_sentiment_history().
    """
    if to_date is None:
        to_date = datetime.date.today().isoformat()

    total  = len(symbols)
    frames = []

    for i, sym in enumerate(symbols, 1):
        logger.info("Finnhub backfill: %d/%d — %s", i, total, sym)
        df = fetch_symbol_sentiment(sym, from_date, to_date, api_key)
        if not df.empty:
            frames.append(df)
        # Log progress every 20 symbols
        if i % 20 == 0:
            logger.info("  %d symbols done, %d with data so far.", i, len(frames))

    if not frames:
        logger.warning("Finnhub backfill: no data returned for any symbol.")
        return pd.DataFrame(columns=_SENT_COLUMNS)

    result = pd.concat(frames, ignore_index=True)
    logger.info(
        "Finnhub backfill complete — %d rows, %d symbols, %d days.",
        len(result), result["Symbol"].nunique(), result["Date"].nunique(),
    )
    return result
