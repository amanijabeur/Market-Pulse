"""
========================================================================
sentiment.py — Financial Sentiment Analysis Module
========================================================================
Fetches recent news headlines for a list of stock symbols using yfinance,
scores each headline with FinBERT (ProsusAI/finbert), aggregates scores
per symbol, and returns a clean DataFrame that can be imported by
dashboard.py or any other module.

Scoring model
-------------
- Primary:  FinBERT (ProsusAI/finbert) — trained on financial text,
            returns positive_prob − negative_prob mapped to [-1, +1].
            Requires: pip install transformers torch
- Fallback: VADER — used automatically when transformers/torch are not
            installed.  Same [-1, +1] output scale.

Architecture notes
------------------
- This module is self-contained and adds zero coupling to eda.py /
  stats_results.py.  It is imported by both scraper.py (which fetches
  and persists sentiment daily) and dashboard.py (which reads the stored
  results and falls back to run() only when today is not yet stored).
- All network calls are wrapped in try/except so a connectivity failure
  or an API change in yfinance never crashes the caller.
- Score thresholds (+0.05 / -0.05) apply to both FinBERT and VADER output:
    score >= +0.05  →  Bullish
    score <= -0.05  →  Bearish
    otherwise       →  Neutral

Output DataFrame columns
------------------------
  Symbol          : ticker (str)
  Date            : date of analysis (datetime.date)
  Sentiment_Score : mean score across headlines (float, -1 to +1)
  Sentiment_Label : "Bullish" | "Bearish" | "Neutral" (str)
  Headlines_Used  : number of headlines scored (int)
  Top_Headline    : representative headline string (str)

Public API
----------
  run(symbols, max_headlines=10) -> pd.DataFrame
      Primary entry point.  Pass a list of ticker strings; returns the
      sentiment DataFrame described above.  Always returns a non-empty
      DataFrame — symbols with no news receive Sentiment_Score=0.0 and
      Sentiment_Label="Neutral".

Shared constants and utilities (imported by scraper.py and dashboard.py)
-------------------------------------------------------------------------
  SENT_SHEET          : sheet name for the sentiment history Excel sheet
  BULLISH_THRESHOLD   : +0.05 — Bullish threshold
  BEARISH_THRESHOLD   : -0.05 — Bearish threshold
  empty_sentiment_df(): returns an empty DataFrame with the correct schema
  normalise_dates(df) : converts Date column to plain "YYYY-MM-DD" strings
  daily_sentiment_avg(): groups sent_history by date, computes rolling avg

Pip requirements (one-time install)
------------------------------------
  pip install transformers torch vaderSentiment yfinance
========================================================================
"""

import datetime
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

from config import ROLLING, SENTIMENT as _SENTIMENT_CFG

# ── Module logger ──────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Sentiment model — FinBERT preferred, VADER fallback ──────────────
# FinBERT (ProsusAI/finbert) is trained on 10-K filings, earnings calls,
# and financial news — far better calibrated than VADER for finance.
# Falls back to VADER automatically if transformers/torch are not installed.

def _load_sentiment_model():
    """
    Load the best available headline scoring model.
    Returns (model_type, model_object):
      ("finbert", pipeline) — if transformers + torch are installed
      ("vader",   analyser) — fallback
    """
    try:
        from transformers import pipeline as hf_pipeline
        logger.info("sentiment: loading FinBERT (%s)...", _SENTIMENT_CFG.NLP_MODEL)
        model = hf_pipeline(
            "text-classification",
            model=_SENTIMENT_CFG.NLP_MODEL,
            top_k=None,         # return all label probabilities
            device=-1,          # CPU; set to 0 for GPU if available
        )
        logger.info("sentiment: FinBERT loaded — financial NLP active.")
        return "finbert", model
    except Exception as exc:
        logger.info(
            "sentiment: FinBERT unavailable (%s) — install `transformers torch` "
            "to enable it. Using VADER.", type(exc).__name__,
        )
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return "vader", SentimentIntensityAnalyzer()


_ANALYSER_TYPE, _ANALYSER = _load_sentiment_model()

# Lock ensures FinBERT batch calls are serialised across threads so
# concurrent symbol processing doesn't corrupt shared model state.
_MODEL_LOCK = threading.Lock()

# ── Thresholds — single source of truth in config.SENTIMENT ──────────
# Re-exported here so dashboard.py and ai_narratives.py can keep their
# existing `from sentiment import BULLISH_THRESHOLD` imports unchanged.
BULLISH_THRESHOLD = _SENTIMENT_CFG.BULLISH_THRESHOLD
BEARISH_THRESHOLD = _SENTIMENT_CFG.BEARISH_THRESHOLD

# ── Sheet name — single definition imported by scraper and dashboard ──
# Changing this string here propagates everywhere automatically.
SENT_SHEET = "sentiment_history"

# ── Rate-limit guard: pause between yfinance ticker fetches ───────────
_FETCH_DELAY_SECONDS = _SENTIMENT_CFG.FETCH_DELAY


# ══════════════════════════════════════════════════════════════════════
# SHARED UTILITIES  (imported by scraper.py and dashboard.py)
# ══════════════════════════════════════════════════════════════════════

# Column schema — sourced from config so it stays in sync with all callers.
_SENT_COLUMNS = list(_SENTIMENT_CFG.COLUMNS)


def empty_sentiment_df() -> pd.DataFrame:
    """
    Return an empty DataFrame with the correct sentiment column schema.

    Used as the fallback when the sentiment_history sheet does not yet
    exist.  Both scraper.py and dashboard.py call this instead of
    duplicating the column list inline.
    """
    return pd.DataFrame(columns=_SENT_COLUMNS)


def normalise_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the Date column of a sentiment (or price) DataFrame to
    plain "YYYY-MM-DD" strings in-place and return the DataFrame.

    openpyxl reads dates as datetime objects; this function makes the
    comparison always string vs string regardless of how the file was
    written, preventing silent double-write bugs.
    """
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.date.astype(str)
    return df


def daily_sentiment_avg(sent_history: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-day average sentiment score from the full sentiment history.

    Returns a DataFrame sorted ascending by Date with columns:
        Date      : "YYYY-MM-DD" string
        avg_score : mean compound score across all symbols that day
        rolling_7d: 7-day rolling mean of avg_score

    Called by dashboard.py once after loading sent_history so the
    groupby is never duplicated across chart sections.
    """
    if sent_history.empty:
        return pd.DataFrame(columns=["Date", "avg_score", "rolling_7d"])

    history = sent_history.copy()
    history["Sentiment_Score"] = pd.to_numeric(
        history["Sentiment_Score"], errors="coerce"
    )
    history = history.dropna(subset=["Sentiment_Score"])

    if history.empty:
        return pd.DataFrame(columns=["Date", "avg_score", "rolling_7d"])

    daily = (
        history.groupby("Date")["Sentiment_Score"]
        .mean()
        .reset_index()
        .sort_values("Date")
        .rename(columns={"Sentiment_Score": "avg_score"})
    )
    daily["rolling_7d"] = daily["avg_score"].rolling(ROLLING.SHORT_WINDOW, min_periods=1).mean()
    return daily.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════

def _classify(compound: float) -> str:
    """Map a sentiment score to a human-readable label."""
    if compound >= BULLISH_THRESHOLD:
        return "Bullish"
    if compound <= BEARISH_THRESHOLD:
        return "Bearish"
    return "Neutral"


def _score_headlines(headlines: List[str]) -> List[float]:
    """
    Score a batch of headlines, returning compound-style scores in [-1, +1].

    FinBERT path: positive_probability − negative_probability per headline.
    VADER path:   standard compound score per headline.

    Always returns the same length as the input list.
    Falls back to per-headline VADER scoring if FinBERT batch call fails.
    """
    if not headlines:
        return []

    if _ANALYSER_TYPE == "finbert":
        try:
            with _MODEL_LOCK:
                results = _ANALYSER(
                    headlines,
                    truncation=True,
                    max_length=_SENTIMENT_CFG.NLP_MAX_LENGTH,
                    batch_size=_SENTIMENT_CFG.NLP_BATCH_SIZE,
                )
            scores = []
            for result in results:
                probs = {r["label"]: r["score"] for r in result}
                scores.append(round(probs.get("positive", 0.0) - probs.get("negative", 0.0), 4))
            return scores
        except Exception as exc:
            logger.warning("FinBERT batch failed (%s) — falling back to VADER.", exc)
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader = SentimentIntensityAnalyzer()
            return [round(_vader.polarity_scores(h)["compound"], 4) for h in headlines]
    else:
        return [round(_ANALYSER.polarity_scores(h)["compound"], 4) for h in headlines]


def _score_headline(headline: str) -> float:
    """Score a single headline (calls _score_headlines for consistency)."""
    results = _score_headlines([headline])
    return results[0] if results else 0.0


def _fetch_headlines(symbol: str, max_headlines: int) -> List[str]:
    """
    Fetch recent news headlines for *symbol* via yfinance.

    Returns a (possibly empty) list of headline strings.  Never raises —
    all exceptions are caught and logged so callers never crash.

    Parameters
    ----------
    symbol       : NYSE/NASDAQ ticker string, e.g. "AAPL"
    max_headlines: upper bound on headlines to return per ticker
    """
    try:
        ticker = yf.Ticker(symbol)
        news   = ticker.news  # list[dict] or empty list

        if not news:
            logger.debug("%s: no news returned by yfinance.", symbol)
            return []

        headlines: List[str] = []
        for item in news[:max_headlines]:
            # yfinance news items are dicts; headline may live under
            # different keys depending on the yfinance version.
            title = (
                item.get("title")
                or item.get("headline")
                or item.get("content", {}).get("title", "")
                if isinstance(item, dict)
                else ""
            )
            if title and isinstance(title, str) and len(title.strip()) > 3:
                headlines.append(title.strip())

        return headlines

    except Exception as exc:
        # Any network error, rate-limit, or API change — degrade gracefully.
        logger.warning("%s: headline fetch failed (%s: %s)", symbol, type(exc).__name__, exc)
        return []


def _analyse_symbol(symbol: str, max_headlines: int, today: datetime.date) -> dict:
    """
    Fetch headlines for one symbol and return a sentiment result dict.

    Always returns a complete dict even when no headlines are available.

    Parameters
    ----------
    symbol        : ticker string
    max_headlines : upper bound on headlines to fetch
    today         : date resolved once by run() — prevents the clock
                    ticking past midnight mid-run from producing mixed
                    dates across the 100-symbol batch
    """
    headlines = _fetch_headlines(symbol, max_headlines)

    if not headlines:
        return {
            "Symbol":          symbol,
            "Date":            today.isoformat(),
            "Sentiment_Score": np.nan,
            "Sentiment_Label": "No Data",
            "Headlines_Used":  0,
            "Top_Headline":    "No headlines available",
        }

    # Score all headlines in a single batch call (FinBERT is faster batched)
    scores     = _score_headlines(headlines)
    mean_score = float(np.mean(scores))

    # Pick the headline closest to the mean compound as the representative
    best_idx = int(np.argmin(np.abs(np.array(scores) - mean_score)))
    top_headline = headlines[best_idx]

    return {
        "Symbol":          symbol,
        "Date":            today.isoformat(),
        "Sentiment_Score": round(mean_score, 4),
        "Sentiment_Label": _classify(mean_score),
        "Headlines_Used":  len(headlines),
        "Top_Headline":    top_headline,
    }


# ══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════

def run(symbols: List[str], max_headlines: int = 10) -> pd.DataFrame:
    """
    Analyse sentiment for every symbol in *symbols*.

    Parameters
    ----------
    symbols       : list of ticker strings (e.g. ["AAPL", "TSLA", ...])
    max_headlines : max headlines to score per ticker (default 10)

    Returns
    -------
    pd.DataFrame with columns:
        Symbol, Date, Sentiment_Score, Sentiment_Label,
        Headlines_Used, Top_Headline

    Date is returned as a plain "YYYY-MM-DD" string, matching the format
    the scraper uses for the price dataset so joins and comparisons are
    always type-safe without further conversion.

    The DataFrame always has one row per symbol.  Symbols with no
    available news are included with Score=NaN and Label="No Data".
    """
    if not symbols:
        logger.warning("run() called with empty symbol list — returning empty DataFrame.")
        return empty_sentiment_df()

    # Resolve date once here so every symbol in the batch gets the same
    # date string even if the run crosses midnight.
    today = datetime.date.today()

    logger.info("Fetching sentiment for %d symbols on %s ...", len(symbols), today.isoformat())
    results = [None] * len(symbols)
    completed_count = 0

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            pool.submit(_analyse_symbol, sym, max_headlines, today): i
            for i, sym in enumerate(symbols)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            completed_count += 1
            if completed_count % 10 == 0:
                logger.info("  Processed %d / %d", completed_count, len(symbols))

    df = pd.DataFrame(results)

    # Enforce expected dtypes.
    # Date stays as a plain string — callers (scraper, dashboard) both
    # work with "YYYY-MM-DD" strings and must not need to convert.
    df["Sentiment_Score"] = pd.to_numeric(df["Sentiment_Score"], errors="coerce")
    df["Headlines_Used"]  = pd.to_numeric(df["Headlines_Used"],  errors="coerce").fillna(0).astype(int)
    df["Date"]            = df["Date"].astype(str)

    # Deduplicate on Symbol (in case caller passed duplicates)
    df = df.drop_duplicates(subset=["Symbol"]).reset_index(drop=True)

    bullish = (df["Sentiment_Label"] == "Bullish").sum()
    bearish = (df["Sentiment_Label"] == "Bearish").sum()
    neutral = (df["Sentiment_Label"] == "Neutral").sum()
    no_data = (df["Sentiment_Label"] == "No Data").sum()
    logger.info(
        "Sentiment complete -- Bullish: %d | Bearish: %d | Neutral: %d | No Data: %d",
        bullish, bearish, neutral, no_data,
    )

    return df


# ══════════════════════════════════════════════════════════════════════
# CONVENIENCE HELPERS (used by dashboard.py)
# ══════════════════════════════════════════════════════════════════════

def extended_daily_sentiment_avg(sent_history: pd.DataFrame) -> pd.DataFrame:
    """
    Like daily_sentiment_avg() but extends the baseline with VIX-derived
    sentiment for dates not yet covered by NLP-scored history.

    Strategy
    --------
    1. Compute NLP-based daily averages from stored sentiment history.
    2. Fetch VIX history (2 years) from yfinance via finnhub_sentiment.
    3. Keep only VIX rows for dates with no NLP data.
    4. Concatenate and recompute rolling_7d across the full series.

    Returns
    -------
    pd.DataFrame with columns: Date, avg_score, rolling_7d, source
    Same schema as daily_sentiment_avg() plus a "source" column
    ("nlp" | "vix") so callers can distinguish the two.
    """
    import finnhub_sentiment as _fh

    nlp_daily = daily_sentiment_avg(sent_history)
    vix_daily = _fh.fetch_vix_history("2y")

    if vix_daily.empty:
        nlp_daily["source"] = "nlp"
        return nlp_daily

    nlp_daily["source"] = "nlp"
    nlp_dates = set(nlp_daily["Date"].astype(str))

    # Only use VIX for dates not covered by NLP-scored history
    vix_prior = vix_daily[~vix_daily["Date"].isin(nlp_dates)].copy()

    if vix_prior.empty:
        return nlp_daily

    combined = pd.concat(
        [vix_prior[["Date", "avg_score", "rolling_7d", "source"]],
         nlp_daily[["Date", "avg_score", "rolling_7d", "source"]]],
        ignore_index=True,
    ).sort_values("Date").reset_index(drop=True)

    # Recompute rolling_7d across the full merged series
    combined["rolling_7d"] = (
        combined["avg_score"]
        .rolling(ROLLING.SHORT_WINDOW, min_periods=1)
        .mean()
        .round(4)
    )

    logger.info(
        "extended_daily_sentiment_avg: %d total days (%d VIX-prior + %d NLP).",
        len(combined), len(vix_prior), len(nlp_daily),
    )
    return combined


def top_bullish(sentiment_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the *n* stocks with the highest sentiment scores."""
    return (
        sentiment_df[sentiment_df["Sentiment_Label"] == "Bullish"]
        .nlargest(n, "Sentiment_Score")
        .reset_index(drop=True)
    )


def top_bearish(sentiment_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the *n* stocks with the lowest (most negative) sentiment scores."""
    return (
        sentiment_df[sentiment_df["Sentiment_Label"] == "Bearish"]
        .nsmallest(n, "Sentiment_Score")
        .reset_index(drop=True)
    )


def breadth_summary(sentiment_df: pd.DataFrame) -> dict:
    """
    Compute high-level sentiment breadth metrics.

    Returns
    -------
    dict with keys:
        n_total, n_bullish, n_bearish, n_neutral,
        pct_bullish, pct_bearish, pct_neutral,
        mean_score, regime
    """
    n = len(sentiment_df)
    if n == 0:
        return {
            "n_total": 0, "n_bullish": 0, "n_bearish": 0, "n_neutral": 0,
            "n_no_data": 0, "pct_bullish": 0.0, "pct_bearish": 0.0,
            "pct_neutral": 0.0, "pct_no_data": 0.0,
            "mean_score": 0.0, "regime": "Neutral",
        }

    n_bull = int((sentiment_df["Sentiment_Label"] == "Bullish").sum())
    n_bear = int((sentiment_df["Sentiment_Label"] == "Bearish").sum())
    n_neut = int((sentiment_df["Sentiment_Label"] == "Neutral").sum())
    n_no_data = int((sentiment_df["Sentiment_Label"] == "No Data").sum())
    mean = float(pd.to_numeric(sentiment_df["Sentiment_Score"], errors="coerce").mean())
    if np.isnan(mean):
        mean = 0.0

    # Regime: whichever class dominates; ties resolved by mean score
    if n_bull > n_bear and n_bull > n_neut:
        regime = "Risk-On"
    elif n_bear > n_bull and n_bear > n_neut:
        regime = "Risk-Off"
    elif mean >= BULLISH_THRESHOLD:
        regime = "Cautiously Bullish"
    elif mean <= BEARISH_THRESHOLD:
        regime = "Cautiously Bearish"
    else:
        regime = "Mixed"

    return {
        "n_total":     n,
        "n_bullish":   n_bull,
        "n_bearish":   n_bear,
        "n_neutral":   n_neut,
        "n_no_data":   n_no_data,
        "pct_bullish": round(n_bull / n * 100, 1),
        "pct_bearish": round(n_bear / n * 100, 1),
        "pct_neutral": round(n_neut / n * 100, 1),
        "pct_no_data": round(n_no_data / n * 100, 1),
        "mean_score":  round(mean, 4),
        "regime":      regime,
    }


# ══════════════════════════════════════════════════════════════════════
# STANDALONE EXECUTION — for testing outside the dashboard
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    TEST_SYMBOLS = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "AMZN", "META"]
    print(f"\nRunning sentiment analysis on {len(TEST_SYMBOLS)} test symbols...\n")

    result_df = run(TEST_SYMBOLS, max_headlines=5)
    print(result_df[["Symbol", "Sentiment_Score", "Sentiment_Label", "Headlines_Used"]].to_string(index=False))

    summary = breadth_summary(result_df)
    print(f"\nMarket Sentiment Regime : {summary['regime']}")
    print(f"Bullish / Bearish / Neutral : {summary['n_bullish']} / {summary['n_bearish']} / {summary['n_neutral']}")
    print(f"Mean Compound Score         : {summary['mean_score']:+.4f}")