"""
========================================================================
pipelines/indicators.py — Indicator Computation Pipeline
========================================================================
Orchestrates the full indicator computation workflow for historical
OHLCV data. Wires together preprocess.py and
technical_indicators.compute_ohlcv_indicators() into one callable that
produces a feature-rich DataFrame and persists it via data_loader.

This is the layer main.py calls; it does not contain indicator math
(that lives in technical_indicators.py, reusing existing helpers).

Public API
----------
  run_indicators(symbol, force)        -> pd.DataFrame
  run_indicators_batch(symbols, force) -> dict
========================================================================
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from config import HIST_FETCH
import data_loader
from technical_indicators import compute_ohlcv_indicators
from pipelines.preprocess import preprocess_ohlcv

logger = logging.getLogger(__name__)


def run_indicators(symbol: str, force: bool = False) -> pd.DataFrame:
    """
    Load OHLCV data for a symbol, preprocess it, compute all indicators,
    and persist the result as a features parquet.

    Incremental: if features already exist and OHLCV has not changed,
    returns the cached features without recomputing.

    Parameters
    ----------
    symbol : ticker string
    force  : recompute even if features are current

    Returns
    -------
    pd.DataFrame with OHLCV + full indicator columns.
    Empty DataFrame if OHLCV data is not available.
    """
    sym_up = symbol.upper()

    # -- Load raw OHLCV -----------------------------------------------
    ohlcv = data_loader.load_ohlcv(sym_up)
    if ohlcv.empty:
        logger.warning(
            "indicators [%s]: no OHLCV data — run fetch_historical first.", sym_up
        )
        return pd.DataFrame()

    # -- Incremental check: compare OHLCV latest vs features latest ---
    if not force:
        existing_features = data_loader.load_features(sym_up)
        if not existing_features.empty and "Date" in existing_features.columns:
            ohlcv_latest     = pd.to_datetime(ohlcv["Date"].max()).normalize()
            features_latest  = pd.to_datetime(existing_features["Date"].max()).normalize()
            if features_latest >= ohlcv_latest:
                logger.info(
                    "indicators [%s]: features current. Skipping recompute.", sym_up
                )
                return existing_features

    # -- Preprocess ---------------------------------------------------
    clean = preprocess_ohlcv(ohlcv, symbol=sym_up)
    if clean.empty:
        logger.warning("indicators [%s]: preprocessing failed.", sym_up)
        return pd.DataFrame()

    # -- Compute indicators -------------------------------------------
    features = compute_ohlcv_indicators(clean, symbol=sym_up)
    if features.empty:
        return pd.DataFrame()

    # -- Persist ------------------------------------------------------
    data_loader.save_features(features, sym_up)
    logger.info(
        "indicators [%s]: %d rows, %d columns saved.",
        sym_up, len(features), len(features.columns),
    )
    return features


def run_indicators_batch(
    symbols: list[str] = None,
    force: bool = False,
) -> dict:
    """
    Run the indicator pipeline for a list of symbols.

    Parameters
    ----------
    symbols : list of ticker strings
              (defaults to all symbols with local OHLCV data)
    force   : recompute all features

    Returns
    -------
    dict with keys:
        succeeded : list of symbols successfully processed
        failed    : list of symbols that failed
        n_rows    : total feature rows across all symbols
    """
    if symbols is None:
        symbols = data_loader.list_available_ohlcv()

    if not symbols:
        logger.warning("indicators: no symbols with OHLCV data found.")
        return {"succeeded": [], "failed": [], "n_rows": 0}

    succeeded: list[str] = []
    failed:    list[str] = []
    n_rows     = 0

    def _run_one(sym: str) -> tuple[str, pd.DataFrame]:
        return sym, run_indicators(sym, force=force)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_run_one, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                _, df = future.result()
                if not df.empty:
                    succeeded.append(sym)
                    n_rows += len(df)
                else:
                    failed.append(sym)
            except Exception as exc:
                logger.warning("indicators: %s failed — %s", sym, exc)
                failed.append(sym)

    logger.info(
        "indicators batch complete: %d/%d succeeded, %d total rows.",
        len(succeeded), len(symbols), n_rows,
    )
    return {"succeeded": succeeded, "failed": failed, "n_rows": n_rows}
