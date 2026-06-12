"""
========================================================================
validators.py -- Market Pulse Data Validation Layer
========================================================================
Pre-flight checks that run before each pipeline stage to catch data
quality issues early and produce clear, actionable error messages
instead of cryptic pandas stack traces deep inside the pipeline.

Design
------
- Every validator returns a ValidationResult (passed, warnings, errors).
- Validators never raise exceptions -- they collect problems and return them.
- main.py decides whether to abort or continue based on severity.
- All thresholds come from config.py so they are easy to tune.

Public API
----------
  validate_price_data(df)      -> ValidationResult
  validate_sentiment_data(df)  -> ValidationResult
  validate_file_exists(path)   -> ValidationResult
  validate_parquet(path)       -> ValidationResult
  run_preflight(args)          -> bool
      Master check called by main.py before the pipeline starts.
      Returns True if safe to proceed, False if critical files missing.
========================================================================
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List

import pandas as pd

from config import PATHS, VALIDATION

logger = logging.getLogger(__name__)


# ======================================================================
# RESULT CONTAINER
# ======================================================================

@dataclass
class ValidationResult:
    """
    Holds the outcome of a single validation check.

    Attributes
    ----------
    passed   : True if no errors were found (warnings are allowed)
    warnings : non-fatal issues -- logged but pipeline continues
    errors   : fatal issues -- pipeline should stop if any are present
    """
    passed:   bool        = True
    warnings: List[str]   = field(default_factory=list)
    errors:   List[str]   = field(default_factory=list)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning("VALIDATION WARNING: %s", msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False
        logger.error("VALIDATION ERROR: %s", msg)

    def log_summary(self, context: str) -> None:
        if self.passed and not self.warnings:
            logger.info("Validation passed: %s", context)
        elif self.passed:
            logger.info(
                "Validation passed with %d warning(s): %s",
                len(self.warnings), context,
            )
        else:
            logger.error(
                "Validation FAILED (%d error(s), %d warning(s)): %s",
                len(self.errors), len(self.warnings), context,
            )


# ======================================================================
# FILE VALIDATORS
# ======================================================================

def validate_file_exists(path: str, label: str = "") -> ValidationResult:
    """
    Check that a required file exists on disk.

    Parameters
    ----------
    path  : file path to check
    label : human-readable name shown in error messages
    """
    result = ValidationResult()
    name   = label or path

    if not os.path.exists(path):
        result.add_error(f"Required file not found: {name} ({path})")
    else:
        size_kb = os.path.getsize(path) / 1024
        if size_kb < 1:
            result.add_warning(f"{name} exists but is very small ({size_kb:.1f} KB) -- may be empty.")
        logger.debug("File check passed: %s (%.1f KB)", name, size_kb)

    result.log_summary(f"file_exists: {name}")
    return result


def validate_parquet(path: str, label: str = "") -> ValidationResult:
    """
    Check that a Parquet file exists and can be read without errors.

    Parameters
    ----------
    path  : Parquet file path
    label : human-readable name for messages
    """
    result = ValidationResult()
    name   = label or path

    if not os.path.exists(path):
        result.add_warning(f"Parquet cache not found: {name} -- will be built on first run.")
        result.log_summary(f"parquet: {name}")
        return result

    try:
        df = pd.read_parquet(path, engine="pyarrow")
        if df.empty:
            result.add_warning(f"Parquet file is empty: {name}")
        else:
            logger.debug(
                "Parquet check passed: %s (%d rows, %d cols)",
                name, len(df), len(df.columns),
            )
    except Exception as exc:
        result.add_error(
            f"Parquet file is unreadable: {name} -- {type(exc).__name__}: {exc}. "
            "Delete the file to force a rebuild."
        )

    result.log_summary(f"parquet: {name}")
    return result


# ======================================================================
# DATA VALIDATORS
# ======================================================================

def validate_price_data(df: pd.DataFrame) -> ValidationResult:
    """
    Validate the price DataFrame loaded from data_loader.

    Checks
    ------
    - Required columns present
    - No nulls in key columns
    - Price values within sane bounds
    - No duplicate Date+Symbol pairs
    - Minimum row count per latest day
    """
    result = ValidationResult()

    if df is None or df.empty:
        result.add_error("Price DataFrame is None or empty.")
        result.log_summary("price_data")
        return result

    # -- Required columns --------------------------------------------
    missing_cols = [
        c for c in VALIDATION.PRICE_REQUIRED_COLS if c not in df.columns
    ]
    if missing_cols:
        result.add_error(f"Price data missing required columns: {missing_cols}")

    # Stop here if columns are missing -- further checks will crash
    if not result.passed:
        result.log_summary("price_data")
        return result

    # -- Null check --------------------------------------------------
    key_cols = ["Symbol", "Price", "Change", "% Change"]
    for col in key_cols:
        n_nulls = df[col].isnull().sum()
        if n_nulls > 0:
            result.add_warning(f"Price data: {n_nulls} null(s) in column '{col}'.")

    # -- Price sanity ------------------------------------------------
    invalid_low  = (df["Price"] < VALIDATION.MIN_VALID_PRICE).sum()
    invalid_high = (df["Price"] > VALIDATION.MAX_VALID_PRICE).sum()
    if invalid_low > 0:
        result.add_warning(
            f"{invalid_low} row(s) have Price < ${VALIDATION.MIN_VALID_PRICE}."
        )
    if invalid_high > 0:
        result.add_warning(
            f"{invalid_high} row(s) have Price > ${VALIDATION.MAX_VALID_PRICE:,.0f} -- verify outliers."
        )

    # -- Duplicate check ---------------------------------------------
    if "Date" in df.columns:
        n_dupes = df.duplicated(subset=["Date", "Symbol"]).sum()
        dupe_rate = n_dupes / len(df)
        if n_dupes > 0:
            if dupe_rate > VALIDATION.MAX_DUPE_RATE:
                result.add_error(
                    f"{n_dupes} duplicate Date+Symbol rows ({dupe_rate:.1%}) -- "
                    "data integrity issue. Run data_loader.invalidate_cache()."
                )
            else:
                result.add_warning(
                    f"{n_dupes} duplicate Date+Symbol row(s) detected -- "
                    "will be removed by deduplication."
                )

    # -- Minimum rows per day ----------------------------------------
    if "Date" in df.columns:
        latest = df["Date"].max()
        latest_count = (df["Date"] == latest).sum()
        if latest_count < VALIDATION.MIN_ROWS_PER_DAY:
            result.add_warning(
                f"Latest day ({latest}) has only {latest_count} rows "
                f"(expected >= {VALIDATION.MIN_ROWS_PER_DAY}). "
                "Scrape may have been incomplete."
            )

    result.log_summary("price_data")
    return result


def validate_sentiment_data(df: pd.DataFrame) -> ValidationResult:
    """
    Validate the sentiment history DataFrame.

    Checks
    ------
    - Required columns present
    - Score values within [-1, +1]
    - Labels are valid categories
    - No duplicate Date+Symbol pairs
    """
    result = ValidationResult()

    if df is None or df.empty:
        # Sentiment is non-critical on first run -- warn, do not fail
        result.add_warning(
            "Sentiment history is empty. "
            "Sentiment charts will be unavailable until scraper runs."
        )
        result.log_summary("sentiment_data")
        return result

    # -- Required columns --------------------------------------------
    missing_cols = [
        c for c in VALIDATION.SENTIMENT_REQUIRED_COLS if c not in df.columns
    ]
    if missing_cols:
        result.add_error(f"Sentiment data missing required columns: {missing_cols}")

    if not result.passed:
        result.log_summary("sentiment_data")
        return result

    # -- Score bounds ------------------------------------------------
    if "Sentiment_Score" in df.columns:
        out_of_range = ((df["Sentiment_Score"] < -1) | (df["Sentiment_Score"] > 1)).sum()
        if out_of_range > 0:
            result.add_warning(
                f"{out_of_range} Sentiment_Score value(s) outside [-1, +1] range."
            )

    # -- Label validity ----------------------------------------------
    if "Sentiment_Label" in df.columns:
        valid_labels = {"Bullish", "Bearish", "Neutral", "No Data"}
        bad_labels = df[~df["Sentiment_Label"].isin(valid_labels)]["Sentiment_Label"].unique()
        if len(bad_labels) > 0:
            result.add_warning(f"Unexpected Sentiment_Label values: {bad_labels.tolist()}")

    # -- Duplicate check ---------------------------------------------
    n_dupes = df.duplicated(subset=["Date", "Symbol"]).sum()
    if n_dupes > 0:
        result.add_warning(
            f"{n_dupes} duplicate Date+Symbol sentiment row(s) detected."
        )

    result.log_summary("sentiment_data")
    return result


def validate_forecast_data(df: pd.DataFrame) -> ValidationResult:
    """Validate Phase 2 forecast cache integrity without failing first-run pipelines."""
    result = ValidationResult()
    if df is None or df.empty:
        result.add_warning("Forecast cache is empty; forecasting stage has not produced rows yet.")
        result.log_summary("forecast_data")
        return result
    required = {"Date", "metric", "forecast", "lower_bound", "upper_bound", "method"}
    missing = required - set(df.columns)
    if missing:
        result.add_error(f"Forecast data missing required columns: {sorted(missing)}")
        result.log_summary("forecast_data")
        return result
    numeric = df[["forecast", "lower_bound", "upper_bound"]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        result.add_warning("Forecast cache contains NaN/non-numeric forecast bounds.")
    invalid_bounds = (numeric["lower_bound"] > numeric["forecast"]) | (numeric["forecast"] > numeric["upper_bound"])
    if invalid_bounds.any():
        result.add_warning(f"{int(invalid_bounds.sum())} forecast row(s) have invalid confidence bounds.")
    result.log_summary("forecast_data")
    return result


def validate_signal_data(df: pd.DataFrame) -> ValidationResult:
    """Validate signal consistency and score hygiene."""
    result = ValidationResult()
    if df is None or df.empty:
        result.add_warning("Signal cache is empty; no active signals or stage not run.")
        result.log_summary("signal_data")
        return result
    required = {"Date", "Symbol", "signal_type", "direction", "strength", "score", "reason"}
    missing = required - set(df.columns)
    if missing:
        result.add_error(f"Signal data missing required columns: {sorted(missing)}")
        result.log_summary("signal_data")
        return result
    bad_dir = ~df["direction"].isin(["Bullish", "Bearish", "Neutral"])
    if bad_dir.any():
        result.add_warning(f"{int(bad_dir.sum())} signal row(s) have unexpected direction labels.")
    if pd.to_numeric(df["score"], errors="coerce").isna().any():
        result.add_warning("Signal cache contains NaN/non-numeric scores.")
    result.log_summary("signal_data")
    return result


def validate_anomaly_data(df: pd.DataFrame) -> ValidationResult:
    """Validate anomaly score ranges and severity labels."""
    result = ValidationResult()
    if df is None or df.empty:
        result.add_warning("Anomaly cache is empty; no anomalies or stage not run.")
        result.log_summary("anomaly_data")
        return result
    required = {"Date", "anomaly_type", "severity", "z_score", "value", "reason"}
    missing = required - set(df.columns)
    if missing:
        result.add_error(f"Anomaly data missing required columns: {sorted(missing)}")
        result.log_summary("anomaly_data")
        return result
    bad_sev = ~df["severity"].isin(["Low", "Moderate", "High", "Critical"])
    if bad_sev.any():
        result.add_warning(f"{int(bad_sev.sum())} anomaly row(s) have unexpected severity labels.")
    z = pd.to_numeric(df["z_score"], errors="coerce")
    if z.isna().any():
        result.add_warning("Anomaly cache contains NaN/non-numeric z_score values.")
    # cluster_spike z_score is a participation fraction [0, 1]; other detectors
    # use real z-scores where > 10 indicates a data error, not a real event.
    if "anomaly_type" in df.columns:
        non_cluster = df["anomaly_type"] != "cluster_spike"
        if (z[non_cluster].abs() > 10).any():
            result.add_warning("Anomaly z_score contains extreme values above 10; inspect cache/source data.")
        if (z[~non_cluster].abs() > 1.0).any():
            result.add_warning("cluster_spike z_score exceeds 1.0; participation fraction is corrupted.")
    else:
        if (z.abs() > 10).any():
            result.add_warning("Anomaly z_score contains extreme values above 10; inspect cache/source data.")
    result.log_summary("anomaly_data")
    return result


def validate_historical_depth(df: pd.DataFrame, min_days: int = 3) -> ValidationResult:
    """Warn when rolling/forecasting windows do not have enough history."""
    result = ValidationResult()
    if df is None or df.empty or "Date" not in df.columns:
        result.add_warning("Historical depth could not be assessed; Date column missing or empty.")
        result.log_summary("historical_depth")
        return result
    n_days = pd.to_datetime(df["Date"], errors="coerce").nunique()
    if n_days < min_days:
        result.add_warning(f"Only {n_days} trading day(s) available; rolling signals and forecasts are low-confidence.")
    result.log_summary("historical_depth")
    return result


# ======================================================================
# MASTER PRE-FLIGHT CHECK
# Called by main.py before the pipeline starts.
# ======================================================================

def run_preflight(skip_scraper: bool = False, dashboard_only: bool = False) -> bool:
    """
    Run all pre-flight checks and report results.

    Parameters
    ----------
    skip_scraper   : if True, Excel file must already exist
    dashboard_only : if True, all data files must already exist

    Returns
    -------
    True  : safe to proceed
    False : critical files missing -- pipeline should not start
    """
    logger.info("Running pre-flight validation checks...")
    all_passed = True
    any_warnings = False

    def _check(result: ValidationResult, label: str) -> None:
        nonlocal all_passed, any_warnings
        if not result.passed:
            all_passed = False
            print(f"  [ERROR]   {label}")
            for e in result.errors:
                print(f"            {e}")
        if result.warnings:
            any_warnings = True
            for w in result.warnings:
                print(f"  [WARN]    {w}")

    print("\n  Pre-flight checks:")

    # -- Parquet integrity (non-critical -- rebuilt automatically) -----
    _check(
        validate_parquet(PATHS.PRICE_PARQUET, "Price Parquet cache"),
        "Price Parquet",
    )
    _check(
        validate_parquet(PATHS.SENTIMENT_PARQUET, "Sentiment Parquet cache"),
        "Sentiment Parquet",
    )
    _check(
        validate_parquet(PATHS.HIST_PARQUET, "Historical metrics cache"),
        "Historical metrics Parquet",
    )
    for path, label in [
        (PATHS.TECHNICAL_PARQUET, "Technical metrics cache"),
        (PATHS.FORECAST_PARQUET, "Forecast metrics cache"),
        (PATHS.SIGNAL_PARQUET, "Signal metrics cache"),
        (PATHS.ANOMALY_PARQUET, "Anomaly metrics cache"),
    ]:
        _check(validate_parquet(path, label), label)

    # -- Excel file (critical if not doing a fresh scrape) -----------
    if skip_scraper or dashboard_only:
        result = validate_file_exists(PATHS.EXCEL_FILE, "Price dataset (Excel)")
        _check(result, "Excel dataset")
        if not result.passed:
            print(
                f"\n  CRITICAL: Excel dataset not found at '{PATHS.EXCEL_FILE}'.\n"
                "  Run without --skip-scraper or --dashboard-only first."
            )
            all_passed = False

    # -- stats_results.py (critical if building dashboard) -----------
    if dashboard_only or skip_scraper:
        result = validate_file_exists(PATHS.STATS_RESULTS, "stats_results.py")
        _check(result, "stats_results.py")
        if not result.passed:
            print(
                "\n  CRITICAL: stats_results.py not found.\n"
                "  Run without --dashboard-only first so EDA can generate it."
            )
            all_passed = False

    # -- Data content validation (only when files exist) --------------
    if os.path.exists(PATHS.EXCEL_FILE) and (skip_scraper or dashboard_only):
        try:
            import data_loader
            df_price = data_loader.load_price_data()
            _check(validate_price_data(df_price), "Price data content")

            df_sent = data_loader.load_sentiment_history()
            _check(validate_sentiment_data(df_sent), "Sentiment data content")
            _check(validate_historical_depth(df_price), "Historical depth")
            _check(validate_rolling_window_sufficiency(df_price, "Price data"), "Rolling window")
            _check(validate_forecast_data(data_loader.load_forecast_data()), "Forecast cache content")
            _check(validate_signal_data(data_loader.load_signal_data()), "Signal cache content")
            _check(validate_anomaly_data(data_loader.load_anomaly_data()), "Anomaly cache content")
        except Exception as exc:
            logger.warning("Could not load data for validation: %s", exc)

    if all_passed and not any_warnings:
        print("  All checks passed.\n")
    elif all_passed:
        print("  Checks passed with warnings (see above).\n")
    else:
        print("  Pre-flight FAILED. Fix errors above before running the pipeline.\n")

    return all_passed


if __name__ == "__main__":
    # Allow running validators standalone for debugging
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ok = run_preflight()
    sys.exit(0 if ok else 1)



# ======================================================================
# ROLLING WINDOW & NaN VALIDATORS
# Warn when datasets are too thin for meaningful analytics.
# ======================================================================

def validate_rolling_window_sufficiency(
    df: pd.DataFrame,
    context: str = "hist_df",
) -> ValidationResult:
    """
    Warn when the number of rows is below the rolling windows used
    by technical_indicators.py and forecasting.py.

    Parameters
    ----------
    df      : hist_df or similar one-row-per-day DataFrame
    context : label used in warning messages
    """
    result = ValidationResult()

    if df is None or df.empty:
        result.add_warning(f"{context}: empty — rolling window check skipped.")
        result.log_summary(f"rolling_window: {context}")
        return result

    from config import ROLLING
    n  = len(df)
    sw = ROLLING.SHORT_WINDOW
    lw = ROLLING.LONG_WINDOW

    if n < sw:
        result.add_warning(
            f"{context} has only {n} row(s) — fewer than the {sw}-day short window. "
            "SMA/EMA and signal-strength values will be heavily smoothed."
        )
    elif n < lw:
        result.add_warning(
            f"{context} has {n} row(s) — fewer than the {lw}-day long window. "
            "SMA-30 and linear forecast confidence intervals are not yet meaningful."
        )

    result.log_summary(f"rolling_window: {context}")
    return result


def validate_nan_propagation(
    df: pd.DataFrame,
    key_cols: list = None,
    nan_threshold: float = 0.5,
    context: str = "DataFrame",
) -> ValidationResult:
    """
    Warn when a DataFrame has a high NaN density in key numeric columns.

    A NaN rate above nan_threshold in any key column usually means
    a rolling computation failed upstream or data alignment went wrong.

    Parameters
    ----------
    df            : DataFrame to check
    key_cols      : columns to inspect (None = all numeric columns)
    nan_threshold : fraction above which a warning is emitted
    context       : label used in warning messages
    """
    result = ValidationResult()

    if df is None or df.empty:
        result.log_summary(f"nan_propagation: {context}")
        return result

    cols = key_cols or df.select_dtypes(include="number").columns.tolist()
    for col in cols:
        if col not in df.columns:
            continue
        nan_rate = df[col].isna().mean()
        if nan_rate > nan_threshold:
            result.add_warning(
                f"{context}['{col}']: {nan_rate:.0%} NaN — "
                "possible rolling/alignment failure upstream."
            )

    result.log_summary(f"nan_propagation: {context}")
    return result


# ======================================================================
# HISTORICAL DATA VALIDATORS
# Validate OHLCV and feature DataFrames from the historical layer.
# Consistent with the existing ValidationResult architecture.
# ======================================================================

def validate_ohlcv_data(df: pd.DataFrame, symbol: str = "") -> ValidationResult:
    """
    Validate an OHLCV DataFrame from data_loader.load_ohlcv().

    Checks
    ------
    - Required columns present (Date, Open, High, Low, Close, Volume)
    - No all-NaN Close column
    - No negative Close prices
    - High >= Low on every row
    - Minimum row count for meaningful indicators
    - No duplicate dates
    """
    from config import HIST_FETCH
    result = ValidationResult()
    label  = symbol or "OHLCV"

    if df is None or df.empty:
        result.add_warning(f"{label}: OHLCV cache is empty — run fetch_historical.")
        result.log_summary(f"ohlcv_data: {label}")
        return result

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        result.add_error(f"{label}: OHLCV missing required columns: {missing}")
        result.log_summary(f"ohlcv_data: {label}")
        return result

    if df["Close"].isna().all():
        result.add_error(f"{label}: Close column is all NaN.")
        result.log_summary(f"ohlcv_data: {label}")
        return result

    n_neg_close = int((df["Close"].dropna() <= 0).sum())
    if n_neg_close:
        result.add_warning(f"{label}: {n_neg_close} row(s) with Close <= 0.")

    inv = int((df["High"] < df["Low"]).sum())
    if inv:
        result.add_warning(f"{label}: {inv} row(s) where High < Low.")

    if "Date" in df.columns:
        n_dupes = df.duplicated(subset=["Date"]).sum()
        if n_dupes:
            result.add_warning(f"{label}: {n_dupes} duplicate Date row(s).")

    n_rows = len(df)
    if n_rows < HIST_FETCH.MIN_OHLCV_ROWS:
        result.add_warning(
            f"{label}: only {n_rows} rows (minimum {HIST_FETCH.MIN_OHLCV_ROWS} "
            "recommended for meaningful indicators)."
        )

    result.log_summary(f"ohlcv_data: {label}")
    return result


def validate_feature_data(df: pd.DataFrame, symbol: str = "") -> ValidationResult:
    """
    Validate a feature/indicator DataFrame from data_loader.load_features().

    Checks
    ------
    - Date and Close columns present
    - Key indicator columns present (rsi_14, macd, bb_upper, atr_14)
    - NaN propagation check on key features
    """
    from pipelines.feature_engineering import FEATURE_NAMES
    result = ValidationResult()
    label  = symbol or "features"

    if df is None or df.empty:
        result.add_warning(f"{label}: feature cache empty — run indicator pipeline.")
        result.log_summary(f"feature_data: {label}")
        return result

    key_indicators = ["rsi_14", "macd", "bb_upper", "atr_14", "drawdown_pct"]
    missing = [c for c in key_indicators if c not in df.columns]
    if missing:
        result.add_warning(
            f"{label}: indicator columns missing: {missing}. "
            "Re-run pipelines/indicators.run_indicators()."
        )

    # NaN density check on present key features
    for col in [c for c in key_indicators if c in df.columns]:
        nan_rate = df[col].isna().mean()
        if nan_rate > 0.5:
            result.add_warning(
                f"{label}['{col}']: {nan_rate:.0%} NaN — possible insufficient history."
            )

    result.log_summary(f"feature_data: {label}")
    return result