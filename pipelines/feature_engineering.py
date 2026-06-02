"""
========================================================================
pipelines/feature_engineering.py — Feature Engineering for AI/ML
========================================================================
Transforms indicator-rich OHLCV DataFrames into feature sets ready for
forecasting, anomaly detection, regime classification, and AI models.

Builds on top of the indicator layer — expects the output of
pipelines/indicators.run_indicators() as input.

Architecture
------------
- Pure transformation. No I/O, no network calls.
- All features are derived from existing indicator columns; no new
  data sources are introduced here.
- Produces a clean, model-ready DataFrame where all columns are numeric,
  NaNs are handled, and the feature set is documented.
- Future ML modules (forecasting.py Phase 3, regime engine, AI
  narratives) can consume the output of build_feature_set() directly.

Public API
----------
  build_feature_set(df, symbol)         -> pd.DataFrame
  build_cross_asset_features(dfs_dict)  -> pd.DataFrame
  get_feature_names()                   -> list[str]
========================================================================
"""

import logging

import numpy as np
import pandas as pd

from config import ROLLING
from pipelines.preprocess import normalise_series

logger = logging.getLogger(__name__)

# Feature columns produced by build_feature_set()
# Exported so forecasting and AI modules can reference them without
# importing the full engineering pipeline.
FEATURE_NAMES: list[str] = [
    # Price-based
    "returns_pct", "log_returns", "gap_pct", "intraday_range_pct",
    # Trend
    "sma_20", "sma_50", "ema_12", "ema_26",
    "price_vs_sma20",    # Close / sma_20 - 1
    "price_vs_sma50",    # Close / sma_50 - 1
    "sma_crossover",     # sma_20 / sma_50 - 1 (golden/death cross proxy)
    # Momentum
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "momentum_10",
    "signal_strength",
    # Volatility
    "atr_14", "volatility_21", "bb_width",
    "atr_vs_sma20",      # atr_14 / sma_20 — normalised volatility
    # Risk
    "drawdown_pct",
    # Volume
    "volume_sma_20",
    "volume_ratio",      # Volume / volume_sma_20 — relative volume
    # Rolling stats (short window)
    f"return_rolling_mean_{ROLLING.SHORT_WINDOW}d",
    f"return_rolling_std_{ROLLING.SHORT_WINDOW}d",
    f"return_rolling_skew_{ROLLING.SHORT_WINDOW}d",
    # Bollinger %B
    "bb_pct_b",
    # ADX trend strength
    "adx_14", "plus_di", "minus_di",
    # Stochastic RSI
    "stochrsi_k", "stochrsi_d",
    # Interaction terms — capture non-linear relationships
    "rsi_x_vol",           # RSI × volatility: momentum in volatile regime
    "momentum_x_volume",   # momentum × relative volume: volume-confirmed breakout
    "macd_x_bb",           # MACD histogram × BB width: signal in expanding market
    "price_vs_sma20_x_rsi",# price deviation × RSI: trend + momentum alignment
    # Lag features — 5-day returns, 3-day RSI and MACD for AR structure
    "returns_lag1", "returns_lag2", "returns_lag3", "returns_lag4", "returns_lag5",
    "rsi_lag1",     "rsi_lag2",     "rsi_lag3",
    "macd_hist_lag1","macd_hist_lag2","macd_hist_lag3",
]


def build_feature_set(
    df: pd.DataFrame,
    symbol: str = "",
    normalise: bool = False,
) -> pd.DataFrame:
    """
    Build a complete feature set from an indicator-enriched OHLCV DataFrame.

    Adds derived features (ratios, lags, rolling stats) on top of the
    base indicators already computed by compute_ohlcv_indicators().
    The result is the input to any forecasting, regime, or AI module.

    Parameters
    ----------
    df        : output of pipelines/indicators.run_indicators()
    symbol    : ticker label (added as a column)
    normalise : if True, apply z-score normalisation to all numeric
                feature columns (useful for ML model inputs)

    Returns
    -------
    pd.DataFrame with Date, Symbol, and all FEATURE_NAMES columns.
    Rows with insufficient history for rolling features are kept but
    will contain NaN in those columns — callers decide whether to drop.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if symbol:
        out["Symbol"] = symbol

    close  = out["Close"]
    vol    = out.get("Volume", pd.Series(dtype=float))

    # ── Price vs moving averages (ratio - 1, so 0 = at MA) ─────────
    if "sma_20" in out.columns:
        out["price_vs_sma20"] = (close / out["sma_20"].replace(0, np.nan) - 1).round(4)
    if "sma_50" in out.columns:
        out["price_vs_sma50"] = (close / out["sma_50"].replace(0, np.nan) - 1).round(4)
    if "sma_20" in out.columns and "sma_50" in out.columns:
        out["sma_crossover"] = (
            out["sma_20"] / out["sma_50"].replace(0, np.nan) - 1
        ).round(4)

    # ── ATR normalised by price ─────────────────────────────────────
    if "atr_14" in out.columns and "sma_20" in out.columns:
        out["atr_vs_sma20"] = (
            out["atr_14"] / out["sma_20"].replace(0, np.nan)
        ).round(4)

    # ── Relative volume ─────────────────────────────────────────────
    if "volume_sma_20" in out.columns and not vol.empty:
        out["volume_ratio"] = (
            vol.astype(float) / out["volume_sma_20"].replace(0, np.nan)
        ).round(4)

    # ── Rolling stats on returns (7-day) ────────────────────────────
    if "returns_pct" in out.columns:
        ret = out["returns_pct"]
        out[f"return_rolling_mean_{ROLLING.SHORT_WINDOW}d"] = ret.rolling(ROLLING.SHORT_WINDOW, min_periods=1).mean().round(4)
        out[f"return_rolling_std_{ROLLING.SHORT_WINDOW}d"]  = ret.rolling(ROLLING.SHORT_WINDOW, min_periods=1).std().round(4)
        out[f"return_rolling_skew_{ROLLING.SHORT_WINDOW}d"] = ret.rolling(ROLLING.SHORT_WINDOW, min_periods=1).skew().round(4)

    # ── Interaction terms — non-linear signal combinations ──────────
    if "rsi_14" in out.columns and "volatility_21" in out.columns:
        # High RSI in a volatile regime signals momentum that may reverse
        out["rsi_x_vol"] = (out["rsi_14"] / 100 * out["volatility_21"]).round(4)

    if "momentum_10" in out.columns and "volume_ratio" in out.columns:
        # Volume-confirmed momentum: breakout with high participation is stronger
        out["momentum_x_volume"] = (out["momentum_10"] * out["volume_ratio"]).round(4)

    if "macd_hist" in out.columns and "bb_width" in out.columns:
        # MACD signal amplified by Bollinger expansion (trending market context)
        out["macd_x_bb"] = (out["macd_hist"] * out["bb_width"]).round(4)

    if "price_vs_sma20" in out.columns and "rsi_14" in out.columns:
        # (rsi_14 - 50) / 50 centres RSI at 0 (neutral=0, overbought≈+1, oversold≈-1)
        out["price_vs_sma20_x_rsi"] = (
            out["price_vs_sma20"] * ((out["rsi_14"] - 50) / 50)
        ).round(4)

    # ── Lag features (5-day returns, 3-day RSI and MACD) ────────────
    for lag in range(1, 6):
        if "returns_pct" in out.columns:
            out[f"returns_lag{lag}"] = out["returns_pct"].shift(lag).round(4)
    for lag in range(1, 4):
        if "rsi_14" in out.columns:
            out[f"rsi_lag{lag}"] = out["rsi_14"].shift(lag).round(4)
        if "macd_hist" in out.columns:
            out[f"macd_hist_lag{lag}"] = out["macd_hist"].shift(lag).round(4)

    # ── Optional normalisation ───────────────────────────────────────
    if normalise:
        numeric_cols = [c for c in FEATURE_NAMES if c in out.columns]
        for col in numeric_cols:
            out[col] = normalise_series(out[col], method="zscore")

    # ── Keep only expected feature columns + metadata ───────────────
    meta_cols    = [c for c in ["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]
                    if c in out.columns]
    feature_cols = [c for c in FEATURE_NAMES if c in out.columns]
    final_cols   = meta_cols + [c for c in feature_cols if c not in meta_cols]
    out = out[[c for c in final_cols if c in out.columns]]

    logger.info(
        "feature_engineering [%s]: %d rows, %d feature columns.",
        symbol or "unknown", len(out), len(feature_cols),
    )
    return out


def build_cross_asset_features(
    dfs_dict: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build cross-asset features by aligning multiple symbol feature sets on Date.

    Useful for regime detection and portfolio-level analytics where the
    relationship between symbols matters (e.g. NVDA vs INTC spread,
    tech sector average RSI, correlated drawdowns).

    Parameters
    ----------
    dfs_dict : {symbol -> feature_DataFrame} from build_feature_set()

    Returns
    -------
    Wide pd.DataFrame indexed by Date with columns prefixed by symbol.
    E.g. "AAPL_returns_pct", "NVDA_rsi_14", ...
    Only dates where at least one symbol has data are included.
    """
    if not dfs_dict:
        return pd.DataFrame()

    frames = []
    for sym, df in dfs_dict.items():
        if df is None or df.empty or "Date" not in df.columns:
            continue
        sub = df.set_index("Date")
        # Keep only numeric feature columns
        num_cols = [c for c in FEATURE_NAMES if c in sub.columns]
        sub = sub[num_cols].add_prefix(f"{sym.upper()}_")
        frames.append(sub)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1).sort_index()
    logger.info(
        "feature_engineering: cross-asset frame — %d dates, %d columns.",
        len(combined), len(combined.columns),
    )
    return combined.reset_index()


def get_feature_names() -> list[str]:
    """Return the canonical list of feature column names."""
    return list(FEATURE_NAMES)


# ======================================================================
# MUTUAL INFORMATION FEATURE RANKING
# Ranks features by their non-linear dependency with a target column.
# No sklearn dependency — uses a pure numpy/pandas discretisation approach.
# ======================================================================

def _mutual_info(x_bins: pd.Series, y_bins: pd.Series) -> float:
    """
    Compute mutual information between two discretised Series.

    MI(X;Y) = Σ_x Σ_y P(x,y) · log( P(x,y) / (P(x)·P(y)) )

    Returns a non-negative float.  Higher = stronger (possibly non-linear)
    dependency.  Zero = statistically independent.
    """
    n = len(x_bins)
    if n == 0:
        return 0.0

    joint = pd.crosstab(x_bins, y_bins).values.astype(float)
    pxy   = joint / n
    px    = pxy.sum(axis=1, keepdims=True)   # shape (n_x, 1)
    py    = pxy.sum(axis=0, keepdims=True)   # shape (1, n_y)
    denom = px * py                           # shape (n_x, n_y) via broadcasting

    mask = pxy > 0   # log is only defined for positive joint probabilities
    mi   = float(np.sum(pxy[mask] * np.log(pxy[mask] / denom[mask])))
    return max(0.0, round(mi, 6))


def rank_features_by_mi(
    df: pd.DataFrame,
    feature_cols: list[str] = None,
    target_col: str = "returns_pct",
    n_bins: int = None,
) -> pd.DataFrame:
    """
    Rank features by mutual information (MI) with a target column.

    MI captures non-linear relationships that pairwise correlation misses,
    making it the standard pre-selection step before tree-based or neural
    models.  Both features and target are discretised into quantile bins
    before MI is estimated, so the estimator is distribution-free.

    Parameters
    ----------
    df          : feature DataFrame (output of build_feature_set())
    feature_cols: columns to rank; defaults to all FEATURE_NAMES in df
    target_col  : column to measure dependency against (default: returns_pct)
    n_bins      : number of quantile bins (default: FEATURE_SELECTION.MI_N_BINS)

    Returns
    -------
    pd.DataFrame with columns: feature, mutual_information
    Sorted descending by mutual_information.
    """
    from config import FEATURE_SELECTION
    n_bins = n_bins if n_bins is not None else FEATURE_SELECTION.MI_N_BINS

    if feature_cols is None:
        feature_cols = [c for c in FEATURE_NAMES if c in df.columns]

    if target_col not in df.columns:
        logger.warning("rank_features_by_mi: target '%s' not in df — returning empty.", target_col)
        return pd.DataFrame(columns=["feature", "mutual_information"])

    present = [c for c in feature_cols if c in df.columns and c != target_col]
    sub     = df[[target_col] + present].dropna()

    if sub.empty:
        return pd.DataFrame(columns=["feature", "mutual_information"])

    try:
        target_bins = pd.qcut(sub[target_col], q=n_bins, labels=False, duplicates="drop")
    except Exception:
        logger.warning("rank_features_by_mi: could not bin target '%s'.", target_col)
        return pd.DataFrame(columns=["feature", "mutual_information"])

    results = []
    for col in present:
        try:
            feat_bins = pd.qcut(sub[col], q=n_bins, labels=False, duplicates="drop")
            mi        = _mutual_info(feat_bins, target_bins)
        except Exception:
            mi = 0.0
        results.append({"feature": col, "mutual_information": mi})

    result_df = (
        pd.DataFrame(results)
        .sort_values("mutual_information", ascending=False)
        .reset_index(drop=True)
    )
    logger.info(
        "rank_features_by_mi: ranked %d features vs '%s' (top: %s, MI=%.4f).",
        len(result_df), target_col,
        result_df["feature"].iloc[0] if not result_df.empty else "N/A",
        result_df["mutual_information"].iloc[0] if not result_df.empty else 0.0,
    )
    return result_df


# ======================================================================
# FEATURE SELECTION
# Reduce the feature set before model training by removing columns that
# add no predictive signal (near-zero variance) or are near-duplicates
# of other features (high pairwise correlation).
# Thresholds are read from config.FEATURE_SELECTION — never hardcoded.
# ======================================================================

def select_features_by_variance(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_variance: float = None,
) -> list[str]:
    """
    Remove features whose variance falls below min_variance.

    Near-constant columns (e.g. a lag feature that is always NaN except
    for a handful of rows) add no signal and can destabilise training.

    Parameters
    ----------
    df           : feature DataFrame
    feature_cols : columns to evaluate
    min_variance : drop columns with var < this (default: FEATURE_SELECTION.MIN_VARIANCE)

    Returns
    -------
    Subset of feature_cols with sufficient variance.
    """
    from config import FEATURE_SELECTION
    threshold = min_variance if min_variance is not None else FEATURE_SELECTION.MIN_VARIANCE

    present = [c for c in feature_cols if c in df.columns]
    sub     = df[present].dropna()
    if sub.empty:
        return present

    var     = sub.var()
    kept    = [c for c in present if var.get(c, 0.0) >= threshold]
    dropped = [c for c in present if c not in kept]
    if dropped:
        logger.info(
            "select_features_by_variance: dropped %d near-zero-variance feature(s): %s",
            len(dropped), dropped,
        )
    return kept


def select_features_by_correlation(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = None,
) -> list[str]:
    """
    Remove features that are highly correlated with an earlier feature.

    For each pair (A, B) where A appears before B in feature_cols and
    |corr(A, B)| >= threshold, B is dropped (A is kept as the more
    fundamental feature). This is the standard greedy forward-selection
    approach used before linear and tree-based models.

    Parameters
    ----------
    df           : feature DataFrame
    feature_cols : columns to evaluate (order matters — earlier = higher priority)
    threshold    : drop if |corr| >= this (default: FEATURE_SELECTION.CORR_THRESHOLD)

    Returns
    -------
    Subset of feature_cols with pairwise |corr| < threshold.
    """
    from config import FEATURE_SELECTION
    cutoff = threshold if threshold is not None else FEATURE_SELECTION.CORR_THRESHOLD

    present = [c for c in feature_cols if c in df.columns]
    if len(present) < 2:
        return present

    sub  = df[present].dropna()
    if sub.empty:
        return present

    corr  = sub.corr().abs()
    # Upper triangle: corr[i,j] where j > i.
    # corr[col_j] in the upper matrix = correlations of col_j with all
    # earlier features. If any >= cutoff, col_j is redundant → drop it.
    upper    = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop  = {col for col in upper.columns if (upper[col] >= cutoff).any()}
    kept     = [c for c in present if c not in to_drop]
    if to_drop:
        logger.info(
            "select_features_by_correlation: dropped %d highly correlated feature(s) "
            "(|r| >= %.2f): %s",
            len(to_drop), cutoff, sorted(to_drop),
        )
    return kept


def select_features(
    df: pd.DataFrame,
    feature_cols: list[str] = None,
    corr_threshold: float = None,
    min_variance: float = None,
) -> list[str]:
    """
    Return a clean, non-redundant feature list ready for model training.

    Applies two passes in sequence:
    1. Variance filter  — drops near-constant columns.
    2. Correlation filter — drops near-duplicate predictors.

    Both thresholds come from config.FEATURE_SELECTION by default so they
    are never hardcoded.

    Parameters
    ----------
    df             : feature DataFrame (output of build_feature_set())
    feature_cols   : columns to consider; defaults to all FEATURE_NAMES in df
    corr_threshold : override FEATURE_SELECTION.CORR_THRESHOLD
    min_variance   : override FEATURE_SELECTION.MIN_VARIANCE

    Returns
    -------
    List of feature column names that passed both filters.
    """
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_NAMES if c in df.columns]

    n_in = len(feature_cols)
    cols = select_features_by_variance(df, feature_cols, min_variance)
    cols = select_features_by_correlation(df, cols, corr_threshold)

    logger.info(
        "select_features: %d → %d features after variance + correlation filtering.",
        n_in, len(cols),
    )
    return cols
