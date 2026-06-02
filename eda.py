"""
========================================================================
TASK 2: Exploratory Data Analysis (EDA) -- 100 Most Active Stocks
========================================================================
Single source of truth for ALL statistics in this project.
Outputs: console report + stats_results.py (imported by visualization
         and dashboard so nothing is recalculated elsewhere)

Architecture note
-----------------
Reads data via data_loader so it benefits from the Parquet cache
instead of hitting openpyxl on every run. Called automatically by
scraper.py after each daily scrape -- no manual execution needed.
========================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings

from config import PATHS
import data_loader

warnings.filterwarnings("ignore")


def _py_float(value, digits=4, scientific=False):
    """Format a numeric value as valid Python for stats_results.py."""
    if pd.isna(value):
        return 'float("nan")'
    fmt = f".{digits}{'e' if scientific else 'f'}"
    return format(float(value), fmt)


def main():
    print("\n" + "=" * 60)
    print("      STEP 1: INITIAL DATA INGESTION & TYPE CHECK")
    print("=" * 60)

    try:
        df_full = data_loader.load_price_data()
        print(f"Success: Loaded {df_full.shape[0]} rows and {df_full.shape[1]} columns.")

        if "Date" in df_full.columns:
            df_full["Date"] = pd.to_datetime(df_full["Date"])
            n_days = df_full["Date"].nunique()
            dates  = sorted(df_full["Date"].dt.date.unique())
            latest = df_full["Date"].max()
            df     = df_full[df_full["Date"] == latest].copy().reset_index(drop=True)
            print(f"  Days in dataset : {n_days}  ({dates[0]} to {dates[-1]})")
            print(f"  Analysing latest: {latest.date()}  ({len(df)} stocks)")
        else:
            df = df_full.copy()

        print("\n--- Column Data Types ---")
        print(df.dtypes)
        print("\n--- First 5 Rows ---")
        print(df.head(5).to_string(index=False))

    except Exception as e:
        print(f"\nError: Could not load dataset -- {e}")
        print("Run scraper.py first to generate the dataset.")
        raise RuntimeError("Could not load dataset for EDA.") from e

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("      STEP 2: CORE QUESTIONS GUIDING THIS EDA")
    print("=" * 60)

    questions = [
        "What is the average and typical price of a heavily traded stock today?",
        "Do high-priced stocks see larger absolute price swings than lower-priced stocks?",
        "Is market activity leaning positive (gainers) or negative (losers) overall?",
        "Are there any massive price anomalies or extreme outliers in our dataset?",
        "Are there any formatting gaps or data quality issues that require attention?",
    ]
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("      STEP 3: DATA INTEGRITY & QUALITY CHECK")
    print("=" * 60)

    original_cols = ["Symbol", "Company Name", "Price", "Change", "% Change"]
    null_counts = df[original_cols].isnull().sum()
    total_nulls = int(null_counts.sum())
    print("--- Null / Blank Values per Column ---")
    print(null_counts.to_string())

    df["Market_Action"] = np.where(df["Change"] > 0, "Gainer", "Loser")
    df.loc[df["Change"] == 0, "Market_Action"] = "Flat"

    duplicates = df.duplicated(subset=["Symbol"]).sum()
    print(f"\nDuplicate Tickers Found      : {duplicates}")

    broken_prices = df[df["Price"] <= 0]
    print(f"Invalid Prices (<= $0)       : {len(broken_prices)}")

    df["_prev_price"] = df["Price"] - df["Change"]
    df["_computed_pct"] = np.where(
        df["_prev_price"] != 0,
        (df["Change"] / df["_prev_price"]) * 100,
        np.nan,
    )
    df["_pct_discrepancy"] = (df["_computed_pct"] - df["% Change"]).abs()
    problem_rows = df[df["_pct_discrepancy"] > 0.1]

    if not problem_rows.empty:
        print(f"\n  Data Quality Note -- % Change Rounding Discrepancy:")
        print(
            f"   {len(problem_rows)} low-priced stocks have up to "
            f"{df['_pct_discrepancy'].max():.2f}pp discrepancy."
        )
        print("   Cause: Yahoo Finance rounds Change to 2 decimal places.")
        print(f"   Affected: {', '.join(problem_rows['Symbol'].tolist())}")
        print("   Impact: Negligible. Using stored % Change values.")

    df.drop(columns=["_prev_price", "_computed_pct", "_pct_discrepancy"], inplace=True)

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("      STEP 4: STATISTICAL DISTRIBUTION & DESCRIPTIVES")
    print("=" * 60)

    print("--- Summary Statistics ---")
    print(df[["Price", "Change", "% Change"]].describe().round(2))

    price_mean   = df["Price"].mean()
    price_median = df["Price"].median()
    price_std    = df["Price"].std()
    price_skew   = df["Price"].skew()

    print(f"\n--- Price Distribution ---")
    print(f"  Mean     : ${price_mean:.2f}")
    print(f"  Median   : ${price_median:.2f}")
    print(f"  Std Dev  : ${price_std:.2f}")
    print(f"  Skewness : {price_skew:.2f}")

    if price_skew > 1:
        print(f"\n  Interpretation: Heavily right-skewed (skew={price_skew:.2f}).")
        print(f"  Mean (${price_mean:.2f}) pulled above median (${price_median:.2f}) by mega-caps.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("      STEP 5: MARKET BREADTH & TREND DISCOVERY")
    print("=" * 60)

    gainers = df[df["Market_Action"] == "Gainer"]
    losers  = df[df["Market_Action"] == "Loser"]
    flat    = df[df["Market_Action"] == "Flat"]

    n_gain = len(gainers)
    n_loss = len(losers)
    n_flat = len(flat)
    gain_pct = n_gain / len(df) * 100 if len(df) else 0.0
    loss_pct = n_loss / len(df) * 100 if len(df) else 0.0
    flat_pct = n_flat / len(df) * 100 if len(df) else 0.0

    top_sym  = df.loc[df["% Change"].idxmax(), "Symbol"]
    top_name = df.loc[df["% Change"].idxmax(), "Company Name"]
    top_pct  = df["% Change"].max()
    bot_sym  = df.loc[df["% Change"].idxmin(), "Symbol"]
    bot_name = df.loc[df["% Change"].idxmin(), "Company Name"]
    bot_pct  = df["% Change"].min()
    avg_chg  = df["% Change"].mean()

    gainer_pct_median   = gainers["% Change"].median()
    loser_pct_median    = losers["% Change"].median()
    gainer_price_median = gainers["Price"].median()
    loser_price_median  = losers["Price"].median()
    price_max = df["Price"].max()
    price_min = df["Price"].min()

    print("--- Session Breadth ---")
    print(f"  Advancing (Gainers)  : {n_gain:>3}  ({gain_pct:.1f}%)")
    print(f"  Declining (Losers)   : {n_loss:>3}  ({loss_pct:.1f}%)")
    print(f"  Unchanged (Flat)     : {n_flat:>3}  ({flat_pct:.1f}%)")
    print(f"\n  Top Gainer : {top_sym} {top_pct:+.2f}%  ({top_name})")
    print(f"  Top Loser  : {bot_sym} {bot_pct:.2f}%  ({bot_name})")

    print("\n--- Group Averages ---")
    print(
        df.groupby("Market_Action")[["Price", "Change", "% Change"]]
        .mean()
        .round(2)
        .to_string()
    )

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("      STEP 6: HYPOTHESIS TESTING")
    print("=" * 60)

    print("--- Normality Check (Shapiro-Wilk on Price) ---")
    price_sample = df["Price"].dropna()
    if len(price_sample) >= 3:
        sw_stat, sw_p = stats.shapiro(price_sample)
        print(f"  W-statistic : {sw_stat:.4f}")
        print(f"  P-value     : {sw_p:.4e}")
        non_normal = sw_p < 0.05
        if non_normal:
            print("  Result: NOT normally distributed (p < 0.05). Using non-parametric tests.")
    else:
        sw_stat, sw_p, non_normal = np.nan, np.nan, False
        print("  Not enough non-null prices for Shapiro-Wilk (need at least 3).")

    print("\n--- Test 1: Spearman Rank Correlation (Price vs % Change) ---")
    if df["Price"].nunique(dropna=True) > 1 and df["% Change"].nunique(dropna=True) > 1:
        spearman_corr, spearman_p = stats.spearmanr(df["Price"], df["% Change"])
    else:
        spearman_corr, spearman_p = np.nan, np.nan
    abs_corr  = abs(spearman_corr)
    strength  = (
        "undefined"  if pd.isna(abs_corr) else
        "negligible" if abs_corr < 0.1 else
        "weak"       if abs_corr < 0.3 else
        "moderate"   if abs_corr < 0.5 else
        "strong"
    )
    direction = "negative" if spearman_corr < 0 else "positive" if spearman_corr >= 0 else "undefined"

    print(f"  Spearman rho : {spearman_corr:.4f}")
    print(f"  P-value      : {spearman_p:.4f}")
    print(f"  Strength     : {strength} {direction} correlation")
    if pd.isna(spearman_p):
        print("  Significant  : N/A - correlation undefined for constant or insufficient data.")
    elif spearman_p < 0.05:
        print(f"  Significant  : YES (p < 0.05) - but {strength}, explains little variance.")
    else:
        print(f"  Significant  : NO (p >= 0.05) - {strength}, not reliable for this run.")

    corr_cols = ["Price", "Change", "% Change"]
    spearman_matrix = []
    for c1 in corr_cols:
        row = []
        for c2 in corr_cols:
            if df[c1].nunique(dropna=True) > 1 and df[c2].nunique(dropna=True) > 1:
                row.append(round(stats.spearmanr(df[c1], df[c2])[0], 2))
            else:
                row.append(np.nan)
        spearman_matrix.append(row)

    print("\n--- Test 2: Mann-Whitney U Test (Gainer Prices vs Loser Prices) ---")
    g_prices = gainers["Price"].dropna().values
    l_prices = losers["Price"].dropna().values
    if len(g_prices) > 0 and len(l_prices) > 0:
        u_stat, mw_p = stats.mannwhitneyu(g_prices, l_prices, alternative="two-sided")
        mw_reject = mw_p < 0.05
    else:
        u_stat, mw_p, mw_reject = np.nan, np.nan, False

    print(f"  U-statistic         : {u_stat:.1f}")
    print(f"  P-value             : {mw_p:.4f}")
    print(f"  Gainer median price : ${np.median(g_prices):.2f}" if len(g_prices) else "  Gainer median price : N/A")
    print(f"  Loser  median price : ${np.median(l_prices):.2f}" if len(l_prices) else "  Loser  median price : N/A")
    if pd.isna(mw_p):
        print("  Result: N/A. Need at least one gainer and one loser.")
    elif mw_reject:
        print("  Result: REJECT H0. Gainer and loser prices differ significantly.")
    else:
        print("  Result: FAIL TO REJECT H0. Price does not predict direction.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("      STEP 7: OUTLIER DETECTION (IQR METHOD)")
    print("=" * 60)

    q1          = df["Price"].quantile(0.25)
    q3          = df["Price"].quantile(0.75)
    iqr         = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    outliers   = df[
        (df["Price"] < lower_fence) | (df["Price"] > upper_fence)
    ].sort_values("Price", ascending=False)
    n_outliers = len(outliers)

    print(f"  Q1 (25th pct)  : ${q1:.2f}")
    print(f"  Q3 (75th pct)  : ${q3:.2f}")
    print(f"  IQR            : ${iqr:.2f}")
    print(f"  Lower fence    : ${lower_fence:.2f}  (negative -- no lower outliers possible)")
    print(f"  Upper fence    : ${upper_fence:.2f}")
    print(f"  Outliers found : {n_outliers}")
    print("\n  All Price Outliers:")
    print(
        outliers[["Symbol", "Company Name", "Price", "Change", "% Change"]]
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("      STEP 8: DATA LIMITATIONS")
    print("=" * 60)

    print(
        "\n  1. Latest-Day Snapshot: EDA analyses the latest scrape only;"
        " historical trend analytics are handled by historical_metrics.py."
        "\n  2. No Sector Classification: Cannot isolate sector-driven movements."
        "\n  3. No Fundamental Data: P/E, market cap, volume absent."
        "\n  4. % Change Rounding: Minor (<0.16pp) errors on sub-$5 stocks.\n"
    )

    # ------------------------------------------------------------------
    print("=" * 60)
    print("      STEP 9: EXECUTIVE SUMMARY")
    print("=" * 60)

    spearman_sig_text = (
        "undefined" if pd.isna(spearman_p)
        else "significant" if spearman_p < 0.05
        else "not significant"
    )
    mw_text = (
        "prices differ by direction" if mw_reject
        else "price does NOT predict daily direction"
    )

    print(
        f"\n  Dataset: {len(df)} stocks, {len(original_cols)} columns."
        f" {total_nulls} null values. {duplicates} duplicate tickers."
        f"\n\n  PRICE STRUCTURE"
        f"\n  Skewness = {price_skew:.2f}."
        f" Mean ${price_mean:.2f} >> Median ${price_median:.2f}."
        f"\n  {n_outliers} mega-cap outliers above IQR fence of ${upper_fence:.0f}."
        f"\n\n  MARKET BREADTH"
        f"\n  {n_gain} advancing ({gain_pct:.1f}%) / "
        f"{n_loss} declining ({loss_pct:.1f}%) / {n_flat} flat ({flat_pct:.1f}%)."
        f"\n  Top: {top_sym} +{top_pct:.2f}%  |  Worst: {bot_sym} {bot_pct:.2f}%"
        f"\n\n  HYPOTHESIS RESULTS"
        f"\n  Spearman rho = {spearman_corr:.4f} (p = {spearman_p:.4f})"
        f" - {strength} {direction}, {spearman_sig_text}."
        f"\n  Mann-Whitney p = {mw_p:.4f} - {mw_text}.\n"
    )
    print("=" * 60)

    # ------------------------------------------------------------------
    # EXPORT RESULTS -- used by visualization.py and dashboard.py
    # ------------------------------------------------------------------
    results = (
        '"""\n'
        "Auto-generated by eda.py -- do not edit manually.\n"
        "Single source of truth for all statistics in this project.\n"
        "visualization.py and dashboard.py import from here.\n"
        '"""\n'
        "\n"
        "# -- Price descriptives --\n"
        f"PRICE_MEAN     = {price_mean:.4f}\n"
        f"PRICE_MEDIAN   = {price_median:.4f}\n"
        f"PRICE_STD      = {price_std:.4f}\n"
        f"PRICE_SKEW     = {price_skew:.4f}\n"
        f"PRICE_MAX      = {price_max:.2f}\n"
        f"PRICE_MIN      = {price_min:.2f}\n"
        "\n"
        "# -- Market breadth --\n"
        f"N_TOTAL        = {len(df)}\n"
        f"N_GAIN         = {n_gain}\n"
        f"N_LOSS         = {n_loss}\n"
        f"N_FLAT         = {n_flat}\n"
        f"AVG_CHG        = {avg_chg:.4f}\n"
        f'TOP_SYM        = "{top_sym}"\n'
        f'TOP_NAME       = "{top_name}"\n'
        f"TOP_PCT        = {top_pct:.2f}\n"
        f'BOT_SYM        = "{bot_sym}"\n'
        f'BOT_NAME       = "{bot_name}"\n'
        f"BOT_PCT        = {bot_pct:.2f}\n"
        f"GAINER_PCT_MED = {_py_float(gainer_pct_median)}\n"
        f"LOSER_PCT_MED  = {_py_float(loser_pct_median)}\n"
        f"GAINER_PRC_MED = {_py_float(gainer_price_median)}\n"
        f"LOSER_PRC_MED  = {_py_float(loser_price_median)}\n"
        "\n"
        "# -- Normality (Shapiro-Wilk) --\n"
        f"SW_STAT        = {_py_float(sw_stat)}\n"
        f"SW_P           = {_py_float(sw_p, digits=6, scientific=True)}\n"
        f"NON_NORMAL     = {non_normal}\n"
        "\n"
        "# -- Spearman correlation --\n"
        f"SP_CORR        = {_py_float(spearman_corr)}\n"
        f"SP_P           = {_py_float(spearman_p)}\n"
        f'SP_STRENGTH    = "{strength}"\n'
        f'SP_DIRECTION   = "{direction}"\n'
        "# Full 3x3 matrix: [Price, Change, % Change]\n"
        f"SPEARMAN_MATRIX = {[[None if pd.isna(v) else float(v) for v in row] for row in spearman_matrix]}\n"
        "\n"
        "# -- Mann-Whitney U --\n"
        f"U_STAT         = {_py_float(u_stat, digits=1)}\n"
        f"MW_P           = {_py_float(mw_p)}\n"
        f"MW_REJECT      = {mw_reject}\n"
        "\n"
        "# -- IQR Outlier Detection --\n"
        f"Q1             = {q1:.4f}\n"
        f"Q3             = {q3:.4f}\n"
        f"IQR_VAL        = {iqr:.4f}\n"
        f"LOWER_FENCE    = {lower_fence:.4f}\n"
        f"UPPER_FENCE    = {upper_fence:.4f}\n"
        f"N_OUTLIERS     = {n_outliers}\n"
        f"OUTLIER_SYMS   = {outliers['Symbol'].tolist()}\n"
        f"OUTLIER_PRICES = {outliers['Price'].tolist()}\n"
        "\n"
        "# -- Phase 2 lightweight intelligence labels --\n"
        'TECHNICAL_SUMMARY = "Technical cache generated by technical_indicators.py when the full pipeline runs."\n'
        'FORECAST_SUMMARY = "Forecast cache generated by forecasting.py when the full pipeline runs."\n'
        'SIGNAL_SUMMARY = "Signal cache generated by signal_engine.py when the full pipeline runs."\n'
        'VOLATILITY_REGIME_LABEL = "See historical_metrics.volatility_regime for rolling regime state."\n'
        'AI_SUMMARY_OUTPUT = "AI briefing generated in ai_narratives.py and rendered by dashboard.py."\n'
        'MARKET_INTELLIGENCE_LABEL = "Data -> Intelligence -> Forecasting -> Signals -> AI Interpretation -> Dashboard"\n'
    )

    with open(PATHS.STATS_RESULTS, "w", encoding="utf-8") as f:
        f.write(results)

    print(f"\nstats_results.py saved -- imported by visualization.py and dashboard.py\n")


if __name__ == "__main__":
    main()
