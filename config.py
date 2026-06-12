"""
========================================================================
config.py -- Market Pulse Platform Configuration
========================================================================
Single source of truth for every constant, path, threshold, and
setting used across the platform.

Rules
-----
- All modules import from here instead of defining their own constants.
- Changing a value here propagates everywhere automatically.
- No business logic lives here -- only configuration values.
- All paths are relative to the project root (the folder containing
  main.py). Modules resolve them at runtime via BASE_DIR.

Usage
-----
    from config import PATHS, SENTIMENT, ROLLING, LOGGING, DASHBOARD
    excel_file = PATHS.EXCEL_FILE
    threshold  = SENTIMENT.BULLISH_THRESHOLD
========================================================================
"""

import os
from dataclasses import dataclass
from typing import List


# -- Project root -- all relative paths resolve from here ---------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ======================================================================
# PATH CONFIGURATION
# Every file the platform reads or writes is declared here.
# ======================================================================

@dataclass(frozen=True)
class _Paths:
    # -- Input / storage -----------------------------------------------
    EXCEL_FILE:         str = os.path.join(BASE_DIR, "most_active_stocks_dataset.xlsx")
    PRICE_PARQUET:      str = os.path.join(BASE_DIR, "most_active_stocks_dataset.parquet")
    SENTIMENT_PARQUET:  str = os.path.join(BASE_DIR, "sentiment_history.parquet")
    HIST_PARQUET:       str = os.path.join(BASE_DIR, "historical_metrics.parquet")
    TECHNICAL_PARQUET:  str = os.path.join(BASE_DIR, "technical_metrics.parquet")
    FORECAST_PARQUET:   str = os.path.join(BASE_DIR, "forecast_metrics.parquet")
    SIGNAL_PARQUET:     str = os.path.join(BASE_DIR, "signal_metrics.parquet")
    ANOMALY_PARQUET:    str = os.path.join(BASE_DIR, "anomaly_metrics.parquet")
    STATS_RESULTS:      str = os.path.join(BASE_DIR, "stats_results.py")

    # -- Output --------------------------------------------------------
    DASHBOARD_HTML:     str = os.path.join(BASE_DIR, "market_pulse_dashboard.html")
    OUTPUTS_DIR:        str = os.path.join(BASE_DIR, "outputs")

    # -- Logs ----------------------------------------------------------
    LOGS_DIR:           str = os.path.join(BASE_DIR, "logs")

    # -- Excel sheet names ---------------------------------------------
    PRICE_SHEET:        str = "Sheet1"
    SENTIMENT_SHEET:    str = "sentiment_history"


PATHS = _Paths()


# ======================================================================
# SENTIMENT CONFIGURATION
# ======================================================================

@dataclass(frozen=True)
class _Sentiment:
    # VADER compound score thresholds (standard VADER documentation)
    BULLISH_THRESHOLD:  float = 0.05
    BEARISH_THRESHOLD:  float = -0.05

    # Maximum headlines to fetch per ticker per run
    MAX_HEADLINES:      int   = 10

    # Seconds to wait between yfinance API calls (rate-limit guard)
    FETCH_DELAY:        float = 0.25

    # % of universe that must be bullish for a risk-off regime label
    # (different from BREADTH.BULLISH_PCT which classifies price breadth)
    RISK_OFF_BULL_PCT:  float = 30.0

    # NLP model for headline scoring. "ProsusAI/finbert" is the gold standard
    # for financial text; requires `pip install transformers torch`.
    # Falls back to VADER automatically if transformers is not installed.
    NLP_MODEL:      str = "ProsusAI/finbert"
    NLP_BATCH_SIZE: int = 16    # headlines per FinBERT forward pass
    NLP_MAX_LENGTH: int = 512   # token limit (model maximum)

    # Column schema for the sentiment history DataFrame
    COLUMNS: tuple = (
        "Date", "Symbol", "Sentiment_Score",
        "Sentiment_Label", "Headlines_Used", "Top_Headline",
    )


SENTIMENT = _Sentiment()


# ======================================================================
# ROLLING WINDOW CONFIGURATION
# ======================================================================

@dataclass(frozen=True)
class _Rolling:
    # Windows used in historical_metrics.py and dashboard.py
    SHORT_WINDOW:  int = 7   # 7-day rolling metrics
    LONG_WINDOW:   int = 30  # 30-day rolling metrics

    # Minimum appearances for cumulative performance chart
    MIN_APPEARANCE_PCT: float = 0.50  # symbol must appear in >= 50% of days

    # Top N symbols to track in frequent movers analysis
    TOP_MOVERS_N:       int   = 15
    TOP_MOVERS_PER_DAY: int   = 10

    # Top N sessions to highlight in extreme sessions analysis
    EXTREME_SESSIONS_N: int   = 5

    # Volatility regime thresholds (ratio vs rolling baseline)
    VOL_ELEVATED:       float = 1.5
    VOL_SLIGHTLY_HIGH:  float = 1.1
    VOL_NORMAL_LOW:     float = 0.9

    # Tail-risk quantile for VaR and CVaR rolling metrics in historical_metrics.py
    # 0.05 = 5th percentile (worst 5% of market days)
    VAR_QUANTILE:       float = 0.05


ROLLING = _Rolling()


# ======================================================================
# MARKET BREADTH THRESHOLDS
# Used by historical_metrics.py and dashboard.py for regime labelling.
# ======================================================================

@dataclass(frozen=True)
class _Breadth:
    BULLISH_PCT:    float = 60.0   # >= this -> Bullish regime (static)
    BEARISH_PCT:    float = 40.0   # <= this -> Bearish regime (static)
    BROAD_ADVANCE:  float = 70.0   # breadth label thresholds
    MOD_ADVANCE:    float = 55.0
    MIXED:          float = 45.0
    MOD_DECLINE:    float = 30.0

    # Adaptive regime: classify using rolling breadth percentile rank so
    # the same raw breadth reading means different things in different environments.
    REGIME_LOOKBACK: int   = 63    # rolling window for percentile (~1 quarter)
    REGIME_BULL_PCT: float = 0.65  # rank >= this → adaptive Bullish
    REGIME_BEAR_PCT: float = 0.35  # rank <= this → adaptive Bearish


BREADTH = _Breadth()


# ======================================================================
# PRICE TIER CONFIGURATION
# Used by dashboard.py for price bucket charts.
# ======================================================================

@dataclass(frozen=True)
class _PriceTiers:
    BINS:   tuple = (0, 10, 25, 50, 100, 200, 9999)
    LABELS: tuple = ("<$10", "$10-25", "$25-50", "$50-100", "$100-200", ">$200")


PRICE_TIERS = _PriceTiers()


# ======================================================================
# SCRAPER CONFIGURATION
# ======================================================================

@dataclass(frozen=True)
class _Scraper:
    # Primary data source: JSON screener API (structured, layout-independent)
    SCREENER_API_URL:   str   = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
    SCREENER_SCRID:     str   = "most_actives"

    # Fallback: HTML page scraper (used only when JSON API is unavailable)
    TARGET_URL:         str   = "https://finance.yahoo.com/markets/stocks/most-active/?count=100"

    RETRY_ATTEMPTS:     int   = 3
    RETRY_DELAY_SEC:    int   = 3
    REQUEST_TIMEOUT:    int   = 15
    MIN_CHANGE_RECOMP:  float = 0.01  # min $ change to trigger % recomputation
    EXPECTED_STOCKS:    int   = 100


SCRAPER = _Scraper()


# ======================================================================
# LOGGING CONFIGURATION
# ======================================================================

@dataclass(frozen=True)
class _Logging:
    # Log format used by all handlers
    FORMAT:     str = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    DATE_FMT:   str = "%H:%M:%S"

    # Rotating file handler settings
    MAX_BYTES:  int = 5 * 1024 * 1024   # 5 MB per log file
    BACKUP_COUNT: int = 7               # keep last 7 log files

    # Console handler level (file handler always logs at DEBUG)
    CONSOLE_LEVEL: str = "INFO"


LOGGING_CFG = _Logging()


# ======================================================================
# DASHBOARD CONFIGURATION
# ======================================================================

@dataclass(frozen=True)
class _Dashboard:
    # Output filename
    OUTPUT_FILE:    str = os.path.join(BASE_DIR, "market_pulse_dashboard.html")

    # Scatter bubble size multiplier for sentiment chart
    BUBBLE_MULTIPLIER: float = 1.5
    BUBBLE_MIN_SIZE:   float = 2.0

    # Sentiment vs price correlation minimum sample size
    MIN_CORR_SAMPLE:   int   = 3

    # Number of top/bottom stocks shown in mini-panels
    MINI_PANEL_N:      int   = 5

    # Screener table row height (px)
    SCREENER_ROW_HEIGHT: int = 27


DASHBOARD = _Dashboard()


# ======================================================================
# VALIDATION CONFIGURATION
# Used by validators.py to define what constitutes valid data.
# ======================================================================

@dataclass(frozen=True)
class _Validation:
    # Required columns in the price dataset
    PRICE_REQUIRED_COLS: tuple = (
        "Date", "Symbol", "Company Name", "Price", "Change", "% Change"
    )

    # Required columns in the sentiment dataset
    SENTIMENT_REQUIRED_COLS: tuple = (
        "Date", "Symbol", "Sentiment_Score", "Sentiment_Label",
        "Headlines_Used", "Top_Headline",
    )

    # Price sanity bounds
    MIN_VALID_PRICE:    float = 0.01
    MAX_VALID_PRICE:    float = 100_000.0

    # Minimum rows expected per trading day
    MIN_ROWS_PER_DAY:   int   = 50

    # Maximum allowed duplicate rate before raising a warning
    MAX_DUPE_RATE:      float = 0.01   # 1%


VALIDATION = _Validation()


# ======================================================================
# PHASE 2 -- TECHNICAL INDICATORS CONFIGURATION
# ======================================================================

@dataclass(frozen=True)
class _TechnicalIndicators:
    # Trend classification gap threshold (relative to long MA)
    TREND_GAP_THRESHOLD:   float = 0.005   # 0.5% gap to classify as trending

    # Signal strength z-score threshold for breakout classification
    BREAKOUT_Z:            float = 1.5

    # Minimum history required for symbol-level indicators
    MIN_SYMBOL_HISTORY:    int   = 3


TECHNICAL = _TechnicalIndicators()


# ======================================================================
# PHASE 2 -- FORECASTING CONFIGURATION
# ======================================================================

@dataclass(frozen=True)
class _Forecasting:
    # Default forecast horizon in business days
    DEFAULT_HORIZON:       int   = 5

    # Default forecast method
    DEFAULT_METHOD:        str   = "linear"   # "sma" | "ema" | "linear"

    # Minimum observations for a meaningful linear forecast
    MIN_LINEAR_OBS:        int   = 5

    # OLS fit window (last N days used for linear trend estimation)
    LINEAR_FIT_WINDOW:     int   = 30

    # Confidence interval z-score (1.96 = 95%)
    CONFIDENCE_Z:          float = 1.96


FORECASTING = _Forecasting()


# ======================================================================
# PHASE 2 -- SIGNAL ENGINE CONFIGURATION
# ======================================================================

@dataclass(frozen=True)
class _Signals:
    # z-score threshold for momentum breakout signal
    BREAKOUT_Z:            float = 1.5

    # Ratio vs baseline to classify as volatility spike
    VOL_SPIKE_RATIO:       float = 1.5

    # Minimum absolute VADER score for sentiment divergence check
    SENT_DIVERGE_MIN:      float = 0.1

    # Price must be below this % change for bearish divergence
    SENT_DIVERGE_PRICE:    float = -0.5

    # z-score threshold for unusual mover (full history)
    UNUSUAL_MOVER_Z:       float = 2.0

    # Number of stocks returned in opportunity ranking
    OPPORTUNITY_TOP_N:     int   = 10

    # Sentiment scale factor in opportunity_ranking score formula.
    # VADER scores are in [-1, 1]; this scales them to the same order of
    # magnitude as momentum and z-score terms before weighting.
    SENT_SCALE_FACTOR:     float = 20.0


SIGNALS = _Signals()


# ======================================================================
# PHASE 2 -- ANOMALY DETECTION CONFIGURATION
# ======================================================================

@dataclass(frozen=True)
class _Anomaly:
    # z-score for individual stock price gap anomaly
    PRICE_GAP_Z:           float = 2.5

    # z-score for market-wide volatility regime anomaly
    VOL_REGIME_Z:          float = 2.0

    # z-score for breadth extreme anomaly
    BREADTH_EXTREME_Z:     float = 2.0

    # Minimum fraction of stocks moving together for cluster spike
    CLUSTER_MIN_PCT:       float = 0.30

    # Minimum abs % change to count in cluster spike
    CLUSTER_MIN_MOVE:      float = 2.0

    # Rolling correlation threshold for sentiment-volatility divergence
    SENT_VOL_CORR_THRESHOLD: float = -0.5

    # Z-score thresholds for anomaly severity classification
    SEVERITY_CRITICAL_Z:  float = 3.5  # abs(z) >= this → Critical
    SEVERITY_HIGH_Z:      float = 3.0  # abs(z) >= this → High (else Moderate)

    # Minimum baseline observations required before firing a z-score anomaly.
    # With fewer points the standard deviation is unreliable (t-dist df too small).
    # Per-symbol detectors (price_gap) use the same threshold; yfinance history
    # typically provides 400-500 days so this only affects brand-new installs.
    MIN_BASELINE_DAYS:    int   = 20


ANOMALY = _Anomaly()


# ======================================================================
# HISTORICAL DATA PATHS
# Separate directory tree for OHLCV and macro data so it never
# conflicts with the daily scrape parquets.
# ======================================================================

@dataclass(frozen=True)
class _HistoricalPaths:
    # Root directories
    DATA_DIR:        str = os.path.join(BASE_DIR, "data")
    RAW_DIR:         str = os.path.join(BASE_DIR, "data", "raw")
    PROCESSED_DIR:   str = os.path.join(BASE_DIR, "data", "processed")
    HISTORICAL_DIR:  str = os.path.join(BASE_DIR, "data", "historical")
    MACRO_DIR:       str = os.path.join(BASE_DIR, "data", "macro")

    # OHLCV parquet pattern: data/historical/{symbol}.parquet
    OHLCV_DIR:       str = os.path.join(BASE_DIR, "data", "historical")

    # Processed feature parquet pattern: data/processed/{symbol}_features.parquet
    FEATURES_DIR:    str = os.path.join(BASE_DIR, "data", "processed")

    # FRED macro parquet: data/macro/{series_id}.parquet
    FRED_DIR:        str = os.path.join(BASE_DIR, "data", "macro")

    # SQLite database (optional — used only when USE_SQLITE=True)
    SQLITE_DB:       str = os.path.join(BASE_DIR, "data", "market_pulse.db")


HISTORICAL_PATHS = _HistoricalPaths()


# ======================================================================
# HISTORICAL DATA FETCH CONFIGURATION
# ======================================================================

@dataclass(frozen=True)
class _HistoricalFetch:
    # Default lookback period for yfinance OHLCV pulls
    DEFAULT_PERIOD:      str   = "2y"        # yfinance period string
    DEFAULT_INTERVAL:    str   = "1d"        # daily bars

    # Minimum trading days required before indicators are meaningful
    MIN_OHLCV_ROWS:      int   = 60

    # Rate-limit guard between yfinance ticker fetches (seconds)
    FETCH_DELAY_SEC:     float = 0.3

    # Retry attempts for failed fetches
    RETRY_ATTEMPTS:      int   = 3
    RETRY_DELAY_SEC:     float = 2.0

    # Whether to use SQLite in addition to parquet
    USE_SQLITE:          bool  = False

    # Outlier detection (used by pipelines/preprocess.flag_outliers)
    OUTLIER_WINDOW:      int   = 30    # rolling window for vol baseline
    OUTLIER_N_SIGMA:     float = 3.0   # z-score threshold to flag an outlier

    # Default tickers to fetch historical OHLCV for.
    # Populated from the daily scrape symbols at runtime;
    # this list is the static fallback.
    DEFAULT_TICKERS: tuple = (
        "NVDA", "AAPL", "MSFT", "AMZN", "TSLA",
        "META", "GOOGL", "AMD", "INTC", "PLTR",
    )


HIST_FETCH = _HistoricalFetch()


# ======================================================================
# FRED MACRO CONFIGURATION
# ======================================================================

@dataclass(frozen=True)
class _Fred:
    # Series to fetch. Key = local alias, value = FRED series ID.
    SERIES: dict = None   # populated below (dataclass limitation)

    # Default lookback
    DEFAULT_PERIOD_YEARS: int = 5

    # Rate-limit guard (FRED allows 120 req/min for free keys)
    FETCH_DELAY_SEC: float = 0.5


# Frozen dataclasses can't have mutable defaults — use a module-level dict.
FRED_SERIES: dict[str, str] = {
    "fed_funds_rate":   "FEDFUNDS",    # Federal Funds Effective Rate
    "cpi_yoy":          "CPIAUCSL",    # CPI All Urban Consumers
    "unemployment":     "UNRATE",      # Unemployment Rate
    "us_10y_yield":     "DGS10",       # 10-Year Treasury Constant Maturity
    "us_2y_yield":      "DGS2",        # 2-Year Treasury Constant Maturity
    "vix":              "VIXCLS",      # CBOE Volatility Index
    "sp500":            "SP500",       # S&P 500 Index
    "gdp_growth":       "A191RL1Q225SBEA",  # Real GDP Growth Rate
}

FRED_CFG = _Fred()


# ======================================================================
# INDICATOR CONFIGURATION
# Parameters for RSI, MACD, Bollinger Bands — new in historical layer.
# SMA/EMA windows reuse ROLLING.SHORT_WINDOW / ROLLING.LONG_WINDOW.
# ======================================================================

@dataclass(frozen=True)
class _Indicators:
    # RSI
    RSI_WINDOW:          int   = 14

    # MACD
    MACD_FAST:           int   = 12
    MACD_SLOW:           int   = 26
    MACD_SIGNAL:         int   = 9

    # Bollinger Bands
    BB_WINDOW:           int   = 20
    BB_STD:              float = 2.0

    # ATR (Average True Range)
    ATR_WINDOW:          int   = 14

    # Momentum lookback
    MOM_WINDOW:          int   = 10

    # Drawdown: rolling window for max-price lookback
    DRAWDOWN_WINDOW:     int   = 252   # ~1 trading year

    # Fixed SMA/EMA windows used in compute_ohlcv_indicators.
    # These are market-standard reference levels (20-day and 50-day SMAs
    # are widely tracked by institutional traders) so they're fixed, not
    # regime-adaptive. EMA_FAST and EMA_SLOW mirror the MACD periods.
    SMA_FAST_WINDOW:     int   = 20
    SMA_SLOW_WINDOW:     int   = 50

    # Regime-adaptive window parameters.
    # When recent volatility is elevated, shorter windows are more responsive.
    # When suppressed, longer windows reduce false signals.
    ADAPTIVE_LOOKBACK:   int   = 63    # vol percentile lookback (~1 quarter)
    ADAPTIVE_HIGH_PCT:   float = 0.80  # vol percentile >= this  → fast (shorter) windows
    ADAPTIVE_LOW_PCT:    float = 0.20  # vol percentile <= this  → slow (longer)  windows
    ADAPTIVE_FAST_MULT:  float = 0.70  # multiply base window in elevated-vol regime
    ADAPTIVE_SLOW_MULT:  float = 1.35  # multiply base window in suppressed-vol regime


INDICATORS = _Indicators()


# ======================================================================
# FEATURE SELECTION CONFIGURATION
# Thresholds used by feature_engineering.select_features().
# ======================================================================

@dataclass(frozen=True)
class _FeatureSelection:
    # Drop features whose pairwise absolute correlation >= this value.
    # 0.92 removes near-duplicate predictors while retaining complementary ones.
    CORR_THRESHOLD:  float = 0.92

    # Drop features whose variance falls below this floor.
    # Constant or near-constant columns add no predictive signal.
    MIN_VARIANCE:    float = 1e-5

    # Number of quantile bins used when discretising features for
    # mutual information estimation in rank_features_by_mi().
    MI_N_BINS:       int   = 10


FEATURE_SELECTION = _FeatureSelection()


# ======================================================================
# FINNHUB CONFIGURATION
# Used by finnhub_sentiment.py for historical news backfill.
# ======================================================================

@dataclass(frozen=True)
class _Finnhub:
    BASE_URL:       str   = "https://finnhub.io/api/v1"
    CALL_DELAY_SEC: float = 1.1    # keeps under 60 req/min free-tier limit
    REQUEST_TIMEOUT: int  = 12


FINNHUB_CFG = _Finnhub()


# ======================================================================
# NARRATIVE THRESHOLDS
# Phrase-mapping cutoffs used by ai_narratives.py. Kept in config so
# the language the system uses is consistent with the analytics thresholds
# and can be tuned without editing prose logic.
# ======================================================================

@dataclass(frozen=True)
class _Narrative:
    # Volatility z-score → descriptive phrase (_vol_phrase)
    VOL_SEVERE_Z:     float =  2.5
    VOL_ELEVATED_Z:   float =  1.5
    VOL_SLIGHT_Z:     float =  0.5
    VOL_CALM_Z:       float = -0.5
    VOL_SUPPRESSED_Z: float = -1.5

    # Average % change → momentum descriptor (_momentum_phrase)
    MOM_STRONG_POS:   float =  3.0
    MOM_POS:          float =  1.0
    MOM_NEG:          float = -1.0
    MOM_STRONG_NEG:   float = -3.0


NARRATIVE = _Narrative()