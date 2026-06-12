# Market Pulse

A Python-based financial analytics platform that runs a fully automated daily pipeline on the 100 most-active US stocks. Scrapes live price data, scores news sentiment, computes technical indicators, detects anomalies and signals, forecasts key metrics, and produces a self-contained interactive HTML dashboard, all from a single command.

---

## Features

- **Live data scraping:** Yahoo Finance most-active list with US Eastern date handling
- **NLP sentiment scoring:** FinBERT (ProsusAI/finbert) primary, VADER fallback; per-symbol headline analysis
- **Historical baseline extension:** merges 2 years of OHLCV data with daily scrapes so analytics have meaningful baselines from day one
- **Technical indicators:** SMA, EMA, RSI, MACD, Bollinger Bands, ATR, momentum, rolling z-scores
- **Signal engine:** momentum breakouts, volatility spikes, sentiment divergence, unusual movers, risk scores, opportunity rankings
- **Anomaly detection:** price gap anomalies, volatility regime shifts, breadth extremes, cluster spikes, sentiment-volatility divergence
- **Forecasting:** OLS linear, SMA, and EMA projections with widening confidence intervals
- **AI narratives:** deterministic session summaries, volatility commentary, sentiment briefings
- **Interactive dashboard:** Plotly-based HTML with Technical, Forecast, Signals, Anomaly, and AI Briefing tabs
- **Automated scheduling:** Windows Task Scheduler (`run_daily.bat`) and Linux cron (`run_daily.sh`)
- **Self-healing pipeline:** auto-backfills missed trading days and sparse sentiment history on every run

---

## Architecture

```
main.py  (pipeline orchestrator)
│
├── Stage 0  pipelines/fetch_historical.py   OHLCV + FRED macro fetch (yfinance, skips if already run today)
├── Stage 1  scraper.py                       Yahoo Finance price scrape + EDA
├── Stage 2  sentiment.py                     FinBERT / VADER headline scoring
├── Stage 3  pipelines/preprocess.py          OHLCV cleaning, returns, outlier flagging
├── Stage 4  pipelines/indicators.py          Technical indicator computation
├── Stage 5  pipelines/feature_engineering.py 44-feature ML-ready feature set
├── Stage 6  historical_metrics.py            Per-day aggregate metrics + rolling intelligence
├── Stage 7  signal_engine.py                 Signal detection + risk/opportunity rankings
├── Stage 8  anomaly_detection.py             Statistical anomaly detection (z-score based)
├── Stage 9  forecasting.py                   SMA / EMA / OLS forward projections
└── Stage 10 dashboard.py                     Plotly HTML dashboard build
```

**Data layer:**  all I/O goes through `data_loader.py`. Parquet-first with Excel as human-readable backup. In-process cache avoids repeated file reads within a single run.

**Config:**  every threshold, path, and constant lives in `config.py` as frozen dataclasses. No magic numbers in module code.

---

## Installation

```bash
git clone https://github.com/amanijabeur/Market-Pulse.git
cd Market-Pulse
pip install requests beautifulsoup4 pandas numpy scipy openpyxl pyarrow \
            plotly vaderSentiment yfinance colorama fredapi
```

**Optional: FinBERT (recommended for better sentiment accuracy):**
```bash
pip install transformers torch
```
Without this, the platform falls back to VADER automatically.

**Optional: Finnhub historical sentiment backfill:**

Create a `.env` file in the project root:
```
FINNHUB_API_KEY=your_key_here
```
Free tier at [finnhub.io](https://finnhub.io) is sufficient.

---

## Usage

```bash
# Full daily pipeline (normal use)
python main.py

# Skip scraping, reuse today's already-scraped data
python main.py --skip-scraper

# Rebuild the dashboard from existing cached data only
python main.py --dashboard-only

# Force full rebuild of all Parquet caches and metrics
python main.py --force-rebuild

# Run without opening the browser on completion
python main.py --no-browser
```

The dashboard is written to `market_pulse_dashboard.html` and opens automatically in your default browser.

---

## Automated Daily Runs

**Windows:**
```bash
python setup_scheduler.py   # registers a Task Scheduler task (Mon–Fri 22:30)
# or run manually:
run_daily.bat
```

**Linux / Cloud:**
```bash
bash run_daily.sh --install   # installs cron job (Mon–Fri 21:30 UTC)
bash run_daily.sh --remove    # removes it
```

---

## Backfill Utilities

```bash
# Fill a missed trading day from yfinance
python backfill_date.py 2025-05-30

# Seed sentiment history from Finnhub (one-time setup)
python backfill_sentiment.py --from 2024-01-01
```

---

## Project Structure

```
Market-Pulse/
├── main.py                        Pipeline entry point
├── config.py                      All constants and thresholds
├── scraper.py                     Yahoo Finance scraper
├── sentiment.py                   FinBERT / VADER sentiment scoring
├── historical_metrics.py          Daily aggregate metrics
├── signal_engine.py               Signal and ranking engine
├── anomaly_detection.py           Anomaly detection
├── forecasting.py                 Time series forecasting
├── technical_indicators.py        Indicator computations
├── time_series.py                 Decomposition and trend analysis
├── ai_narratives.py               Narrative generation
├── data_loader.py                 Centralised I/O layer
├── validators.py                  Pre-flight and data validation
├── finnhub_sentiment.py           Finnhub + VIX historical sentiment
├── visualization.py               Static chart exports
├── dashboard.py                   Dashboard orchestrator
├── dashboard_components/          Per-tab Plotly components
│   ├── technical_tab.py
│   ├── forecast_tab.py
│   ├── signal_tab.py
│   ├── anomaly_tab.py
│   └── ai_briefing_tab.py
├── pipelines/
│   ├── fetch_historical.py        OHLCV + FRED macro fetch
│   ├── preprocess.py              OHLCV cleaning
│   ├── indicators.py              Indicator pipeline
│   └── feature_engineering.py    44-feature ML feature set
├── backfill_date.py               Missed-day backfill utility
├── backfill_sentiment.py          Historical sentiment seeder
├── run_daily.bat                  Windows scheduler script
├── run_daily.sh                   Linux / cron script
└── setup_scheduler.py             Windows Task Scheduler registration
```

---

## Data Storage

| File | Contents |
|------|----------|
| `most_active_stocks_dataset.xlsx` | Daily price data (human-readable source of truth) |
| `most_active_stocks_dataset.parquet` | Fast-read price mirror |
| `sentiment_history.parquet` | Per-symbol daily sentiment scores |
| `historical_metrics.parquet` | Per-day market aggregate metrics |
| `technical_metrics.parquet` | Computed technical indicators |
| `signal_metrics.parquet` | Detected signals and rankings |
| `anomaly_metrics.parquet` | Detected anomalies |
| `forecast_metrics.parquet` | Forward projections |
| `data/ohlcv/` | Per-symbol OHLCV parquet files (2-year history) |

---

## Requirements

- Python 3.10+
- Internet access for Yahoo Finance and optionally Finnhub / FRED
