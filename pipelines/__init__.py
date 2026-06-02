"""
Market Pulse historical data pipeline package.

Modules
-------
fetch_historical   : OHLCV and FRED data fetching via yfinance / fredapi
preprocess         : cleaning, return calculation, normalisation
indicators         : full indicator computation on OHLCV DataFrames
feature_engineering: rolling features ready for forecasting / AI models
"""
