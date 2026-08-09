# Crypto Range Screener

## Project Overview

This project detects three crypto trading signals:

- Range Signal
- Sweep Signal
- Failure Signal

It also uses a Random Forest machine learning model to forecast price movement and includes stablecoin market data from DefiLlama for on-chain analysis.

---

## Features

- Detects Range, Sweep and Failure signals
- Generates Buy/Sell entries
- Creates Matplotlib charts
- Random Forest price forecasting
- On-chain stablecoin flow analysis
- Supports BTC, ETH and SOL

---

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- DefiLlama API

---

## Dataset (from Kaggle)

- BTC
- ETH
- SOL

Hourly OHLCV data generated from Binance minute data.

---

## Machine Learning

Model:
- Random Forest Regressor

Evaluation:
- MAE
- Directional Accuracy

---

## On-Chain Analytics

The project uses stablecoin market capitalization data from DefiLlama to improve prediction performance.
