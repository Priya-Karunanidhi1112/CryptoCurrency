### Crypto Range Screener & ML Dashboard

### Project Overview

This project detects three crypto trading signals, performs predictive pricing using machine learning, and integrates macroeconomic on-chain liquidity indicators to analyze performance. 
The system streams high-resolution historical market candles automatically using the **Kaggle API** and provides an interactive web dashboard powered by **Streamlit** to filter signals and evaluate predictive accuracy in real-time. 

### Features

* **Automated Data Streaming:** Direct integration with Kaggle API for seamless data ingestion.
* **Technical Signal Detection:** Tracks Range, Sweep, and Failure trading signals.
* **Automated Trade Entries:** Generates localized Buy/Sell trigger targets.
* **Interactive Visualization:** Renders interactive technical analysis layouts, custom metrics, and dynamic tables.
* **Machine Learning Engine:** Uses an optimized Random Forest framework to forecast price movements.
* **On-Chain Volume Analysis:** Dynamically tracks global stablecoin flows directly from the DefiLlama data ecosystem.
* **Multi-Asset Architecture:** Seamless native support for Bitcoin (**BTC**), Ethereum (**ETH**), and Solana (**SOL**).

### Dataset Ingestion (Via Kaggle API)

* **Assets Tracked:** BTC, ETH, SOL
* **Granularity:** Hourly Open-High-Low-Close-Volume (OHLCV) records compiled directly from historical Binance minute transaction streams.
* **Pipeline:** The dataset is fetched directly down into the workspace at runtime via the official Kaggle database client library, bypassing the need to commit bulky raw CSV files into GitHub.

### Machine Learning Framework

* **Model Pipeline:** Random Forest Regressor (sklearn.ensemble.RandomForestRegressor)
* **Features Used:** Engineered technical features combined with global stablecoin market cap data.
* **Validation Protocols:** Evaluated via Mean Absolute Error (MAE) and Directional Market Accuracy metrics using a clear training/testing partition (train_test_split).

### On-Chain Liquidity Analytics

The engine integrates macro-level stablecoin market capitalization metrics extracted from the **DefiLlama API**. These data pipelines are cached inside the app logic to isolate key liquidity inflows and enhance predictive directional confidence. 

### Tech Stack

* **Core Runtime:** Python 3.12+ / 3.13 / 3.14
* **Web UI Framework:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Visualization Engine:** Matplotlib, Pydeck, Altair
* **Machine Learning & APIs:** Scikit-learn, Kaggle API, DefiLlama API
