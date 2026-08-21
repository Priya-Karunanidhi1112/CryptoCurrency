import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import requests

st.set_page_config(page_title="Crypto Range Screener",layout="wide")
st.title("Crypto Range Screener")
st.markdown(
    '''
    **Range / Sweep / Failure Analysis with Machine Learning**
    Market data is retrieved directly from the
    **Binance Public Market Data API**.
    ''')
# CONFIGURATION
SYMBOL_MAP = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",}

INTERVAL_MAP = {
    "1 Hour": ("1h",60*60*1000),
    "6 Hours": ("6h", 6*60*60*1000),
    "1 Day": ("1d", 24*60*60*1000),
    "1 Week": ("1w", 7*24*60*60*1000),
}

st.sidebar.header("Dashboard Controls")

coin = st.sidebar.selectbox("Select Cryptocurrency", ["BTC", "ETH", "SOL"])
symbol = SYMBOL_MAP[coin]
interval_label = st.sidebar.radio(
    "Time Interval",
    list(INTERVAL_MAP.keys()),
    index=0,
)
interval_code, interval_ms = INTERVAL_MAP[interval_label]
today = pd.Timestamp.now(tz="UTC").date()
latest_completed_date = today - pd.Timedelta(days=1)

start_date = st.sidebar.date_input(
    "Start Date",
    pd.Timestamp("2023-01-01").date(),
    min_value=pd.Timestamp("2017-01-01").date(),
    max_value=latest_completed_date,
)

end_date = st.sidebar.date_input(
    "End Date",
    pd.Timestamp("2023-04-01").date(),
    min_value=pd.Timestamp("2017-01-01").date(),
    max_value=latest_completed_date,
)
range_lookback = st.sidebar.number_input(
    "Range Candles", min_value=5, max_value=100, value=24, step=1)

max_range_width = st.sidebar.number_input(
    "Maximum Range Width",
    min_value=0.001,
    max_value=0.20,
    value=0.02,
    step=0.001,
    format="%.3f",)
failure_candles=st.sidebar.number_input("Failure Confirmation Candles", min_value=1, max_value=20, value=6, step=1)

if start_date >= end_date:
    st.error("End Date must be later than Start Date.")
    st.stop()
start_date = str(start_date)
end_date = str(end_date)

@st.cache_data(ttl=300)
def load_price_data(symbol, start_date, end_date, interval_code, interval_ms):
    url = "https://data-api.binance.vision/api/v3/klines"
    start_timestamp = pd.Timestamp(start_date, tz="UTC")
    end_timestamp = (
        pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    )

    start_time_ms = int(start_timestamp.timestamp() * 1000)
    end_time_ms = int(end_timestamp.timestamp() * 1000)
    all_candles=[]
    current_start=start_time_ms
    batch_limit=1000
    while current_start<end_time_ms:
        params={
            "symbol": symbol,
            "interval": interval_code,
            "startTime": current_start,
            "endTime": end_time_ms,
            "limit": batch_limit,}

        try:
            response=requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            batch=response.json()
        except Exception as e:
            st.error(f"Could not retrieve Binance data: {e}")
            return pd.DataFrame()
        if not batch:
            break
        all_candles.extend(batch)
        last_candle_time=batch[-1][0]
        current_start=last_candle_time+interval_ms
        if len(batch)<batch_limit:
            break
    if len(all_candles)==0:
        return pd.DataFrame()
    columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",]
    df=pd.DataFrame(all_candles, columns=columns)
    df["timestamp"]=pd.to_datetime(df["timestamp"],unit="ms",utc=True)
    df=df[["timestamp", "open", "high", "low", "close", "volume"]]
    for column in ["open","high","low","close","volume"]:
        df[column]=pd.to_numeric(df[column],errors="coerce")
    df=df.set_index("timestamp")
    df=df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    start_timestamp = pd.Timestamp(start_date, tz="UTC")
    end_timestamp = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    df = df[(df.index >= start_timestamp) & (df.index < end_timestamp)]
    df = df.dropna()

    return df
# SIGNAL DETECTION

def add_range_and_signal_columns(df):
    df=df.copy()
    # Range high / low over the lookback window
    df["range_high"]=df["high"].rolling(range_lookback).max().shift(1)
    df["range_low"]=df["low"].rolling(range_lookback).min().shift(1)
    # Range width as a percentage of price
    df["range_width"]=(df["range_high"]-df["range_low"])/df["close"]
    df["range_signal"]=df["range_width"]<max_range_width
    # Sweep columns (price briefly pokes outside the range then closes back inside)
    df["sweep_signal"]=False
    df["sweep_direction"]=""
    # Failed breakout columns
    df["failure_signal"]=False
    df["failure_direction"]=""
    rows=df.reset_index()
    last_breakout_direction=None
    last_breakout_index=None
    for i in range(range_lookback,len(rows)):
        range_high=rows.loc[i,"range_high"]
        range_low=rows.loc[i,"range_low"]
        candle_high=rows.loc[i,"high"]
        candle_low=rows.loc[i,"low"]
        candle_close=rows.loc[i,"close"]
        if pd.isna(range_high) or pd.isna(range_low):
            continue
        if candle_high>range_high and candle_close<=range_high:
            df.iloc[i, df.columns.get_loc("sweep_signal")]=True
            df.iloc[i, df.columns.get_loc("sweep_direction")]="up"
        elif candle_low<range_low and candle_close>=range_low:
            df.iloc[i,df.columns.get_loc("sweep_signal")]=True
            df.iloc[i,df.columns.get_loc("sweep_direction")]="down"

        if candle_close>range_high:
            last_breakout_direction="up"
            last_breakout_index = i
        elif candle_close<range_low:
            last_breakout_direction="down"
            last_breakout_index=i
        if (last_breakout_direction is not None and last_breakout_index is not None and (i-last_breakout_index)<=failure_candles):
            if range_low<=candle_close<=range_high and i>last_breakout_index:
                df.iloc[i,df.columns.get_loc("failure_signal")]=True
                df.iloc[i,df.columns.get_loc("failure_direction")]=last_breakout_direction
                last_breakout_direction=None
                last_breakout_index=None
    return df
# TRADING LEVELS
def add_trade_levels(df):
    df = df.copy()

    df["entry"] = np.nan
    df["stop_loss"] = np.nan
    df["take_profit"] = np.nan
    df["side"] = ""

    for i in range(len(df)):
        row = df.iloc[i]
        range_high = row["range_high"]
        range_low = row["range_low"]

        if pd.isna(range_high) or pd.isna(range_low):
            continue

        # Sweep up -> sell
        if row["sweep_signal"] and row["sweep_direction"] == "up":
            df.iloc[i, df.columns.get_loc("entry")] = row["close"]
            df.iloc[i, df.columns.get_loc("stop_loss")] = row["high"] * 1.002
            df.iloc[i, df.columns.get_loc("take_profit")] = range_low
            df.iloc[i, df.columns.get_loc("side")] = "sell"

        # Sweep down -> buy
        elif row["sweep_signal"] and row["sweep_direction"] == "down":
            df.iloc[i, df.columns.get_loc("entry")] = row["close"]
            df.iloc[i, df.columns.get_loc("stop_loss")] = row["low"] * 0.998
            df.iloc[i, df.columns.get_loc("take_profit")] = range_high
            df.iloc[i, df.columns.get_loc("side")] = "buy"

        # Failed upward breakout -> sell
        elif row["failure_signal"] and row["failure_direction"] == "up":
            df.iloc[i, df.columns.get_loc("entry")] = row["close"]
            df.iloc[i, df.columns.get_loc("stop_loss")] = row["high"] * 1.002
            df.iloc[i, df.columns.get_loc("take_profit")] = range_low
            df.iloc[i, df.columns.get_loc("side")] = "sell"

        # Failed downward breakout -> buy
        elif row["failure_signal"] and row["failure_direction"] == "down":
            df.iloc[i, df.columns.get_loc("entry")] = row["close"]
            df.iloc[i, df.columns.get_loc("stop_loss")] = row["low"] * 0.998
            df.iloc[i, df.columns.get_loc("take_profit")] = range_high
            df.iloc[i, df.columns.get_loc("side")] = "buy"

    return df

@st.cache_data(ttl=3600)
def get_onchain_flow(_index, cache_key):
    try:
        url = "https://stablecoins.llama.fi/stablecoincharts/all"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        onchain_df = pd.DataFrame(data)
        onchain_df["date"] = pd.to_datetime(onchain_df["date"].astype(int), unit="s", utc=True)
        onchain_df = onchain_df.set_index("date").sort_index()

        onchain_df["market_cap"] = onchain_df["totalCirculating"].apply(lambda x: x.get("peggedUSD") if isinstance(x,dict) else None)
        onchain_df["flow"] = onchain_df["market_cap"].pct_change()
        result = onchain_df["flow"].reindex(_index, method="ffill")
        if result.isna().all():
            raise ValueError("On-chain dates do not match dataset dates")
        return result.fillna(0)
    except Exception:
        return pd.Series(0.0, index=_index)

# MACHINE LEARNING
def run_ml_models(df, coin_name):
    """Train two Random Forest models (price-only vs. price + on-chain)"""
    price_change_1h = df["close"].pct_change(1)
    price_change_6h = df["close"].pct_change(6)
    range_width = df["range_width"]
    range_signal = df["range_signal"].astype(int)
    sweep_signal = df["sweep_signal"].astype(int)
    failure_signal = df["failure_signal"].astype(int)
    onchain_flow = get_onchain_flow(df.index, (coin_name, interval_code, start_date, end_date))

    target = df["close"].pct_change(6).shift(-6)
    features_df = pd.DataFrame({
        "price_change_1h": price_change_1h,
        "price_change_6h": price_change_6h,
        "range_width": range_width,
        "range_signal": range_signal,
        "sweep_signal": sweep_signal,
        "failure_signal": failure_signal,
        "onchain_flow": onchain_flow,
        "target": target,
    })

    features_df = features_df.dropna()
    minimum_samples = max(20, int(range_lookback) + 10)
    if len(features_df) < minimum_samples:
        return {
            "status": "insufficient_data",
            "sample_count": len(features_df),
            "minimum_samples": minimum_samples,
        }

    price_only_cols = [
        "price_change_1h", "price_change_6h", "range_width",
        "range_signal", "sweep_signal", "failure_signal",
    ]
    price_and_onchain_cols = price_only_cols + ["onchain_flow"]
    # Model 1: price features only
    x_train, x_test, y_train, y_test = train_test_split(
        features_df[price_only_cols], features_df["target"], test_size=0.20, shuffle=False
    )
    price_model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=1)
    price_model.fit(x_train, y_train)
    price_predictions = price_model.predict(x_test)
    price_mae = mean_absolute_error(y_test, price_predictions)
    price_accuracy = (np.sign(price_predictions) == np.sign(y_test)).mean()

    # Model 2: price + on-chain features
    x_train2, x_test2, y_train2, y_test2 = train_test_split(
        features_df[price_and_onchain_cols], features_df["target"], test_size=0.20, shuffle=False
    )
    onchain_model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=1)
    onchain_model.fit(x_train2, y_train2)
    onchain_predictions = onchain_model.predict(x_test2)
    onchain_mae = mean_absolute_error(y_test2, onchain_predictions)
    onchain_accuracy = (np.sign(onchain_predictions) == np.sign(y_test2)).mean()
    return {
        "price_mae": price_mae,
        "price_accuracy": price_accuracy,
        "onchain_mae": onchain_mae,
        "onchain_accuracy": onchain_accuracy,
        "actual": y_test,
        "pred_price": price_predictions,
        "pred_onchain": onchain_predictions,
    }
# LOAD DATA
with st.spinner(f"Loading {coin} data from Binance..."):
    price_df = load_price_data(symbol, start_date, end_date, interval_code, interval_ms)

# DATA VALIDATION
if price_df.empty:
    st.error("No market data was returned from Binance.")
    st.info(
        f"No candles were found for {coin} between "
        f"{start_date} and {end_date} at the {interval_label} interval. "
        "Try an earlier historical period or a different interval."
    )
    st.stop()
st.sidebar.success(f"Loaded {len(price_df):,} {interval_label} candles")

# SIGNAL PROCESSING
price_df = add_range_and_signal_columns(price_df)
price_df = add_trade_levels(price_df)

# CURRENT MARKET INFORMATION
latest_candle = price_df.iloc[-1]
current_price = latest_candle["close"]

# CURRENT SIGNAL
current_signal = "NEUTRAL"
current_side = "-"

if latest_candle["sweep_signal"]:
    if latest_candle["sweep_direction"] == "up":
        current_signal = "SWEEP UP"
        current_side = "SELL"
    elif latest_candle["sweep_direction"] == "down":
        current_signal = "SWEEP DOWN"
        current_side = "BUY"

elif latest_candle["failure_signal"]:
    if latest_candle["failure_direction"] == "up":
        current_signal = "FAILED BREAKOUT UP"
        current_side = "SELL"
    elif latest_candle["failure_direction"] == "down":
        current_signal = "FAILED BREAKOUT DOWN"
        current_side = "BUY"

elif latest_candle["range_signal"]:
    current_signal = "RANGE"

# DASHBOARD METRICS
st.subheader(f"{coin} Market Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Current Price", f"${current_price:,.2f}")
with col2:
    st.metric("Range Signals", int(price_df["range_signal"].sum()))
with col3:
    st.metric("Sweep Signals", int(price_df["sweep_signal"].sum()))
with col4:
    st.metric("Failure Signals", int(price_df["failure_signal"].sum()))

# CURRENT SIGNAL
st.subheader("Current Signal")
signal_col1, signal_col2 = st.columns(2)
with signal_col1:
    st.info(f"Signal: **{current_signal}**")
with signal_col2:
    if current_side == "BUY":
        st.success("Trading Direction: BUY")
    elif current_side == "SELL":
        st.error("Trading Direction: SELL")
    else:
        st.warning("Trading Direction: NEUTRAL")

st.subheader("Range / Sweep / Failure Chart")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(price_df.index, price_df["close"], color="black", label="Price")
range_points = price_df[price_df["range_signal"]]
sweep_up_points = price_df[(price_df["sweep_signal"]) & (price_df["sweep_direction"] == "up")]
sweep_down_points = price_df[(price_df["sweep_signal"]) & (price_df["sweep_direction"] == "down")]
failure_up_points = price_df[(price_df["failure_signal"]) & (price_df["failure_direction"] == "up")]
failure_down_points = price_df[(price_df["failure_signal"]) & (price_df["failure_direction"] == "down")]
ax.scatter(range_points.index, range_points["close"], color="blue", marker="s", label="Range")
ax.scatter(sweep_up_points.index, sweep_up_points["close"], color="red", marker="v", label="Sweep Up")
ax.scatter(sweep_down_points.index, sweep_down_points["close"], color="green", marker="^", label="Sweep Down")
ax.scatter(failure_up_points.index, failure_up_points["close"], color="orange", marker="x", label="Failure Up")
ax.scatter(failure_down_points.index, failure_down_points["close"], color="purple", marker="x", label="Failure Down")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title(f"{coin} - Range / Sweep / Failure ({interval_label})")
ax.legend()
fig.tight_layout()
st.pyplot(fig)
st.subheader("Trading Levels")
active_trades = price_df[price_df["side"] != ""].copy()
if active_trades.empty:
    st.info("No trading setup detected for the selected period.")
else:
    trades_table = active_trades[[
        "close", "sweep_direction", "failure_direction",
        "entry", "stop_loss", "take_profit", "side",
    ]].copy()
    trades_table = trades_table.rename(columns={
        "close": "Price",
        "sweep_direction": "Sweep Direction",
        "failure_direction": "Failure Direction",
        "entry": "Entry",
        "stop_loss": "Stop Loss",
        "take_profit": "Take Profit",
        "side": "Side", })
    st.dataframe(trades_table.tail(20), width="stretch")

st.subheader("Machine Learning Analysis")
with st.spinner("Training Random Forest models..."):
    ml_results = run_ml_models(price_df, coin)
if ml_results is not None and ml_results.get("status") == "insufficient_data":
    st.warning(
        f"Not enough valid samples for ML after feature preparation. "
        f"Available: {ml_results['sample_count']:,}; "
        f"required: {ml_results['minimum_samples']:,}. "
        "Select a wider historical date range or a shorter interval."
    )
elif ml_results is None:
    st.warning("The ML model could not be trained for this date range.")
else:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Price Model MAE", f"{ml_results['price_mae']:.5f}")
    with col2:
        st.metric("Price Model Accuracy", f"{ml_results['price_accuracy'] * 100:.1f}%")
    with col3:
        st.metric("Price + On-chain MAE", f"{ml_results['onchain_mae']:.5f}")
    with col4:
        st.metric("Price + On-chain Accuracy", f"{ml_results['onchain_accuracy'] * 100:.1f}%")
    st.subheader(f"{interval_label} Price Return Forecast")
    forecast_df = pd.DataFrame({
        "Actual": ml_results["actual"].values,
        "Predicted - Price Only": ml_results["pred_price"],
        "Predicted - Price + On-chain": ml_results["pred_onchain"], })
    st.line_chart(forecast_df)

# DATA TABLE
st.subheader("Market Data")
display_columns = [
    "open", "high", "low", "close", "volume",
    "range_high", "range_low", "range_width", "range_signal",
    "sweep_signal", "sweep_direction",
    "failure_signal", "failure_direction",
    "entry", "stop_loss", "take_profit", "side", ]

st.dataframe(price_df[display_columns].tail(100), width="stretch")
st.caption("Data Source: Binance Public Market Data API")
