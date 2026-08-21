import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

btc_path = "BTCUSDT"
eth_path = "ETHUSDT"
sol_path = "SOLUSDT"
START_DATE = "2023-01-01"
END_DATE = "2023-04-01"
n=24
w=0.02
FAILURE_CANDLES = 6
def load_csv(path,start_date,end_date):
    print("\nLoading:",path)
    url = ("https://data-api.binance.vision"
        "/api/v3/klines")
    start_time=int(pd.Timestamp(start_date,tz="UTC").timestamp()*1000)
    end_time=int(pd.Timestamp(end_date,tz="UTC").timestamp()*1000)
    all_data=[]
    current_start=start_time
    limit=1000
    while current_start < end_time:
        params = {"symbol": path,
            "interval": "1h",
            "startTime": current_start,"endTime": end_time,"limit": limit}
        try:
            response = requests.get(url,params=params,timeout=20)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print("Could not retrieve Binance data:",e)
            return pd.DataFrame()
        if not data:
            break
        all_data.extend(data)
        last_timestamp = data[-1][0]
        current_start=(last_timestamp+60*60*1000)
        print(path,"- downloaded candles:",len(all_data))
        if len(data) < limit:
            break

    if len(all_data)==0:
        print("No Binance data found for:",path)
        return pd.DataFrame()
    columns=["timestamp","open","high","low","close","volume","close_time",
        "quote_asset_volume","number_of_trades","taker_buy_base_volume","taker_buy_quote_volume","ignore"]
    d=pd.DataFrame(all_data,columns=columns)
    d["timestamp"] = pd.to_datetime(d["timestamp"],unit="ms",utc=True)
    d=d[["timestamp","open","high","low","close","volume"]]
    d["open"] = pd.to_numeric(d["open"],errors="coerce")
    d["high"] = pd.to_numeric(d["high"],errors="coerce")
    d["low"] = pd.to_numeric(d["low"],errors="coerce")
    d["close"] = pd.to_numeric(d["close"],errors="coerce")
    d["volume"] = pd.to_numeric(d["volume"],errors="coerce")
    d = d.set_index("timestamp")
    d = d[~d.index.duplicated(keep="first")]
    d=d.sort_index()
    d=d.loc[start_date:end_date]
    d=d.dropna()
    print("Hourly candles:",len(d))
    return d

#DETECT RANGE, SWEEP AND FAILURE SIGNALS
def get_signals(d):
    d=d.copy()
    d["rh"]=(d["high"].rolling(n).max().shift(1))
    d["rl"]=(d["low"].rolling(n).min().shift(1))
    d["width"]=((d["rh"]-d["rl"])/d["close"])
    d["range_sig"]=d["width"]<w
    d["sweep_sig"]=False
    d["sweep_dir"]=""
    d["fail_sig"]=False
    d["fail_dir"]=""
    temp = d.reset_index()
    last_dir = None
    last_i = None
    for i in range(n, len(temp)):
        rh=temp.loc[i,"rh"]
        rl=temp.loc[i,"rl"]
        hi=temp.loc[i,"high"]
        lo=temp.loc[i,"low"]
        cl=temp.loc[i,"close"]

        if pd.isna(rh) or pd.isna(rl):
            continue
        if hi > rh and cl <= rh:
            d.iloc[i,d.columns.get_loc("sweep_sig")]=True
            d.iloc[i,d.columns.get_loc("sweep_dir")] = "up"

        elif lo < rl and cl >= rl:
            d.iloc[i,d.columns.get_loc("sweep_sig")] = True
            d.iloc[i,d.columns.get_loc("sweep_dir")] = "down"

        if cl > rh:
            last_dir = "up"
            last_i = i

        elif cl < rl:
            last_dir = "down"
            last_i = i
        if (last_dir is not None and last_i is not None and (i-last_i)<=FAILURE_CANDLES):
            if(rl <= cl <= rh and i > last_i):
                d.iloc[i,d.columns.get_loc("fail_sig")]=True
                d.iloc[i,d.columns.get_loc("fail_dir")]=last_dir
                last_dir = None
                last_i = None

    return d
# GENERATE TRADING LEVELS
def get_trades(d):
    d = d.copy()
    # Create trade columns
    d["entry"] = np.nan
    d["sl"] = np.nan
    d["tp"] = np.nan
    d["side"] = ""
    for i in range(len(d)):
        row = d.iloc[i]
        rh = row["rh"]
        rl = row["rl"]
        if pd.isna(rh) or pd.isna(rl):
            continue
        if (row["sweep_sig"] and row["sweep_dir"] == "up"):
            d.iloc[i,d.columns.get_loc("entry")]=row["close"]
            d.iloc[i, d.columns.get_loc("sl")] = row["high"]*1.002
            d.iloc[i,d.columns.get_loc("tp")]=rl
            d.iloc[i,d.columns.get_loc("side")] = "sell"

        elif(row["sweep_sig"]and row["sweep_dir"] == "down"):
            d.iloc[i,d.columns.get_loc("entry")]=row["close"]
            d.iloc[i,d.columns.get_loc("sl")]=row["low"]*0.998
            d.iloc[i,d.columns.get_loc("tp")]=rh
            d.iloc[i,d.columns.get_loc("side")] = "buy"

        elif(row["fail_sig"] and row["fail_dir"] == "up"):
            d.iloc[i,d.columns.get_loc("entry")]=row["close"]
            d.iloc[i,d.columns.get_loc("sl")] = row["high"] * 1.002
            d.iloc[i,d.columns.get_loc("tp")]=rl
            d.iloc[i,d.columns.get_loc("side")]="sell"

        elif(row["fail_sig"]and row["fail_dir"] == "down"):
            d.iloc[i,d.columns.get_loc("entry")]=row["close"]
            d.iloc[i,d.columns.get_loc("sl")]=row["low"]*0.998
            d.iloc[i,d.columns.get_loc("tp")]=rh
            d.iloc[i,d.columns.get_loc("side")]="buy"
    return d

# CREATE SIGNAL CHART
def make_chart(d, name):
    plt.figure(figsize=(14, 6))
    # Price
    plt.plot(d.index,d["close"],color="black",label="Price")
    # Range
    a = d[d["range_sig"]]
    # Sweep Up
    b = d[(d["sweep_sig"])&(d["sweep_dir"] == "up")]
    # Sweep Down
    c=d[(d["sweep_sig"])& (d["sweep_dir"] == "down")]
    # Failure Up
    e=d[(d["fail_sig"])& (d["fail_dir"] == "up")]
    # Failure Down
    f=d[(d["fail_sig"])& (d["fail_dir"] == "down")]
    plt.scatter(a.index,a["close"],color="blue",marker="s",label="Range")
    plt.scatter(b.index,b["close"],color="red",marker="v",label="Sweep Up")
    plt.scatter(c.index,c["close"],color="green",marker="^",label="Sweep Down")
    plt.scatter(e.index,e["close"],color="orange",marker="x",label="Failure Up")
    plt.scatter(f.index,f["close"],color="purple",marker="x",label="Failure Down")
    plt.title(f"{name} - Range / Sweep / Failure Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(name + "_chart.png")
    plt.close()
# GET BLOCKCHAIN / ON-CHAIN DATA
def get_onchain(idx):
    try:
        import requests
        url = (
            "https://stablecoins.llama.fi/"
            "stablecoincharts/all")
        response = requests.get(url,timeout=10)
        response.raise_for_status()
        data = response.json()
        oc = pd.DataFrame(data)
        # Convert timestamp
        oc["date"] = pd.to_datetime(oc["date"].astype(int),unit="s",utc=True)
        oc=(oc.set_index("date").sort_index())
        # Extract total stablecoin market cap
        oc["mcap"] = oc["totalCirculating"].apply(lambda x: x.get("peggedUSD") if isinstance(x, dict) else None)
        oc["flow"] = oc["mcap"].pct_change()
        # Match blockchain data to crypto timestamps
        result = oc["flow"].reindex(idx,method="ffill")
        if result.isna().all():
            raise ValueError("On-chain dates do not match dataset dates")
        print("Real stablecoin data received from DefiLlama")
        return result.fillna(0)
    except Exception as e:
        print("Could not retrieve on-chain data:",e)
        return pd.Series( 0.0,index=idx)

# MACHINE LEARNING MODEL
def do_ml(d, name):
    # 1-hour return
    x1 = d["close"].pct_change(1)
    # 6-hour return
    x2 = d["close"].pct_change(6)
    # Range width
    x3 = d["width"]
    # Range signal
    x4 = d["range_sig"].astype(int)
    # Sweep signal
    x5 = d["sweep_sig"].astype(int)
    # Failure signal
    x6 = d["fail_sig"].astype(int)
    x7 = get_onchain(d.index)
    y = (d["close"].pct_change(6).shift(-6))
    table = pd.DataFrame({
        "price_return_1h": x1,
        "price_return_6h": x2,
        "range_width": x3,
        "range_signal": x4,
        "sweep_signal": x5,
        "failure_signal": x6,
        "stablecoin_flow": x7,
        "future_6h_return": y})
    table = table.dropna()
    if len(table) < 20:
        print(name,"- Not enough ML samples.")
        return None
    price_features = ["price_return_1h","price_return_6h",
        "range_width","range_signal","sweep_signal","failure_signal"]
    onchain_features = ["price_return_1h","price_return_6h","range_width",
        "range_signal","sweep_signal","failure_signal","stablecoin_flow"]
    x_train, x_test, y_train, y_test = train_test_split(
        table[price_features],table["future_6h_return"],test_size=0.20,shuffle=False)
    model1 = RandomForestRegressor(n_estimators=200,max_depth=5,random_state=1)
    model1.fit(x_train,y_train)
    pred1 = model1.predict(x_test)
    # MAE
    mae1 = mean_absolute_error(y_test,pred1)
    # Directional Accuracy
    acc1 = (np.sign(pred1)==np.sign(y_test)).mean()
    # Train/Test split for Model 2
    x_train2, x_test2, y_train2, y_test2 = train_test_split(table[onchain_features],table["future_6h_return"],test_size=0.20,shuffle=False)
    # Random Forest Model 2
    model2 = RandomForestRegressor(n_estimators=200,max_depth=5,random_state=1)
    model2.fit(x_train2,y_train2)
    pred2 = model2.predict(x_test2)
    mae2 = mean_absolute_error(y_test2,pred2)
    acc2=(np.sign(pred2)==np.sign(y_test2)).mean()
    print(name,"- Price model MAE:",round(mae1,5))
    print(name,"- Price model Directional Accuracy:",round(acc1*100,1),"%")
    print(name,"- Price + On-chain MAE:",round(mae2,5))
    print(name,"- Price + On-chain Directional Accuracy:",round(acc2*100, 1),"%")
    # Forecast plot
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values,color="black",label="Actual")
    plt.plot(pred1,color="red",label="Predicted - Price Only")
    plt.plot(pred2,color="green",label="Predicted - Price + On-chain")
    plt.title(name + " - 6 Hour Price Return Forecast")
    plt.xlabel("Test Samples")
    plt.ylabel("Future 6-Hour Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(name + "_forecast.png")
    plt.close()
    return {"price_mae": mae1,"price_accuracy": acc1,"onchain_mae": mae2,"onchain_accuracy": acc2,"actual": y_test,"pred_price": pred1,"pred_onchain": pred2}
# LOAD DATASETS
print("\n")
print("CRYPTO RANGE SCREENER")
print("\nDataset period:",START_DATE,"to",END_DATE)
btc = load_csv(btc_path,START_DATE,END_DATE)
eth = load_csv(eth_path,START_DATE,END_DATE)
sol = load_csv(sol_path,START_DATE,END_DATE)

#DISPLAY DATASET SIZE
print("\nDataset Summary")
print("BTC hourly rows:",len(btc))
print("ETH hourly rows:",len(eth))
print("SOL hourly rows:",len(sol))

# DETECT SIGNALS
btc = get_signals(btc)
eth = get_signals(eth)
sol = get_signals(sol)

# DISPLAY SIGNAL COUNTS
print("\nSignal Summary")
print("BTC - Range:",btc["range_sig"].sum(),"Sweep:",btc["sweep_sig"].sum(),"Failure:",btc["fail_sig"].sum())
print("ETH - Range:",eth["range_sig"].sum(),"Sweep:",eth["sweep_sig"].sum(),"Failure:",eth["fail_sig"].sum())
print("SOL-Range:",sol["range_sig"].sum(),"Sweep:",sol["sweep_sig"].sum(),"Failure:",sol["fail_sig"].sum())

#GENERATE TRADES
btc = get_trades(btc)
eth = get_trades(eth)
sol = get_trades(sol)

#CREATE SIGNAL CHARTS
make_chart(btc,"BTC")
make_chart(eth,"ETH")
make_chart(sol,"SOL")

# MACHINE LEARNING
btc_ml = do_ml(btc,"BTC")
eth_ml = do_ml(eth,"ETH")
sol_ml = do_ml(sol,"SOL")
print("\n")
print("Analysis Completed Successfully")
print("Data Source:Binance Public Market Data API")
print("\nDataset period used:")
print(START_DATE, "to", END_DATE)
print("\nDone.")