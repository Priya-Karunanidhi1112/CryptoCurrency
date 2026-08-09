import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# paths to my csv files
btc_path="C:\\Users\\jeyar\\Desktop\\RedSandTask\\btc.csv"
eth_path="C:\\Users\\jeyar\\Desktop\\RedSandTask\\eth.csv"
sol_path="C:\\Users\\jeyar\\Desktop\\RedSandTask\\sol.csv"
n=24 #candles to check for a range
w=0.02 #max width to count as a range

# Loads CSV data, converts timestamp and resamples to 1-hour candles
def load_csv(path,s,e):
    d=pd.read_csv(path)
    d["timestamp"]=pd.to_datetime(d["timestamp"],unit="s")
    d=d.set_index("timestamp")
    d=d.loc[s:e]
    d=d.resample("1h").agg({"open": "first","high": "max","low": "min","close": "last","volume": "sum"})
    d=d.dropna()
    return d
#Detects Range, Sweep and Failure signals from OHLCV data
def get_signals(d):
    d["rh"]=d["high"].rolling(n).max().shift(1)   # range high
    d["rl"]=d["low"].rolling(n).min().shift(1)     # range low
    d["width"]=(d["rh"]-d["rl"])/d["close"]
    d["range_sig"]=d["width"]<w
    d["sweep_sig"]=False
    d["sweep_dir"]=""
    d["fail_sig"]=False
    d["fail_dir"]=""
    temp=d.reset_index()
    last_dir=None
    last_i=None
    for i in range(n,len(temp)):
        rh=temp.loc[i,"rh"]
        rl=temp.loc[i,"rl"]
        hi=temp.loc[i,"high"]
        lo=temp.loc[i,"low"]
        cl=temp.loc[i,"close"]
        if pd.isna(rh) or pd.isna(rl):
            continue
        #Detect liquidity sweep

        if hi>rh and cl<=rh:
            d.iloc[i,d.columns.get_loc("sweep_sig")]=True
            d.iloc[i,d.columns.get_loc("sweep_dir")]="up"
        elif lo<rl and cl>=rl:
            d.iloc[i,d.columns.get_loc("sweep_sig")]=True
            d.iloc[i,d.columns.get_loc("sweep_dir")]="down"
        # breakout - candle closes outside the range
        if cl>rh:
            last_dir="up"
            last_i=i
        elif cl<rl:
            last_dir="down"
            last_i=i
        # failure - breakout comes back inside within 6 candles
        if last_dir is not None and (i-last_i)<=6:
            if rl<=cl<=rh and i>last_i:
                d.iloc[i,d.columns.get_loc("fail_sig")]=True
                d.iloc[i,d.columns.get_loc("fail_dir")]=last_dir
                last_dir=None
    return d

#Generates trade entries, stop-loss and target levels
def get_trades(d):
    d["entry"]=np.nan
    d["sl"]=np.nan
    d["tp"]=np.nan
    d["side"]=""
    for i in range(len(d)):
        row=d.iloc[i]
        rh=row["rh"]
        rl=row["rl"]
        if pd.isna(rh) or pd.isna(rl):
            continue

        if row["sweep_sig"] and row["sweep_dir"]=="up":
            d.iloc[i,d.columns.get_loc("entry")]=row["close"]
            d.iloc[i,d.columns.get_loc("sl")]=row["high"]*1.002
            d.iloc[i,d.columns.get_loc("tp")]=rl
            d.iloc[i,d.columns.get_loc("side")]="sell"

        elif row["sweep_sig"] and row["sweep_dir"]=="down":
            d.iloc[i,d.columns.get_loc("entry")]=row["close"]
            d.iloc[i,d.columns.get_loc("sl")]=row["low"]*0.998
            d.iloc[i,d.columns.get_loc("tp")]=rh
            d.iloc[i,d.columns.get_loc("side")]="buy"

        elif row["fail_sig"] and row["fail_dir"]=="up":
            d.iloc[i,d.columns.get_loc("entry")]=row["close"]
            d.iloc[i,d.columns.get_loc("sl")]=row["high"]*1.002
            d.iloc[i,d.columns.get_loc("tp")]=rl
            d.iloc[i,d.columns.get_loc("side")]="sell"

        elif row["fail_sig"] and row["fail_dir"]=="down":
            d.iloc[i,d.columns.get_loc("entry")]=row["close"]
            d.iloc[i,d.columns.get_loc("sl")]=row["low"]*0.998
            d.iloc[i,d.columns.get_loc("tp")]=rh
            d.iloc[i,d.columns.get_loc("side")]="buy"

    return d

# Creates and saves charts showing all detected signals
def make_chart(d,name):
    plt.figure(figsize=(14,6))
    plt.plot(d.index,d["close"],color="black",label="price")
    a=d[d["range_sig"]]
    b=d[(d["sweep_sig"])&(d["sweep_dir"]=="up")]
    c=d[(d["sweep_sig"])&(d["sweep_dir"]=="down")]
    e=d[(d["fail_sig"])&(d["fail_dir"]=="up")]
    f=d[(d["fail_sig"])&(d["fail_dir"]=="down")]
    plt.scatter(a.index,a["close"],color="blue",marker="s",label="range")
    plt.scatter(b.index,b["close"],color="red",marker="v",label="sweep up")
    plt.scatter(c.index,c["close"],color="green",marker="^",label="sweep down")
    plt.scatter(e.index,e["close"],color="orange",marker="x",label="fail up")
    plt.scatter(f.index,f["close"],color="purple",marker="x",label="fail down")
    plt.title(name)
    plt.xlabel("date")
    plt.ylabel("price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(name+"_chart.png")
    plt.close()

#stablecoin market cap data from DefiLlama
def get_onchain(idx):
    try:
        import requests
        res=requests.get("https://stablecoins.llama.fi/stablecoincharts/all",timeout=10)
        res.raise_for_status()
        j=res.json()
        oc=pd.DataFrame(j)
        oc["date"]=pd.to_datetime(oc["date"].astype(int),unit="s")
        oc=oc.set_index("date").sort_index()
        oc["mcap"]=oc["totalCirculating"].apply(lambda x: x.get("peggedUSD") if isinstance(x,dict) else None)
        oc["flow"]=oc["mcap"].pct_change()
        result=oc["flow"].reindex(idx,method="ffill")
        if result.isna().all():
            raise ValueError("dates dont match up")
        print("got real onchain data")
        return result.fillna(0)
    except Exception as e:
        print("couldnt get onchain data:",e)
        np.random.seed(1)
        return pd.Series(np.random.normal(0,0.01,len(idx)),index=idx)

# Trains Random Forest model and compares prediction accuracy
def do_ml(d,name):
    x1=d["close"].pct_change(1)
    x2=d["close"].pct_change(6)
    x3=d["width"]
    x4=d["range_sig"].astype(int)
    x5=d["sweep_sig"].astype(int)
    x6=d["fail_sig"].astype(int)
    x7=get_onchain(d.index)
    y=d["close"].pct_change(6).shift(-6)

    table=pd.DataFrame({
        "x1": x1,"x2": x2,"x3": x3,"x4": x4,"x5": x5,"x6": x6,"x7": x7,"y": y
    })
    table=table.dropna()

    cols1=["x1","x2","x3","x4","x5","x6"]
    cols2=["x1","x2","x3","x4","x5","x6","x7"]

    x_train,x_test,y_train,y_test=train_test_split(table[cols1],table["y"],test_size=0.2,shuffle=False)
    model1=RandomForestRegressor(n_estimators=200,max_depth=5,random_state=1)
    model1.fit(x_train,y_train)
    pred1=model1.predict(x_test)
    mae1=mean_absolute_error(y_test,pred1)
    acc1=(np.sign(pred1)==np.sign(y_test)).mean()

    x_train2,x_test2,y_train2,y_test2=train_test_split(table[cols2],table["y"],test_size=0.2,shuffle=False)
    model2=RandomForestRegressor(n_estimators=200,max_depth=5,random_state=1)
    model2.fit(x_train2,y_train2)
    pred2=model2.predict(x_test2)
    mae2=mean_absolute_error(y_test2,pred2)
    acc2=(np.sign(pred2)==np.sign(y_test2)).mean()

    print(name,"no onchain - mae:",round(mae1,5),"acc:",round(acc1*100,1))
    print(name,"with onchain - mae:",round(mae2,5),"acc:",round(acc2*100,1))

    plt.figure(figsize=(12,5))
    plt.plot(y_test.values,color="black",label="actual")
    plt.plot(pred1,color="red",label="predicted no onchain")
    plt.plot(pred2,color="green",label="predicted with onchain")
    plt.title(name+" forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(name+"_forecast.png")
    plt.close()
btc=load_csv(btc_path,"2023-01-01","2023-04-01")
eth=load_csv(eth_path,"2023-01-01","2023-04-01")
sol=load_csv(sol_path,"2022-01-01","2022-04-01")
print("btc rows:",len(btc))
print("eth rows:",len(eth))
print("sol rows:",len(sol))
btc=get_signals(btc)
eth=get_signals(eth)
sol=get_signals(sol)
print("btc range/sweep/fail:",btc["range_sig"].sum(),btc["sweep_sig"].sum(),btc["fail_sig"].sum())
print("eth range/sweep/fail:",eth["range_sig"].sum(),eth["sweep_sig"].sum(),eth["fail_sig"].sum())
print("sol range/sweep/fail:",sol["range_sig"].sum(),sol["sweep_sig"].sum(),sol["fail_sig"].sum())
btc=get_trades(btc)
eth=get_trades(eth)
sol=get_trades(sol)
make_chart(btc,"BTC")
make_chart(eth,"ETH")
make_chart(sol,"SOL")
do_ml(btc,"BTC")
do_ml(eth,"ETH")
do_ml(sol,"SOL")
print("Completed")
