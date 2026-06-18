import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

FEATURES = ["Trend", "Breadth", "Liquidity", "Valuation", "Sentiment", "Macro"]

def get_close(df):
    if isinstance(df.columns, pd.MultiIndex):
        return df["Close"].iloc[:, 0]
    return df["Close"]

def rsi(s, period=14):
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def label_phase(row):
    t, b, l, v, s, m = row["Trend"], row["Breadth"], row["Liquidity"], row["Valuation"], row["Sentiment"], row["Macro"]

    if v <= 35 and s <= 35 and l >= 50:
        return "Accumulation"
    if 45 <= t <= 72 and m >= 55 and l >= 50 and v <= 65:
        return "Early Bull"
    if t >= 65 and b >= 60 and m >= 50 and v <= 75 and s <= 75:
        return "Mature Bull"
    if t >= 70 and v >= 75 and s >= 70:
        return "Late Bull"
    if t >= 50 and b <= 55 and v >= 65 and s >= 60:
        return "Distribution"
    if t <= 55 and b <= 55 and s <= 65:
        return "Early Bear"
    if t <= 40 and b <= 45 and l <= 50:
        return "Mature Bear"
    if t <= 35 and s <= 30 and v <= 40:
        return "Late Bear"

    return "Mature Bull"

def main():
    tickers = {
        "nifty": "^NSEI",
        "bank": "^NSEBANK",
        "midcap": "MID150BEES.NS",
        "vix": "^INDIAVIX",
        "usd_inr": "INR=X",
        "crude": "CL=F",
        "gold": "GC=F",
        "us10y": "^TNX",
    }

    prices = pd.DataFrame()

    for name, ticker in tickers.items():
        print(f"Downloading {ticker}...")
        data = yf.download(
            ticker,
            period="20y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        close = get_close(data)
        prices[name] = pd.to_numeric(close, errors="coerce")

    prices = prices.dropna()
    print("Aligned rows:", len(prices))

    n = prices["nifty"]
    bank = prices["bank"]
    mid = prices["midcap"]
    vix = prices["vix"]
    usd = prices["usd_inr"]
    crude = prices["crude"]
    gold = prices["gold"]
    us10y = prices["us10y"]

    df = pd.DataFrame(index=prices.index)

    sma20 = n.rolling(20).mean()
    sma50 = n.rolling(50).mean()
    sma200 = n.rolling(200).mean()
    rsi_v = rsi(n)

    df["Trend"] = (
        (n > sma20).astype(int) * 15 +
        (n > sma50).astype(int) * 20 +
        (n > sma200).astype(int) * 25 +
        (sma50 > sma200).astype(int) * 20 +
        (n.pct_change(20) > 0).astype(int) * 10 +
        ((rsi_v >= 45) & (rsi_v <= 70)).astype(int) * 10
    ).clip(0, 100)

    breadth = pd.Series(0, index=prices.index)

    for col in ["nifty", "bank", "midcap"]:
        s = prices[col]
        breadth += (s > s.rolling(50).mean()).astype(int) * 12
        breadth += (s > s.rolling(200).mean()).astype(int) * 16
        breadth += (s.pct_change(20) > 0).astype(int) * 5

    df["Breadth"] = (breadth / 99 * 100).clip(0, 100)

    liquidity = pd.Series(50, index=prices.index)
    liquidity += np.where(n.pct_change(60) > 0, 15, -10)
    liquidity += np.where(usd.pct_change(60) < 0.025, 15, -15)
    liquidity += np.where(us10y.diff(60) < 0, 15, -10)
    df["Liquidity"] = liquidity.clip(0, 100)

    one_year_return = n.pct_change(252) * 100
    distance_high = n / n.rolling(252).max() * 100 - 100

    valuation = pd.Series(50, index=prices.index)
    valuation[one_year_return > 35] = 95
    valuation[(one_year_return > 25) & (one_year_return <= 35)] = 85
    valuation[(one_year_return > 15) & (one_year_return <= 25)] = 70
    valuation[(one_year_return > 5) & (one_year_return <= 15)] = 55
    valuation[(one_year_return > -10) & (one_year_return <= 5)] = 35
    valuation[one_year_return <= -10] = 20
    valuation += np.where(distance_high > -3, 8, 0)
    valuation -= np.where(distance_high < -20, 10, 0)
    df["Valuation"] = valuation.clip(0, 100)

    vix_percentile = vix.rolling(252).apply(lambda x: (x < x.iloc[-1]).mean() * 100)
    drawdown = n / n.rolling(252).max() * 100 - 100

    sentiment = 100 - vix_percentile
    sentiment += np.where(drawdown > -3, 10, 0)
    sentiment -= np.where(drawdown < -15, 15, 0)
    df["Sentiment"] = sentiment.clip(0, 100)

    macro = pd.Series(55, index=prices.index)
    macro += np.where(crude.pct_change(120) < 0.10, 8, -8)
    macro += np.where(gold.pct_change(120) < 0.15, 4, -4)
    macro += np.where(usd.pct_change(120) < 0.03, 8, -8)
    macro += np.where(us10y.diff(120) < 0, 8, -6)
    macro += np.where(n.pct_change(120) > 0, 8, -8)
    df["Macro"] = macro.clip(0, 100)

    df = df.dropna()
    df["Phase"] = df.apply(label_phase, axis=1)

    print("\nPhase distribution:")
    print(df["Phase"].value_counts())

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["Phase"])
    X = df[FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=42, stratify=y
    )

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(encoder.classes_),
        n_estimators=350,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.88,
        colsample_bytree=0.88,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=1,
    )

    print("\nTraining XGBoost...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\nValidation accuracy:", round(accuracy_score(y_test, preds) * 100, 2), "%")
    print(classification_report(y_test, preds, target_names=encoder.classes_))

    joblib.dump(model, "xgb_market_cycle_model.pkl")
    joblib.dump(encoder, "phase_label_encoder.pkl")
    joblib.dump(FEATURES, "model_features.pkl")
    joblib.dump(df, "historical_market_cycle_dataset.pkl")

    print("\nSaved model files successfully.")

if __name__ == "__main__":
    main()
