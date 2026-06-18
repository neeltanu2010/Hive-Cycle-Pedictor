import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier


PHASES = [
    "Accumulation",
    "Early Bull",
    "Mature Bull",
    "Late Bull",
    "Distribution",
    "Early Bear",
    "Mature Bear",
    "Late Bear",
]

FEATURES = ["Trend", "Breadth", "Liquidity", "Valuation", "Sentiment", "Macro"]


def clamp(x, low=0, high=100):
    return max(low, min(high, float(x)))


def get_close(df):
    if isinstance(df.columns, pd.MultiIndex):
        return df["Close"].iloc[:, 0].dropna()
    return df["Close"].dropna()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def pct(a, b):
    if b == 0 or pd.isna(b):
        return 0
    return (a / b - 1) * 100


def score_trend(close):
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    rsi_v = rsi(close)
    ret20 = close.pct_change(20) * 100
    ret60 = close.pct_change(60) * 100

    score = (
        (close > sma20).astype(int) * 15 +
        (close > sma50).astype(int) * 20 +
        (close > sma200).astype(int) * 25 +
        (sma50 > sma200).astype(int) * 20 +
        (ret20 > 0).astype(int) * 10 +
        ((rsi_v >= 45) & (rsi_v <= 70)).astype(int) * 10
    )

    return score.clip(0, 100)


def score_breadth(nifty, bank, midcap):
    data = pd.DataFrame({
        "nifty": nifty,
        "bank": bank,
        "midcap": midcap,
    }).dropna()

    score = pd.Series(0, index=data.index)

    for col in data.columns:
        s = data[col]
        score += (s > s.rolling(50).mean()).astype(int) * 12
        score += (s > s.rolling(200).mean()).astype(int) * 16
        score += (s.pct_change(20) > 0).astype(int) * 5

    return (score / 99 * 100).clip(0, 100)


def score_liquidity(nifty, usd_inr, us10y):
    score = pd.Series(50, index=nifty.index)

    nifty60 = nifty.pct_change(60) * 100
    usd60 = usd_inr.pct_change(60) * 100
    yield60 = us10y.diff(60)

    score += np.where(nifty60 > 0, 15, -10)
    score += np.where(usd60 < 2.5, 15, -15)
    score += np.where(yield60 < 0, 15, -10)

    return pd.Series(score, index=nifty.index).clip(0, 100)


def score_valuation(close):
    one_year_return = close.pct_change(252) * 100
    distance_high = close / close.rolling(252).max() * 100 - 100

    score = pd.Series(50, index=close.index)

    score = np.where(one_year_return > 35, 95, score)
    score = np.where((one_year_return > 25) & (one_year_return <= 35), 85, score)
    score = np.where((one_year_return > 15) & (one_year_return <= 25), 70, score)
    score = np.where((one_year_return > 5) & (one_year_return <= 15), 55, score)
    score = np.where((one_year_return > -10) & (one_year_return <= 5), 35, score)
    score = np.where(one_year_return <= -10, 20, score)

    score = pd.Series(score, index=close.index)
    score += np.where(distance_high > -3, 8, 0)
    score -= np.where(distance_high < -20, 10, 0)

    return score.clip(0, 100)


def score_sentiment(vix, close):
    vix_percentile = vix.rolling(252).apply(lambda x: (x < x.iloc[-1]).mean() * 100)
    drawdown = close / close.rolling(252).max() * 100 - 100

    score = 100 - vix_percentile
    score += np.where(drawdown > -3, 10, 0)
    score -= np.where(drawdown < -15, 15, 0)

    return pd.Series(score, index=vix.index).clip(0, 100)


def score_macro(nifty, crude, gold, usd_inr, us10y):
    score = pd.Series(55, index=nifty.index)

    crude6m = crude.pct_change(120) * 100
    gold6m = gold.pct_change(120) * 100
    usd6m = usd_inr.pct_change(120) * 100
    yield6m = us10y.diff(120)
    nifty6m = nifty.pct_change(120) * 100

    score += np.where(crude6m < 10, 8, -8)
    score += np.where(gold6m < 15, 4, -4)
    score += np.where(usd6m < 3, 8, -8)
    score += np.where(yield6m < 0, 8, -6)
    score += np.where(nifty6m > 0, 8, -8)

    return pd.Series(score, index=nifty.index).clip(0, 100)


def label_phase(row):
    trend = row["Trend"]
    breadth = row["Breadth"]
    liquidity = row["Liquidity"]
    valuation = row["Valuation"]
    sentiment = row["Sentiment"]
    macro = row["Macro"]

    if valuation <= 35 and sentiment <= 35 and liquidity >= 50:
        return "Accumulation"

    if trend >= 45 and trend <= 72 and macro >= 55 and liquidity >= 50 and valuation <= 65:
        return "Early Bull"

    if trend >= 65 and breadth >= 60 and macro >= 50 and valuation <= 75 and sentiment <= 75:
        return "Mature Bull"

    if trend >= 70 and valuation >= 75 and sentiment >= 70:
        return "Late Bull"

    if trend >= 50 and breadth <= 55 and valuation >= 65 and sentiment >= 60:
        return "Distribution"

    if trend <= 55 and breadth <= 55 and sentiment <= 65:
        return "Early Bear"

    if trend <= 40 and breadth <= 45 and liquidity <= 50:
        return "Mature Bear"

    if trend <= 35 and sentiment <= 30 and valuation <= 40:
        return "Late Bear"

    return "Mature Bull"


def main():
    print("Downloading 20 years of market data...")

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

    data = {}

    for name, ticker in tickers.items():
        print(f"Downloading {ticker}")
        data[name] = yf.download(
            ticker,
            period="20y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )

    nifty = get_close(data["nifty"])
    bank = get_close(data["bank"])
    midcap = get_close(data["midcap"])
    vix = get_close(data["vix"])
    usd_inr = get_close(data["usd_inr"])
    crude = get_close(data["crude"])
    gold = get_close(data["gold"])
    us10y = get_close(data["us10y"])

    df = pd.DataFrame(index=nifty.index)

    df["Trend"] = score_trend(nifty)
    df["Breadth"] = score_breadth(nifty, bank, midcap)
    df["Liquidity"] = score_liquidity(nifty, usd_inr, us10y)
    df["Valuation"] = score_valuation(nifty)
    df["Sentiment"] = score_sentiment(vix, nifty)
    df["Macro"] = score_macro(nifty, crude, gold, usd_inr, us10y)

    df = df.dropna()
    df["Phase"] = df.apply(label_phase, axis=1)

    print("\nPhase distribution:")
    print(df["Phase"].value_counts())

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["Phase"])
    X = df[FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.22,
        random_state=42,
        stratify=y,
    )

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(encoder.classes_),
        n_estimators=350,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.88,
        colsample_bytree=0.88,
        min_child_weight=2,
        reg_lambda=2.0,
        reg_alpha=0.15,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=1,
    )

    print("\nTraining XGBoost...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\nAccuracy:", round(accuracy_score(y_test, preds) * 100, 2), "%")
    print("\nClassification report:")
    print(classification_report(y_test, preds, target_names=encoder.classes_))

    joblib.dump(model, "xgb_market_cycle_model.pkl")
    joblib.dump(encoder, "phase_label_encoder.pkl")
    joblib.dump(FEATURES, "model_features.pkl")
    joblib.dump(df, "historical_market_cycle_dataset.pkl")

    print("\nSaved files:")
    print("xgb_market_cycle_model.pkl")
    print("phase_label_encoder.pkl")
    print("model_features.pkl")
    print("historical_market_cycle_dataset.pkl")


if __name__ == "__main__":
    main()
