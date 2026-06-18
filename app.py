import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Madness of Money Bees",
    page_icon="🐝",
    layout="wide"
)


# =====================================================
# THEME CSS
# =====================================================
st.markdown("""
<style>
.stApp {
    background-color: #050505;
    color: #F8F8F8;
    background-image:
      linear-gradient(30deg, rgba(255,210,31,.08) 12%, transparent 12.5%, transparent 87%, rgba(255,210,31,.08) 87.5%, rgba(255,210,31,.08)),
      linear-gradient(150deg, rgba(255,210,31,.08) 12%, transparent 12.5%, transparent 87%, rgba(255,210,31,.08) 87.5%, rgba(255,210,31,.08)),
      linear-gradient(30deg, rgba(255,210,31,.08) 12%, transparent 12.5%, transparent 87%, rgba(255,210,31,.08) 87.5%, rgba(255,210,31,.08)),
      linear-gradient(150deg, rgba(255,210,31,.08) 12%, transparent 12.5%, transparent 87%, rgba(255,210,31,.08) 87.5%, rgba(255,210,31,.08));
    background-size: 70px 120px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111111, #050505);
    border-right: 1px solid rgba(255,210,31,.35);
}

.hero {
    padding: 36px;
    border-radius: 32px;
    background: linear-gradient(135deg, #FFD21F, #F5B700);
    color: #050505;
    text-align: center;
    box-shadow: 0 0 50px rgba(255,210,31,.40);
    margin-bottom: 24px;
}

.hero h1 {
    font-size: 48px;
    font-weight: 950;
    margin-bottom: 4px;
}

.card {
    background: rgba(17,17,17,.94);
    padding: 24px;
    border-radius: 24px;
    border: 1px solid rgba(255,210,31,.35);
    box-shadow: 0 0 24px rgba(255,210,31,.12);
    margin-bottom: 18px;
}

.phase-card {
    background: linear-gradient(135deg, rgba(255,210,31,.25), rgba(17,17,17,.96));
    padding: 28px;
    border-radius: 28px;
    border: 1px solid rgba(255,210,31,.55);
    box-shadow: 0 0 34px rgba(255,210,31,.25);
}

.big-score {
    font-size: 42px;
    font-weight: 950;
    color: #FFD21F;
}

.phase {
    font-size: 34px;
    font-weight: 950;
    color: #FFD21F;
}

.small {
    color: #D8D8D8;
    font-size: 15px;
}

.explain {
    background: rgba(255,210,31,.12);
    border-left: 5px solid #FFD21F;
    padding: 20px;
    border-radius: 18px;
    margin: 18px 0;
}

.warning {
    background: rgba(255,152,0,.14);
    border-left: 5px solid #FF9800;
    padding: 18px;
    border-radius: 16px;
    margin: 16px 0;
}

.danger {
    background: rgba(255,61,0,.14);
    border-left: 5px solid #FF3D00;
    padding: 18px;
    border-radius: 16px;
    margin: 16px 0;
}

.good {
    background: rgba(0,200,83,.13);
    border-left: 5px solid #00C853;
    padding: 18px;
    border-radius: 16px;
    margin: 16px 0;
}

@media only screen and (max-width: 768px) {
    .hero h1 {font-size: 30px;}
    .phase {font-size: 25px;}
    .big-score {font-size: 30px;}
    .card {padding: 18px;}
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# CONSTANTS
# =====================================================
PHASES = [
    "Accumulation",
    "Early Bull",
    "Mature Bull",
    "Late Bull",
    "Distribution",
    "Early Bear",
    "Mature Bear",
    "Late Bear"
]

FEATURES = [
    "Trend",
    "Breadth",
    "Liquidity",
    "Valuation",
    "Sentiment",
    "Macro"
]

PHASE_TEXT = {
    "Accumulation": "Cheap valuation, depressed sentiment, improving liquidity, and early long-term buying.",
    "Early Bull": "Economy and liquidity improve before the crowd fully believes the recovery.",
    "Mature Bull": "Strong trend, broad participation, and healthy macro conditions.",
    "Late Bull": "Investor psychology dominates. Euphoria and expensive valuation increase risk.",
    "Distribution": "Index may still look strong, but breadth and liquidity weaken underneath.",
    "Early Bear": "Trend and breadth start breaking. Investors often mistake this for a correction.",
    "Mature Bear": "Weak trend, weak breadth, poor liquidity, and fearful sentiment dominate.",
    "Late Bear": "Panic or exhaustion. Valuation improves, but trend may still be weak."
}

PHASE_MODELS = {
    "Accumulation": {
        "weights": {"Trend": .10, "Breadth": .16, "Liquidity": .22, "Valuation": .26, "Sentiment": .18, "Macro": .08},
        "ideal": {"Trend": (15,45), "Breadth": (20,50), "Liquidity": (55,90), "Valuation": (0,35), "Sentiment": (0,35), "Macro": (35,65)}
    },
    "Early Bull": {
        "weights": {"Trend": .18, "Breadth": .18, "Liquidity": .16, "Valuation": .12, "Sentiment": .10, "Macro": .26},
        "ideal": {"Trend": (45,70), "Breadth": (45,75), "Liquidity": (50,80), "Valuation": (35,65), "Sentiment": (35,60), "Macro": (55,85)}
    },
    "Mature Bull": {
        "weights": {"Trend": .24, "Breadth": .24, "Liquidity": .14, "Valuation": .08, "Sentiment": .12, "Macro": .18},
        "ideal": {"Trend": (65,90), "Breadth": (65,90), "Liquidity": (55,85), "Valuation": (40,70), "Sentiment": (50,75), "Macro": (55,85)}
    },
    "Late Bull": {
        "weights": {"Trend": .18, "Breadth": .10, "Liquidity": .08, "Valuation": .18, "Sentiment": .34, "Macro": .12},
        "ideal": {"Trend": (75,100), "Breadth": (45,75), "Liquidity": (35,65), "Valuation": (75,100), "Sentiment": (80,100), "Macro": (45,75)}
    },
    "Distribution": {
        "weights": {"Trend": .16, "Breadth": .28, "Liquidity": .16, "Valuation": .18, "Sentiment": .16, "Macro": .06},
        "ideal": {"Trend": (55,85), "Breadth": (25,55), "Liquidity": (25,55), "Valuation": (65,100), "Sentiment": (65,95), "Macro": (35,65)}
    },
    "Early Bear": {
        "weights": {"Trend": .25, "Breadth": .25, "Liquidity": .14, "Valuation": .08, "Sentiment": .18, "Macro": .10},
        "ideal": {"Trend": (30,55), "Breadth": (25,55), "Liquidity": (25,60), "Valuation": (45,75), "Sentiment": (35,65), "Macro": (30,60)}
    },
    "Mature Bear": {
        "weights": {"Trend": .24, "Breadth": .22, "Liquidity": .18, "Valuation": .08, "Sentiment": .18, "Macro": .10},
        "ideal": {"Trend": (10,40), "Breadth": (10,40), "Liquidity": (10,45), "Valuation": (25,55), "Sentiment": (15,45), "Macro": (20,50)}
    },
    "Late Bear": {
        "weights": {"Trend": .14, "Breadth": .14, "Liquidity": .18, "Valuation": .24, "Sentiment": .22, "Macro": .08},
        "ideal": {"Trend": (0,35), "Breadth": (0,35), "Liquidity": (35,70), "Valuation": (0,35), "Sentiment": (0,25), "Macro": (20,55)}
    }
}


# =====================================================
# HELPERS
# =====================================================
def clamp(x, low=0, high=100):
    return float(max(low, min(high, x)))


def get_close(df):
    if df is None or df.empty:
        return pd.Series(dtype=float)

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            s = df["Close"].iloc[:, 0]
        else:
            s = df.iloc[:, 0]
    else:
        s = df["Close"] if "Close" in df.columns else df.iloc[:, 0]

    return pd.to_numeric(s, errors="coerce").dropna()


def get_volume(df):
    if df is None or df.empty:
        return pd.Series(dtype=float)

    if isinstance(df.columns, pd.MultiIndex):
        if "Volume" in df.columns.get_level_values(0):
            s = df["Volume"].iloc[:, 0]
        else:
            return pd.Series(dtype=float)
    else:
        if "Volume" not in df.columns:
            return pd.Series(dtype=float)
        s = df["Volume"]

    return pd.to_numeric(s, errors="coerce").dropna()


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


def ideal_range_score(value, low, high):
    if low <= value <= high:
        return 100
    if value < low:
        return clamp(100 - (low - value) * 2.4)
    return clamp(100 - (value - high) * 2.4)


def score_color(score):
    if score >= 70:
        return "#00C853"
    if score >= 45:
        return "#FFD21F"
    if score >= 25:
        return "#FF9800"
    return "#FF3D00"


# =====================================================
# LIVE DATA FETCH
# =====================================================
@st.cache_data(ttl=60 * 60)
def download_ticker(ticker, period="2y"):
    return yf.download(
        ticker,
        period=period,
        interval="1d",
        progress=False,
        auto_adjust=True,
        threads=True
    )


@st.cache_data(ttl=60 * 60)
def fetch_all_market_data():
    tickers = {
        "nifty": "^NSEI",
        "nifty_bank": "^NSEBANK",
        "nifty_bees": "NIFTYBEES.NS",
        "junior_bees": "JUNIORBEES.NS",
        "midcap_bees": "MID150BEES.NS",
        "india_vix": "^INDIAVIX",
        "usd_inr": "INR=X",
        "crude": "CL=F",
        "gold": "GC=F",
        "us10y": "^TNX"
    }

    data = {}
    for key, ticker in tickers.items():
        try:
            data[key] = download_ticker(ticker)
        except Exception:
            data[key] = pd.DataFrame()

    return data


# =====================================================
# AUTOMATIC SCORE CALCULATION
# =====================================================
def calculate_trend_score(data):
    nifty = get_close(data.get("nifty"))
    if len(nifty) < 220:
        return 50, {}

    price = nifty.iloc[-1]
    sma20 = nifty.rolling(20).mean().iloc[-1]
    sma50 = nifty.rolling(50).mean().iloc[-1]
    sma200 = nifty.rolling(200).mean().iloc[-1]
    rsi_now = rsi(nifty).iloc[-1]
    ret20 = pct(nifty.iloc[-1], nifty.iloc[-21])
    ret60 = pct(nifty.iloc[-1], nifty.iloc[-61])
    distance_200 = pct(price, sma200)

    score = 0
    score += 20 if price > sma20 else 5
    score += 25 if price > sma50 else 5
    score += 25 if price > sma200 else 5
    score += 15 if sma50 > sma200 else 3
    score += 10 if ret20 > 0 else 2
    score += 5 if 45 <= rsi_now <= 70 else 2 if rsi_now > 70 else 0

    return clamp(score), {
        "Nifty Price": round(price, 2),
        "20 DMA": round(sma20, 2),
        "50 DMA": round(sma50, 2),
        "200 DMA": round(sma200, 2),
        "RSI": round(rsi_now, 1),
        "20D Return %": round(ret20, 2),
        "60D Return %": round(ret60, 2),
        "Distance From 200DMA %": round(distance_200, 2)
    }


def calculate_breadth_score(data):
    nifty = get_close(data.get("nifty"))
    bank = get_close(data.get("nifty_bank"))
    junior = get_close(data.get("junior_bees"))
    midcap = get_close(data.get("midcap_bees"))

    series_list = [s for s in [nifty, bank, junior, midcap] if len(s) > 220]
    if len(series_list) < 2:
        return 50, {}

    score = 0
    details = {}

    positive_20 = 0
    above_50 = 0
    above_200 = 0

    names = ["Nifty", "Bank Nifty", "Junior Bees", "Midcap Bees"]
    for name, s in zip(names, [nifty, bank, junior, midcap]):
        if len(s) > 220:
            ret20 = pct(s.iloc[-1], s.iloc[-21])
            sma50 = s.rolling(50).mean().iloc[-1]
            sma200 = s.rolling(200).mean().iloc[-1]

            if ret20 > 0:
                positive_20 += 1
            if s.iloc[-1] > sma50:
                above_50 += 1
            if s.iloc[-1] > sma200:
                above_200 += 1

            details[f"{name} 20D Return %"] = round(ret20, 2)

    total = len(series_list)
    score += positive_20 / total * 30
    score += above_50 / total * 30
    score += above_200 / total * 40

    details["Markets Positive 20D"] = f"{positive_20}/{total}"
    details["Markets Above 50DMA"] = f"{above_50}/{total}"
    details["Markets Above 200DMA"] = f"{above_200}/{total}"

    return clamp(score), details


def calculate_liquidity_score(data):
    nifty_bees = get_close(data.get("nifty_bees"))
    volume = get_volume(data.get("nifty_bees"))
    usd_inr = get_close(data.get("usd_inr"))
    us10y = get_close(data.get("us10y"))

    score = 50
    details = {}

    if len(nifty_bees) > 120:
        ret60 = pct(nifty_bees.iloc[-1], nifty_bees.iloc[-61])
        score += 15 if ret60 > 0 else -10
        details["NiftyBees 60D Return %"] = round(ret60, 2)

    if len(volume) > 120:
        vol20 = volume.tail(20).mean()
        vol120 = volume.tail(120).mean()
        if vol20 > vol120:
            score += 10
        else:
            score -= 5
        details["Volume 20D / 120D"] = round(vol20 / vol120, 2) if vol120 else 0

    if len(usd_inr) > 60:
        inr_move = pct(usd_inr.iloc[-1], usd_inr.iloc[-61])
        score += 10 if inr_move < 2 else -10
        details["USDINR 60D Move %"] = round(inr_move, 2)

    if len(us10y) > 60:
        yield_move = us10y.iloc[-1] - us10y.iloc[-61]
        score += 10 if yield_move < 0 else -5
        details["US10Y 60D Change"] = round(yield_move, 2)

    return clamp(score), details


def calculate_valuation_score(data):
    nifty = get_close(data.get("nifty"))
    if len(nifty) < 252:
        return 50, {}

    one_year_return = pct(nifty.iloc[-1], nifty.iloc[-252])
    distance_from_high = pct(nifty.iloc[-1], nifty.rolling(252).max().iloc[-1])

    if one_year_return > 30:
        score = 88
    elif one_year_return > 20:
        score = 75
    elif one_year_return > 10:
        score = 60
    elif one_year_return > 0:
        score = 45
    elif one_year_return > -12:
        score = 30
    else:
        score = 18

    if distance_from_high > -3:
        score += 8
    elif distance_from_high < -20:
        score -= 10

    return clamp(score), {
        "Nifty 1Y Return %": round(one_year_return, 2),
        "Distance From 1Y High %": round(distance_from_high, 2),
        "Note": "Valuation proxy. For production, connect live Nifty PE/PB."
    }


def calculate_sentiment_score(data):
    vix = get_close(data.get("india_vix"))
    nifty = get_close(data.get("nifty"))

    if len(vix) < 120:
        return 50, {}

    current_vix = vix.iloc[-1]
    vix_percentile = (vix.tail(252) < current_vix).mean() * 100

    # Higher score means more greed / comfort.
    score = 100 - vix_percentile

    details = {
        "India VIX": round(current_vix, 2),
        "VIX Percentile": round(vix_percentile, 1)
    }

    if len(nifty) > 252:
        drawdown = pct(nifty.iloc[-1], nifty.rolling(252).max().iloc[-1])
        if drawdown > -3:
            score += 10
        elif drawdown < -15:
            score -= 15
        details["Nifty Drawdown From 1Y High %"] = round(drawdown, 2)

    return clamp(score), details


def calculate_macro_score(data):
    usd_inr = get_close(data.get("usd_inr"))
    crude = get_close(data.get("crude"))
    gold = get_close(data.get("gold"))
    us10y = get_close(data.get("us10y"))
    nifty = get_close(data.get("nifty"))

    score = 55
    details = {}

    if len(usd_inr) > 120:
        move = pct(usd_inr.iloc[-1], usd_inr.iloc[-121])
        score += 8 if move < 3 else -8
        details["USDINR 6M Move %"] = round(move, 2)

    if len(crude) > 120:
        move = pct(crude.iloc[-1], crude.iloc[-121])
        score += 8 if move < 10 else -8
        details["Crude 6M Move %"] = round(move, 2)

    if len(us10y) > 120:
        move = us10y.iloc[-1] - us10y.iloc[-121]
        score += 8 if move < 0 else -6
        details["US10Y 6M Change"] = round(move, 2)

    if len(gold) > 120:
        move = pct(gold.iloc[-1], gold.iloc[-121])
        score += 4 if move < 15 else -4
        details["Gold 6M Move %"] = round(move, 2)

    if len(nifty) > 120:
        move = pct(nifty.iloc[-1], nifty.iloc[-121])
        score += 8 if move > 0 else -8
        details["Nifty 6M Move %"] = round(move, 2)

    return clamp(score), details


def calculate_all_live_scores():
    data = fetch_all_market_data()

    trend, trend_d = calculate_trend_score(data)
    breadth, breadth_d = calculate_breadth_score(data)
    liquidity, liquidity_d = calculate_liquidity_score(data)
    valuation, valuation_d = calculate_valuation_score(data)
    sentiment, sentiment_d = calculate_sentiment_score(data)
    macro, macro_d = calculate_macro_score(data)

    scores = {
        "Trend": trend,
        "Breadth": breadth,
        "Liquidity": liquidity,
        "Valuation": valuation,
        "Sentiment": sentiment,
        "Macro": macro
    }

    details = {
        "Trend": trend_d,
        "Breadth": breadth_d,
        "Liquidity": liquidity_d,
        "Valuation": valuation_d,
        "Sentiment": sentiment_d,
        "Macro": macro_d
    }

    return scores, details, data


# =====================================================
# RULE ENGINE
# =====================================================
def phase_rule_score(phase, raw):
    model = PHASE_MODELS[phase]
    adjusted = {}

    for f in FEATURES:
        low, high = model["ideal"][f]
        adjusted[f] = ideal_range_score(raw[f], low, high)

    base = sum(adjusted[f] * model["weights"][f] for f in FEATURES)
    penalty = 0

    if phase in ["Early Bull", "Mature Bull", "Late Bull"]:
        if raw["Trend"] < 35:
            penalty += 8
        if raw["Breadth"] < 35:
            penalty += 8
        if raw["Liquidity"] < 25:
            penalty += 5

    if phase in ["Early Bear", "Mature Bear", "Late Bear"]:
        if raw["Trend"] > 70:
            penalty += 8
        if raw["Breadth"] > 70:
            penalty += 8

    if phase == "Late Bull":
        if raw["Sentiment"] < 70:
            penalty += 10
        if raw["Valuation"] < 65:
            penalty += 10

    if phase == "Distribution":
        if raw["Breadth"] > 70:
            penalty += 12
        if raw["Valuation"] < 55:
            penalty += 8

    if phase == "Accumulation":
        if raw["Valuation"] > 45:
            penalty += 10
        if raw["Sentiment"] > 45:
            penalty += 8

    return clamp(base - penalty), base, penalty, adjusted


def get_rule_results(raw):
    rows = []
    for phase in PHASES:
        final, base, penalty, adjusted = phase_rule_score(phase, raw)
        rows.append({
            "Phase": phase,
            "Rule Score": final,
            "Base Score": base,
            "Noise Penalty": penalty,
            "Adjusted": adjusted
        })
    return sorted(rows, key=lambda x: x["Rule Score"], reverse=True)


# =====================================================
# XGBOOST MODEL
# =====================================================
def create_synthetic_training_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    for phase in PHASES:
        model = PHASE_MODELS[phase]

        for _ in range(n // len(PHASES)):
            row = {}
            for f in FEATURES:
                low, high = model["ideal"][f]
                center = (low + high) / 2
                spread = max((high - low) / 2, 8)
                row[f] = clamp(rng.normal(center, spread))
            row["Phase"] = phase
            rows.append(row)

    df = pd.DataFrame(rows)
    df[FEATURES] = np.clip(df[FEATURES] + rng.normal(0, 5, df[FEATURES].shape), 0, 100)
    return df


@st.cache_resource
def train_model():
    df = create_synthetic_training_data()

    le = LabelEncoder()
    y = le.fit_transform(df["Phase"])
    X = df[FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.22, random_state=42, stratify=y
    )

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(PHASES),
        n_estimators=420,
        max_depth=4,
        learning_rate=.035,
        subsample=.88,
        colsample_bytree=.88,
        min_child_weight=2,
        reg_lambda=2.0,
        reg_alpha=.15,
        eval_metric="mlogloss",
        random_state=42
    )

    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, le, acc


def xgb_predict(model, le, scores):
    X_live = pd.DataFrame([scores], columns=FEATURES)
    probs = model.predict_proba(X_live)[0]
    classes = le.inverse_transform(np.arange(len(probs)))

    return pd.DataFrame({
        "Phase": classes,
        "XGBoost Probability": probs * 100
    }).sort_values("XGBoost Probability", ascending=False)


def hybrid_prediction(xgb_df, rule_rows):
    rule_df = pd.DataFrame([
        {"Phase": r["Phase"], "Rule Score": r["Rule Score"]}
        for r in rule_rows
    ])

    merged = xgb_df.merge(rule_df, on="Phase", how="left")
    merged["Hybrid Score"] = merged["XGBoost Probability"] * .65 + merged["Rule Score"] * .35
    return merged.sort_values("Hybrid Score", ascending=False)


# =====================================================
# UI HEADER
# =====================================================
st.markdown("""
<div class="hero">
    <h1>🐝 Madness of Money Bees</h1>
    <h3>Automatic XGBoost Market Cycle Predictor</h3>
    <p>Live market data • Dynamic phase scoring • Noise filtering • Premium cycle intelligence</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🐝 Money Bees Control Room")
st.sidebar.caption("This version is automatic. Visitors do not need to enter market scores.")

refresh = st.sidebar.button("Refresh Live Data")

if refresh:
    st.cache_data.clear()

previous_phase = st.sidebar.selectbox(
    "Previous Confirmed Phase",
    ["None"] + PHASES,
    index=0
)


# =====================================================
# RUN DATA + MODEL
# =====================================================
try:
    live_scores, live_details, raw_data = calculate_all_live_scores()
    data_status = "Live data loaded successfully"
except Exception as e:
    st.error("Live market data failed. Check internet/data source availability.")
    st.exception(e)
    st.stop()

model, le, acc = train_model()
rule_rows = get_rule_results(live_scores)
xgb_df = xgb_predict(model, le, live_scores)


# Transition smoothing
if previous_phase != "None":
    adjusted = xgb_df.copy()
    prev_i = PHASES.index(previous_phase)

    for idx, row in adjusted.iterrows():
        i = PHASES.index(row["Phase"])
        distance = min(abs(i - prev_i), len(PHASES) - abs(i - prev_i))

        if distance == 0:
            mult = 1.05
        elif distance == 1:
            mult = 1.00
        elif distance == 2:
            mult = .78
        else:
            mult = .45

        adjusted.loc[idx, "XGBoost Probability"] *= mult

    adjusted["XGBoost Probability"] = adjusted["XGBoost Probability"] / adjusted["XGBoost Probability"].sum() * 100
    xgb_df = adjusted.sort_values("XGBoost Probability", ascending=False)

hybrid_df = hybrid_prediction(xgb_df, rule_rows)

winner = hybrid_df.iloc[0]
runner = hybrid_df.iloc[1]

current_phase = winner["Phase"]
final_score = winner["Hybrid Score"]
confidence = clamp(55 + (winner["Hybrid Score"] - runner["Hybrid Score"]) * 2.4, 50, 96)


# =====================================================
# TOP CARDS
# =====================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="phase-card">
        <p class="small">Predicted Market Phase</p>
        <div class="phase">{current_phase}</div>
        <p class="small">{PHASE_TEXT[current_phase]}</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card">
        <p class="small">Final Hybrid Score</p>
        <div class="big-score">{final_score:.1f}/100</div>
        <p class="small">65% XGBoost + 35% phase-rule engine.</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card">
        <p class="small">Prediction Confidence</p>
        <div class="big-score">{confidence:.0f}%</div>
        <p class="small">Based on winner strength and runner-up gap.</p>
    </div>
    """, unsafe_allow_html=True)


st.markdown(f"""
<div class="good">
<b>{data_status}</b><br>
Last refreshed: {datetime.now().strftime("%d %b %Y, %I:%M %p")}
</div>
""", unsafe_allow_html=True)


# =====================================================
# MARKET CYCLE WAVE
# =====================================================
st.markdown("## 🐝 Market Cycle Wave")

phase_positions = {
    "Accumulation": 8,
    "Early Bull": 22,
    "Mature Bull": 38,
    "Late Bull": 52,
    "Distribution": 64,
    "Early Bear": 76,
    "Mature Bear": 88,
    "Late Bear": 96
}

x = np.linspace(0, 100, 700)
y = np.sin(x / 7.5) * 10 + x * .23

cx = phase_positions[current_phase]
cy = np.sin(cx / 7.5) * 10 + cx * .23

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x, y=y,
    mode="lines",
    line=dict(width=5, color="#FFD21F"),
    name="Market Cycle"
))

fig.add_trace(go.Scatter(
    x=[cx], y=[cy],
    mode="markers+text",
    marker=dict(size=25, color="#050505", line=dict(color="#FFD21F", width=5)),
    text=[current_phase],
    textposition="top center",
    name="Current Phase"
))

for phase, px in phase_positions.items():
    py = np.sin(px / 7.5) * 10 + px * .23
    fig.add_annotation(
        x=px, y=py, text=phase,
        showarrow=False,
        font=dict(color="white", size=11),
        bgcolor="rgba(17,17,17,.85)",
        bordercolor="#FFD21F",
        borderwidth=1
    )

fig.update_layout(
    paper_bgcolor="#050505",
    plot_bgcolor="#050505",
    font=dict(color="white"),
    height=460,
    margin=dict(l=20, r=20, t=40, b=30),
    xaxis=dict(title="Market Cycle Journey", showticklabels=False, gridcolor="rgba(255,255,255,.08)"),
    yaxis=dict(title="Cycle Momentum", gridcolor="rgba(255,255,255,.08)")
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class="explain">
<h3>How to read this chart</h3>
<p>
The yellow wave represents the market cycle. Bull phases appear on the rising part.
Distribution appears near the top, where the index may look strong but internal indicators weaken.
Bear phases appear on the falling part. Accumulation appears near the bottom, where sentiment is weak
but long-term opportunity may start forming.
</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# LIVE SCORE CARDS
# =====================================================
st.markdown("## 📊 Automatically Calculated Live Scores")

descriptions = {
    "Trend": "Price trend, moving averages, RSI, 20D/60D momentum, and 200DMA distance.",
    "Breadth": "Participation using Nifty, Bank Nifty, Junior Bees, and Midcap Bees proxies.",
    "Liquidity": "Volume, ETF flow proxy, USDINR pressure, and global yield pressure.",
    "Valuation": "Automatic valuation proxy using 1Y return and distance from highs.",
    "Sentiment": "India VIX percentile and drawdown-based fear/greed proxy.",
    "Macro": "USDINR, crude, gold, US yields, and broad market macro pressure."
}

cols = st.columns(3)

for i, f in enumerate(FEATURES):
    v = live_scores[f]
    with cols[i % 3]:
        st.markdown(f"""
        <div class="card">
            <h3>{f}</h3>
            <div class="big-score" style="color:{score_color(v)};">{v:.1f}/100</div>
            <p class="small">{descriptions[f]}</p>
        </div>
        """, unsafe_allow_html=True)


# =====================================================
# PHASE RANKING
# =====================================================
st.markdown("## 🧠 Final Phase Ranking")

rank_df = hybrid_df.copy()
rank_df["XGBoost Probability"] = rank_df["XGBoost Probability"].round(1)
rank_df["Rule Score"] = rank_df["Rule Score"].round(1)
rank_df["Hybrid Score"] = rank_df["Hybrid Score"].round(1)

st.dataframe(rank_df, use_container_width=True, hide_index=True)

if winner["Hybrid Score"] - runner["Hybrid Score"] < 7:
    st.markdown(f"""
    <div class="warning">
    <h3>Close Phase Warning</h3>
    <p>
    The model selected <b>{current_phase}</b>, but <b>{runner["Phase"]}</b> is close.
    This means the market may be transitioning between phases.
    </p>
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# FEATURE IMPORTANCE
# =====================================================
st.markdown("## 🔍 XGBoost Feature Importance")

imp = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
})

imp["Importance"] = imp["Importance"] / imp["Importance"].sum() * 100
imp = imp.sort_values("Importance", ascending=True)

fig_imp = go.Figure()
fig_imp.add_trace(go.Bar(
    x=imp["Importance"],
    y=imp["Feature"],
    orientation="h",
    marker=dict(color="#FFD21F")
))

fig_imp.update_layout(
    paper_bgcolor="#050505",
    plot_bgcolor="#050505",
    font=dict(color="white"),
    height=380,
    xaxis=dict(title="Importance %", gridcolor="rgba(255,255,255,.08)"),
    yaxis=dict(title=""),
    margin=dict(l=20, r=20, t=30, b=30)
)

st.plotly_chart(fig_imp, use_container_width=True)


# =====================================================
# CURRENT PHASE WEIGHTS
# =====================================================
st.markdown("## ⚖️ Current Phase Weightage")

weights = PHASE_MODELS[current_phase]["weights"]
ideal = PHASE_MODELS[current_phase]["ideal"]

weight_df = pd.DataFrame({
    "Indicator": FEATURES,
    "Current Live Score": [round(live_scores[f], 1) for f in FEATURES],
    "Weight In This Phase %": [round(weights[f] * 100, 1) for f in FEATURES],
    "Ideal Range For This Phase": [f"{ideal[f][0]} - {ideal[f][1]}" for f in FEATURES]
})

st.dataframe(weight_df, use_container_width=True, hide_index=True)


# =====================================================
# LIVE DATA DETAILS
# =====================================================
st.markdown("## 🧾 Live Data Breakdown")

for section, values in live_details.items():
    with st.expander(f"{section} Details"):
        if values:
            st.dataframe(
                pd.DataFrame(list(values.items()), columns=["Metric", "Value"]),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.write("Not enough live data available for detailed breakdown.")


# =====================================================
# RAW CHART
# =====================================================
st.markdown("## 📈 Nifty 50 Price Chart")

nifty_close = get_close(raw_data.get("nifty"))

if len(nifty_close) > 20:
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=nifty_close.index,
        y=nifty_close.values,
        mode="lines",
        line=dict(color="#FFD21F", width=3),
        name="Nifty 50"
    ))

    fig_price.update_layout(
        paper_bgcolor="#050505",
        plot_bgcolor="#050505",
        font=dict(color="white"),
        height=420,
        xaxis=dict(gridcolor="rgba(255,255,255,.08)"),
        yaxis=dict(title="Price", gridcolor="rgba(255,255,255,.08)")
    )

    st.plotly_chart(fig_price, use_container_width=True)


# =====================================================
# NOTE
# =====================================================
st.markdown("""
<div class="warning">
<h3>Important Production Note</h3>
<p>
This version is fully automatic and does not require user input. It uses public market data and automatic proxies.
For the most professional paid version, later connect official Nifty PE/PB, FII/DII cash flow, PMI, CPI, GDP,
and advance-decline breadth data. That will improve accuracy.
</p>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="danger">
<h3>Disclaimer</h3>
<p>
This dashboard is for education, analytics, and research only. It is not financial advice.
Market cycles are probabilistic and can change quickly.
</p>
</div>
""", unsafe_allow_html=True)
