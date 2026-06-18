import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
import joblib

from datetime import datetime


st.set_page_config(
    page_title="Madness of Money Bees",
    page_icon="🐝",
    layout="wide"
)


st.markdown("""
<style>
.stApp {
    background-color: #050505;
    color: #F8F8F8;
    background-image:
      radial-gradient(circle at 25px 25px, rgba(255,210,31,.22) 2px, transparent 2px),
      radial-gradient(circle at 55px 55px, rgba(245,183,0,.14) 2px, transparent 2px);
    background-size: 60px 60px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111111, #050505);
    border-right: 1px solid rgba(255,210,31,.35);
}

.hero {
    padding: 34px;
    border-radius: 32px;
    background: linear-gradient(135deg, #FFD21F, #F5B700);
    color: #050505;
    text-align: center;
    box-shadow: 0 0 50px rgba(255,210,31,.40);
    margin-bottom: 24px;
}

.hero h1 {
    font-size: 46px;
    font-weight: 950;
    margin-bottom: 6px;
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

PHASE_TEXT = {
    "Accumulation": "Cheap valuation, depressed sentiment, improving liquidity, and early long-term buying.",
    "Early Bull": "Economy and liquidity improve before the crowd fully believes the recovery.",
    "Mature Bull": "Strong trend, broad participation, and healthy macro conditions.",
    "Late Bull": "Investor psychology dominates. Euphoria and expensive valuation increase risk.",
    "Distribution": "Index may still look strong, but breadth and liquidity weaken underneath.",
    "Early Bear": "Trend and breadth start breaking. Investors often mistake this for a correction.",
    "Mature Bear": "Weak trend, weak breadth, poor liquidity, and fearful sentiment dominate.",
    "Late Bear": "Panic or exhaustion. Valuation improves, but trend may still be weak.",
}


def clamp(value, low=0, high=100):
    return float(max(low, min(high, value)))


def get_close(df):
    if df is None or df.empty:
        return pd.Series(dtype=float)

    if isinstance(df.columns, pd.MultiIndex):
        return df["Close"].iloc[:, 0].dropna()

    return df["Close"].dropna()


def pct(a, b):
    if b == 0 or pd.isna(b):
        return 0
    return (a / b - 1) * 100


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def score_color(score):
    if score >= 70:
        return "#00C853"
    if score >= 45:
        return "#FFD21F"
    if score >= 25:
        return "#FF9800"
    return "#FF3D00"


@st.cache_resource
def load_model():
    files = [
        "xgb_market_cycle_model.pkl",
        "phase_label_encoder.pkl",
        "model_features.pkl",
    ]

    missing = [f for f in files if not os.path.exists(f)]

    if missing:
        return None, None, None, missing

    return (
        joblib.load("xgb_market_cycle_model.pkl"),
        joblib.load("phase_label_encoder.pkl"),
        joblib.load("model_features.pkl"),
        [],
    )


@st.cache_data(ttl=60 * 60)
def download_ticker(ticker, period="20y"):
    return yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )


@st.cache_data(ttl=60 * 60)
def fetch_market_data():
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

    for key, ticker in tickers.items():
        try:
            data[key] = download_ticker(ticker)
        except Exception:
            data[key] = pd.DataFrame()

    return data


def score_trend(close):
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    rsi_v = rsi(close)
    ret20 = close.pct_change(20) * 100

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


def calculate_live_scores():
    data = fetch_market_data()

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

    latest = df.iloc[-1]

    scores = {f: float(latest[f]) for f in FEATURES}

    details = {
        "Latest Date": df.index[-1].strftime("%d %b %Y"),
        "Nifty": round(nifty.loc[df.index[-1]], 2),
        "Trend": round(scores["Trend"], 1),
        "Breadth": round(scores["Breadth"], 1),
        "Liquidity": round(scores["Liquidity"], 1),
        "Valuation": round(scores["Valuation"], 1),
        "Sentiment": round(scores["Sentiment"], 1),
        "Macro": round(scores["Macro"], 1),
    }

    return scores, details, df, nifty


def predict_phase(model, encoder, scores):
    X = pd.DataFrame([scores], columns=FEATURES)
    probabilities = model.predict_proba(X)[0]
    classes = encoder.inverse_transform(np.arange(len(probabilities)))

    result = pd.DataFrame({
        "Phase": classes,
        "Probability": probabilities * 100,
    }).sort_values("Probability", ascending=False)

    return result


st.markdown("""
<div class="hero">
    <h1>🐝 Madness of Money Bees</h1>
    <h3>20-Year Trained XGBoost Market Cycle Predictor</h3>
    <p>Automatic live prediction using historical market-trained regime intelligence.</p>
</div>
""", unsafe_allow_html=True)


st.sidebar.title("🐝 Control Room")

if st.sidebar.button("Refresh Live Data"):
    st.cache_data.clear()

model, encoder, model_features, missing = load_model()

if missing:
    st.error("Model files missing. Run `python train.py` first and commit the generated .pkl files.")
    st.write(missing)
    st.stop()

try:
    scores, details, history_df, nifty = calculate_live_scores()
except Exception as e:
    st.error("Could not fetch live market data.")
    st.exception(e)
    st.stop()

prediction = predict_phase(model, encoder, scores)

winner = prediction.iloc[0]
runner = prediction.iloc[1]

current_phase = winner["Phase"]
confidence = winner["Probability"]

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="phase-card">
        <p class="small">Current Market Phase</p>
        <div class="phase">{current_phase}</div>
        <p class="small">{PHASE_TEXT[current_phase]}</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card">
        <p class="small">XGBoost Confidence</p>
        <div class="big-score">{confidence:.1f}%</div>
        <p class="small">Based on 20-year historical feature training.</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card">
        <p class="small">Runner-up Phase</p>
        <div class="big-score">{runner["Phase"]}</div>
        <p class="small">{runner["Probability"]:.1f}% probability.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div class="good">
<b>Live market data loaded.</b><br>
Latest market date: {details["Latest Date"]}<br>
Last app refresh: {datetime.now().strftime("%d %b %Y, %I:%M %p")}
</div>
""", unsafe_allow_html=True)


st.markdown("## 🐝 Market Cycle Wave")

phase_positions = {
    "Accumulation": 8,
    "Early Bull": 22,
    "Mature Bull": 38,
    "Late Bull": 52,
    "Distribution": 64,
    "Early Bear": 76,
    "Mature Bear": 88,
    "Late Bear": 96,
}

x = np.linspace(0, 100, 700)
y = np.sin(x / 7.5) * 10 + x * 0.23

cx = phase_positions[current_phase]
cy = np.sin(cx / 7.5) * 10 + cx * 0.23

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode="lines",
    line=dict(width=5, color="#FFD21F"),
    name="Market Cycle",
))

fig.add_trace(go.Scatter(
    x=[cx],
    y=[cy],
    mode="markers+text",
    marker=dict(
        size=25,
        color="#050505",
        line=dict(color="#FFD21F", width=5),
    ),
    text=[current_phase],
    textposition="top center",
    name="Current Phase",
))

for phase, px in phase_positions.items():
    py = np.sin(px / 7.5) * 10 + px * 0.23
    fig.add_annotation(
        x=px,
        y=py,
        text=phase,
        showarrow=False,
        font=dict(color="white", size=11),
        bgcolor="rgba(17,17,17,.85)",
        bordercolor="#FFD21F",
        borderwidth=1,
    )

fig.update_layout(
    paper_bgcolor="#050505",
    plot_bgcolor="#050505",
    font=dict(color="white"),
    height=460,
    margin=dict(l=20, r=20, t=40, b=30),
    xaxis=dict(showticklabels=False, gridcolor="rgba(255,255,255,.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,.08)"),
)

st.plotly_chart(fig, use_container_width=True)


st.markdown("## 📊 Live Indicator Scores")

cols = st.columns(3)

for i, f in enumerate(FEATURES):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="card">
            <h3>{f}</h3>
            <div class="big-score" style="color:{score_color(scores[f])};">{scores[f]:.1f}/100</div>
        </div>
        """, unsafe_allow_html=True)


st.markdown("## 🧠 Phase Probability Ranking")

rank_df = prediction.copy()
rank_df["Probability"] = rank_df["Probability"].round(2)

st.dataframe(rank_df, use_container_width=True, hide_index=True)


st.markdown("## 🔍 XGBoost Feature Importance")

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_,
})

importance["Importance"] = importance["Importance"] / importance["Importance"].sum() * 100
importance = importance.sort_values("Importance", ascending=True)

fig_imp = go.Figure()

fig_imp.add_trace(go.Bar(
    x=importance["Importance"],
    y=importance["Feature"],
    orientation="h",
    marker=dict(color="#FFD21F"),
))

fig_imp.update_layout(
    paper_bgcolor="#050505",
    plot_bgcolor="#050505",
    font=dict(color="white"),
    height=380,
    xaxis=dict(title="Importance %", gridcolor="rgba(255,255,255,.08)"),
    yaxis=dict(title=""),
)

st.plotly_chart(fig_imp, use_container_width=True)


st.markdown("## 📈 Nifty 50 Chart")

fig_price = go.Figure()

fig_price.add_trace(go.Scatter(
    x=nifty.index,
    y=nifty.values,
    mode="lines",
    line=dict(color="#FFD21F", width=3),
    name="Nifty 50",
))

fig_price.update_layout(
    paper_bgcolor="#050505",
    plot_bgcolor="#050505",
    font=dict(color="white"),
    height=420,
    xaxis=dict(gridcolor="rgba(255,255,255,.08)"),
    yaxis=dict(title="Price", gridcolor="rgba(255,255,255,.08)"),
)

st.plotly_chart(fig_price, use_container_width=True)


st.markdown("## 🧾 Live Data Snapshot")

st.dataframe(
    pd.DataFrame(list(details.items()), columns=["Metric", "Value"]),
    use_container_width=True,
    hide_index=True,
)


st.markdown("""
<div class="warning">
<h3>Important Note</h3>
<p>
This model is trained on 20 years of historical market-derived features.
However, phase labels are generated using rule-based market-regime logic.
For institutional-grade accuracy, manually label historical phases or add official PE/PB, FII/DII flows,
PMI, CPI, GDP, and full Nifty 500 breadth.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="danger">
<h3>Disclaimer</h3>
<p>
This tool is for education and analytics only. It is not financial advice.
</p>
</div>
""", unsafe_allow_html=True)
