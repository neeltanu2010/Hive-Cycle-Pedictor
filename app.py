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
from protect import require_tool_access, record_tool_use


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Madness of Money Bees",
    page_icon="🐝",
    layout="wide"
)


# =====================================================
# FINANCIFY ACCESS CONTROL
# =====================================================
require_tool_access(
    tool_name="hive-cycle-predictor",
    display_name="Madness of Money Bees",
    free_limit=5,
    pro_limit=100,
)


# =====================================================
# CSS
# =====================================================
st.markdown("""
<style>
:root {
    --bee-black: #090600;
    --bee-ink: #151007;
    --bee-yellow: #FFD21F;
    --bee-gold: #F5B700;
    --bee-orange: #FF9800;
    --bee-text: #FFF8DA;
    --bee-muted: #D9CFA6;
}
.stApp {
    color: var(--bee-text);
    background-color: #0A0700;
    background-image:
        radial-gradient(circle at 18% 10%, rgba(255,210,31,.22), transparent 24rem),
        radial-gradient(circle at 92% 22%, rgba(245,183,0,.16), transparent 20rem),
        linear-gradient(30deg, rgba(255,210,31,.08) 12%, transparent 12.5%, transparent 87%, rgba(255,210,31,.08) 87.5%, rgba(255,210,31,.08)),
        linear-gradient(150deg, rgba(255,210,31,.08) 12%, transparent 12.5%, transparent 87%, rgba(255,210,31,.08) 87.5%, rgba(255,210,31,.08)),
        linear-gradient(30deg, rgba(255,210,31,.08) 12%, transparent 12.5%, transparent 87%, rgba(255,210,31,.08) 87.5%, rgba(255,210,31,.08)),
        linear-gradient(150deg, rgba(255,210,31,.08) 12%, transparent 12.5%, transparent 87%, rgba(255,210,31,.08) 87.5%, rgba(255,210,31,.08)),
        linear-gradient(60deg, rgba(255,210,31,.045) 25%, transparent 25.5%, transparent 75%, rgba(255,210,31,.045) 75%, rgba(255,210,31,.045)),
        linear-gradient(60deg, rgba(255,210,31,.045) 25%, transparent 25.5%, transparent 75%, rgba(255,210,31,.045) 75%, rgba(255,210,31,.045));
    background-position: 0 0, 0 0, 0 0, 0 0, 42px 72px, 42px 72px, 0 0, 42px 72px;
    background-size: auto, auto, 84px 144px, 84px 144px, 84px 144px, 84px 144px, 84px 144px, 84px 144px;
}
.block-container { max-width: 1460px; padding: 1.4rem 2.8rem 3rem 2.8rem; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #171006, #080500) !important; border-right: 1px solid rgba(255,210,31,.45); }
[data-testid="stSidebar"] * { color: #FFF4B8 !important; }
[data-testid="stSidebar"] button { background: linear-gradient(135deg, #FFD21F, #F5B700) !important; color: #100B00 !important; border: 0 !important; font-weight: 900 !important; border-radius: 14px !important; }
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small { color: #D9CFA6 !important; }
h1, h2, h3 { letter-spacing: -.025em; color: #FFF8DA; }
p, li, span, div { line-height: 1.55; }
.hero { padding: 42px 36px; border-radius: 34px; background: radial-gradient(circle at 12% 18%, rgba(255,255,255,.34), transparent 10rem), linear-gradient(135deg, #FFD21F, #F5B700 54%, #FFB000); color: #100B00; text-align: center; box-shadow: 0 22px 70px rgba(255,210,31,.30); margin-bottom: 26px; border: 1px solid rgba(255,255,255,.32); }
.hero h1 { font-size: 48px; font-weight: 950; margin-bottom: 10px; color:#100B00; }
.hero h3 { font-size: 24px; font-weight: 850; color:#241900; margin-bottom: 8px; }
.hero p { font-size: 16px; color:#3A2A00; }
.card, .phase-card { background: linear-gradient(145deg, rgba(28,20,6,.95), rgba(10,7,0,.92)); padding: 24px; border-radius: 26px; border: 1px solid rgba(255,210,31,.32); box-shadow: 0 18px 48px rgba(0,0,0,.24), inset 0 1px 0 rgba(255,255,255,.07); margin-bottom: 18px; }
.phase-card { background: linear-gradient(145deg, rgba(255,210,31,.19), rgba(14,10,2,.96)); min-height: 190px; }
.big-score { font-size: 42px; font-weight: 950; color: #FFD21F; line-height: 1.1; }
.phase { font-size: 34px; font-weight: 950; color: #FFD21F; line-height: 1.12; }
.small { color: #D9CFA6; font-size: 15px; }
.explain, .warning, .danger, .good { padding: 20px 22px; border-radius: 22px; margin: 18px 0; color: #FFF8DA; background: linear-gradient(135deg, rgba(255,210,31,.12), rgba(24,17,5,.90)); border: 1px solid rgba(255,210,31,.28); border-left: 5px solid #FFD21F; }
.good { border-left-color:#00C853; background: linear-gradient(135deg, rgba(0,200,83,.12), rgba(24,17,5,.90)); }
.warning { border-left-color:#FF9800; }
.danger { border-left-color:#FF3D00; background: linear-gradient(135deg, rgba(255,61,0,.12), rgba(24,17,5,.90)); }
[data-testid="stMetric"] { background: linear-gradient(145deg, rgba(28,20,6,.96), rgba(10,7,0,.92)); border: 1px solid rgba(255,210,31,.26); border-radius: 22px; padding: 18px 20px; box-shadow: 0 14px 36px rgba(0,0,0,.22); }
[data-testid="stMetricLabel"] { color: #D9CFA6 !important; font-weight: 850; }
[data-testid="stMetricValue"] { color: #FFD21F !important; font-weight: 950; }
.stDataFrame, [data-testid="stDataFrame"] { border: 1px solid rgba(255,210,31,.28) !important; border-radius: 22px !important; overflow: hidden !important; background: rgba(255,248,218,.96) !important; }
div[data-testid="stProgress"] > div { background-color: rgba(255,248,218,.18) !important; }
.section-title { font-size: 2rem; font-weight: 950; color: #FFF8DA; margin: 34px 0 6px 0; }
.section-note { color: #D9CFA6; font-size: 1rem; margin-bottom: 16px; }
.prob-card { background: linear-gradient(145deg, rgba(255,210,31,.13), rgba(16,11,2,.94)); border: 1px solid rgba(255,210,31,.30); border-radius: 22px; padding: 18px 20px; margin-bottom: 14px; min-height: 120px; }
.prob-card h4 { color:#FFF8DA; margin:0 0 6px 0; font-size:1.05rem; }
.prob-card .num { color:#FFD21F; font-size:1.7rem; font-weight:950; margin-bottom:8px; }
@media only screen and (max-width: 900px) { .block-container {padding-left: 1rem; padding-right: 1rem;} .hero h1 {font-size: 32px;} .hero h3 {font-size: 18px;} .phase {font-size: 25px;} .big-score {font-size: 30px;} .card {padding: 18px;} }
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


# =====================================================
# HELPER FUNCTIONS
# =====================================================
def clamp(value, low=0, high=100):
    return float(max(low, min(high, value)))


def get_close(df):
    if df is None or df.empty:
        return pd.Series(dtype=float)

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            return pd.to_numeric(df["Close"].iloc[:, 0], errors="coerce").dropna()
        return pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()

    if "Close" in df.columns:
        return pd.to_numeric(df["Close"], errors="coerce").dropna()

    return pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()


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


def plot_layout(fig, height=560):
    fig.update_layout(
        paper_bgcolor="rgba(5,5,5,0)",
        plot_bgcolor="rgba(5,5,5,0.35)",
        font=dict(color="#F8F8F8", size=15),
        height=height,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(17,17,17,.88)",
            bordercolor="rgba(255,210,31,.24)",
            borderwidth=1,
            font=dict(size=13),
        ),
        margin=dict(l=70, r=30, t=82, b=60),
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,.09)",
        zerolinecolor="rgba(255,255,255,.14)",
        tickfont=dict(size=13),
        title_font=dict(size=15),
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,.09)",
        zerolinecolor="rgba(255,255,255,.14)",
        tickfont=dict(size=13),
        title_font=dict(size=15),
    )
    return fig


def render_phase_probability_cards(df):
    st.markdown('<div class="section-title">🧠 Phase Probability Ranking</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Model probabilities ranked from highest to lowest. Higher values mean stronger evidence for that cycle phase.</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    for i, row in df.iterrows():
        phase = str(row["Phase"])
        prob = float(row["Probability"])
        with cols[i % 4]:
            st.markdown(f"""
            <div class="prob-card">
                <h4>{phase}</h4>
                <div class="num">{prob:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(min(max(prob / 100, 0), 1), text=f"{prob:.2f}% probability")

    st.dataframe(df, use_container_width=True, hide_index=True, height=330)


def render_snapshot(details_dict):
    st.markdown('<div class="section-title">🧾 Live Data Snapshot</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Latest market inputs used in the current prediction run.</div>', unsafe_allow_html=True)

    items = list(details_dict.items())
    for row_start in range(0, len(items), 4):
        cols = st.columns(4)
        for col, (label, value) in zip(cols, items[row_start:row_start + 4]):
            with col:
                st.metric(label=label, value=value)

    snapshot_df = pd.DataFrame(items, columns=["Metric", "Value"])
    st.dataframe(snapshot_df, use_container_width=True, hide_index=True, height=330)

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    required_files = [
        "xgb_market_cycle_model.pkl",
        "phase_label_encoder.pkl",
        "model_features.pkl",
    ]

    missing = [file for file in required_files if not os.path.exists(file)]

    if missing:
        return None, None, None, missing

    model = joblib.load("xgb_market_cycle_model.pkl")
    encoder = joblib.load("phase_label_encoder.pkl")
    features = joblib.load("model_features.pkl")

    return model, encoder, features, []


# =====================================================
# LIVE DATA
# =====================================================
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


# =====================================================
# FIXED LIVE SCORE ENGINE
# =====================================================
def calculate_live_scores():
    data = fetch_market_data()

    prices = pd.DataFrame({
        "nifty": get_close(data["nifty"]),
        "bank": get_close(data["bank"]),
        "midcap": get_close(data["midcap"]),
        "vix": get_close(data["vix"]),
        "usd_inr": get_close(data["usd_inr"]),
        "crude": get_close(data["crude"]),
        "gold": get_close(data["gold"]),
        "us10y": get_close(data["us10y"]),
    }).dropna()

    if len(prices) < 500:
        raise RuntimeError("Not enough aligned live market data. Try refreshing after some time.")

    n = prices["nifty"]
    bank = prices["bank"]
    midcap = prices["midcap"]
    vix = prices["vix"]
    usd = prices["usd_inr"]
    crude = prices["crude"]
    gold = prices["gold"]
    us10y = prices["us10y"]

    df = pd.DataFrame(index=prices.index)

    # ---------------- Trend Score ----------------
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

    # ---------------- Breadth Score ----------------
    breadth = pd.Series(0, index=prices.index)

    for col in ["nifty", "bank", "midcap"]:
        s = prices[col]
        breadth += (s > s.rolling(50).mean()).astype(int) * 12
        breadth += (s > s.rolling(200).mean()).astype(int) * 16
        breadth += (s.pct_change(20) > 0).astype(int) * 5

    df["Breadth"] = (breadth / 99 * 100).clip(0, 100)

    # ---------------- Liquidity Score ----------------
    liquidity = pd.Series(50.0, index=prices.index)
    liquidity += pd.Series(np.where(n.pct_change(60) > 0, 15, -10), index=prices.index)
    liquidity += pd.Series(np.where(usd.pct_change(60) < 0.025, 15, -15), index=prices.index)
    liquidity += pd.Series(np.where(us10y.diff(60) < 0, 15, -10), index=prices.index)

    df["Liquidity"] = liquidity.clip(0, 100)

    # ---------------- Valuation Proxy ----------------
    one_year_return = n.pct_change(252) * 100
    distance_high = n / n.rolling(252).max() * 100 - 100

    valuation = pd.Series(50.0, index=prices.index)
    valuation[one_year_return > 35] = 95
    valuation[(one_year_return > 25) & (one_year_return <= 35)] = 85
    valuation[(one_year_return > 15) & (one_year_return <= 25)] = 70
    valuation[(one_year_return > 5) & (one_year_return <= 15)] = 55
    valuation[(one_year_return > -10) & (one_year_return <= 5)] = 35
    valuation[one_year_return <= -10] = 20
    valuation += pd.Series(np.where(distance_high > -3, 8, 0), index=prices.index)
    valuation -= pd.Series(np.where(distance_high < -20, 10, 0), index=prices.index)

    df["Valuation"] = valuation.clip(0, 100)

    # ---------------- Sentiment Score ----------------
    vix_percentile = vix.rolling(252).apply(
        lambda x: (x < x.iloc[-1]).mean() * 100,
        raw=False
    )
    drawdown = n / n.rolling(252).max() * 100 - 100

    sentiment = 100 - vix_percentile
    sentiment += pd.Series(np.where(drawdown > -3, 10, 0), index=prices.index)
    sentiment -= pd.Series(np.where(drawdown < -15, 15, 0), index=prices.index)

    df["Sentiment"] = sentiment.clip(0, 100)

    # ---------------- Macro Score ----------------
    macro = pd.Series(55.0, index=prices.index)
    macro += pd.Series(np.where(crude.pct_change(120) < 0.10, 8, -8), index=prices.index)
    macro += pd.Series(np.where(gold.pct_change(120) < 0.15, 4, -4), index=prices.index)
    macro += pd.Series(np.where(usd.pct_change(120) < 0.03, 8, -8), index=prices.index)
    macro += pd.Series(np.where(us10y.diff(120) < 0, 8, -6), index=prices.index)
    macro += pd.Series(np.where(n.pct_change(120) > 0, 8, -8), index=prices.index)

    df["Macro"] = macro.clip(0, 100)

    df = df.dropna()

    if df.empty:
        raise RuntimeError("Live score calculation failed because final aligned dataframe is empty.")

    latest = df.iloc[-1]

    scores = {
        "Trend": float(latest["Trend"]),
        "Breadth": float(latest["Breadth"]),
        "Liquidity": float(latest["Liquidity"]),
        "Valuation": float(latest["Valuation"]),
        "Sentiment": float(latest["Sentiment"]),
        "Macro": float(latest["Macro"]),
    }

    latest_date = df.index[-1]

    details = {
        "Latest Market Date": latest_date.strftime("%d %b %Y"),
        "Nifty": round(float(n.loc[latest_date]), 2),
        "Trend": round(scores["Trend"], 1),
        "Breadth": round(scores["Breadth"], 1),
        "Liquidity": round(scores["Liquidity"], 1),
        "Valuation": round(scores["Valuation"], 1),
        "Sentiment": round(scores["Sentiment"], 1),
        "Macro": round(scores["Macro"], 1),
    }

    return scores, details, df, n


# =====================================================
# PREDICTION
# =====================================================
def predict_phase(model, encoder, scores):
    X_live = pd.DataFrame([scores], columns=FEATURES)
    probabilities = model.predict_proba(X_live)[0]
    classes = encoder.inverse_transform(np.arange(len(probabilities)))

    result = pd.DataFrame({
        "Phase": classes,
        "Probability": probabilities * 100,
    }).sort_values("Probability", ascending=False)

    return result


 
# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="hero">
    <h1>🐝 Madness of Money Bees</h1>
    <h3>20-Year Trained XGBoost Market Cycle Predictor</h3>
    <p>Automatic live prediction using historical market-trained regime intelligence.</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🐝 Control Room")
st.sidebar.caption("Automatic live market-cycle prediction.")

if st.sidebar.button("Refresh Live Data"):
    st.cache_data.clear()


# =====================================================
# LOAD MODEL
# =====================================================
model, encoder, model_features, missing = load_model()

if missing:
    st.error("Model files missing. Run `python train.py` first and commit the generated `.pkl` files.")
    st.write("Missing files:", missing)
    st.stop()


# =====================================================
# RUN APP ENGINE
# =====================================================
try:
    scores, details, history_df, nifty = calculate_live_scores()
except Exception as e:
    st.error("Could not fetch live market data.")
    st.exception(e)
    st.stop()

prediction = predict_phase(model, encoder, scores)

try:
    record_tool_use("hive-cycle-predictor")
except Exception:
    pass

winner = prediction.iloc[0]
runner = prediction.iloc[1]

current_phase = winner["Phase"]
confidence = winner["Probability"]

# =====================================================
# PHASE DURATION INTELLIGENCE - SIMPLE SAFE VERSION
# =====================================================

next_phase_map = {
    "Accumulation": "Early Bull",
    "Early Bull": "Mature Bull",
    "Mature Bull": "Late Bull",
    "Late Bull": "Distribution",
    "Distribution": "Early Bear",
    "Early Bear": "Mature Bear",
    "Mature Bear": "Late Bear",
    "Late Bear": "Accumulation",
}

next_probable_phase = next_phase_map.get(current_phase, "Unknown")

try:
    phase_history_rows = []

    for date, row in history_df[["Trend", "Breadth", "Liquidity", "Valuation", "Sentiment", "Macro"]].dropna().iterrows():
        hist_scores = row.to_dict()
        hist_pred = predict_phase(model, encoder, hist_scores)
        hist_phase = hist_pred.iloc[0]["Phase"]

        phase_history_rows.append({
            "Date": date,
            "Phase": hist_phase
        })

    phase_history = pd.DataFrame(phase_history_rows)

    same_phase = phase_history["Phase"] == current_phase
    last_break = same_phase[::-1].idxmin() if (~same_phase).any() else phase_history.index[0]

    if (~same_phase).any():
        phase_start_date = phase_history.loc[last_break + 1, "Date"]
    else:
        phase_start_date = phase_history["Date"].iloc[0]

    current_phase_days = (phase_history["Date"].iloc[-1] - phase_start_date).days
    current_phase_months = round(current_phase_days / 30.44, 1)

    blocks = []
    start_date = phase_history["Date"].iloc[0]
    start_phase = phase_history["Phase"].iloc[0]

    for i in range(1, len(phase_history)):
        if phase_history["Phase"].iloc[i] != start_phase:
            end_date = phase_history["Date"].iloc[i - 1]
            months = (end_date - start_date).days / 30.44

            if months > 0.25:
                blocks.append({
                    "Phase": start_phase,
                    "Months": months
                })

            start_date = phase_history["Date"].iloc[i]
            start_phase = phase_history["Phase"].iloc[i]

    blocks_df = pd.DataFrame(blocks)

    if not blocks_df.empty:
        avg_duration_map = blocks_df.groupby("Phase")["Months"].mean().to_dict()
    else:
        avg_duration_map = {}

    avg_current_duration = avg_duration_map.get(current_phase, 6.0)
    avg_next_duration = avg_duration_map.get(next_probable_phase, 6.0)
    remaining_current_duration = max(avg_current_duration - current_phase_months, 0)

except Exception:
    phase_start_date = datetime.now()
    current_phase_months = 0.0
    remaining_current_duration = 0.0
    avg_next_duration = 6.0   



# =====================================================
# TOP CARDS
# =====================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="phase-card">
        <p class="small">Current Market Phase</p>
        <div class="phase">{current_phase}</div>
        <p class="small">{PHASE_TEXT.get(current_phase, "")}</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card">
        <p class="small">XGBoost Confidence</p>
        <div class="big-score">{confidence:.1f}%</div>
        <p class="small">Based on trained market-regime features.</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card">
        <p class="small">Runner-Up Phase</p>
        <div class="big-score">{runner["Phase"]}</div>
        <p class="small">{runner["Probability"]:.1f}% probability.</p>
    </div>
    """, unsafe_allow_html=True)


st.markdown(f"""
<div class="good">
<b>Live market data loaded successfully.</b><br>
Latest market date: {details["Latest Market Date"]}<br>
Last app refresh: {datetime.now().strftime("%d %b %Y, %I:%M %p")}
</div>
""", unsafe_allow_html=True)
# =====================================================
# PHASE DURATION UI
# =====================================================
st.markdown("## ⏳ Phase Duration Intelligence")

d1, d2, d3 = st.columns(3)

with d1:
    st.markdown(f"""
    <div class="card">
        <p class="small">Current Phase Duration</p>
        <div class="big-score">{current_phase_months:.1f} months</div>
        <p class="small">Started around {phase_start_date.strftime("%d %b %Y")}</p>
    </div>
    """, unsafe_allow_html=True)

with d2:
    st.markdown(f"""
    <div class="card">
        <p class="small">Estimated Remaining Duration</p>
        <div class="big-score">{remaining_current_duration:.1f} months</div>
        <p class="small">Based on historical average duration of {current_phase}</p>
    </div>
    """, unsafe_allow_html=True)

with d3:
    st.markdown(f"""
    <div class="card">
        <p class="small">Next Probable Phase</p>
        <div class="phase">{next_probable_phase}</div>
        <p class="small">Expected duration: {avg_next_duration:.1f} months</p>
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
    "Late Bear": 96,
}

x = np.linspace(0, 100, 700)
y = np.sin(x / 7.5) * 10 + x * 0.23

cx = phase_positions.get(current_phase, 50)
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
        font=dict(color="white", size=13),
        bgcolor="rgba(17,17,17,.92)",
        bordercolor="#FFD21F",
        borderwidth=1,
    )

fig.update_layout(
    paper_bgcolor="#050505",
    plot_bgcolor="#050505",
    font=dict(color="white", size=15),
    height=620,
    margin=dict(l=70, r=30, t=60, b=60),
    xaxis=dict(
        showticklabels=False,
        title="Market Cycle Journey",
        gridcolor="rgba(255,255,255,.08)",
    ),
    yaxis=dict(
        title="Cycle Momentum",
        gridcolor="rgba(255,255,255,.08)",
    ),
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})


st.markdown("""
<div class="explain">
<h3>How to read this chart</h3>
<p>
The yellow wave represents the market cycle. Bull phases usually sit on the rising part of the curve.
Distribution appears near the top, where the index may still look strong but internal strength starts weakening.
Bear phases appear on the declining part. Accumulation appears near the bottom, where sentiment is weak
but long-term opportunity may begin forming.
</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# LIVE SCORES
# =====================================================
st.markdown("## 📊 Live Indicator Scores")

cols = st.columns(3)

for i, feature in enumerate(FEATURES):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="card">
            <h3>{feature}</h3>
            <div class="big-score" style="color:{score_color(scores[feature])};">{scores[feature]:.1f}/100</div>
        </div>
        """, unsafe_allow_html=True)


# =====================================================
# PHASE PROBABILITIES - PREMIUM READABLE TABLE
# =====================================================
rank_df = prediction.copy()
rank_df["Probability"] = rank_df["Probability"].round(2)

render_phase_probability_cards(rank_df)


# =====================================================
# FEATURE IMPORTANCE
# =====================================================
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

fig_imp.update_traces(
    text=importance["Importance"].round(1).astype(str) + "%",
    textposition="outside",
    cliponaxis=False,
)

fig_imp.update_layout(
    title=dict(text="Feature importance used by the model", font=dict(size=21)),
    xaxis=dict(title="Importance %", gridcolor="rgba(255,255,255,.08)"),
    yaxis=dict(title="", categoryorder="array", categoryarray=importance["Feature"].tolist()),
    showlegend=False,
)
plot_layout(fig_imp, height=480)

st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False, "responsive": True})


# =====================================================
# NIFTY CHART
# =====================================================
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
    title=dict(text="Nifty 50 long-term price trend", font=dict(size=21)),
    xaxis=dict(gridcolor="rgba(255,255,255,.08)"),
    yaxis=dict(title="Price", gridcolor="rgba(255,255,255,.08)"),
    showlegend=False,
)
plot_layout(fig_price, height=520)

st.plotly_chart(fig_price, use_container_width=True, config={"displayModeBar": False, "responsive": True})


# =====================================================
# HISTORICAL SCORE TREND - DESKTOP READABLE
# =====================================================
st.markdown('<div class="section-title">📉 Indicator History</div>', unsafe_allow_html=True)
st.markdown('<div class="section-note">Last 3 years shown by default. Use the legend to focus on one indicator and the range buttons to zoom.</div>', unsafe_allow_html=True)

history_display = history_df[FEATURES].tail(756).copy()
fig_hist = go.Figure()

line_colors = {
    "Trend": "#FFD21F",
    "Breadth": "#00C853",
    "Liquidity": "#29B6F6",
    "Valuation": "#FF9800",
    "Sentiment": "#E040FB",
    "Macro": "#FF5252",
}

for feature in FEATURES:
    fig_hist.add_trace(go.Scatter(
        x=history_display.index,
        y=history_display[feature],
        mode="lines",
        name=feature,
        line=dict(width=3.2, color=line_colors.get(feature)),
        hovertemplate=f"<b>{feature}</b><br>%{{x|%d %b %Y}}<br>Score: %{{y:.1f}}/100<extra></extra>",
    ))

fig_hist.update_layout(
    title=dict(text="Readable score history", font=dict(size=22)),
    xaxis=dict(
        title="Date",
        gridcolor="rgba(255,255,255,.08)",
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=3, label="3Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ]),
            bgcolor="rgba(255,210,31,.12)",
            activecolor="#FFD21F",
            font=dict(color="#FFFFFF", size=12),
        ),
    ),
    yaxis=dict(title="Score / 100", gridcolor="rgba(255,255,255,.08)", range=[0, 100]),
)
plot_layout(fig_hist, height=660)

st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": True, "responsive": True})


# =====================================================
# SNAPSHOT TABLE - READABLE SUMMARY
# =====================================================
render_snapshot(details)


# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="warning">
<h3>Important Note</h3>
<p>
This model uses historical market-derived features and an XGBoost classifier.
The current version uses automatic proxies for valuation, liquidity, sentiment, breadth, and macro data.
For an institutional-grade version, connect official Nifty PE/PB, FII/DII flows, PMI, CPI, GDP,
and full Nifty 500 advance-decline breadth data.
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
