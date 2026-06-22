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
from urllib.parse import quote
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
access_info = require_tool_access(
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
[data-testid="stSidebar"] button { background: linear-gradient(135deg, #FFD21F, #F5B700) !important; color: #100B00 !important; border: 0 !important; font-weight: 950 !important; border-radius: 16px !important; min-height: 46px !important; box-shadow: 0 12px 28px rgba(255,210,31,.24) !important; }
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small { color: #D9CFA6 !important; }
.stButton > button, .stLinkButton > a {
    background: linear-gradient(135deg, #FFD21F, #F5B700) !important;
    color: #120D00 !important;
    border: 1px solid rgba(255,255,255,.35) !important;
    border-radius: 18px !important;
    font-weight: 950 !important;
    min-height: 48px !important;
    box-shadow: 0 14px 34px rgba(255,210,31,.22), inset 0 1px 0 rgba(255,255,255,.45) !important;
}
.stButton > button:hover, .stLinkButton > a:hover { transform: translateY(-1px); box-shadow: 0 18px 42px rgba(255,210,31,.30) !important; }
.stButton > button p, .stLinkButton > a p { color:#120D00 !important; font-weight:950 !important; }
div[data-testid="column"] .stButton button, div[data-testid="column"] .stLinkButton a { width: 100%; background: linear-gradient(135deg, #FFD21F 0%, #F5B700 58%, #FFB000 100%) !important; color: #0F0A00 !important; border-radius: 18px !important; border: 1px solid rgba(255,255,255,.32) !important; }

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
.snapshot-card { background: linear-gradient(145deg, rgba(255,210,31,.14), rgba(14,10,2,.96)); border: 1px solid rgba(255,210,31,.32); border-radius: 22px; padding: 18px 20px; min-height: 112px; margin-bottom: 16px; box-shadow: 0 14px 34px rgba(0,0,0,.22); }
.snapshot-label { color:#D9CFA6; font-size:.92rem; font-weight:850; margin-bottom:10px; }
.snapshot-value { color:#FFD21F; font-size:1.65rem; font-weight:950; line-height:1.15; }


.hook-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;margin:20px 0 8px 0;}
.hook-card {background:linear-gradient(145deg,rgba(255,210,31,.16),rgba(11,8,1,.96));border:1px solid rgba(255,210,31,.34);border-radius:26px;padding:22px 24px;box-shadow:0 18px 46px rgba(0,0,0,.24);min-height:170px;}
.hook-label {color:#D9CFA6;font-size:.9rem;font-weight:900;text-transform:uppercase;letter-spacing:.04em;margin-bottom:10px;}
.hook-value {color:#FFD21F;font-size:2.1rem;font-weight:950;line-height:1.08;margin-bottom:8px;}
.hook-sub {color:#FFF8DA;font-size:1rem;line-height:1.55;margin:0;}
.action-card {background:linear-gradient(145deg,rgba(255,248,218,.08),rgba(14,10,2,.96));border:1px solid rgba(255,210,31,.30);border-radius:26px;padding:22px 24px;min-height:235px;margin-bottom:16px;box-shadow:0 16px 42px rgba(0,0,0,.22);}
.action-card h3 {color:#FFD21F;margin:0 0 12px 0;font-size:1.2rem;}
.action-card ul {margin:0;padding-left:1.1rem;color:#FFF8DA;}
.action-card li {margin:.45rem 0;}
.locked-card {position:relative;background:linear-gradient(145deg,rgba(255,210,31,.12),rgba(12,8,0,.96));border:1px solid rgba(255,210,31,.34);border-radius:26px;padding:24px;min-height:190px;overflow:hidden;box-shadow:0 16px 42px rgba(0,0,0,.24);}
.locked-card:after {content:"PRO";position:absolute;top:16px;right:18px;background:linear-gradient(135deg,#FFD21F,#F5B700);color:#100B00;font-weight:950;border-radius:999px;padding:7px 11px;font-size:.75rem;}
.locked-card h3 {color:#FFD21F;margin-top:0;}
.bee-summary {background:radial-gradient(circle at 92% 12%,rgba(255,210,31,.22),transparent 13rem),linear-gradient(145deg,rgba(255,210,31,.18),rgba(11,8,1,.98));border:1px solid rgba(255,210,31,.42);border-radius:30px;padding:28px 30px;margin:24px 0;box-shadow:0 22px 58px rgba(0,0,0,.28);}
.bee-summary h2 {margin-top:0;color:#FFD21F;}
.badge-pill {display:inline-block;padding:8px 12px;border-radius:999px;background:rgba(255,210,31,.16);border:1px solid rgba(255,210,31,.28);color:#FFF8DA;font-weight:850;margin:5px 6px 5px 0;}

.financify-score-card, .buffett-card, .share-card {
    background: radial-gradient(circle at 90% 12%, rgba(255,210,31,.22), transparent 14rem), linear-gradient(145deg, rgba(255,210,31,.18), rgba(10,7,0,.96));
    border: 1px solid rgba(255,210,31,.40);
    border-radius: 30px;
    padding: 28px 30px;
    margin: 24px 0;
    box-shadow: 0 22px 58px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.07);
}
.financify-score-card h2, .buffett-card h2, .share-card h2 { color:#FFD21F; margin-top:0; }
.mega-score { font-size: 58px; font-weight: 950; color:#FFD21F; line-height:1.05; margin:10px 0; }
.share-btn {
    display:inline-block;
    margin: 10px 10px 0 0;
    padding: 13px 18px;
    border-radius:999px;
    background:linear-gradient(135deg,#FFD21F,#F5B700);
    color:#100B00 !important;
    font-weight:950;
    text-decoration:none !important;
    box-shadow:0 12px 28px rgba(255,210,31,.24);
}
.share-btn:hover { transform: translateY(-1px); box-shadow:0 16px 34px rgba(255,210,31,.30); }

@media only screen and (max-width: 900px) { .hook-grid {grid-template-columns:1fr;} }


.financify-score-card, .buffett-card, .share-card {
    background: radial-gradient(circle at 90% 12%, rgba(255,210,31,.22), transparent 14rem), linear-gradient(145deg, rgba(255,210,31,.18), rgba(10,7,0,.96));
    border: 1px solid rgba(255,210,31,.40);
    border-radius: 30px;
    padding: 28px 30px;
    margin: 24px 0;
    box-shadow: 0 22px 58px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.07);
}
.financify-score-card h2, .buffett-card h2, .share-card h2 { color:#FFD21F; margin-top:0; }
.mega-score { font-size: 58px; font-weight: 950; color:#FFD21F; line-height:1.05; margin:10px 0; }
.share-btn {
    display:inline-block;
    margin: 10px 10px 0 0;
    padding: 13px 18px;
    border-radius:999px;
    background:linear-gradient(135deg,#FFD21F,#F5B700);
    color:#100B00 !important;
    font-weight:950;
    text-decoration:none !important;
    box-shadow:0 12px 28px rgba(255,210,31,.24);
}
.share-btn:hover { transform: translateY(-1px); box-shadow:0 16px 34px rgba(255,210,31,.30); }

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
        paper_bgcolor="rgba(12,8,0,0.96)",
        plot_bgcolor="rgba(12,8,0,0.92)",
        font=dict(color="#FFF8DA", size=18, family="Arial, sans-serif"),
        height=height,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
            bgcolor="rgba(20,14,2,.96)",
            bordercolor="rgba(255,210,31,.45)",
            borderwidth=1,
            font=dict(size=15, color="#FFF8DA"),
        ),
        margin=dict(l=92, r=44, t=92, b=74),
        title_font=dict(size=25, color="#FFF8DA"),
    )
    fig.update_xaxes(
        gridcolor="rgba(255,210,31,.12)",
        zerolinecolor="rgba(255,210,31,.22)",
        tickfont=dict(size=15, color="#FFF8DA"),
        title_font=dict(size=17, color="#FFF8DA"),
        linecolor="rgba(255,210,31,.35)",
    )
    fig.update_yaxes(
        gridcolor="rgba(255,210,31,.12)",
        zerolinecolor="rgba(255,210,31,.22)",
        tickfont=dict(size=15, color="#FFF8DA"),
        title_font=dict(size=17, color="#FFF8DA"),
        linecolor="rgba(255,210,31,.35)",
    )
    return fig


def render_phase_probability_cards(df):
    st.markdown('<div class="section-title">🧠 Phase Probability Ranking</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Model probabilities ranked from highest to lowest. The cards below are the only probability view for cleaner reading.</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    for i, row in df.reset_index(drop=True).iterrows():
        phase = str(row["Phase"])
        prob = float(row["Probability"])
        rank = i + 1
        badge = "Top phase" if rank == 1 else f"Rank #{rank}"
        with cols[i % 4]:
            html = f'''<div class="prob-card">
                <div style="font-size:.78rem;font-weight:950;color:#110B00;background:linear-gradient(135deg,#FFD21F,#F5B700);display:inline-block;padding:6px 10px;border-radius:999px;margin-bottom:12px;">{badge}</div>
                <h4>{phase}</h4>
                <div class="num">{prob:.2f}%</div>
                <p class="small" style="margin:0;">Model confidence for this market-cycle phase.</p>
            </div>'''
            st.markdown(html, unsafe_allow_html=True)


def render_snapshot(details_dict):
    st.markdown('<div class="section-title">🧾 Live Market Snapshot</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Latest market inputs used in the current prediction run. Shown as cards only for a cleaner premium view.</div>', unsafe_allow_html=True)

    items = list(details_dict.items())
    for row_start in range(0, len(items), 4):
        cols = st.columns(4)
        for col, (label, value) in zip(cols, items[row_start:row_start + 4]):
            with col:
                st.markdown(f'''<div class="snapshot-card">
                    <div class="snapshot-label">{label}</div>
                    <div class="snapshot-value">{value}</div>
                </div>''', unsafe_allow_html=True)



# =====================================================
# PREMIUM DECISION LAYER HELPERS
# =====================================================
def market_weather(current_phase, confidence, scores):
    trend = scores.get("Trend", 50)
    breadth = scores.get("Breadth", 50)
    valuation = scores.get("Valuation", 50)
    sentiment = scores.get("Sentiment", 50)
    liquidity = scores.get("Liquidity", 50)
    macro = scores.get("Macro", 50)
    sunshine = clamp((trend * .28) + (breadth * .20) + (liquidity * .18) + (macro * .16) + ((100 - valuation) * .08) + (confidence * .10))
    correction_risk_score = clamp((valuation * .28) + (sentiment * .24) + ((100 - breadth) * .20) + ((100 - liquidity) * .16) + ((100 - macro) * .12))
    euphoria = clamp((sentiment * .40) + (valuation * .34) + (trend * .16) + (breadth * .10))
    risk = "High" if correction_risk_score >= 68 else "Medium" if correction_risk_score >= 44 else "Low"
    eup = "High" if euphoria >= 72 else "Moderate" if euphoria >= 52 else "Low"
    icon = "☀️" if sunshine >= 68 else "⛅" if sunshine >= 45 else "🌧️"
    return {"sunshine": sunshine, "correction_risk": risk, "correction_score": correction_risk_score, "euphoria": eup, "euphoria_score": euphoria, "icon": icon}


def phase_duration_label(current_phase):
    labels = {
        "Accumulation": "3-9 months usually, but can be slow and frustrating.",
        "Early Bull": "4-10 months when liquidity and macro support improve.",
        "Mature Bull": "6-18 months when earnings and participation stay healthy.",
        "Late Bull": "2-8 months; can be powerful but emotionally dangerous.",
        "Distribution": "1-6 months; index strength often hides internal weakness.",
        "Early Bear": "2-6 months; false rebounds are common.",
        "Mature Bear": "3-9 months; capital protection matters most.",
        "Late Bear": "1-5 months; panic can create long-term opportunities.",
    }
    return labels.get(current_phase, "Cycle duration varies with liquidity, sentiment, and earnings.")


def smart_money_actions(current_phase):
    actions = {
        "Accumulation": ["Build a watchlist of quality leaders", "Prefer staggered buying over emotional lump-sum bets", "Ignore panic headlines and focus on balance-sheet strength"],
        "Early Bull": ["Increase exposure gradually through SIPs or phased entries", "Prefer quality cyclicals and earnings recovery names", "Avoid waiting for perfect comfort because early bull markets rarely feel safe"],
        "Mature Bull": ["Stay invested in quality but start checking valuation comfort", "Trim weak/speculative positions that ran only due to momentum", "Use dips to add to durable compounders, not random stories"],
        "Late Bull": ["Avoid chasing hot tips, IPO mania, and leverage", "Book partial profits in stretched positions", "Keep a cash buffer for future corrections"],
        "Distribution": ["Reduce concentration risk", "Prefer strong cash-flow businesses over fragile momentum names", "Do not confuse index strength with broad market strength"],
        "Early Bear": ["Protect capital before hunting bargains", "Stop averaging down in weak fundamentals", "Keep SIPs running only if horizon and emergency fund are clear"],
        "Mature Bear": ["Stay defensive and avoid leverage", "Look for debt-free leaders becoming reasonably valued", "Build cash and patience; opportunity comes after forced selling"],
        "Late Bear": ["Prepare a high-quality buy list", "Start phased accumulation if fundamentals are intact", "Expect volatility even after the best opportunities appear"],
    }
    return actions.get(current_phase, ["Focus on quality", "Avoid emotional decisions", "Use valuation and risk controls"])


def investor_action_cards(current_phase, weather):
    risk = weather["correction_risk"]
    if current_phase in ["Late Bull", "Distribution", "Early Bear"] or risk == "High":
        return {"Conservative Investor": ["Protect capital first", "Avoid fresh lump-sum buying in expensive stocks", "Keep cash for better risk-reward opportunities"], "Moderate Investor": ["Continue SIPs but avoid over-allocation", "Trim weak holdings and rebalance", "Add only to quality stocks after valuation checks"], "Aggressive Investor": ["Trade smaller position sizes", "Use strict stop-loss or exit rules", "Avoid leverage and crowded momentum bets"]}
    if current_phase in ["Accumulation", "Late Bear"]:
        return {"Conservative Investor": ["Start with index funds or high-quality leaders", "Use small staggered entries", "Keep emergency cash untouched"], "Moderate Investor": ["Build positions in quality compounders", "Use 3-6 month phased buying", "Prefer clean balance sheets and durable ROCE"], "Aggressive Investor": ["Look for beaten-down leaders with intact fundamentals", "Do not buy junk only because it is down", "Scale in gradually instead of all at once"]}
    return {"Conservative Investor": ["Stay invested through diversified funds", "Avoid panic exits on normal corrections", "Review allocation once a month"], "Moderate Investor": ["Add selectively on dips", "Prefer earnings growth plus reasonable valuation", "Avoid portfolio over-concentration"], "Aggressive Investor": ["Use momentum only with risk control", "Book profits from overheated names", "Rotate from weak stories to strong fundamentals"]}


def professor_bee_summary(current_phase, confidence, weather, scores):
    strongest = max(scores, key=scores.get)
    weakest = min(scores, key=scores.get)
    return [f"The model currently reads the market as {current_phase} with {confidence:.1f}% confidence.", f"The strongest signal is {strongest}, while the weakest pressure point is {weakest}.", f"Correction risk is {weather['correction_risk']}; act with discipline, not FOMO."]


def render_market_weather(weather, current_phase):
    st.markdown('<div class="section-title">🌦️ Market Weather Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">A plain-English decision layer built on top of the cycle prediction.</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="hook-grid">
      <div class="hook-card"><div class="hook-label">Sunshine Probability</div><div class="hook-value">{weather['icon']} {weather['sunshine']:.0f}%</div><p class="hook-sub">How supportive the current regime looks for staying invested.</p></div>
      <div class="hook-card"><div class="hook-label">Correction Risk</div><div class="hook-value">⚠️ {weather['correction_risk']}</div><p class="hook-sub">Risk score: {weather['correction_score']:.0f}/100 based on valuation, sentiment, breadth, liquidity and macro.</p></div>
      <div class="hook-card"><div class="hook-label">Euphoria Alert</div><div class="hook-value">🧠 {weather['euphoria']}</div><p class="hook-sub">Investor psychology pressure for the current {current_phase} setup.</p></div>
    </div>
    """, unsafe_allow_html=True)


def render_action_cards(action_map):
    st.markdown('<div class="section-title">🎯 What Should You Do Today?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Not a buy/sell call. This converts the market phase into practical behaviour rules.</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (title, points) in zip(cols, action_map.items()):
        with col:
            lis = "".join([f"<li>{p}</li>" for p in points])
            st.markdown(f'<div class="action-card"><h3>{title}</h3><ul>{lis}</ul></div>', unsafe_allow_html=True)


def render_smart_money(current_phase):
    st.markdown('<div class="section-title">🧠 What Smart Money May Be Doing</div>', unsafe_allow_html=True)
    points = smart_money_actions(current_phase)
    pills = "".join([f'<span class="badge-pill">✓ {p}</span>' for p in points])
    st.markdown(f'<div class="card"><h3 style="color:#FFD21F;margin-top:0;">Current behaviour map for {current_phase}</h3>{pills}</div>', unsafe_allow_html=True)


def render_premium_locks(plan):
    plan = str(plan or "free").lower()
    note = "You are viewing the full Pro decision layer." if plan == "pro" else "Free view shows the core phase. Pro unlocks duration intelligence, smart-money actions, investor-specific action cards, and shareable reports."
    st.markdown('<div class="section-title">🚀 Financify Pro Edge</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-note">{note}</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    locked = [("Historical Similarity", "Compare today with earlier cycle setups like overheated bull runs, early recoveries, or bear-market exhaustion."), ("Sector Opportunity Map", "Add sector strength/weakness to see where the cycle is flowing instead of only reading index direction."), ("Downloadable Bee Report", "Turn the live result into a shareable weekly report for Instagram, WhatsApp, or your blog audience.")]
    for col, (title, text) in zip(cols, locked):
        with col:
            st.markdown(f'<div class="locked-card"><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)



def calculate_financify_score(scores):
    return round(
        (scores["Trend"] * 0.22) +
        (scores["Breadth"] * 0.18) +
        (scores["Liquidity"] * 0.16) +
        ((100 - scores["Valuation"]) * 0.14) +
        (scores["Sentiment"] * 0.15) +
        (scores["Macro"] * 0.15),
        1,
    )


def financify_score_text(financify_score):
    if financify_score >= 75:
        return "Strong market setup, but still avoid blind excitement. Quality and valuation discipline matter most."
    if financify_score >= 55:
        return "Balanced setup. Selective investing is better than aggressive chasing."
    if financify_score >= 35:
        return "Cautious zone. Protect capital, avoid weak stocks, and prefer staggered decisions."
    return "Risk-heavy zone. Cash, patience, and quality matter more than speed."


def render_financify_score(financify_score):
    st.markdown(f"""
    <div class="financify-score-card">
        <h2>🐝 Financify Market Score</h2>
        <div class="mega-score">{financify_score}/100</div>
        <p>{financify_score_text(financify_score)}</p>
        <p class="small">This combines trend, breadth, liquidity, valuation pressure, sentiment and macro into one simple market-readiness score.</p>
    </div>
    """, unsafe_allow_html=True)


def render_buffett_box(current_phase):
    buffett_map = {
        "Accumulation": "Buffett would likely look for wonderful businesses available at sensible prices while the crowd is fearful.",
        "Early Bull": "Buffett would still focus on business quality, not market excitement. He would avoid buying just because prices are rising.",
        "Mature Bull": "Buffett would stay invested in strong businesses but become stricter about valuation.",
        "Late Bull": "Buffett would likely avoid hype, keep cash ready, and refuse to overpay for average businesses.",
        "Distribution": "Buffett would become highly selective and avoid businesses whose prices are running ahead of fundamentals.",
        "Early Bear": "Buffett would not panic. He would wait for quality businesses to become cheaper.",
        "Mature Bear": "Buffett would likely prepare his shopping list and slowly accumulate durable businesses.",
        "Late Bear": "Buffett would become interested, because fear often creates long-term opportunity.",
    }
    st.markdown(f"""
    <div class="buffett-card">
        <h2>🧓 What Would Buffett Do?</h2>
        <p>{buffett_map.get(current_phase, "Buffett would focus on quality, valuation, and patience.")}</p>
        <p class="small">This is an educational behavioural lens, not a direct quote or investment recommendation.</p>
    </div>
    """, unsafe_allow_html=True)


def render_share_buttons(current_phase, confidence, financify_score, next_phase, weather):
    share_text = f"""🐝 Financify Market Cycle Update

Current Phase: {current_phase}
Confidence: {confidence:.1f}%
Financify Market Score: {financify_score}/100
Market Sunshine: {weather['sunshine']:.0f}%
Correction Risk: {weather['correction_risk']}
Next Probable Phase: {next_phase}

Professor Bee says: Markets reward patience, not panic.

Try it on Financify Notes:
https://financify.blog/financifytools"""
    encoded_share = quote(share_text)
    st.markdown(f"""
    <div class="share-card">
        <h2>📤 Share This Market View</h2>
        <p>Let your audience see today’s market phase in simple Money Bees language.</p>
        <a class="share-btn" target="_blank" href="https://wa.me/?text={encoded_share}">WhatsApp</a>
        <a class="share-btn" target="_blank" href="https://twitter.com/intent/tweet?text={encoded_share}">X / Twitter</a>
        <a class="share-btn" target="_blank" href="https://www.linkedin.com/sharing/share-offsite/?url=https%3A%2F%2Ffinancify.blog%2Ffinancifytools">LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)
    st.text_area("Ready-to-copy caption", value=share_text, height=210)


def render_professor_bee(summary_lines):
    lis = "".join([f"<li>{line}</li>" for line in summary_lines])
    st.markdown(f'<div class="bee-summary"><h2>🐝 Professor Bee Says</h2><ul>{lis}</ul></div>', unsafe_allow_html=True)


def render_share_box(current_phase, confidence, weather, next_phase):
    share_text = f"🐝 Financify Hive Cycle Predictor update:\nCurrent phase: {current_phase}\nModel confidence: {confidence:.1f}%\nMarket sunshine: {weather['sunshine']:.0f}%\nCorrection risk: {weather['correction_risk']}\nNext probable phase: {next_phase}\n\nFollow Financify Notes for practical finance."
    st.markdown('<div class="section-title">📲 Shareable Result Caption</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Copy this into Instagram, WhatsApp, LinkedIn or your blog update.</div>', unsafe_allow_html=True)
    st.text_area("Ready-to-copy caption", value=share_text, height=190)


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
current_plan = str(access_info.get("plan", "free") if isinstance(access_info, dict) else "free").lower()
weather = market_weather(current_phase, confidence, scores)
action_map = investor_action_cards(current_phase, weather)
financify_score = calculate_financify_score(scores)
professor_lines = professor_bee_summary(current_phase, confidence, weather, scores)

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

render_market_weather(weather, current_phase)
render_smart_money(current_phase)
render_action_cards(action_map)
render_financify_score(financify_score)
render_buffett_box(current_phase)
render_professor_bee(professor_lines)

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

st.markdown(f"""
<div class="explain">
<h3>⏱️ Plain-English duration note</h3>
<p>{phase_duration_label(current_phase)}</p>
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
st.markdown('<div class="section-title">🔍 XGBoost Feature Importance</div>', unsafe_allow_html=True)
st.markdown('<div class="section-note">Shows which inputs influenced the trained model most. Larger bars have more weight in the prediction.</div>', unsafe_allow_html=True)

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
    textfont=dict(size=16, color="#FFF8DA"),
    cliponaxis=False,
)

fig_imp.update_layout(
    title=dict(text="Feature importance used by the model", font=dict(size=25, color="#FFF8DA")),
    xaxis=dict(title="Importance %", range=[0, max(25, float(importance["Importance"].max()) + 4)]),
    yaxis=dict(title="", categoryorder="array", categoryarray=importance["Feature"].tolist(), tickfont=dict(size=17, color="#FFF8DA")),
    showlegend=False,
)
plot_layout(fig_imp, height=480)

st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False, "responsive": True})


# =====================================================
# NIFTY CHART
# =====================================================
st.markdown('<div class="section-title">📈 Nifty 50 Chart</div>', unsafe_allow_html=True)
st.markdown('<div class="section-note">Long-term Nifty 50 trend used as the main market reference.</div>', unsafe_allow_html=True)

fig_price = go.Figure()

fig_price.add_trace(go.Scatter(
    x=nifty.index,
    y=nifty.values,
    mode="lines",
    line=dict(color="#FFD21F", width=3),
    name="Nifty 50",
))

fig_price.update_layout(
    title=dict(text="Nifty 50 long-term price trend", font=dict(size=25, color="#FFF8DA")),
    xaxis=dict(title="Date"),
    yaxis=dict(title="Price"),
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
    title=dict(text="Readable score history", font=dict(size=25, color="#FFF8DA")),
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
            font=dict(color="#100B00", size=13),
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

render_premium_locks(current_plan)
render_share_buttons(current_phase, confidence, financify_score, next_probable_phase, weather)


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
