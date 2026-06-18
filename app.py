import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Madness of Money Bees",
    page_icon="🐝",
    layout="wide"
)


# =========================================================
# CSS THEME
# =========================================================
st.markdown("""
<style>
.stApp {
    background-color: #050505;
    background-image:
        linear-gradient(30deg, rgba(255,210,31,0.08) 12%, transparent 12.5%, transparent 87%, rgba(255,210,31,0.08) 87.5%, rgba(255,210,31,0.08)),
        linear-gradient(150deg, rgba(255,210,31,0.08) 12%, transparent 12.5%, transparent 87%, rgba(255,210,31,0.08) 87.5%, rgba(255,210,31,0.08)),
        linear-gradient(30deg, rgba(255,210,31,0.08) 12%, transparent 12.5%, transparent 87%, rgba(255,210,31,0.08) 87.5%, rgba(255,210,31,0.08)),
        linear-gradient(150deg, rgba(255,210,31,0.08) 12%, transparent 12.5%, transparent 87%, rgba(255,210,31,0.08) 87.5%, rgba(255,210,31,0.08));
    background-size: 70px 120px;
    color: #F8F8F8;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111111, #050505);
    border-right: 1px solid rgba(255,210,31,0.35);
}

.hero {
    padding: 36px;
    border-radius: 32px;
    background: linear-gradient(135deg, #FFD21F, #F5B700);
    color: #050505;
    text-align: center;
    box-shadow: 0 0 45px rgba(255,210,31,0.42);
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 46px;
    margin-bottom: 6px;
    font-weight: 950;
}

.card {
    background: rgba(17,17,17,0.94);
    padding: 24px;
    border-radius: 24px;
    border: 1px solid rgba(255,210,31,0.35);
    box-shadow: 0 0 24px rgba(255,210,31,0.12);
    margin-bottom: 18px;
}

.phase-card {
    background: linear-gradient(135deg, rgba(255,210,31,0.24), rgba(17,17,17,0.96));
    padding: 28px;
    border-radius: 28px;
    border: 1px solid rgba(255,210,31,0.55);
    box-shadow: 0 0 32px rgba(255,210,31,0.22);
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

.small-text {
    color: #D8D8D8;
    font-size: 15px;
}

.explain {
    background: rgba(255,210,31,0.12);
    border-left: 5px solid #FFD21F;
    padding: 20px;
    border-radius: 18px;
    color: #F8F8F8;
    margin-top: 18px;
    margin-bottom: 18px;
}

.warning-box {
    background: rgba(255,152,0,0.13);
    border-left: 5px solid #FF9800;
    padding: 18px;
    border-radius: 16px;
}

.red-box {
    background: rgba(255,61,0,0.12);
    border-left: 5px solid #FF3D00;
    padding: 18px;
    border-radius: 16px;
}

.green-box {
    background: rgba(0,200,83,0.12);
    border-left: 5px solid #00C853;
    padding: 18px;
    border-radius: 16px;
}

@media only screen and (max-width: 768px) {
    .hero h1 {
        font-size: 30px;
    }

    .big-score {
        font-size: 30px;
    }

    .phase {
        font-size: 25px;
    }

    .card {
        padding: 18px;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# CONSTANTS
# =========================================================
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


PHASE_MODELS = {
    "Accumulation": {
        "weights": {"Trend": 0.10, "Breadth": 0.16, "Liquidity": 0.22, "Valuation": 0.26, "Sentiment": 0.18, "Macro": 0.08},
        "ideal": {"Trend": (15, 45), "Breadth": (20, 50), "Liquidity": (55, 90), "Valuation": (0, 35), "Sentiment": (0, 35), "Macro": (35, 65)}
    },
    "Early Bull": {
        "weights": {"Trend": 0.18, "Breadth": 0.18, "Liquidity": 0.16, "Valuation": 0.12, "Sentiment": 0.10, "Macro": 0.26},
        "ideal": {"Trend": (45, 70), "Breadth": (45, 75), "Liquidity": (50, 80), "Valuation": (35, 65), "Sentiment": (35, 60), "Macro": (55, 85)}
    },
    "Mature Bull": {
        "weights": {"Trend": 0.24, "Breadth": 0.24, "Liquidity": 0.14, "Valuation": 0.08, "Sentiment": 0.12, "Macro": 0.18},
        "ideal": {"Trend": (65, 90), "Breadth": (65, 90), "Liquidity": (55, 85), "Valuation": (40, 70), "Sentiment": (50, 75), "Macro": (55, 85)}
    },
    "Late Bull": {
        "weights": {"Trend": 0.18, "Breadth": 0.10, "Liquidity": 0.08, "Valuation": 0.18, "Sentiment": 0.34, "Macro": 0.12},
        "ideal": {"Trend": (75, 100), "Breadth": (45, 75), "Liquidity": (35, 65), "Valuation": (75, 100), "Sentiment": (80, 100), "Macro": (45, 75)}
    },
    "Distribution": {
        "weights": {"Trend": 0.16, "Breadth": 0.28, "Liquidity": 0.16, "Valuation": 0.18, "Sentiment": 0.16, "Macro": 0.06},
        "ideal": {"Trend": (55, 85), "Breadth": (25, 55), "Liquidity": (25, 55), "Valuation": (65, 100), "Sentiment": (65, 95), "Macro": (35, 65)}
    },
    "Early Bear": {
        "weights": {"Trend": 0.25, "Breadth": 0.25, "Liquidity": 0.14, "Valuation": 0.08, "Sentiment": 0.18, "Macro": 0.10},
        "ideal": {"Trend": (30, 55), "Breadth": (25, 55), "Liquidity": (25, 60), "Valuation": (45, 75), "Sentiment": (35, 65), "Macro": (30, 60)}
    },
    "Mature Bear": {
        "weights": {"Trend": 0.24, "Breadth": 0.22, "Liquidity": 0.18, "Valuation": 0.08, "Sentiment": 0.18, "Macro": 0.10},
        "ideal": {"Trend": (10, 40), "Breadth": (10, 40), "Liquidity": (10, 45), "Valuation": (25, 55), "Sentiment": (15, 45), "Macro": (20, 50)}
    },
    "Late Bear": {
        "weights": {"Trend": 0.14, "Breadth": 0.14, "Liquidity": 0.18, "Valuation": 0.24, "Sentiment": 0.22, "Macro": 0.08},
        "ideal": {"Trend": (0, 35), "Breadth": (0, 35), "Liquidity": (35, 70), "Valuation": (0, 35), "Sentiment": (0, 25), "Macro": (20, 55)}
    }
}


PHASE_TEXT = {
    "Accumulation": "Smart money may be entering quietly. Valuations and sentiment are usually depressed.",
    "Early Bull": "Economy and liquidity start improving before the crowd fully trusts the recovery.",
    "Mature Bull": "Trend, breadth, liquidity, and macro data are broadly supportive.",
    "Late Bull": "Investor psychology dominates. Euphoria and expensive valuation become key risks.",
    "Distribution": "The index may look strong, but breadth and liquidity often weaken underneath.",
    "Early Bear": "Trend and breadth start breaking down. Many investors still think it is only a correction.",
    "Mature Bear": "Weak trend, weak breadth, poor liquidity, and fearful sentiment dominate.",
    "Late Bear": "Panic or exhaustion may appear. Valuation becomes attractive, but trend may still be weak."
}


# =========================================================
# UTILS
# =========================================================
def clamp(value, low=0, high=100):
    return max(low, min(high, value))


def ideal_range_score(value, low, high):
    if low <= value <= high:
        return 100
    if value < low:
        return clamp(100 - (low - value) * 2.4)
    return clamp(100 - (value - high) * 2.4)


def phase_rule_score(phase, raw):
    model = PHASE_MODELS[phase]
    adjusted = {}

    for feature in FEATURES:
        low, high = model["ideal"][feature]
        adjusted[feature] = ideal_range_score(raw[feature], low, high)

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

    final = clamp(base - penalty)

    return final, base, penalty, adjusted


def get_rule_engine_results(raw):
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

    rows = sorted(rows, key=lambda x: x["Rule Score"], reverse=True)
    return rows


def create_synthetic_training_data(n=3500, random_state=42):
    """
    Creates starter training data using phase-specific ideal ranges.
    Later, replace this with real historical labelled data.
    """
    rng = np.random.default_rng(random_state)
    rows = []

    for phase in PHASES:
        model = PHASE_MODELS[phase]

        for _ in range(n // len(PHASES)):
            row = {}

            for feature in FEATURES:
                low, high = model["ideal"][feature]
                center = (low + high) / 2
                spread = max((high - low) / 2, 8)

                value = rng.normal(center, spread)
                row[feature] = clamp(value)

            row["Phase"] = phase
            rows.append(row)

    df = pd.DataFrame(rows)

    noise = rng.normal(0, 5, size=df[FEATURES].shape)
    df[FEATURES] = np.clip(df[FEATURES].values + noise, 0, 100)

    return df


@st.cache_resource
def train_xgboost_model(training_df):
    df = training_df.copy()

    le = LabelEncoder()
    y = le.fit_transform(df["Phase"])
    X = df[FEATURES]

    stratify = y if len(np.unique(y)) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.22,
        random_state=42,
        stratify=stratify
    )

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(le.classes_),
        n_estimators=350,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.88,
        colsample_bytree=0.88,
        min_child_weight=2,
        reg_lambda=2.0,
        reg_alpha=0.15,
        eval_metric="mlogloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    return model, le, accuracy


def xgb_predict(model, le, raw):
    X_live = pd.DataFrame([raw], columns=FEATURES)

    probabilities = model.predict_proba(X_live)[0]
    classes = le.inverse_transform(np.arange(len(probabilities)))

    prob_df = pd.DataFrame({
        "Phase": classes,
        "XGBoost Probability": probabilities * 100
    })

    return prob_df.sort_values("XGBoost Probability", ascending=False)


def transition_filter(prob_df, previous_phase=None):
    """
    Prevents impossible phase jumps.
    Example: Late Bull should not jump directly to Accumulation.
    """
    if not previous_phase or previous_phase == "None":
        return prob_df

    order = PHASES
    previous_index = order.index(previous_phase)

    adjusted = prob_df.copy()

    for idx, row in adjusted.iterrows():
        phase_index = order.index(row["Phase"])
        circular_distance = min(
            abs(phase_index - previous_index),
            len(order) - abs(phase_index - previous_index)
        )

        if circular_distance == 0:
            multiplier = 1.05
        elif circular_distance == 1:
            multiplier = 1.00
        elif circular_distance == 2:
            multiplier = 0.78
        else:
            multiplier = 0.45

        adjusted.loc[idx, "XGBoost Probability"] *= multiplier

    total = adjusted["XGBoost Probability"].sum()
    adjusted["XGBoost Probability"] = adjusted["XGBoost Probability"] / total * 100

    return adjusted.sort_values("XGBoost Probability", ascending=False)


def hybrid_prediction(xgb_df, rule_rows):
    rule_df = pd.DataFrame([
        {"Phase": r["Phase"], "Rule Score": r["Rule Score"]}
        for r in rule_rows
    ])

    merged = xgb_df.merge(rule_df, on="Phase", how="left")

    merged["Hybrid Score"] = (
        merged["XGBoost Probability"] * 0.65 +
        merged["Rule Score"] * 0.35
    )

    merged = merged.sort_values("Hybrid Score", ascending=False)
    return merged


def score_color(score):
    if score >= 70:
        return "#00C853"
    if score >= 45:
        return "#FFD21F"
    if score >= 25:
        return "#FF9800"
    return "#FF3D00"


# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🐝 Madness of Money Bees</h1>
    <h3>XGBoost Market Cycle Predictor</h3>
    <p>Premium phase-aware market-cycle intelligence with noise filtering and dynamic scoring.</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🐝 Inputs")

mode = st.sidebar.radio(
    "Training Data Mode",
    ["Use Starter Synthetic Data", "Upload Historical CSV"]
)

uploaded = None
if mode == "Upload Historical CSV":
    uploaded = st.sidebar.file_uploader(
        "Upload CSV with Trend, Breadth, Liquidity, Valuation, Sentiment, Macro, Phase",
        type=["csv"]
    )

previous_phase = st.sidebar.selectbox(
    "Previous Confirmed Phase",
    ["None"] + PHASES,
    index=0
)

st.sidebar.markdown("### Live Market Scores")

raw_scores = {
    "Trend": st.sidebar.slider("Trend Score", 0, 100, 65),
    "Breadth": st.sidebar.slider("Breadth Score", 0, 100, 60),
    "Liquidity": st.sidebar.slider("Liquidity Score", 0, 100, 55),
    "Valuation": st.sidebar.slider("Valuation Score", 0, 100, 50),
    "Sentiment": st.sidebar.slider("Sentiment Score", 0, 100, 58),
    "Macro": st.sidebar.slider("Macro / Economy Score", 0, 100, 52)
}


# =========================================================
# DATA
# =========================================================
if uploaded is not None:
    training_df = pd.read_csv(uploaded)

    missing = [c for c in FEATURES + ["Phase"] if c not in training_df.columns]

    if missing:
        st.error(f"CSV is missing columns: {missing}")
        st.stop()

    training_df = training_df[FEATURES + ["Phase"]].dropna()

else:
    training_df = create_synthetic_training_data()


# =========================================================
# MODEL
# =========================================================
model, label_encoder, model_accuracy = train_xgboost_model(training_df)

rule_rows = get_rule_engine_results(raw_scores)

xgb_df = xgb_predict(model, label_encoder, raw_scores)
xgb_df = transition_filter(xgb_df, previous_phase)

hybrid_df = hybrid_prediction(xgb_df, rule_rows)

winner = hybrid_df.iloc[0]
runner = hybrid_df.iloc[1]

current_phase = winner["Phase"]
final_score = winner["Hybrid Score"]
confidence = clamp(55 + (winner["Hybrid Score"] - runner["Hybrid Score"]) * 2.4, 50, 96)


# =========================================================
# TOP CARDS
# =========================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="phase-card">
        <p class="small-text">Predicted Market Phase</p>
        <div class="phase">{current_phase}</div>
        <p class="small-text">{PHASE_TEXT[current_phase]}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <p class="small-text">Final Hybrid Score</p>
        <div class="big-score">{final_score:.1f}/100</div>
        <p class="small-text">65% XGBoost + 35% rule engine.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card">
        <p class="small-text">Prediction Confidence</p>
        <div class="big-score">{confidence:.0f}%</div>
        <p class="small-text">Based on gap from runner-up phase.</p>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# MODEL INFO
# =========================================================
st.markdown(f"""
<div class="explain">
<h3>Model Engine</h3>
<p>
This tool uses an XGBoost multi-class classifier plus a phase-aware rule engine.
XGBoost is designed for classification and ranking problems and is widely used for tabular machine-learning tasks.
The rule engine helps reduce market-cycle noise by checking whether each phase's expected conditions are actually present.
</p>
<p>
Current training accuracy on validation split: <b>{model_accuracy * 100:.1f}%</b>.
</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# MARKET CYCLE CHART
# =========================================================
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
y = np.sin(x / 7.5) * 10 + x * 0.23

current_x = phase_positions[current_phase]
current_y = np.sin(current_x / 7.5) * 10 + current_x * 0.23

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode="lines",
    line=dict(width=5, color="#FFD21F"),
    name="Market Cycle"
))

fig.add_trace(go.Scatter(
    x=[current_x],
    y=[current_y],
    mode="markers+text",
    marker=dict(
        size=25,
        color="#050505",
        line=dict(color="#FFD21F", width=5)
    ),
    text=[current_phase],
    textposition="top center",
    name="Current Phase"
))

for phase, px in phase_positions.items():
    py = np.sin(px / 7.5) * 10 + px * 0.23
    fig.add_annotation(
        x=px,
        y=py,
        text=phase,
        showarrow=False,
        font=dict(color="white", size=11),
        bgcolor="rgba(17,17,17,0.85)",
        bordercolor="#FFD21F",
        borderwidth=1
    )

fig.update_layout(
    paper_bgcolor="#050505",
    plot_bgcolor="#050505",
    font=dict(color="white"),
    height=460,
    margin=dict(l=20, r=20, t=40, b=30),
    xaxis=dict(
        title="Market Cycle Journey",
        gridcolor="rgba(255,255,255,0.08)",
        showticklabels=False
    ),
    yaxis=dict(
        title="Cycle Momentum",
        gridcolor="rgba(255,255,255,0.08)"
    )
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class="explain">
<h3>How to Read This Chart</h3>
<p>
The yellow wave represents the market cycle. Bull phases usually sit on the rising part of the curve.
Distribution appears near the top when the market may still look strong but internal strength weakens.
Bear phases appear on the declining part. Accumulation appears near the bottom, where sentiment is weak
but long-term opportunity may begin forming.
</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# LIVE INDICATOR SCORES
# =========================================================
st.markdown("## 📊 Live Indicator Scores")

descriptions = {
    "Trend": "Moving averages, RSI, MACD, ADX, and price momentum.",
    "Breadth": "Advance/decline, stocks above 50/200 DMA, and new highs/lows.",
    "Liquidity": "FII/DII flows, interest rates, money supply, and credit conditions.",
    "Valuation": "PE, PB, earnings yield, and market cap/GDP.",
    "Sentiment": "VIX, put-call ratio, IPO activity, retail excitement, and fear/greed.",
    "Macro": "GDP, CPI, PMI, crude oil, USDINR, and employment."
}

cols = st.columns(3)

for i, feature in enumerate(FEATURES):
    value = raw_scores[feature]

    with cols[i % 3]:
        st.markdown(f"""
        <div class="card">
            <h3>{feature}</h3>
            <div class="big-score" style="color:{score_color(value)};">{value}/100</div>
            <p class="small-text">{descriptions[feature]}</p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# HYBRID SCORE TABLE
# =========================================================
st.markdown("## 🧠 Final Phase Ranking")

display_df = hybrid_df.copy()
display_df["XGBoost Probability"] = display_df["XGBoost Probability"].round(1)
display_df["Rule Score"] = display_df["Rule Score"].round(1)
display_df["Hybrid Score"] = display_df["Hybrid Score"].round(1)

st.dataframe(display_df, use_container_width=True, hide_index=True)

if winner["Hybrid Score"] - runner["Hybrid Score"] < 7:
    st.markdown(f"""
    <div class="warning-box">
    <h3>Close Phase Warning</h3>
    <p>
    The model selected <b>{current_phase}</b>, but <b>{runner["Phase"]}</b> is close.
    This means the market may be transitioning. Wait for confirmation from trend, breadth, and liquidity.
    </p>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================
st.markdown("## 🔍 XGBoost Feature Importance")

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
})

importance["Importance"] = importance["Importance"] / importance["Importance"].sum() * 100
importance = importance.sort_values("Importance", ascending=True)

fig_imp = go.Figure()

fig_imp.add_trace(go.Bar(
    x=importance["Importance"],
    y=importance["Feature"],
    orientation="h",
    marker=dict(color="#FFD21F")
))

fig_imp.update_layout(
    paper_bgcolor="#050505",
    plot_bgcolor="#050505",
    font=dict(color="white"),
    height=380,
    xaxis=dict(title="Importance %", gridcolor="rgba(255,255,255,0.08)"),
    yaxis=dict(title=""),
    margin=dict(l=20, r=20, t=30, b=30)
)

st.plotly_chart(fig_imp, use_container_width=True)


# =========================================================
# CURRENT PHASE WEIGHTS
# =========================================================
st.markdown("## ⚖️ Current Phase Weightage")

weights = PHASE_MODELS[current_phase]["weights"]
ideal = PHASE_MODELS[current_phase]["ideal"]

weight_df = pd.DataFrame({
    "Indicator": FEATURES,
    "Phase Weight %": [weights[f] * 100 for f in FEATURES],
    "Live Score": [raw_scores[f] for f in FEATURES],
    "Ideal Range": [f"{ideal[f][0]} - {ideal[f][1]}" for f in FEATURES]
})

weight_df["Phase Weight %"] = weight_df["Phase Weight %"].round(1)

st.dataframe(weight_df, use_container_width=True, hide_index=True)


# =========================================================
# UPLOAD FORMAT
# =========================================================
st.markdown("## 📁 CSV Format for Real Training")

st.markdown("""
<div class="explain">
<h3>Required CSV Columns</h3>
<p>
To train on real historical data, upload a CSV with these columns:
</p>
<p>
<b>Trend, Breadth, Liquidity, Valuation, Sentiment, Macro, Phase</b>
</p>
<p>
The Phase column should contain one of:
Accumulation, Early Bull, Mature Bull, Late Bull, Distribution, Early Bear, Mature Bear, Late Bear.
</p>
</div>
""", unsafe_allow_html=True)

sample = pd.DataFrame({
    "Trend": [62, 82, 58],
    "Breadth": [68, 50, 35],
    "Liquidity": [70, 48, 32],
    "Valuation": [42, 88, 75],
    "Sentiment": [48, 92, 82],
    "Macro": [72, 58, 45],
    "Phase": ["Early Bull", "Late Bull", "Distribution"]
})

st.dataframe(sample, use_container_width=True, hide_index=True)


# =========================================================
# DISCLAIMER
# =========================================================
st.markdown("""
<div class="red-box">
<h3>Disclaimer</h3>
<p>
This tool is for education, analytics, and research only. It is not financial advice.
The starter synthetic data is only for demo purposes. For serious use, train the model on real historical labelled data.
</p>
</div>
""", unsafe_allow_html=True)
