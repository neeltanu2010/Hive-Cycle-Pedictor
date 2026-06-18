import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

PHASE_MODELS = {
    "Accumulation": {
        "ideal": {
            "Trend": (15, 45),
            "Breadth": (20, 50),
            "Liquidity": (55, 90),
            "Valuation": (0, 35),
            "Sentiment": (0, 35),
            "Macro": (35, 65),
        }
    },
    "Early Bull": {
        "ideal": {
            "Trend": (45, 70),
            "Breadth": (45, 75),
            "Liquidity": (50, 80),
            "Valuation": (35, 65),
            "Sentiment": (35, 60),
            "Macro": (55, 85),
        }
    },
    "Mature Bull": {
        "ideal": {
            "Trend": (65, 90),
            "Breadth": (65, 90),
            "Liquidity": (55, 85),
            "Valuation": (40, 70),
            "Sentiment": (50, 75),
            "Macro": (55, 85),
        }
    },
    "Late Bull": {
        "ideal": {
            "Trend": (75, 100),
            "Breadth": (45, 75),
            "Liquidity": (35, 65),
            "Valuation": (75, 100),
            "Sentiment": (80, 100),
            "Macro": (45, 75),
        }
    },
    "Distribution": {
        "ideal": {
            "Trend": (55, 85),
            "Breadth": (25, 55),
            "Liquidity": (25, 55),
            "Valuation": (65, 100),
            "Sentiment": (65, 95),
            "Macro": (35, 65),
        }
    },
    "Early Bear": {
        "ideal": {
            "Trend": (30, 55),
            "Breadth": (25, 55),
            "Liquidity": (25, 60),
            "Valuation": (45, 75),
            "Sentiment": (35, 65),
            "Macro": (30, 60),
        }
    },
    "Mature Bear": {
        "ideal": {
            "Trend": (10, 40),
            "Breadth": (10, 40),
            "Liquidity": (10, 45),
            "Valuation": (25, 55),
            "Sentiment": (15, 45),
            "Macro": (20, 50),
        }
    },
    "Late Bear": {
        "ideal": {
            "Trend": (0, 35),
            "Breadth": (0, 35),
            "Liquidity": (35, 70),
            "Valuation": (0, 35),
            "Sentiment": (0, 25),
            "Macro": (20, 55),
        }
    },
}


def clamp(value, low=0, high=100):
    return max(low, min(high, float(value)))


def create_training_data(samples=5000, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    for phase in PHASES:
        ideal = PHASE_MODELS[phase]["ideal"]

        for _ in range(samples // len(PHASES)):
            row = {}

            for feature in FEATURES:
                low, high = ideal[feature]
                center = (low + high) / 2
                spread = max((high - low) / 2, 8)
                row[feature] = clamp(rng.normal(center, spread))

            row["Phase"] = phase
            rows.append(row)

    df = pd.DataFrame(rows)

    noise = rng.normal(0, 5, size=df[FEATURES].shape)
    df[FEATURES] = np.clip(df[FEATURES].values + noise, 0, 100)

    return df


def main():
    df = create_training_data()

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
        num_class=len(PHASES),
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.88,
        colsample_bytree=0.88,
        min_child_weight=2,
        reg_lambda=2.0,
        reg_alpha=0.15,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=1,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    joblib.dump(model, "xgb_market_cycle_model.pkl")
    joblib.dump(encoder, "phase_label_encoder.pkl")
    joblib.dump(FEATURES, "model_features.pkl")

    print("Model trained successfully.")
    print(f"Validation accuracy: {accuracy * 100:.2f}%")
    print("Saved files:")
    print("- xgb_market_cycle_model.pkl")
    print("- phase_label_encoder.pkl")
    print("- model_features.pkl")


if __name__ == "__main__":
    main()
