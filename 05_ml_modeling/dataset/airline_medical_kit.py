"""
airline_medical_kit.py
======================
Airline Medical Kit — Predict Kit Usefulness
Binary classification: predict whether a passenger will use the kit (kit_used = 1)

Usage
-----
    python airline_medical_kit.py

Output
------
    - Console: CV scores, test metrics, business insights
    - Plots: EDA dashboard, evaluation dashboard, feature importance
    - Files: predictions.csv  (test set predictions)
"""

# ── Imports ───────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)

# ══════════════════════════════════════════════════════════════
# 1. GENERATE DATASET
# ══════════════════════════════════════════════════════════════
def generate_dataset(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Generate synthetic airline medical kit dataset."""
    rng = np.random.default_rng(seed)

    age                    = rng.integers(18, 80, n).astype(float)
    flight_duration_h      = rng.uniform(0.5, 16, n)
    has_chronic_condition  = rng.integers(0, 2, n)
    previous_kit_purchases = rng.integers(0, 6, n).astype(float)
    altitude_sensitivity   = rng.integers(1, 11, n).astype(float)
    travel_class           = rng.choice(["Economy", "Business", "First"], n, p=[0.70, 0.20, 0.10])
    flight_type            = rng.choice(["Domestic", "International"], n, p=[0.45, 0.55])
    seat_position          = rng.choice(["Window", "Middle", "Aisle"], n)
    meal_type              = rng.choice(
        ["Standard", "Vegetarian", "Diabetic", "Low-sodium"], n,
        p=[0.55, 0.20, 0.15, 0.10]
    )

    # Inject missing values
    age[rng.random(n) < 0.04]                 = np.nan
    flight_duration_h[rng.random(n) < 0.03]   = np.nan
    altitude_sensitivity[rng.random(n) < 0.05] = np.nan

    # Target: logistic function with realistic drivers
    logit = (
        -2.0
        + 0.015 * np.where(np.isnan(age), 40, age)
        + 0.12  * np.where(np.isnan(flight_duration_h), 5, flight_duration_h)
        + 1.20  * has_chronic_condition
        + 0.35  * previous_kit_purchases
        + 0.10  * np.where(np.isnan(altitude_sensitivity), 5, altitude_sensitivity)
        + 0.40  * (meal_type == "Diabetic").astype(int)
        + 0.30  * (meal_type == "Low-sodium").astype(int)
        + 0.20  * (flight_type == "International").astype(int)
        + rng.normal(0, 0.4, n)
    )
    kit_used = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)

    return pd.DataFrame({
        "age":                    age,
        "flight_duration_h":      flight_duration_h,
        "travel_class":           travel_class,
        "has_chronic_condition":  has_chronic_condition,
        "previous_kit_purchases": previous_kit_purchases,
        "altitude_sensitivity":   altitude_sensitivity,
        "flight_type":            flight_type,
        "seat_position":          seat_position,
        "meal_type":              meal_type,
        "kit_used":               kit_used,
    })


# ══════════════════════════════════════════════════════════════
# 2. EDA PLOT
# ══════════════════════════════════════════════════════════════
def plot_eda(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    num_cols = ["age", "flight_duration_h", "altitude_sensitivity", "previous_kit_purchases"]
    cat_cols = ["travel_class", "flight_type", "meal_type", "has_chronic_condition"]

    for ax, col in zip(axes[0], num_cols):
        for label, grp in df.groupby("kit_used"):
            ax.hist(grp[col].dropna(), bins=25, alpha=0.6, label=f"kit_used={label}")
        ax.set_title(col)
        ax.legend(fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    for ax, col in zip(axes[1], cat_cols):
        rates = df.groupby(col)["kit_used"].mean().sort_values()
        ax.barh(rates.index.astype(str), rates.values, color="steelblue")
        ax.axvline(df["kit_used"].mean(), color="crimson", ls="--", label="overall mean")
        ax.set_title(f"Kit use rate by {col}")
        ax.set_xlabel("P(kit used)")
        ax.legend(fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("EDA — Airline Medical Kit", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════
# 3. BUILD PREPROCESSING + PIPELINE
# ══════════════════════════════════════════════════════════════
NUM_COLS = ["age", "flight_duration_h", "altitude_sensitivity",
            "previous_kit_purchases", "has_chronic_condition"]
CAT_COLS = ["travel_class", "flight_type", "seat_position", "meal_type"]


def build_pipeline(model) -> Pipeline:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, NUM_COLS),
        ("cat", cat_pipe, CAT_COLS),
    ])
    return Pipeline([("prep", preprocessor), ("model", model)])


# ══════════════════════════════════════════════════════════════
# 4. TRAIN & EVALUATE
# ══════════════════════════════════════════════════════════════
def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest":       RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
    }

    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best  = {"name": None, "pipe": None, "auc": 0}

    print("\n── Model comparison ──────────────────────────────────")
    for name, model in models.items():
        pipe   = build_pipeline(model)
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        test_auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        print(f"  {name:25s}  CV={cv_auc.mean():.4f}±{cv_auc.std():.4f}  Test-AUC={test_auc:.4f}")
        if test_auc > best["auc"]:
            best = {"name": name, "pipe": pipe, "auc": test_auc}

    print(f"\n  ✅ Best: {best['name']}  (Test-AUC={best['auc']:.4f})")
    return best


def plot_evaluation(pipe, name, X_test, y_test) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    y_proba   = pipe.predict_proba(X_test)[:, 1]

    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test,
                                          normalize="true", cmap="Blues", ax=axes[0])
    axes[0].set_title(f"Confusion Matrix\n{name}")

    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=axes[1])
    axes[1].set_title(f"ROC  AUC={roc_auc_score(y_test, y_proba):.3f}")

    PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test, ax=axes[2])
    axes[2].set_title(f"PR  AP={average_precision_score(y_test, y_proba):.3f}")

    plt.suptitle(f"Evaluation — {name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    print("\n── Classification Report ──")
    print(classification_report(y_test, pipe.predict(X_test),
                                target_names=["Not Used", "Used"]))


def plot_feature_importance(pipe, name) -> None:
    cat_names = (pipe.named_steps["prep"]
                 .named_transformers_["cat"]["encoder"]
                 .get_feature_names_out(CAT_COLS))
    feat_names = NUM_COLS + list(cat_names)

    try:
        imp      = pipe.named_steps["model"].feature_importances_
        feat_imp = pd.Series(imp, index=feat_names).sort_values(ascending=False)
    except AttributeError:
        imp      = np.abs(pipe.named_steps["model"].coef_[0])
        feat_imp = pd.Series(imp, index=feat_names).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    feat_imp.head(15).plot.barh(ax=ax, color="steelblue")
    ax.invert_yaxis()
    ax.set_title(f"Top 15 Features — {name}", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()

    print("\n── Top 10 features ──")
    print(feat_imp.head(10).round(4))


# ══════════════════════════════════════════════════════════════
# 5. SAVE PREDICTIONS
# ══════════════════════════════════════════════════════════════
def save_predictions(pipe, X_test, y_test, path: str = "predictions.csv") -> None:
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred  = pipe.predict(X_test)
    out = X_test.copy().reset_index(drop=True)
    out["kit_used_actual"]   = y_test.values
    out["kit_used_predicted"] = y_pred
    out["kit_used_proba"]     = y_proba.round(4)
    out.to_csv(path, index=False)
    print(f"\n✅ Predictions saved → {path}  ({len(out)} rows)")


# ══════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 55)
    print("  Airline Medical Kit — ML Pipeline")
    print("=" * 55)

    # Data
    df = generate_dataset(n=3000)
    print(f"\nDataset: {df.shape}  |  Target balance: {df['kit_used'].mean():.1%} used kit")

    X = df.drop(columns="kit_used")
    y = df["kit_used"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # EDA
    plot_eda(df)

    # Train
    best = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Evaluate
    plot_evaluation(best["pipe"], best["name"], X_test, y_test)
    plot_feature_importance(best["pipe"], best["name"])

    # Save
    save_predictions(best["pipe"], X_test, y_test,
                     path="05_ml_modeling/dataset/predictions.csv")

    # Business insights
    print("\n" + "=" * 55)
    print("  BUSINESS INSIGHTS")
    print("=" * 55)
    for insight in [
        "✅ Chronic condition → strongest predictor of kit usage",
        "✅ Longer flights → significantly more kit usage",
        "✅ Diabetic / Low-sodium meal → health-aware passengers",
        "✅ Repeat buyers → already found the kit useful before",
        "💡 Recommendation: target marketing to chronic + long-haul",
    ]:
        print(f"  {insight}")


if __name__ == "__main__":
    main()
