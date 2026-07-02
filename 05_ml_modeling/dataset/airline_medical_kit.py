"""
airline_medical_kit.py
======================
Airline Medical Kit — Predict Kit Usefulness
Binary classification: predict Target (1 = claim filed, 0 = no claim)

Dataset
-------
    train.csv          — labelled rows (ID, features, Target)
    test.csv           — unlabelled rows (ID, features)
    sample_submission.csv — expected output format (ID, Target)

Usage
-----
    python airline_medical_kit.py

Output
------
    - Console : CV scores, evaluation metrics
    - Plots   : EDA, evaluation dashboard, feature importance
    - File    : submission.csv  (ID, Target) — same format as sample_submission.csv
"""

# ── Imports ───────────────────────────────────────────────────
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
TEST_PATH  = os.path.join(BASE_DIR, "test.csv")
SUB_PATH   = os.path.join(BASE_DIR, "submission.csv")

# ── Feature config ────────────────────────────────────────────
# Gender: 70% missing → treat as numeric with median imputation
NUM_COLS = ["Distributor", "Product", "Duration", "Destination",
            "Sales", "Commission", "Gender", "Age"]


# ══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    print(f"Train : {train.shape}  |  Target balance: {train['Target'].mean():.2%} positive")
    print(f"Test  : {test.shape}")
    print(f"Missing Gender — train: {train['Gender'].isna().sum()}  "
          f"test: {test['Gender'].isna().sum()}")
    return train, test


# ══════════════════════════════════════════════════════════════
# 2. EDA
# ══════════════════════════════════════════════════════════════
def plot_eda(train: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    plot_cols = ["Age", "Sales", "Commission", "Duration",
                 "Distributor", "Product", "Destination", "Gender"]

    for ax, col in zip(axes.flat, plot_cols):
        for label, grp in train.groupby("Target"):
            ax.hist(grp[col].dropna(), bins=30, alpha=0.6, label=f"Target={label}")
        ax.set_title(col)
        ax.legend(fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("EDA — Airline Medical Kit (train.csv)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    print("\n── Correlation with Target ──")
    print(train[NUM_COLS + ["Target"]].corr()["Target"].drop("Target").sort_values().round(3))


# ══════════════════════════════════════════════════════════════
# 3. PIPELINE
# ══════════════════════════════════════════════════════════════
def build_pipeline(model) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), NUM_COLS),
    ])
    return Pipeline([("prep", preprocessor), ("model", model)])


# ══════════════════════════════════════════════════════════════
# 4. TRAIN & EVALUATE (on train/val split)
# ══════════════════════════════════════════════════════════════
def train_and_evaluate(X_train, X_val, y_train, y_val):
    # Class weights — dataset is highly imbalanced (~4.7% positive)
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42),
    }

    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best = {"name": None, "pipe": None, "auc": 0}

    print("\n── Model comparison ──────────────────────────────────")
    for name, model in models.items():
        pipe   = build_pipeline(model)
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        val_auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
        print(f"  {name:25s}  CV={cv_auc.mean():.4f}±{cv_auc.std():.4f}  Val-AUC={val_auc:.4f}")
        if val_auc > best["auc"]:
            best = {"name": name, "pipe": pipe, "auc": val_auc}

    print(f"\n  ✅ Best: {best['name']}  (Val-AUC={best['auc']:.4f})")
    return best


def plot_evaluation(pipe, name, X_val, y_val) -> None:
    y_proba = pipe.predict_proba(X_val)[:, 1]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    ConfusionMatrixDisplay.from_estimator(pipe, X_val, y_val,
                                          normalize="true", cmap="Blues", ax=axes[0])
    axes[0].set_title(f"Confusion Matrix\n{name}")

    RocCurveDisplay.from_estimator(pipe, X_val, y_val, ax=axes[1])
    axes[1].set_title(f"ROC  AUC={roc_auc_score(y_val, y_proba):.3f}")

    PrecisionRecallDisplay.from_estimator(pipe, X_val, y_val, ax=axes[2])
    axes[2].set_title(f"PR  AP={average_precision_score(y_val, y_proba):.3f}")

    plt.suptitle(f"Evaluation — {name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    print("\n── Classification Report ──")
    print(classification_report(y_val, pipe.predict(X_val),
                                target_names=["No Claim (0)", "Claim (1)"]))


def plot_feature_importance(pipe, name) -> None:
    try:
        imp      = pipe.named_steps["model"].feature_importances_
    except AttributeError:
        imp      = np.abs(pipe.named_steps["model"].coef_[0])
    feat_imp = pd.Series(imp, index=NUM_COLS).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    feat_imp.plot.barh(ax=ax, color="steelblue")
    ax.invert_yaxis()
    ax.set_title(f"Feature Importances — {name}", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()
    print("\n── Feature importances ──\n", feat_imp.round(4))


# ══════════════════════════════════════════════════════════════
# 5. RETRAIN ON FULL TRAIN + PREDICT TEST
# ══════════════════════════════════════════════════════════════
def make_submission(best_model_name, train, test) -> None:
    """Retrain best model on full train set, predict test, save submission.csv."""
    pos_weight = (train["Target"] == 0).sum() / (train["Target"] == 1).sum()

    model_map = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest":       RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42),
    }

    pipe = build_pipeline(model_map[best_model_name])
    pipe.fit(train[NUM_COLS], train["Target"])

    test_pred = pipe.predict(test[NUM_COLS])

    submission = pd.DataFrame({"ID": test["ID"], "Target": test_pred})
    submission.to_csv(SUB_PATH, index=False)
    print(f"\n✅ submission.csv saved → {SUB_PATH}")
    print(f"   Rows: {len(submission)}  |  Predicted positives: {test_pred.sum()} ({test_pred.mean():.2%})")
    print(submission.head(8).to_string(index=False))


# ══════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 55)
    print("  Airline Medical Kit — ML Pipeline")
    print("=" * 55)

    # Load
    train, test = load_data()

    # EDA
    plot_eda(train)

    # Split train → train / val
    X = train[NUM_COLS]
    y = train["Target"]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train & evaluate
    best = train_and_evaluate(X_tr, X_val, y_tr, y_val)
    plot_evaluation(best["pipe"], best["name"], X_val, y_val)
    plot_feature_importance(best["pipe"], best["name"])

    # Retrain on full data + generate submission
    make_submission(best["name"], train, test)


if __name__ == "__main__":
    main()
