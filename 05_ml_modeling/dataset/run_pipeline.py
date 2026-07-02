import warnings; warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no plt.show() needed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)

import os
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)))

# ── Load data ─────────────────────────────────────────────────
train = pd.read_csv(os.path.join(BASE, "train.csv"))
test  = pd.read_csv(os.path.join(BASE, "test.csv"))
print(f"Train : {train.shape}  Target balance: {train['Target'].mean():.2%} positive")
print(f"Test  : {test.shape}")

NUM_COLS = ["Distributor", "Product", "Duration", "Destination",
            "Sales", "Commission", "Gender", "Age"]

# ── Split ─────────────────────────────────────────────────────
X, y = train[NUM_COLS], train["Target"]
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ── Pipeline factory ─────────────────────────────────────────
def make_pipe(model):
    prep = ColumnTransformer([
        ("n", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
        ]), NUM_COLS),
    ])
    return Pipeline([("prep", prep), ("model", model)])

# ── Train & compare ────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest":       RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42),
}
cv   = StratifiedKFold(5, shuffle=True, random_state=42)
best = {"name": None, "pipe": None, "auc": 0}

print("\n── Model comparison ──────────────────────────────────")
for name, model in models.items():
    p      = make_pipe(model)
    cv_auc = cross_val_score(p, X_tr, y_tr, cv=cv, scoring="roc_auc")
    p.fit(X_tr, y_tr)
    vauc   = roc_auc_score(y_val, p.predict_proba(X_val)[:, 1])
    print(f"  {name:25s}  CV={cv_auc.mean():.4f}+-{cv_auc.std():.4f}  Val-AUC={vauc:.4f}")
    if vauc > best["auc"]:
        best = {"name": name, "pipe": p, "auc": vauc}

print(f"\n  Best: {best['name']}  (Val-AUC={best['auc']:.4f})")

# ── Evaluation plots (saved as PNG) ───────────────────────────
pipe = best["pipe"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
ConfusionMatrixDisplay.from_estimator(pipe, X_val, y_val, normalize="true", cmap="Blues", ax=axes[0])
RocCurveDisplay.from_estimator(pipe, X_val, y_val, ax=axes[1])
PrecisionRecallDisplay.from_estimator(pipe, X_val, y_val, ax=axes[2])
plt.suptitle(f"Evaluation — {best['name']}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(BASE, "evaluation.png"), dpi=120, bbox_inches="tight")
print("\n  evaluation.png saved")

print("\n── Classification Report ──")
print(classification_report(y_val, pipe.predict(X_val), target_names=["No Claim", "Claim"]))

# ── Retrain on FULL train → predict test ──────────────────────
pipe.fit(X, y)
pred = pipe.predict(test[NUM_COLS])

submission = pd.DataFrame({"ID": test["ID"], "Target": pred})
submission.to_csv(os.path.join(BASE, "submission.csv"), index=False)

print(f"\n  submission.csv saved  rows={len(submission)}  positives={pred.sum()} ({pred.mean():.2%})")
print(submission.head(10).to_string(index=False))
