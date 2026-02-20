"""
03_train_model.py
=================
Trains an XGBoost Classifier to predict whether next week
will be a heavy-rain week (Rainy_Next_Week = 1 iff PRCP > 15mm).

Hyperparameter tuning: RandomizedSearchCV (5-fold StratifiedKFold on train set).

Produces:
  models/xgb_model.pkl
  reports/metrics.json
  reports/figures/03_confusion_matrix.png
  reports/figures/03_roc_curve.png
  reports/figures/03_precision_recall_curve.png
  reports/figures/03_xgb_feature_importance.png

Run from project root:
    python src/03_train_model.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score,
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED  = os.path.join("data", "processed")
MODEL_DIR  = "models"
FIG_DIR    = os.path.join("reports", "figures")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── Aesthetics ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       150,
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})
RAINY_CLR = "#1d7fe8"
DRY_CLR   = "#e87d1d"


# =============================================================================
# 1.  Load Data
# =============================================================================
def load_splits():
    with open(os.path.join(PROCESSED, "metadata.json")) as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    target_col   = meta["target_col"]
    spw          = meta["scale_pos_weight"]

    def _load(name):
        df = pd.read_csv(os.path.join(PROCESSED, f"{name}.csv"))
        X  = df[feature_cols].values
        y  = df[target_col].values
        return X, y

    X_train, y_train = _load("train")
    X_val,   y_val   = _load("val")
    X_test,  y_test  = _load("test")

    print(f"[load]  Train: {X_train.shape}  |  Val: {X_val.shape}  |  Test: {X_test.shape}")
    print(f"[load]  scale_pos_weight = {spw}")
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, spw


# =============================================================================
# 2.  Hyperparameter Tuning (RandomizedSearchCV on train set only)
# =============================================================================
PARAM_DIST = {
    "n_estimators":     [100, 150, 200, 300],
    "max_depth":        [3, 4, 5, 6],
    "learning_rate":    [0.01, 0.05, 0.08, 0.1],
    "subsample":        [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma":            [0, 0.1, 0.3],
    "reg_alpha":        [0, 0.1, 0.5],
    "reg_lambda":       [1, 1.5, 2],
}

def tune_model(X_train, y_train, scale_pos_weight):
    base = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=False)  # no shuffle: chronological data

    search = RandomizedSearchCV(
        base, PARAM_DIST,
        n_iter=40,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42,
        refit=True,
    )
    print("\n[tune]  Running RandomizedSearchCV (40 iterations, 5-fold StratifiedKFold)...")
    search.fit(X_train, y_train, verbose=0)
    print(f"[tune]  Best CV ROC-AUC = {search.best_score_:.4f}")
    print(f"[tune]  Best params:\n{json.dumps(search.best_params_, indent=4)}")
    return search.best_estimator_, search.best_params_, search.best_score_


# =============================================================================
# 3.  Evaluate on a split
# =============================================================================
def evaluate(model, X, y, split_name: str, threshold: float = 0.5):
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    acc    = accuracy_score(y, preds)
    f1     = f1_score(y, preds, average="macro")
    f1_cls = f1_score(y, preds, average=None).tolist()
    auc    = roc_auc_score(y, proba)

    print(f"\n── {split_name} Metrics ──────────────────────────────")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Macro : {f1:.4f}  (Dry={f1_cls[0]:.3f}, Rainy={f1_cls[1]:.3f})")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(classification_report(y, preds, target_names=["Dry", "Rainy"]))

    return {
        "accuracy":       round(acc, 4),
        "f1_macro":       round(f1, 4),
        "f1_dry":         round(f1_cls[0], 4),
        "f1_rainy":       round(f1_cls[1], 4),
        "roc_auc":        round(auc, 4),
    }, proba, preds


# =============================================================================
# 4.  Confusion Matrix
# =============================================================================
def plot_confusion_matrix(y_true, y_pred, split_name: str, ax=None):
    cm = confusion_matrix(y_true, y_pred)
    save_here = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Dry (0)", "Rainy (1)"],
                yticklabels=["Dry (0)", "Rainy (1)"],
                ax=ax, linewidths=0.5, linecolor="white",
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title(f"Confusion Matrix — {split_name}", fontweight="bold")

    if save_here:
        path = os.path.join(FIG_DIR, f"03_confusion_matrix_{split_name.lower()}.png")
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {path}")


# =============================================================================
# 5.  ROC + Precision-Recall curves
# =============================================================================
def plot_curves(y_val, proba_val, y_test, proba_test):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for y_true, proba, label, color in [
        (y_val,  proba_val,  "Validation", "#8b47f2"),
        (y_test, proba_test, "Test",        RAINY_CLR),
    ]:
        # ROC
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        axes[0].plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})", color=color, lw=2)

        # PR
        prec, rec, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        axes[1].plot(rec, prec, label=f"{label} (AP={ap:.3f})", color=color, lw=2)

    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate",
                title="ROC Curve")
    axes[0].legend(loc="lower right")

    axes[1].axhline(y=0.5, color="k", linestyle="--", lw=1, alpha=0.5)
    axes[1].set(xlabel="Recall", ylabel="Precision",
                title="Precision–Recall Curve")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "03_roc_pr_curves.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# =============================================================================
# 6.  Native XGBoost Feature Importance
# =============================================================================
def plot_xgb_importance(model, feature_cols: list):
    importance = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(importance["feature"], importance["importance"],
                   color=RAINY_CLR, edgecolor="white")
    ax.set_xlabel("Gain-based Feature Importance (XGBoost)", fontweight="bold")
    ax.set_title("XGBoost Native Feature Importance", fontweight="bold")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "03_xgb_feature_importance.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Colombo Weather — Model Training Pipeline")
    print("=" * 60)

    # 1.  Load
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, spw = load_splits()

    # 2.  Tune
    best_model, best_params, best_cv_auc = tune_model(X_train, y_train, spw)

    # 3.  Evaluate on val & test
    val_metrics,  proba_val,  preds_val  = evaluate(best_model, X_val,   y_val,   "Validation")
    test_metrics, proba_test, preds_test = evaluate(best_model, X_test,  y_test,  "Test")

    # 4.  Plots
    print("\n[plots]  Generating evaluation figures...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_confusion_matrix(y_val,  preds_val,  "Validation", ax=axes[0])
    plot_confusion_matrix(y_test, preds_test, "Test",       ax=axes[1])
    plt.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(FIG_DIR, "03_confusion_matrices.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {cm_path}")

    plot_curves(y_val, proba_val, y_test, proba_test)
    plot_xgb_importance(best_model, feature_cols)

    # 5.  Save model
    model_path = os.path.join(MODEL_DIR, "xgb_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n[save]  Model saved → {model_path}")

    # 6.  Save metrics JSON
    metrics_all = {
        "algorithm":        "XGBoostClassifier",
        "best_cv_roc_auc":  round(best_cv_auc, 4),
        "best_params":      best_params,
        "validation":       val_metrics,
        "test":             test_metrics,
        "feature_cols":     feature_cols,
    }
    metrics_path = os.path.join("reports", "metrics.json")
    os.makedirs("reports", exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics_all, f, indent=2)
    print(f"[save]  Metrics saved → {metrics_path}")

    print("\n✅  Training complete.")
    print(f"    Test  → Accuracy={test_metrics['accuracy']:.3f}  "
          f"F1={test_metrics['f1_macro']:.3f}  ROC-AUC={test_metrics['roc_auc']:.3f}")
