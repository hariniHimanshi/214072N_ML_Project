"""
04_explainability.py
====================
Generates SHAP-based and Partial Dependence Plot (PDP) explanations
for the trained XGBoost Classifier.

Uses XGBoost's native SHAP (pred_contribs=True) for full compatibility
with XGBoost >= 2.x and SHAP >= 0.49.

Outputs (to reports/figures/):
  04_shap_bar_global.png
  04_shap_waterfall_tp.png   (an actual heavy-rain week correctly predicted)
  04_shap_waterfall_tn.png   (an actual dry week correctly predicted)
  04_pdp_top3_features.png
  04_xgb_native_importance.png

Run from project root:
    python src/04_explainability.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Windows console UTF-8 fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED = os.path.join("data", "processed")
MODEL_DIR = "models"
FIG_DIR   = os.path.join("reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
RAINY_CLR  = "#1d7fe8"
DRY_CLR    = "#e87d1d"
POS_SHAP   = "#d73027"   # pushes toward Rainy
NEG_SHAP   = "#4575b4"   # pushes toward Dry


# =============================================================================
# 1.  Load assets
# =============================================================================
def load_assets():
    with open(os.path.join(PROCESSED, "metadata.json")) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    target_col   = meta["target_col"]

    model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    print(f"[load]  Model: {type(model).__name__}")

    df_test  = pd.read_csv(os.path.join(PROCESSED, "test.csv"))
    df_train = pd.read_csv(os.path.join(PROCESSED, "train.csv"))

    X_test  = df_test[feature_cols].values.astype(float)
    X_train = df_train[feature_cols].values.astype(float)
    y_test  = df_test[target_col].values

    return model, X_test, X_train, y_test, feature_cols


# =============================================================================
# 2.  Compute XGBoost native SHAP values
# =============================================================================
def compute_shap(model, X, feature_cols):
    """Use XGBoost's built-in SHAP decomposition (pred_contribs=True).
    Returns shap_values of shape (n_samples, n_features) — bias excluded.
    Also returns bias vector.
    """
    booster = model.get_booster()
    dmat    = xgb.DMatrix(X, feature_names=feature_cols)
    # Last column = bias (expected value)
    contribs  = booster.predict(dmat, pred_contribs=True)     # (n, n_feats+1)
    shap_vals = contribs[:, :-1]                              # (n, n_feats)
    bias      = contribs[:, -1]                               # (n,)
    print(f"[shap]  SHAP values shape: {shap_vals.shape}")
    return shap_vals, bias


# =============================================================================
# 3.  Global SHAP bar chart
# =============================================================================
def plot_shap_bar(shap_vals, feature_cols):
    mean_abs = np.abs(shap_vals).mean(axis=0)
    df_imp   = pd.DataFrame({"feature": feature_cols, "mean_shap": mean_abs})
    df_imp   = df_imp.sort_values("mean_shap", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(df_imp["feature"], df_imp["mean_shap"],
                   color=RAINY_CLR, edgecolor="white")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Mean |SHAP value|  (log-odds contribution)")
    ax.set_title("Global SHAP Feature Importance\n(XGBoost Native TreeSHAP)",
                 fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "04_shap_bar_global.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 04_shap_bar_global.png")

    # Return feature names sorted by importance descending
    return df_imp["feature"].iloc[::-1].tolist()


# =============================================================================
# 4.  SHAP Waterfall plot (manual matplotlib)
# =============================================================================
def _waterfall(shap_row, bias, feature_cols, prob, true_label,
               title, filename):
    """Draw a tidy matplotlib waterfall chart for one prediction."""
    # Sort by absolute SHAP descending, take top 10
    order    = np.argsort(np.abs(shap_row))[::-1][:10]
    feats    = [feature_cols[i] for i in order]
    vals     = shap_row[order]

    # Cumulative "running total" starting from bias (log-odds)
    running = bias
    starts  = []
    ends    = []
    for v in vals:
        starts.append(running)
        running += v
        ends.append(running)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (feat, v, start, end) in enumerate(zip(feats, vals, starts, ends)):
        color = POS_SHAP if v > 0 else NEG_SHAP
        ax.barh(i, abs(v), left=min(start, end), color=color,
                edgecolor="white", height=0.55)
        ax.text(max(start, end) + 0.005, i, f"{v:+.3f}",
                va="center", fontsize=8.5)

    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=9)
    ax.axvline(0, color="#999", linewidth=0.8, linestyle="--")

    patch_pos = mpatches.Patch(color=POS_SHAP, label="Pushes -> Rainy")
    patch_neg = mpatches.Patch(color=NEG_SHAP, label="Pushes -> Dry")
    ax.legend(handles=[patch_pos, patch_neg], loc="lower right", fontsize=8)

    ax.set_xlabel("SHAP value (log-odds contribution)")
    ax.set_title(f"{title}\nTrue label={true_label}  |  P(Rainy)={prob:.2f}",
                 fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, filename)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def plot_waterfalls(model, X_test, y_test, shap_vals, bias, feature_cols):
    proba = model.predict_proba(X_test)[:, 1]

    rainy_ok = np.where((y_test == 1) & (proba > 0.55))[0]
    dry_ok   = np.where((y_test == 0) & (proba < 0.45))[0]

    tp_idx = rainy_ok[0] if len(rainy_ok) > 0 else int(y_test.argmax())
    tn_idx = dry_ok[0]   if len(dry_ok)   > 0 else int((1 - y_test).argmax())

    _waterfall(shap_vals[tp_idx], bias[tp_idx], feature_cols,
               proba[tp_idx], y_test[tp_idx],
               "SHAP Waterfall - Correctly Predicted RAINY Week (True Positive)",
               "04_shap_waterfall_tp.png")

    _waterfall(shap_vals[tn_idx], bias[tn_idx], feature_cols,
               proba[tn_idx], y_test[tn_idx],
               "SHAP Waterfall - Correctly Predicted DRY Week (True Negative)",
               "04_shap_waterfall_tn.png")


# =============================================================================
# 5.  Partial Dependence Plots (manual / no sklearn dep)
# =============================================================================
def plot_pdp(model, X_test, feature_cols, top_features, n_points=60):
    top3   = top_features[:3]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    col_idx = {f: i for i, f in enumerate(feature_cols)}

    for ax, feat in zip(axes, top3):
        fidx  = col_idx[feat]
        fmin  = X_test[:, fidx].min()
        fmax  = X_test[:, fidx].max()
        grid  = np.linspace(fmin, fmax, n_points)
        mean_p = []
        for val in grid:
            X_tmp           = X_test.copy()
            X_tmp[:, fidx]  = val
            mean_p.append(model.predict_proba(X_tmp)[:, 1].mean())

        ax.plot(grid, mean_p, color=RAINY_CLR, linewidth=2)
        ax.fill_between(grid, mean_p, alpha=0.15, color=RAINY_CLR)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.9)
        ax.set_xlabel(feat, fontweight="bold")
        ax.set_ylabel("Avg P(Rainy_Next_Week=1)")
        ax.set_title(f"PDP: {feat}", fontweight="bold")
        ax.set_ylim(0, 1)

    fig.suptitle("Partial Dependence Plots - Top 3 Most Influential Features",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "04_pdp_top3_features.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 04_pdp_top3_features.png")


# =============================================================================
# 6.  Native XGBoost Gain-Based Feature Importance
# =============================================================================
def plot_native_importance(model, feature_cols):
    importance = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(importance["feature"], importance["importance"],
                   color="#8b47f2", edgecolor="white")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Gain-based Importance (XGBoost built-in)")
    ax.set_title("XGBoost Native Feature Importance (Gain)",
                 fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "04_xgb_native_importance.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 04_xgb_native_importance.png")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Colombo Weather - Explainability Pipeline")
    print("=" * 60)

    model, X_test, X_train, y_test, feature_cols = load_assets()

    print("\n[SHAP]  Computing XGBoost native SHAP values...")
    shap_vals, bias = compute_shap(model, X_test, feature_cols)

    print("\n[1/4]  Global SHAP bar chart...")
    top_features = plot_shap_bar(shap_vals, feature_cols)

    print("[2/4]  SHAP Waterfall plots (TP and TN)...")
    plot_waterfalls(model, X_test, y_test, shap_vals, bias, feature_cols)

    print("[3/4]  Partial Dependence Plots (top 3)...")
    plot_pdp(model, X_test, feature_cols, top_features)

    print("[4/4]  Native XGBoost feature importance...")
    plot_native_importance(model, feature_cols)

    print(f"\n[OK]  All explainability figures saved to {FIG_DIR}/")
