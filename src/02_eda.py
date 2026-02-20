"""
02_eda.py
=========
Exploratory Data Analysis for the Colombo weather dataset.
Generates publication-quality figures and saves them to reports/figures/.

Run from project root:
    python src/02_eda.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Windows console UTF-8 fix
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
PROCESSED   = os.path.join("data", "processed")
RAW_PATH    = os.path.join("data", "daily_colombo_weather.csv")
FIG_DIR     = os.path.join("reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Aesthetics ───────────────────────────────────────────────────────────────
RAINY_CLR = "#1d7fe8"
DRY_CLR   = "#e87d1d"
plt.rcParams.update({
    "figure.dpi":         150,
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})


def _save(fig, filename):
    out = os.path.join(FIG_DIR, filename)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")


# =============================================================================
# 1.  Missing Value Heatmap  (raw data — before imputation)
# =============================================================================
def plot_missing(raw_path):
    df = pd.read_csv(raw_path, parse_dates=["DATE"])
    df = df[["DATE", "PRCP", "TAVG", "TMAX", "TMIN"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    missing_counts = df.isnull().sum()
    missing_pct    = (missing_counts / len(df) * 100).round(1)
    ax = axes[0]
    bars = ax.bar(missing_counts.index, missing_counts.values,
                  color=["#888888" if v == 0 else RAINY_CLR for v in missing_counts.values])
    for bar, pct in zip(bars, missing_pct.values):
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 20, f"{pct}%",
                    ha="center", va="bottom", fontsize=9)
    ax.set_title("Missing Values per Column (raw)", fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Column")

    ax2 = axes[1]
    sample_idx  = np.linspace(0, len(df) - 1, 300, dtype=int)
    null_matrix = df.iloc[sample_idx].isnull().astype(int).T
    sns.heatmap(null_matrix, cmap="Blues", cbar=False, ax=ax2,
                yticklabels=null_matrix.index, xticklabels=False)
    ax2.set_title("Null Pattern (300-row sample)", fontweight="bold")
    ax2.set_xlabel("Records (chronological)")

    plt.tight_layout()
    _save(fig, "eda_01_missing_values.png")


# =============================================================================
# 2.  Weekly Time-Series of All Variables
# =============================================================================
def plot_time_series(wf):
    wf["week_start"] = pd.to_datetime(wf["week_start"])
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    pairs = [
        ("tavg_mean", "Avg Temp (deg C)",           RAINY_CLR),
        ("tmax_mean", "Max Temp (deg C)",             "#f28b47"),
        ("prcp_sum",  "Total Precip (mm/week)",       "#3bbfdc"),
        ("prcp_days", "Rainy Days per Week",           "#8b47f2"),
    ]
    for ax, (col, label, color) in zip(axes, pairs):
        ax.plot(wf["week_start"], wf[col], color=color, linewidth=0.8, alpha=0.85)
        ax.fill_between(wf["week_start"], wf[col], alpha=0.15, color=color)
        ax.set_ylabel(label, fontsize=9)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    axes[-1].set_xlabel("Year")
    fig.suptitle("Colombo Weekly Weather - 18-Year Time Series",
                 fontsize=13, fontweight="bold", y=1.0)
    plt.tight_layout()
    _save(fig, "eda_02_time_series.png")


# =============================================================================
# 3.  Seasonal Box Plots (by month)
# =============================================================================
def plot_seasonal_box(wf):
    wf["week_start"] = pd.to_datetime(wf["week_start"])
    wf["month_name"] = pd.Categorical(
        wf["week_start"].dt.strftime("%b"),
        categories=["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"],
        ordered=True
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    cols_labels = [
        ("tavg_mean", "Avg Temp (deg C)",  RAINY_CLR),
        ("tmax_mean", "Max Temp (deg C)",   "#f28b47"),
        ("prcp_sum",  "Total Precip (mm)",  "#3bbfdc"),
        ("prcp_days", "Rainy Days",          "#8b47f2"),
    ]
    for ax, (col, label, color) in zip(axes.flat, cols_labels):
        sns.boxplot(data=wf, x="month_name", y=col,
                    color=color, ax=ax, width=0.6, linewidth=0.8, fliersize=2)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("Seasonal Patterns - Monthly Box Plots (Weekly Data)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "eda_03_seasonal_boxplots.png")


# =============================================================================
# 4.  Target Class Distribution
# =============================================================================
def plot_target_distribution(wf):
    vc     = wf["Rainy_Next_Week"].value_counts().sort_index()
    labels = ["Dry (0)", "Heavy Rain (1)"]
    colors = [DRY_CLR, RAINY_CLR]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    bars = ax.bar(labels, vc.values, color=colors, width=0.5, edgecolor="white")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5, str(int(bar.get_height())),
                ha="center", va="bottom", fontweight="bold")
    ax.set_title("Target Variable Counts", fontweight="bold")
    ax.set_ylabel("Number of Weeks")

    ax2 = axes[1]
    wedge_props = dict(width=0.45, edgecolor="white")
    _, _, autotexts = ax2.pie(
        vc.values, labels=labels, colors=colors,
        autopct="%1.1f%%", wedgeprops=wedge_props, startangle=90
    )
    for t in autotexts:
        t.set_fontsize(11); t.set_fontweight("bold")
    ax2.set_title("Target Class Balance (donut)", fontweight="bold")

    fig.suptitle("Rainy_Next_Week  (threshold: PRCP > 15 mm/week)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "eda_04_target_distribution.png")


# =============================================================================
# 5.  Correlation Heatmap
# =============================================================================
def plot_correlation(wf):
    with open(os.path.join(PROCESSED, "metadata.json")) as f:
        meta = json.load(f)
    cols = meta["feature_cols"] + [meta["target_col"]]
    corr = wf[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.5, ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Matrix (weekly features + target)",
                 fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    _save(fig, "eda_05_correlation_heatmap.png")


# =============================================================================
# 6.  Feature Distributions by class (violin)
# =============================================================================
def plot_feature_dists(wf):
    num_features = [
        "prcp_sum", "tavg_mean", "tmax_mean", "tmin_mean",
        "prcp_days", "temp_range", "lag1_prcp", "lag2_prcp",
    ]
    # Convert target to string so seaborn can handle hue cleanly
    wf["target_label"] = wf["Rainy_Next_Week"].map({0: "Dry (0)", 1: "Rainy (1)"})

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, col in zip(axes.flat, num_features):
        try:
            sns.violinplot(
                data=wf, x="target_label", y=col,
                order=["Dry (0)", "Rainy (1)"],
                palette={"Dry (0)": DRY_CLR, "Rainy (1)": RAINY_CLR},
                ax=ax, inner="box", cut=0,
            )
        except Exception:
            # Fallback: simple boxplot
            sns.boxplot(
                data=wf, x="target_label", y=col,
                order=["Dry (0)", "Rainy (1)"],
                palette={"Dry (0)": DRY_CLR, "Rainy (1)": RAINY_CLR},
                ax=ax,
            )
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("")

    fig.suptitle("Feature Distributions by Target Class",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "eda_06_feature_distributions.png")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Colombo Weather - EDA Pipeline")
    print("=" * 60)

    wf = pd.read_csv(os.path.join(PROCESSED, "weekly_features.csv"),
                     parse_dates=["week_start"])
    print(f"Loaded weekly_features: {wf.shape}")

    print("\n[1/6]  Missing value plot (raw data)...")
    plot_missing(RAW_PATH)

    print("[2/6]  Time-series plot...")
    plot_time_series(wf.copy())

    print("[3/6]  Seasonal box plots...")
    plot_seasonal_box(wf.copy())

    print("[4/6]  Target class distribution...")
    plot_target_distribution(wf.copy())

    print("[5/6]  Correlation heatmap...")
    plot_correlation(wf.copy())

    print("[6/6]  Feature distributions by class...")
    plot_feature_dists(wf.copy())

    print(f"\n[OK]  All EDA figures saved to {FIG_DIR}/")
