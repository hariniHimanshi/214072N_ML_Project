"""
01_preprocessing.py
====================
Preprocesses the raw Colombo daily weather CSV into weekly features
and creates a binary classification target:
    Rainy_Next_Week = 1  if  next week's total PRCP > 15 mm
                     = 0  otherwise

Outputs (saved to data/processed/):
    weekly_features.csv  – full feature-engineered weekly dataset
    train.csv            – chronological 70% split
    val.csv              – chronological 15% split
    test.csv             – chronological 15% split
"""

import os
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# 0.  Paths
# ─────────────────────────────────────────────
RAW_PATH   = os.path.join("data", "daily_colombo_weather.csv")
OUT_DIR    = os.path.join("data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

PRCP_HEAVY_THRESHOLD = 15.0   # mm/week → "heavy rain week"
TMAX_OUTLIER_CAP     = 37.0   # °C cap for sensor outliers


# ─────────────────────────────────────────────
# 1.  Load raw data
# ─────────────────────────────────────────────
def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["DATE"])
    df = df[["DATE", "PRCP", "TAVG", "TMAX", "TMIN"]].copy()
    df = df.sort_values("DATE").reset_index(drop=True)
    print(f"[load]  Raw shape : {df.shape}")
    print(f"[load]  Date range: {df['DATE'].min().date()} → {df['DATE'].max().date()}")
    return df


# ─────────────────────────────────────────────
# 2.  Impute missing values
# ─────────────────────────────────────────────
def impute(df: pd.DataFrame) -> pd.DataFrame:
    before = df.isna().sum()
    print("\n[impute]  Missing before:")
    print(before[before > 0])

    # PRCP: NaN means no measurable precipitation → 0
    df["PRCP"] = df["PRCP"].fillna(0.0)

    # TMAX / TMIN / TAVG: forward-fill + backward-fill using 7-day rolling median
    for col in ["TMAX", "TMIN", "TAVG"]:
        df[col] = df[col].ffill(limit=3).bfill(limit=3)
        df[col] = df[col].fillna(df[col].median())

    after = df.isna().sum()
    print("\n[impute]  Missing after:")
    print(after[after > 0] if after.sum() > 0 else "  → None  ✓")
    return df


# ─────────────────────────────────────────────
# 3.  Cap sensor outliers in TMAX
# ─────────────────────────────────────────────
def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    n_outliers = (df["TMAX"] > TMAX_OUTLIER_CAP).sum()
    print(f"\n[outlier]  TMAX > {TMAX_OUTLIER_CAP}°C records capped: {n_outliers}")
    df["TMAX"] = df["TMAX"].clip(upper=TMAX_OUTLIER_CAP)
    return df


# ─────────────────────────────────────────────
# 4.  Aggregate daily → weekly
# ─────────────────────────────────────────────
def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["DATE"].dt.isocalendar().year.astype(int)
    df["week"] = df["DATE"].dt.isocalendar().week.astype(int)

    # prcp_days = # days per week where it rained (PRCP > 0)
    df["rained_today"] = (df["PRCP"] > 0).astype(int)

    weekly = df.groupby(["year", "week"], as_index=False).agg(
        prcp_sum   =("PRCP",         "sum"),
        tavg_mean  =("TAVG",         "mean"),
        tmax_mean  =("TMAX",         "mean"),
        tmin_mean  =("TMIN",         "mean"),
        prcp_days  =("rained_today", "sum"),
        # keep first DATE of the week for time-ordering
        week_start =("DATE",         "min"),
    )
    weekly = weekly.sort_values("week_start").reset_index(drop=True)
    print(f"\n[weekly]  Weekly shape: {weekly.shape}")
    return weekly


# ─────────────────────────────────────────────
# 5.  Feature engineering
# ─────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # ── Cyclical week-of-year encoding (captures seasonality)
    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)

    # ── Month from week_start
    df["month"] = df["week_start"].dt.month

    # ── Temp range (proxy for cloud cover / humidity)
    df["temp_range"] = df["tmax_mean"] - df["tmin_mean"]

    # ── Lag features: previous 1 and 2 weeks
    df["lag1_prcp"]  = df["prcp_sum"].shift(1)
    df["lag2_prcp"]  = df["prcp_sum"].shift(2)
    df["lag1_tavg"]  = df["tavg_mean"].shift(1)
    df["lag2_tavg"]  = df["tavg_mean"].shift(2)
    df["lag1_prcp_days"] = df["prcp_days"].shift(1)

    # ── TARGET: is next week a heavy rain week?
    df["Rainy_Next_Week"] = (df["prcp_sum"].shift(-1) > PRCP_HEAVY_THRESHOLD).astype(int)

    # Drop rows with NaN introduced by shifts (first 2 rows, last 1 row)
    df = df.dropna().reset_index(drop=True)

    print(f"\n[features]  Shape after engineering: {df.shape}")
    class_dist = df["Rainy_Next_Week"].value_counts()
    print(f"[features]  Target distribution:\n{class_dist}")

    pos = class_dist.get(1, 0)
    neg = class_dist.get(0, 0)
    spw = round(neg / pos, 2) if pos > 0 else 1.0
    print(f"\n[features]  scale_pos_weight (neg/pos) = {spw}  ← use in XGBClassifier")

    return df, spw


# ─────────────────────────────────────────────
# 6.  Chronological train / val / test split
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "prcp_sum", "tavg_mean", "tmax_mean", "tmin_mean",
    "prcp_days", "temp_range",
    "week_sin", "week_cos", "month",
    "lag1_prcp", "lag2_prcp",
    "lag1_tavg", "lag2_tavg",
    "lag1_prcp_days",
]
TARGET_COL = "Rainy_Next_Week"


def chronological_split(df: pd.DataFrame):
    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()

    print(f"\n[split]  Train : {len(train)} rows  "
          f"({train['week_start'].min().date()} → {train['week_start'].max().date()})")
    print(f"[split]  Val   : {len(val)}   rows  "
          f"({val['week_start'].min().date()} → {val['week_start'].max().date()})")
    print(f"[split]  Test  : {len(test)}  rows  "
          f"({test['week_start'].min().date()} → {test['week_start'].max().date()})")

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        rc = split_df[TARGET_COL].value_counts().to_dict()
        print(f"         {split_name} → Rainy=1: {rc.get(1,0)}, Rainy=0: {rc.get(0,0)}")

    return train, val, test


# ─────────────────────────────────────────────
# 7.  Save outputs
# ─────────────────────────────────────────────
def save(weekly: pd.DataFrame, train: pd.DataFrame,
         val: pd.DataFrame, test: pd.DataFrame, spw: float):

    # Save full feature dataset
    weekly.to_csv(os.path.join(OUT_DIR, "weekly_features.csv"), index=False)

    # Save splits (features + target)
    keep_cols = ["week_start"] + FEATURE_COLS + [TARGET_COL]
    train[keep_cols].to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val[keep_cols].to_csv(os.path.join(OUT_DIR, "val.csv"),   index=False)
    test[keep_cols].to_csv(os.path.join(OUT_DIR, "test.csv"),  index=False)

    # Save metadata (scale_pos_weight, feature list) for use by training script
    import json
    meta = {
        "scale_pos_weight": spw,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "prcp_threshold_mm": PRCP_HEAVY_THRESHOLD,
        "train_rows": len(train),
        "val_rows":   len(val),
        "test_rows":  len(test),
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[save]  All files written to  {OUT_DIR}/")
    print(f"        weekly_features.csv, train.csv, val.csv, test.csv, metadata.json")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Colombo Weather — Preprocessing Pipeline")
    print("=" * 60)

    df_raw   = load_raw(RAW_PATH)
    df_imp   = impute(df_raw)
    df_capped = cap_outliers(df_imp)
    df_weekly = aggregate_weekly(df_capped)
    df_feat, scale_pos_weight = feature_engineering(df_weekly)

    train, val, test = chronological_split(df_feat)
    save(df_feat, train, val, test, scale_pos_weight)

    print("\n✅  Preprocessing complete.")
