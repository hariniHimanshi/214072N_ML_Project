"""
streamlit_app.py
================
Interactive front-end for the Colombo Heavy Rain Prediction Model.

User provides ONLY raw weekly weather observations:
  - A date (to derive seasonality automatically)
  - This week's: total rainfall, avg/max/min temp, rainy days
  - Last week's: total rainfall, avg temp, rainy days
  - Two weeks ago: total rainfall, avg temp

All engineered features (temp_range, week_sin, week_cos, month,
lag1_prcp, lag2_prcp, etc.) are computed AUTOMATICALLY inside the app.

Run from project root:
    python -m streamlit run app/streamlit_app.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROCESSED  = os.path.join("data", "processed")
MODEL_PATH = os.path.join("models", "xgb_model.pkl")
META_PATH  = os.path.join("data", "processed", "metadata.json")


# =============================================================================
# Loaders
# =============================================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_meta():
    with open(META_PATH) as f:
        return json.load(f)

@st.cache_data
def load_weekly():
    return pd.read_csv(
        os.path.join(PROCESSED, "weekly_features.csv"),
        parse_dates=["week_start"]
    )


# =============================================================================
# Feature Engineering  (mirrors 01_preprocessing.py logic)
# =============================================================================
def engineer_features(
    input_date,          # datetime.date — the Monday of the current week
    prcp_sum,            # this week's total rainfall (mm)
    tavg_mean,           # this week's average temp (°C)
    tmax_mean,           # this week's max temp (°C)
    tmin_mean,           # this week's min temp (°C)
    prcp_days,           # this week's rainy day count
    lag1_prcp,           # last week's total rainfall (mm)
    lag1_tavg,           # last week's avg temp (°C)
    lag1_prcp_days,      # last week's rainy day count
    lag2_prcp,           # two weeks ago total rainfall (mm)
    lag2_tavg,           # two weeks ago avg temp (°C)
):
    """
    Compute all 14 engineered features from raw user inputs.
    Returns a numpy array compatible with the trained model.
    """
    week_of_year = input_date.isocalendar()[1]
    month        = input_date.month
    week_sin     = np.sin(2 * np.pi * week_of_year / 52)
    week_cos     = np.cos(2 * np.pi * week_of_year / 52)
    temp_range   = tmax_mean - tmin_mean

    # Must match EXACT order in metadata.json / training
    feature_vector = np.array([[
        prcp_sum,
        tavg_mean,
        tmax_mean,
        tmin_mean,
        prcp_days,
        temp_range,      # ← computed automatically
        week_sin,        # ← computed automatically
        week_cos,        # ← computed automatically
        month,           # ← computed automatically
        lag1_prcp,
        lag2_prcp,
        lag1_tavg,
        lag2_tavg,
        lag1_prcp_days,
    ]], dtype=float)
    return feature_vector


# =============================================================================
# SVG Gauge
# =============================================================================
def rain_gauge_html(prob: float) -> str:
    angle = -90 + (180 * prob)
    color = "#1d7fe8" if prob > 0.65 else "#f2a647" if prob > 0.35 else "#4ccc86"
    label = "🌧️ Heavy Rain Likely" if prob > 0.5 else "☀️ Likely Dry"
    pct   = f"{prob*100:.1f}%"
    nx    = 130 + 90 * np.cos(np.radians(angle - 90))
    ny    = 130 + 90 * np.sin(np.radians(angle - 90))
    return f"""
    <div style="text-align:center; margin:10px 0 18px 0;">
      <svg width="260" height="155" viewBox="0 0 260 155">
        <path d="M 30 130 A 100 100 0 0 1 230 130"
              stroke="#dde" stroke-width="22" fill="none" stroke-linecap="round"/>
        <g transform="rotate({angle}, 130, 130)">
          <circle cx="130" cy="30" r="11" fill="{color}"/>
        </g>
        <line x1="130" y1="130" x2="{nx:.1f}" y2="{ny:.1f}"
              stroke="{color}" stroke-width="4" stroke-linecap="round"/>
        <circle cx="130" cy="130" r="9" fill="#fff" stroke="{color}" stroke-width="3"/>
        <text x="20"  y="150" font-size="11" fill="#888">0%</text>
        <text x="114" y="20"  font-size="11" fill="#888">50%</text>
        <text x="220" y="150" font-size="11" fill="#888">100%</text>
        <text x="130" y="115" text-anchor="middle" font-size="26"
              font-weight="bold" fill="{color}">{pct}</text>
        <text x="130" y="148" text-anchor="middle" font-size="12"
              fill="#555">{label}</text>
      </svg>
    </div>"""


# =============================================================================
# Page Config & CSS
# =============================================================================
st.set_page_config(
    page_title="Colombo Rain Forecast",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e8edf3;
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.06);
    border-right: 1px solid rgba(255,255,255,0.1);
}
h1, h2, h3 { color: #7ec8f4 !important; }
.week-section {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    border-left: 3px solid #1d7fe8;
}
.week-label {
    font-size: 0.78rem;
    color: #7ec8f4;
    font-weight: bold;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}
.derived-pill {
    background: rgba(29,127,232,0.18);
    border: 1px solid #1d7fe8;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.78rem;
    color: #7ec8f4;
    display: inline-block;
    margin: 2px 3px;
}
.metric-card {
    background: rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px 22px;
    margin: 8px 0;
    border: 1px solid rgba(255,255,255,0.12);
}
.stButton > button {
    background: linear-gradient(90deg, #1d7fe8, #5c3cbf);
    color: white; font-weight: bold; border: none;
    border-radius: 10px; padding: 0.5rem 1.6rem; font-size: 1rem;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Sidebar — Raw Inputs Only
# =============================================================================
with st.sidebar:
    st.markdown("## 🌦️ Enter Weather Observations")
    st.markdown("Fill in **raw observed values** — the app computes all engineered features automatically.")
    st.markdown("---")

    # ── Date picker (replaces week_of_year slider)
    st.markdown('<div class="week-label">📅 CURRENT WEEK</div>', unsafe_allow_html=True)
    selected_date = st.date_input(
        "Week starting (pick any day)",
        value=datetime.date.today(),
        help="Used to automatically compute: Week of Year, Month, Season (sin/cos)"
    )
    # Compute derived date features and show them as info pills
    week_of_year = selected_date.isocalendar()[1]
    month_val    = selected_date.month
    week_sin     = np.sin(2 * np.pi * week_of_year / 52)
    week_cos     = np.cos(2 * np.pi * week_of_year / 52)

    st.markdown(
        f'<span class="derived-pill">Week {week_of_year}</span>'
        f'<span class="derived-pill">Month {month_val}</span>'
        f'<span class="derived-pill">sin={week_sin:.2f} cos={week_cos:.2f}</span>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ── This week
    st.markdown('<div class="week-label">🌧️ THIS WEEK\'S OBSERVATIONS</div>', unsafe_allow_html=True)
    prcp_sum  = st.number_input("Total Rainfall this week (mm)",  min_value=0.0, max_value=500.0, value=20.0, step=0.5)
    tavg_mean = st.number_input("Average Temperature (°C)",       min_value=18.0, max_value=38.0, value=27.5, step=0.1)
    tmax_mean = st.number_input("Highest daily temp this week (°C)", min_value=18.0, max_value=40.0, value=31.0, step=0.1)
    tmin_mean = st.number_input("Lowest daily temp this week (°C)",  min_value=15.0, max_value=35.0, value=24.5, step=0.1)
    prcp_days = st.slider("Number of rainy days this week", 0, 7, 3)

    # Auto-derived temp_range
    temp_range = tmax_mean - tmin_mean
    st.markdown(
        f'<span class="derived-pill">Temp Range = {temp_range:.1f}°C (auto)</span>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ── Last week
    st.markdown('<div class="week-label">📅 LAST WEEK\'S OBSERVATIONS</div>', unsafe_allow_html=True)
    lag1_prcp      = st.number_input("Total Rainfall last week (mm)",     min_value=0.0, max_value=500.0, value=15.0, step=0.5)
    lag1_tavg      = st.number_input("Average Temperature last week (°C)", min_value=18.0, max_value=38.0, value=27.0, step=0.1)
    lag1_prcp_days = st.slider("Rainy days last week", 0, 7, 3)

    st.markdown("---")

    # ── Two weeks ago
    st.markdown('<div class="week-label">📅 TWO WEEKS AGO</div>', unsafe_allow_html=True)
    lag2_prcp = st.number_input("Total Rainfall 2 weeks ago (mm)",     min_value=0.0, max_value=500.0, value=12.0, step=0.5)
    lag2_tavg = st.number_input("Average Temperature 2 weeks ago (°C)", min_value=18.0, max_value=38.0, value=27.0, step=0.1)

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Next Week's Rainfall")


# =============================================================================
# Main Panel
# =============================================================================
st.markdown("# 🌧️ Colombo Heavy Rain Predictor")
st.markdown(
    "Predicts whether **next week will be a heavy-rain week** (total precipitation > 15 mm) "
    "using an XGBoost Classifier trained on 18 years of NOAA data.  \n"
    "Just enter your **raw weather observations** — all feature engineering happens automatically."
)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📈 Historical Trend", "ℹ️ Model Info"])

# =============================================================================
# Tab 1 — Prediction
# =============================================================================
with tab1:
    try:
        model = load_model()
        meta  = load_meta()

        # ── Compute engineered feature vector automatically
        X_input = engineer_features(
            input_date     = selected_date,
            prcp_sum       = prcp_sum,
            tavg_mean      = tavg_mean,
            tmax_mean      = tmax_mean,
            tmin_mean      = tmin_mean,
            prcp_days      = prcp_days,
            lag1_prcp      = lag1_prcp,
            lag1_tavg      = lag1_tavg,
            lag1_prcp_days = lag1_prcp_days,
            lag2_prcp      = lag2_prcp,
            lag2_tavg      = lag2_tavg,
        )

        prob = model.predict_proba(X_input)[0, 1]
        pred = int(prob >= 0.5)

        col1, col2 = st.columns([1, 1.4])

        with col1:
            st.markdown(rain_gauge_html(prob), unsafe_allow_html=True)

            result_color = "#1d7fe8" if pred == 1 else "#4ccc86"
            result_text  = "🌧️ Heavy Rain Week" if pred == 1 else "☀️ Dry Week"
            st.markdown(f"""
            <div class="metric-card" style="border-color:{result_color}; text-align:center;">
              <div style="font-size:1.5rem; font-weight:bold; color:{result_color}">{result_text}</div>
              <div style="font-size:0.9rem; color:#aaa; margin-top:6px;">
                Rain probability: <b>{prob*100:.1f}%</b>
              </div>
            </div>""", unsafe_allow_html=True)

            # Show the auto-computed features in an expander
            with st.expander("🔧 Auto-computed features (behind the scenes)"):
                derived_df = pd.DataFrame({
                    "Feature":   ["temp_range", "week_of_year", "week_sin", "week_cos", "month"],
                    "Value":     [f"{temp_range:.2f}°C", week_of_year,
                                  f"{week_sin:.4f}", f"{week_cos:.4f}", month_val],
                    "Source":    ["tmax - tmin", "from date picker", "sin(2π×week/52)",
                                  "cos(2π×week/52)", "from date picker"],
                })
                st.dataframe(derived_df, hide_index=True, use_container_width=True)

        with col2:
            st.markdown("### 🔍 Why this prediction?")
            try:
                import xgboost as xgb

                booster  = model.get_booster()
                dmat     = xgb.DMatrix(X_input.astype(float),
                                       feature_names=meta["feature_cols"])
                contribs  = booster.predict(dmat, pred_contribs=True)
                shap_row  = contribs[0, :-1]
                bias_val  = contribs[0, -1]

                order   = np.argsort(np.abs(shap_row))[::-1][:10]
                feats   = [meta["feature_cols"][i] for i in order]
                vals    = shap_row[order]

                running = bias_val
                starts, ends = [], []
                for v in vals:
                    starts.append(running); running += v; ends.append(running)

                fig_wf, ax = plt.subplots(figsize=(8, 5))
                fig_wf.patch.set_facecolor("#0d0d1a")
                ax.set_facecolor("#0d0d1a")
                ax.tick_params(colors="#ccc")
                ax.spines[:].set_color("#444")

                POS_C, NEG_C = "#d73027", "#4575b4"
                for i, (feat, v, s, e) in enumerate(zip(feats, vals, starts, ends)):
                    c = POS_C if v > 0 else NEG_C
                    ax.barh(i, abs(v), left=min(s, e), color=c,
                            edgecolor="none", height=0.55)
                    ax.text(max(s, e) + 0.004, i, f"{v:+.3f}",
                            va="center", fontsize=8, color="#eee")

                ax.set_yticks(range(len(feats)))
                ax.set_yticklabels(feats, fontsize=8.5, color="#ddd")
                ax.axvline(0, color="#888", linewidth=0.8, linestyle="--")
                ax.set_xlabel("SHAP value (log-odds contribution)", color="#aaa")
                ax.set_title("Feature contributions for this prediction",
                             color="#7ec8f4", fontweight="bold", fontsize=10)
                p1 = mpatches.Patch(color=POS_C, label="Pushes → Rainy")
                p2 = mpatches.Patch(color=NEG_C, label="Pushes → Dry")
                ax.legend(handles=[p1, p2], fontsize=7.5,
                          facecolor="#0d0d1a", labelcolor="white", loc="lower right")
                plt.tight_layout()
                st.pyplot(fig_wf, use_container_width=True)
                plt.close(fig_wf)

            except Exception as e:
                st.warning(f"Could not compute SHAP: {e}")

    except FileNotFoundError:
        st.error("⚠️ Model not found. Please run `python src/03_train_model.py` first.")


# =============================================================================
# Tab 2 — Historical Trend
# =============================================================================
with tab2:
    st.markdown("### 📈 Historical Weekly Precipitation — 18 Years")
    try:
        wf = load_weekly()
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        fig.patch.set_facecolor("#0f0c29")
        for ax in axes:
            ax.set_facecolor("#191744")
            ax.tick_params(colors="#b0c8e4")
            ax.spines[:].set_color("#3a3a6a")

        rainy_mask = wf["Rainy_Next_Week"] == 1
        axes[0].fill_between(wf["week_start"], wf["prcp_sum"],
                             alpha=0.45, color="#3bbfdc", label="Weekly PRCP")
        axes[0].scatter(wf.loc[rainy_mask, "week_start"],
                        wf.loc[rainy_mask, "prcp_sum"],
                        color="#1d7fe8", s=8, alpha=0.7, label="Heavy Rain Week (>15mm)")
        axes[0].set_ylabel("Total PRCP (mm/week)", color="#b0c8e4")
        axes[0].legend(loc="upper right", fontsize=8, facecolor="#0f0c29", labelcolor="white")
        axes[0].set_title("Colombo Weekly Precipitation History", color="#7ec8f4", fontweight="bold")

        axes[1].plot(wf["week_start"], wf["tavg_mean"],
                     color="#f28b47", linewidth=0.9, alpha=0.85, label="Avg Temp")
        axes[1].set_ylabel("Avg Temperature (°C)", color="#b0c8e4")
        axes[1].set_xlabel("Year", color="#b0c8e4")
        axes[1].legend(loc="upper right", fontsize=8, facecolor="#0f0c29", labelcolor="white")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.error(f"Could not load historical data: {e}")


# =============================================================================
# Tab 3 — Model Info
# =============================================================================
with tab3:
    st.markdown("### 🧠 Model Information")

    try:
        with open(os.path.join("reports", "metrics.json")) as f:
            mj = json.load(f)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Test Accuracy",   f"{mj['test']['accuracy']*100:.1f}%")
        col_b.metric("Test F1 (Macro)", f"{mj['test']['f1_macro']:.3f}")
        col_c.metric("Test ROC-AUC",    f"{mj['test']['roc_auc']:.3f}")

        st.markdown("#### Best Hyperparameters")
        st.json(mj["best_params"])

        st.markdown("#### What happens when you click Predict")
        st.markdown("""
        | What you enter | What the app computes automatically |
        |---|---|
        | Date picker | `week_of_year`, `month`, `week_sin`, `week_cos` |
        | Max temp − Min temp | `temp_range` |
        | Last week's rainfall | `lag1_prcp` |
        | Two weeks ago rainfall | `lag2_prcp` |
        | Last week's avg temp | `lag1_tavg` |
        | Two weeks ago avg temp | `lag2_tavg` |
        | Rainy days last week | `lag1_prcp_days` |

        All 14 model features are assembled from just **10 raw inputs** you provide.
        """)

    except FileNotFoundError:
        st.info("Run `python src/03_train_model.py` to populate metrics here.")

    st.markdown("""
    ---
    **Dataset**: NOAA Climate Data Online — Colombo Observatory (CEM00043466)  
    **Algorithm**: XGBoost Classifier (gradient-boosted decision trees)  
    **Explainability**: XGBoost native TreeSHAP (additive feature attribution)  
    """)
