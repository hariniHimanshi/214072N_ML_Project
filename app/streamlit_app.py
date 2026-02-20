"""
streamlit_app.py
================
Interactive front-end for the Colombo Heavy Rain Prediction Model.
Uses the trained XGBoost Classifier to predict next-week heavy rain probability.

Run from project root:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore")

# ── Ensure project root is on path ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED   = os.path.join("data", "processed")
MODEL_PATH  = os.path.join("models", "xgb_model.pkl")
META_PATH   = os.path.join("data", "processed", "metadata.json")


# =============================================================================
# Helpers
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

def build_feature_vector(inputs: dict, feature_cols: list) -> np.ndarray:
    """Convert user inputs dict → numpy array matching model feature order."""
    return np.array([[inputs[c] for c in feature_cols]])


def rain_gauge_html(prob: float) -> str:
    """Returns an animated SVG rain-probability gauge."""
    angle = -90 + (180 * prob)
    color = (
        "#1d7fe8" if prob > 0.65
        else "#f2a647" if prob > 0.35
        else "#4ccc86"
    )
    label = "🌧️ Heavy Rain Likely" if prob > 0.5 else "☀️ Likely Dry"
    pct   = f"{prob*100:.1f}%"
    return f"""
    <div style="text-align:center; margin: 10px 0 20px 0;">
      <svg width="260" height="150" viewBox="0 0 260 150">
        <!-- background arc -->
        <path d="M 30 130 A 100 100 0 0 1 230 130"
              stroke="#dde" stroke-width="22" fill="none" stroke-linecap="round"/>
        <!-- filled arc (approx via rotation) -->
        <g transform="rotate({angle}, 130, 130)">
          <circle cx="130" cy="30" r="11" fill="{color}"/>
        </g>
        <!-- needle -->
        <line x1="130" y1="130"
              x2="{130 + 90 * np.cos(np.radians(angle - 90)):.1f}"
              y2="{130 + 90 * np.sin(np.radians(angle - 90)):.1f}"
              stroke="{color}" stroke-width="4" stroke-linecap="round"/>
        <circle cx="130" cy="130" r="9" fill="#fff" stroke="{color}" stroke-width="3"/>
        <!-- labels -->
        <text x="20"  y="148" font-size="12" fill="#888">0%</text>
        <text x="116" y="20"  font-size="12" fill="#888">50%</text>
        <text x="225" y="148" font-size="12" fill="#888">100%</text>
        <!-- probability text -->
        <text x="130" y="115" text-anchor="middle" font-size="26" font-weight="bold"
              fill="{color}">{pct}</text>
        <text x="130" y="145" text-anchor="middle" font-size="13" fill="#444">{label}</text>
      </svg>
    </div>
    """


# =============================================================================
# Page config & custom CSS
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
.metric-card {
    background: rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px 22px;
    margin: 8px 0;
    border: 1px solid rgba(255,255,255,0.12);
}
.stSlider label { color: #b0c8e4 !important; font-size: 0.88rem; }
.stButton > button {
    background: linear-gradient(90deg, #1d7fe8, #5c3cbf);
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1.6rem;
    font-size: 1rem;
}
.stButton > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Sidebar — user inputs
# =============================================================================
with st.sidebar:
    st.image("https://openweathermap.org/img/wn/10d@2x.png", width=70)
    st.markdown("## 🌦️ Input This Week's Data")
    st.markdown("Enter observed weather values to predict **next week's rain**.")
    st.markdown("---")

    week_of_year  = st.slider("🗓️ Week of Year",          1, 52, 20, help="ISO week number")
    tavg_mean     = st.slider("🌡️ Avg Temp (°C)",         22.0, 34.0, 27.5, 0.1)
    tmax_mean     = st.slider("🔺 Max Temp (°C)",          24.0, 37.0, 31.0, 0.1)
    tmin_mean     = st.slider("🔻 Min Temp (°C)",          20.0, 30.0, 24.5, 0.1)
    prcp_sum      = st.slider("🌧️ This week's total Precip (mm)", 0.0, 250.0, 20.0, 0.5)
    prcp_days     = st.slider("🌂 Rainy days this week",   0, 7, 3)
    lag1_prcp     = st.slider("📅 Last week's total Precip (mm)", 0.0, 250.0, 15.0, 0.5)
    lag2_prcp     = st.slider("📅 Two weeks ago Precip (mm)", 0.0, 250.0, 12.0, 0.5)
    lag1_tavg     = st.slider("🌡️ Last week's Avg Temp (°C)", 22.0, 34.0, 27.0, 0.1)
    lag2_tavg     = st.slider("🌡️ Two weeks ago Avg Temp (°C)", 22.0, 34.0, 27.0, 0.1)
    lag1_prcp_days = st.slider("🌂 Rainy days last week", 0, 7, 3)

    predict_btn = st.button("🔮 Predict Next Week's Rain")


# =============================================================================
# Main Panel
# =============================================================================
col_header, _ = st.columns([3, 1])
with col_header:
    st.markdown("# 🌧️ Colombo Heavy Rain Predictor")
    st.markdown(
        "Predicts whether **next week will be a heavy-rain week** (total precipitation > 15 mm) "
        "using an XGBoost Classifier trained on 18 years of NOAA data."
    )
    st.markdown("---")

# ── Derived features ─────────────────────────────────────────────────────────
week_sin   = np.sin(2 * np.pi * week_of_year / 52)
week_cos   = np.cos(2 * np.pi * week_of_year / 52)
month      = ((week_of_year - 1) // 4) + 1
temp_range = tmax_mean - tmin_mean

inputs = {
    "prcp_sum":       prcp_sum,
    "tavg_mean":      tavg_mean,
    "tmax_mean":      tmax_mean,
    "tmin_mean":      tmin_mean,
    "prcp_days":      prcp_days,
    "temp_range":     temp_range,
    "week_sin":       week_sin,
    "week_cos":       week_cos,
    "month":          month,
    "lag1_prcp":      lag1_prcp,
    "lag2_prcp":      lag2_prcp,
    "lag1_tavg":      lag1_tavg,
    "lag2_tavg":      lag2_tavg,
    "lag1_prcp_days": lag1_prcp_days,
}

# ── Prediction section ────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📈 Historical Trend", "ℹ️ Model Info"])

with tab1:
    if predict_btn or True:   # show gauge by default
        try:
            model = load_model()
            meta  = load_meta()

            X_input = build_feature_vector(inputs, meta["feature_cols"])
            prob    = model.predict_proba(X_input)[0, 1]
            pred    = int(prob >= 0.5)

            col1, col2 = st.columns([1, 1.4])
            with col1:
                # Gauge
                st.markdown(rain_gauge_html(prob), unsafe_allow_html=True)

                result_color = "#1d7fe8" if pred == 1 else "#4ccc86"
                result_text  = "🌧️ Heavy Rain Week" if pred == 1 else "☀️ Dry Week"
                st.markdown(f"""
                <div class="metric-card" style="border-color:{result_color}; text-align:center;">
                  <div style="font-size:1.5rem; font-weight:bold; color:{result_color}">
                    {result_text}
                  </div>
                  <div style="font-size:0.9rem; color:#aaa; margin-top:6px;">
                    Rain probability: <b>{prob*100:.1f}%</b>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                # SHAP waterfall — XGBoost native pred_contribs (SHAP-library-free)
                st.markdown("### 🔍 Why this prediction?")
                try:
                    import xgboost as xgb
                    import matplotlib.patches as mpatches

                    booster  = model.get_booster()
                    dmat     = xgb.DMatrix(
                        X_input.astype(float),
                        feature_names=meta["feature_cols"]
                    )
                    contribs  = booster.predict(dmat, pred_contribs=True)
                    shap_row  = contribs[0, :-1]   # exclude bias
                    bias_val  = contribs[0, -1]

                    # Top 10 by absolute magnitude
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

                    p_pos = mpatches.Patch(color=POS_C, label="Pushes → Rainy")
                    p_neg = mpatches.Patch(color=NEG_C, label="Pushes → Dry")
                    ax.legend(handles=[p_pos, p_neg], fontsize=7.5,
                              facecolor="#0d0d1a", labelcolor="white",
                              loc="lower right")
                    plt.tight_layout()
                    st.pyplot(fig_wf, use_container_width=True)
                    plt.close(fig_wf)
                except Exception as shap_err:
                    st.warning(f"Could not compute SHAP: {shap_err}")

        except FileNotFoundError:
            st.error("⚠️  Model not found. Please run `python src/03_train_model.py` first.")

with tab2:
    st.markdown("### 📈 Historical Weekly Precipitation — Past 18 Years")
    try:
        wf = load_weekly()
        wf["rainy_label"] = wf["Rainy_Next_Week"].map({1: "Heavy Rain", 0: "Dry"})

        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        fig.patch.set_facecolor("#0f0c29")

        for ax in axes:
            ax.set_facecolor("#191744")
            ax.tick_params(colors="#b0c8e4")
            ax.spines[:].set_color("#3a3a6a")

        # ── Top: weekly total PRCP
        ax1 = axes[0]
        rainy_mask = wf["Rainy_Next_Week"] == 1
        ax1.fill_between(wf["week_start"], wf["prcp_sum"],
                         alpha=0.45, color="#3bbfdc", label="PRCP")
        ax1.scatter(wf.loc[rainy_mask, "week_start"],
                    wf.loc[rainy_mask, "prcp_sum"],
                    color="#1d7fe8", s=8, alpha=0.7, label="Heavy Rain Week (>15mm)")
        ax1.set_ylabel("Total PRCP (mm/week)", color="#b0c8e4")
        ax1.legend(loc="upper right", fontsize=8, facecolor="#0f0c29",
                   labelcolor="white")
        ax1.set_title("Colombo Weekly Precipitation History",
                      color="#7ec8f4", fontweight="bold")

        # ── Bottom: Avg temp
        ax2 = axes[1]
        ax2.plot(wf["week_start"], wf["tavg_mean"],
                 color="#f28b47", linewidth=0.9, alpha=0.85, label="Avg Temp")
        ax2.set_ylabel("Avg Temperature (°C)", color="#b0c8e4")
        ax2.set_xlabel("Year", color="#b0c8e4")
        ax2.legend(loc="upper right", fontsize=8, facecolor="#0f0c29",
                   labelcolor="white")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.error(f"Could not load historical data: {e}")

with tab3:
    st.markdown("### 🧠 Model Information")
    try:
        metrics_path = os.path.join("reports", "metrics.json")
        with open(metrics_path) as f:
            mj = json.load(f)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Test Accuracy",  f"{mj['test']['accuracy']*100:.1f}%")
        col_b.metric("Test F1 (Macro)", f"{mj['test']['f1_macro']:.3f}")
        col_c.metric("Test ROC-AUC",   f"{mj['test']['roc_auc']:.3f}")

        st.markdown("#### Best Hyperparameters")
        st.json(mj["best_params"])

        st.markdown("#### Feature List")
        st.write(mj["feature_cols"])

    except FileNotFoundError:
        st.info("Run `python src/03_train_model.py` to populate model metrics here.")

    st.markdown("""
    ---
    **Dataset**: NOAA Climate Data Online — Colombo Observatory (CEM00043466)  
    **Training period**: 2007-09-27 → ~2023 (70% of data)  
    **Algorithm**: XGBoost Classifier — gradient-boosted decision trees  
    **Explainability**: SHAP TreeExplainer (additive feature attribution)  
    """)
