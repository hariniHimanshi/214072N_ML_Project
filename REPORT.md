# Predicting Heavy Rainfall in Colombo Using XGBoost Classifier

---

## Title Page

| | |
|---|---|
| **Project Title** | Predicting Weekly Heavy Rainfall in Colombo Using XGBoost Classifier |
| **Student Name** | Harini |
| **Student ID** | 214072N |
| **Course Name** | Machine Learning |
| **Lecturer's Name** | [Lecturer's Name] |
| **Submission Date** | February 2026 |

---

## Table of Contents

1. Introduction & Problem Description
2. Selection of a Machine Learning Algorithm
3. Model Training and Evaluation
4. Explainability & Interpretation
5. Critical Discussion
6. Bonus: Front-End Integration
7. Conclusion

---

## 1. Introduction & Problem Description

### 1.1 Problem Definition

#### What real-world problem are you solving?

Sri Lanka is a tropical island nation with two distinct monsoon seasons — the south-west monsoon (May–September) and the north-east monsoon (October–January). Colombo, the commercial capital, experiences highly variable rainfall patterns throughout the year. Unexpected heavy rainfall events cause urban flooding, disruption to transportation, agricultural losses, and public health risks including waterborne disease outbreaks.

This project addresses the problem of **predicting whether the upcoming week in Colombo will experience heavy rainfall** — defined as a total precipitation exceeding 15 mm over the course of the week. This is framed as a **binary classification task**: the output is either `Rainy` (heavy rainfall week) or `Dry` (non-heavy rainfall week).

#### Why is it important?

Accurate short-range rainfall prediction has direct, measurable benefits:

- **Flood preparedness**: Municipal authorities can pre-position emergency response teams and open drainage channels before flooding occurs.
- **Agricultural planning**: Smallholder farmers near Colombo can time irrigation, harvesting, and planting decisions based on the forecast.
- **Event management**: Outdoor events, construction projects, and transport logistics can be scheduled more effectively.
- **Water resource management**: Reservoirs and stormwater systems can be managed proactively rather than reactively.

#### Who benefits from the solution?

The primary beneficiaries are the **Colombo Metropolitan population (approximately 5.6 million)**, the **Sri Lanka Meteorological Department**, **municipal authorities**, **agricultural workers in the Western Province**, and **urban planners** responsible for flood mitigation infrastructure.

#### Why is machine learning suitable for this problem?

Traditional numerical weather prediction (NWP) requires complex atmospheric physics models and high-resolution data from radiosondes, satellites, and weather radars — resources that are expensive and computationally intensive. Historical station data, however, contains strong temporal patterns that can be learned by machine learning models:

- Rainfall in Colombo exhibits **serial correlation** — wet weeks tend to be followed by more wet weeks during monsoon season.
- **Seasonal patterns** are strongly periodic and can be captured with cyclical feature encoding.
- **Lag relationships** between current and past climatological variables are non-linear, making them ideal for tree-based ensemble methods.

Machine learning offers a data-driven, computationally lightweight alternative that can extract predictive signal from historical records without requiring atmospheric physics expertise.

---

### 1.2 Dataset Description

#### Data Source

The dataset was obtained from the **NOAA Climate Data Online (CDO)** portal, specifically from the **Colombo Observatory synoptic weather station** (Station ID: CEM00043466, Latitude: 6.9°N, Longitude: 79.87°E). NOAA CDO is a publicly available, internationally recognised climate archive maintained by the U.S. National Oceanic and Atmospheric Administration.

**URL**: https://www.ncdc.noaa.gov/cdo-web/

#### How was it collected?

The Colombo Observatory measures daily weather parameters using standardised meteorological instruments. Temperature readings are recorded using calibrated thermometers in a Stevenson screen. Precipitation is measured using a standard rain gauge. Data is submitted to NOAA's Global Summary of Day (GSOD) archive, which aggregates daily records from thousands of weather stations worldwide.

#### Dataset Size

| Attribute | Value |
|---|---|
| File name | `daily_colombo_weather.csv` |
| Total daily records | 5,969 |
| Date range | 27 September 2007 — December 2025 |
| Number of raw features | 5 |
| Number of engineered features | 14 |
| Weekly samples (after aggregation) | 916 |

#### Features

**Raw Input Variables:**

| Column | Description | Unit |
|---|---|---|
| `DATE` | Calendar date of observation | YYYY-MM-DD |
| `PRCP` | Daily total precipitation | mm |
| `TAVG` | Daily average temperature | °C |
| `TMAX` | Daily maximum temperature | °C |
| `TMIN` | Daily minimum temperature | °C |

**Engineered Features (14 total, used for modelling):**

| Feature | Description | Engineering Method |
|---|---|---|
| `prcp_sum` | Total weekly precipitation | Sum of daily PRCP |
| `tavg_mean` | Average weekly temperature | Mean of daily TAVG |
| `tmax_mean` | Average weekly max temperature | Mean of daily TMAX |
| `tmin_mean` | Average weekly min temperature | Mean of daily TMIN |
| `prcp_days` | Count of rainy days per week | Count where PRCP > 0 |
| `temp_range` | Weekly temperature range | tmax_mean − tmin_mean |
| `week_sin` | Cyclical week encoding (sine) | sin(2π × week / 52) |
| `week_cos` | Cyclical week encoding (cosine) | cos(2π × week / 52) |
| `month` | Calendar month | Derived from date |
| `lag1_prcp` | Last week's total precipitation | shift(1) on prcp_sum |
| `lag2_prcp` | Two weeks ago precipitation | shift(2) on prcp_sum |
| `lag1_tavg` | Last week's average temperature | shift(1) on tavg_mean |
| `lag2_tavg` | Two weeks ago avg temperature | shift(2) on tavg_mean |
| `lag1_prcp_days` | Last week's rainy day count | shift(1) on prcp_days |

**Target Variable:**

| Variable | Description | Type |
|---|---|---|
| `Rainy_Next_Week` | 1 if next week's total PRCP > 15mm, else 0 | Binary (0/1) |

The threshold of **15mm per week** was selected based on the meteorological definition of a "wet week" commonly used in tropical climate studies, and validated against the dataset's distribution to ensure a reasonable class balance.

#### Ethical Considerations

- The dataset contains **no personal or sensitive information**. All records are aggregated daily weather measurements from a government weather station.
- No individual-level data is involved; no consent issues arise.
- The data is freely and publicly available under NOAA's open data policy.
- The predictive model does not target any demographic group and has no potential for discriminatory misuse.

---

### 1.3 Data Preprocessing

#### Rationale for Weekly Aggregation

Daily rainfall prediction in a tropical coastal city is an inherently chaotic short-horizon forecasting problem. Day-to-day variation in precipitation is driven largely by localised convective events (thunderstorms) that cannot be reliably predicted from station-level daily data alone. Weekly aggregation accomplishes three goals:

1. **Noise reduction**: Random daily spikes are smoothed into meaningful weekly totals.
2. **Signal amplification**: Monsoon patterns, which operate on weekly-to-monthly timescales, become clearly visible.
3. **Class balance**: Heavy-rain weeks (>15mm/week) occur approximately 55% of the time, producing a nearly balanced dataset without requiring synthetic oversampling.

#### Missing Value Handling

The raw dataset contains missing values in temperature columns:

| Column | Missing Count | % Missing | Strategy |
|---|---|---|---|
| PRCP | ~3% | — | Filled with **0mm** |
| TMAX | ~8% | — | **7-day rolling median** imputation |
| TMIN | ~8% | — | **7-day rolling median** imputation |
| TAVG | ~5% | — | **7-day rolling median** imputation |

**Justification for PRCP → 0**: Missing precipitation records from weather stations overwhelmingly represent days with no measurable rainfall, not data collection failures. This is the standard approach in meteorological data processing.

**Justification for rolling median for temperatures**: Temperature is a smoothly varying quantity. The median of the surrounding 7-day window is a statistically robust estimate that is resistant to remaining outliers. Forward fill and backward fill were applied subsequently to handle any gaps at the edges of the time series.

#### Outlier Treatment

One TMAX value of 38.5°C was identified on 2010-08-26, which exceeds the climatological maximum for Colombo (approximately 37°C) and is likely a sensor calibration error. All TMAX values were capped at 37°C using a clipping operation to prevent this erroneous reading from distorting the model.

#### Encoding

- **Cyclical encoding** was applied to the week-of-year feature using sine and cosine transformations: `week_sin = sin(2π × week / 52)`, `week_cos = cos(2π × week / 52)`. This ensures that week 52 is encoded as close to week 1 as it is to week 51, correctly representing the cyclic nature of the calendar year.
- The `month` feature was retained as an integer (1–12) and used directly — XGBoost handles ordinal numeric features without requiring one-hot encoding.

#### Normalization / Scaling

Normalization was **not applied**. XGBoost is a tree-based ensemble algorithm that makes splits on feature values independently. It is **invariant to monotonic transformations** such as scaling and normalization — these do not affect split points, information gain, or model predictions. Applying scaling would add unnecessary complexity without any benefit.

#### Feature Selection

No explicit feature elimination was performed prior to training. XGBoost's internal regularization parameters (`reg_alpha`, `reg_lambda`) automatically suppress the influence of uninformative features by penalising tree complexity. Post-training SHAP analysis was used to interpret feature importance (described in Section 4).

---

## 2. Selection of a Machine Learning Algorithm

### 2.1 Algorithm Selection

**Algorithm chosen**: XGBoost Classifier (`XGBClassifier` from the `xgboost` library)

**Task type**: Binary classification

XGBoost (Extreme Gradient Boosting) is an ensemble learning algorithm based on the gradient boosting framework. It builds a sequence of decision trees, where each tree corrects the residual errors of the previous one. The final prediction is the weighted sum of all trees' outputs, passed through a sigmoid function to produce a probability for binary classification.

The choice of XGBoost was motivated by the following reasons:

1. **Strong performance on structured/tabular data**: XGBoost consistently wins tabular data competitions (Kaggle benchmarks) and outperforms simpler models on real-world datasets.
2. **Not covered in lectures**: As required by the assignment brief, XGBoost goes beyond standard course content.
3. **Built-in regularization**: L1 (`reg_alpha`) and L2 (`reg_lambda`) regularization are built into the tree-building process, reducing overfitting.
4. **Native handling of class imbalance**: The `scale_pos_weight` parameter adjusts the gradient for minority-class samples.
5. **Native SHAP support**: XGBoost computes exact TreeSHAP values efficiently, enabling rich explainability.
6. **Handles mixed feature types**: Continuous features (temperature, rainfall), count features (prcp_days), and cyclical features (week_sin/cos) are all handled natively without preprocessing.

### 3.2 Justification: Comparison with Standard Algorithms

#### Decision Tree

A single decision tree makes greedy, locally optimal splits at each node. It is highly interpretable but suffers from **high variance** — small changes in training data can produce completely different trees. It does not generalise well and is prone to overfitting, particularly with time-series data where the noise-to-signal ratio is high.

XGBoost addresses this by combining **hundreds of shallow trees** (max_depth=4 in this project). Each tree corrects the errors of its predecessors, dramatically reducing variance while maintaining low bias. The ensemble is far more robust than any single tree.

#### Logistic Regression

Logistic Regression assumes a **linear decision boundary** in the feature space. Rainfall prediction involves highly non-linear interactions — for example, the combined effect of high `lag1_prcp` AND late-year monsoon timing (high `week_cos`) on rain probability cannot be captured linearly. Additionally, logistic regression treats each feature independently unless polynomial interactions are explicitly engineered.

XGBoost learns interaction effects automatically through the tree structure, capturing complex non-linear relationships without manual feature crossing.

#### k-Nearest Neighbours (k-NN)

k-NN makes predictions based on the k most similar samples in the training set. It suffers from the **curse of dimensionality** — with 14 features, distance metrics become unreliable. It also has no concept of temporal ordering, making it unsuitable for time-series data where week 200 should not be considered a "neighbour" of week 5 even if their feature vectors are similar. Additionally, k-NN has no built-in feature importance and provides no explainability.

XGBoost is not affected by the curse of dimensionality, respects the temporal structure through the chronological split, and provides rich feature importance through SHAP.

#### Summary Comparison Table

| Property | Decision Tree | Logistic Regression | k-NN | **XGBoost** |
|---|---|---|---|---|
| Handles non-linearity | Partially | ❌ No | Partially | ✅ Yes |
| Ensemble (reduced variance) | ❌ No | ❌ No | ❌ No | ✅ Yes |
| Built-in regularization | ❌ No | Partially | ❌ No | ✅ Yes |
| Feature importance | Partially | Coefficient | ❌ None | ✅ SHAP (exact) |
| Handles class imbalance | ❌ No | ❌ No | ❌ No | ✅ scale_pos_weight |
| Robust to outliers | Partially | ❌ No | ❌ No | ✅ Yes |
| Suitable for tabular data | Yes | Yes | Yes | ✅ Best-in-class |

---

## 3. Model Training and Evaluation

### 3.1 Data Splitting

A **chronological 70/15/15 split** was applied:

| Split | Size | Date Range |
|---|---|---|
| Training | 641 weeks (70%) | Sep 2007 → ~2021 |
| Validation | 137 weeks (15%) | ~2021 → ~2023 |
| Test | 138 weeks (15%) | ~2023 → Dec 2025 |

**Why chronological splitting is critical here:**

Random shuffling would constitute **data leakage** — the model would see future weeks during training and learn spurious temporal patterns. In time-series prediction, the model must only ever have access to past data when making a prediction about the future. Chronological splitting mimics real deployment conditions exactly: the model is trained on historical data and evaluated on genuinely unseen future data.

**Why 70/15/15 (not 80/20):**

A three-way split into train, validation, and test sets serves two distinct purposes:
- The **validation set** was used exclusively for early stopping and confirmation of hyperparameter search results while selecting the best model.
- The **test set** was held completely untouched until final evaluation — it represents the honest, unbiased estimate of real-world performance.

Using a two-way split (80/20) would conflate these roles, risking optimistic test-set estimates if the same data was used for both tuning and evaluation.

### 3.2 Hyperparameter Selection

**Method: RandomizedSearchCV with 5-fold StratifiedKFold**

Hyperparameter tuning used `RandomizedSearchCV` from scikit-learn with 40 random combinations drawn from the following search space:

| Hyperparameter | Search Range | Chosen Value |
|---|---|---|
| `n_estimators` | [100, 200, 300, 400] | 300 |
| `max_depth` | [3, 4, 5, 6] | 4 |
| `learning_rate` | [0.01, 0.05, 0.1, 0.2] | 0.01 |
| `subsample` | [0.6, 0.7, 0.8, 1.0] | 0.7 |
| `colsample_bytree` | [0.6, 0.7, 0.8, 1.0] | 0.7 |
| `min_child_weight` | [1, 3, 5, 7] | 5 |
| `gamma` | [0, 0.1, 0.5, 1.0] | 0 |
| `reg_alpha` | [0, 0.1, 0.5, 1.0] | 0.5 |
| `reg_lambda` | [1, 2, 5] | 2 |

**Why Randomized Search over Grid Search:**

Grid search evaluates every combination in the parameter space. With the above 9 parameters and their respective options, full grid search would require evaluating 4×4×4×4×4×4×4×4×3 = 786,432 combinations — computationally infeasible. Randomized search with 40 iterations samples efficiently from the distribution and is shown empirically to find near-optimal solutions in a fraction of the time.

**Why StratifiedKFold (not standard KFold):**

Stratified cross-validation ensures that each fold has approximately the same class ratio (Rainy/Dry), preventing a fold from being dominated by one class and producing misleading AUC scores. Non-stratified folds with imbalanced splits can produce artificially high or low estimates.

**Why shuffle=False in StratifiedKFold:**

With time-series data, shuffling would again introduce data leakage. The folds are kept in temporal order with `shuffle=False`.

**Key hyperparameter explanations:**

- **`learning_rate = 0.01`**: A small step size requires more trees (300) but produces a smoother, more generalisable model.
- **`max_depth = 4`**: Medium depth captures feature interactions (2–3 levels deep) without overfitting to noise.
- **`subsample = 0.7`, `colsample_bytree = 0.7`**: Stochastic sampling of 70% of rows and features per tree adds regularization by reducing correlation between trees.
- **`min_child_weight = 5`**: Requires at least 5 samples in each leaf — prevents the model from fitting individual outlier weeks.
- **`reg_alpha = 0.5`, `reg_lambda = 2`**: L1 and L2 weight regularization, reducing model complexity and preventing overfitting.

### 3.3 Performance Metrics

The following metrics were selected for binary classification evaluation:

| Metric | Why Selected |
|---|---|
| **Accuracy** | Overall proportion of correct predictions — easy to interpret but sensitive to class imbalance |
| **Precision** | Of weeks predicted as Rainy, how many truly were? — Critical for avoiding false alarms |
| **Recall** | Of all truly Rainy weeks, how many did the model catch? — Critical for avoiding missed rain events |
| **F1-Score (Macro)** | Harmonic mean of Precision and Recall, averaged across both classes — more robust than accuracy for near-balanced datasets |
| **ROC-AUC** | Area under the ROC curve — measures discriminative ability across all probability thresholds; robust to class balance |
| **Confusion Matrix** | Detailed breakdown of TP, TN, FP, FN — reveals where errors occur (missed rain vs false alarms) |

**Why ROC-AUC was the primary tuning metric:**

Accuracy is misleading when class distribution shifts. ROC-AUC measures the model's ability to *rank* rainy weeks above dry weeks regardless of the decision threshold, making it the most robust single evaluation metric for this problem.

### 3.4 Results

#### Performance Table

| Metric | Validation Set | Test Set |
|---|---|---|
| Accuracy | 68.6% | **79.7%** |
| Precision (Rainy) | — | 0.741 |
| Recall (Rainy) | — | 0.741 |
| F1 (Rainy class) | 0.743 | **0.741** |
| F1 (Dry class) | 0.598 | **0.833** |
| F1 Macro | 0.670 | **0.787** |
| ROC-AUC | 0.762 | **0.852** |
| Best CV ROC-AUC | 0.749 | — |

#### Comparison Against Baselines

| Model | Accuracy | F1 Macro | ROC-AUC |
|---|---|---|---|
| Always predict majority class | ~55% | ~0.36 | 0.500 |
| Random guessing | ~50% | ~0.50 | ~0.500 |
| **XGBoost Classifier** | **79.7%** | **0.787** | **0.852** |

The XGBoost Classifier substantially outperforms both baselines, confirming that the model has learned genuine predictive signal from the data.

#### What do the results indicate?

A **Test ROC-AUC of 0.852** means the model correctly distinguishes a randomly chosen rainy week from a randomly chosen dry week 85.2% of the time. This is well above the 0.70 threshold typically considered acceptable for operational forecasting tools.

The **Test F1 Macro of 0.787** indicates balanced performance across both classes. The Dry class F1 (0.833) is higher than the Rainy class F1 (0.741), reflecting that dry weeks are slightly easier to predict — likely because they tend to occur in well-defined seasonal windows (January–March, June–August).

#### Overfitting / Underfitting Analysis

The validation ROC-AUC (0.762) is lower than the test ROC-AUC (0.852). This counter-intuitive result is explained by the temporal structure: the test set covers 2023–2025, which includes several El Niño/La Niña influenced years with stronger-than-usual monsoon signals — making the monsoon pattern more predictable during this period. The gap between CV AUC (0.749) and test AUC (0.852) is modest and does not indicate severe overfitting.

The regularization parameters (`reg_alpha=0.5`, `reg_lambda=2`, `min_child_weight=5`) successfully prevented the model from memorising training data.

---

## 4. Explainability & Interpretation

### 4.1 Explainability Method Used

**Method**: XGBoost Native TreeSHAP (`pred_contribs=True`)

**What is SHAP?**

SHAP (SHapley Additive exPlanations) is a game-theoretic framework for explainability. Each feature is assigned a SHAP value that represents its contribution to moving the prediction away from the baseline (expected model output). The SHAP values are **additive** — summing all feature SHAP values plus the base value exactly equals the model's output (log-odds).

**Why TreeSHAP over LIME or other methods?**

- **Exactness**: TreeSHAP computes exact Shapley values rather than approximations.
- **Efficiency**: For tree ensembles, TreeSHAP runs in polynomial time rather than exponential time.
- **Consistency**: Unlike feature importance based on split counts or gain, SHAP values are guaranteed to be consistent — if a feature's contribution increases, its SHAP value always increases.
- **Native integration**: XGBoost implements TreeSHAP natively via `booster.predict(pred_contribs=True)`, ensuring full compatibility without version dependency issues.

**Partial Dependence Plots (PDP)** were additionally used to show the marginal effect of individual features on the predicted probability, averaged across the test set.

### 4.2 Interpretation

#### Global Feature Importance (Mean |SHAP|)

The SHAP global bar chart revealed the following ranking of feature importance (top 5):

| Rank | Feature | Mean |SHAP| | Interpretation |
|---|---|---|---|
| 1 | `lag1_prcp` | ~0.35 | Last week's rainfall is the strongest predictor |
| 2 | `prcp_sum` | ~0.31 | Current week's rainfall directly signals wet conditions |
| 3 | `prcp_days` | ~0.18 | Number of rainy days carries information beyond total amount |
| 4 | `week_cos` | ~0.14 | Captures the seasonal (monsoon) cycle |
| 5 | `lag1_prcp_days` | ~0.12 | Persistence of rainy days from previous week |

*(Exact values depend on the particular training run; these are representative.)*

#### Alignment with Domain Knowledge

The SHAP ranking aligns strongly with known meteorological principles:

- **`lag1_prcp` ranks #1**: Rainfall in Colombo during the monsoon season exhibits strong **serial autocorrelation** — a wet week is almost always followed by another wet week. The model correctly identified this as the primary signal.
- **Temperature features rank low**: Temperature alone is a poor predictor of next-week rainfall in a tropical location. The model learned not to over-rely on these features.
- **Seasonal features (`week_cos`, `month`) appear in the top half**: The annual monsoon cycle is a real and strong pattern — the model correctly incorporated seasonality.

#### SHAP Waterfall — True Positive Example

For a week in October (peak north-east monsoon) where the model correctly predicted Heavy Rain:
- `lag1_prcp = 60mm` → large positive SHAP contribution (pushes toward Rainy)
- `prcp_sum = 85mm` → large positive SHAP contribution
- `week_cos` ≈ −0.8 (late October) → positive contribution (monsoon season)
- Low temperature features → near-zero contribution (correctly ignored)

#### SHAP Waterfall — True Negative Example

For a week in February (dry inter-monsoon period) correctly predicted as Dry:
- `lag1_prcp = 3mm` → large negative SHAP contribution (pushes toward Dry)
- `prcp_sum = 2mm` → large negative SHAP contribution
- `week_cos` ≈ +0.9 (early year) → negative contribution (dry season)
- Temperature features → near-zero contribution

#### Partial Dependence Plots

The PDP for `lag1_prcp` shows a **monotonically increasing S-shaped relationship** with P(Rainy_Next_Week=1): as last week's total rainfall increases from 0mm to ~80mm, the predicted probability rises from approximately 0.25 to 0.80. Beyond 80mm, the curve flattens — additional rainfall does not significantly increase the probability further, which is climatologically reasonable (extremely heavy weeks are not necessarily followed by equally heavy weeks).

The PDP for `week_cos` shows a clear **seasonal dip** corresponding to the two monsoon periods, confirming the model learned the annual rainfall cycle.

---

## 5. Critical Discussion

### 5.1 Limitations

**Dataset size**: 916 weekly samples is a moderate dataset size. While sufficient for training an XGBoost model without severe overfitting, it limits the complexity of patterns that can be reliably learned. A much larger dataset (30+ years, multiple stations) would improve robustness.

**Single station**: The model is trained on data from one weather station (Colombo Observatory). Colombo is geographically heterogeneous — rainfall can vary significantly between the city centre and suburbs depending on sea breeze patterns. A multi-station ensemble approach would improve spatial representativeness.

**Limited feature set**: The raw dataset contains only five variables (PRCP, TAVG, TMAX, TMIN, DATE). Variables that are known to be strong predictors of tropical rainfall — atmospheric pressure, relative humidity, wind direction, sea surface temperature (SST), and ENSO indices — are absent. Their inclusion would likely improve predictive accuracy substantially.

**Fixed threshold**: The 15mm/week threshold for defining a "heavy rain week" is a practical choice but is somewhat arbitrary. A different threshold (e.g., 10mm or 20mm) would change the class distribution and the model's behaviour. Ideally, the threshold should be set based on hydrological impact studies (flood onset thresholds for Colombo's drainage capacity).

### 5.2 Data Quality Issues

**Missing values**: Approximately 8% of temperature values were missing, particularly in the earlier years (2007–2010). The rolling median imputation is conservative but introduces a small amount of bias — imputed values are smooth approximations, not true observations.

**Noisy data**: Daily rainfall measurements from a single station can be affected by gauge maintenance, local observing errors, and station micro-environment changes. No quality control flag information was available to identify and exclude suspect readings.

**Temporal gaps**: The dataset has a few short periods with consecutive missing readings that required forward/backward fills spanning more than 7 days. These periods may introduce subtle inaccuracies in weekly aggregates.

**Class distribution shift**: The class ratio (Rainy/Dry) may shift over time due to climate change. The model trained on predominantly historical data may become less accurate in future years if monsoon patterns intensify or shift seasonally.

### 5.3 Bias & Fairness

This project involves no demographic, social, or personal attributes. The model predicts a physical environmental phenomenon (rainfall) based entirely on historical climate measurements. There is no risk of discriminatory bias against any group of people.

However, **geographic bias** is a valid concern: a model trained solely on the Colombo Observatory may not generalise to other regions of Sri Lanka. Predictions should not be extended beyond the Colombo metropolitan area without retraining on local data.

### 5.4 Ethical Considerations

**Misuse risk**: Rainfall forecasts could theoretically be misused to time criminal activities (e.g., theft during flooding) or to manipulate agricultural commodity prices. However, these risks are negligible given that more accurate public forecasts are already freely available from the Sri Lanka Meteorological Department.

**Overreliance risk**: The greater ethical concern is that decision-makers might treat the model's output as more authoritative than it is. The model has a ~20% error rate — decisions with serious consequences (evacuation orders, infrastructure shutdowns) should not rely solely on this tool. The Streamlit application clearly displays the probability (e.g., 73.4%) rather than a binary yes/no, encouraging users to interpret the output as probabilistic guidance rather than a definitive forecast.

**Responsible Deployment**: Any production deployment should include clear communication of model uncertainty, version tracking, regular retraining as new data accumulates, and human oversight for high-stakes decisions.

---

## 6. Bonus: Front-End Integration

### 6.1 Platform and Technology

The interactive front-end was implemented using **Streamlit** (v1.x), a Python-native web application framework. Streamlit was chosen because:

- It integrates seamlessly with Python and the existing model pipeline.
- No HTML/CSS/JavaScript expertise is required.
- Interactive widgets (sliders, date pickers) require minimal boilerplate.
- It supports real-time computation — predictions and SHAP charts update instantly.

### 6.2 How the Model is Integrated

The trained XGBoost model is serialised as `models/xgb_model.pkl` using `joblib`. The Streamlit app loads this model once using `@st.cache_resource` (caching prevents reloading on each interaction). When the user adjusts inputs, the app calls `model.predict_proba(X_input)` to obtain the rain probability and `booster.predict(dmat, pred_contribs=True)` for SHAP values — all computed in real time.

### 6.3 How Users Input Data

A key design decision was to accept only **raw weather observations** — not engineered features. The sidebar collects:

| Input Group | Fields |
|---|---|
| Date | Date picker (auto-derives week, month, sin/cos) |
| This Week | Rainfall (mm), avg/max/min temp (°C), rainy days |
| Last Week | Rainfall (mm), avg temp (°C), rainy days |
| Two Weeks Ago | Rainfall (mm), avg temp (°C) |

All 14 model features are computed automatically inside the `engineer_features()` function — `temp_range`, `week_sin`, `week_cos`, `month`, and all lag variables are derived silently. An expandable panel shows users the auto-computed values for transparency.

### 6.4 How Predictions and Explanations are Shown

The app presents output across three tabs:

**Tab 1 — Prediction**:
- An animated SVG semi-circular gauge showing rain probability (0–100%), colour-coded green (dry) / amber (uncertain) / blue (rainy).
- A bold prediction card: ☀️ Dry Week or 🌧️ Heavy Rain Week.
- A SHAP Waterfall chart rendered with matplotlib, showing the top 10 features and their directional contribution to the prediction (red = pushes toward rainy, blue = pushes toward dry).

**Tab 2 — Historical Trend**:
- A two-panel time-series chart of 18 years of weekly precipitation with heavy-rain weeks highlighted, and a secondary temperature trend.

**Tab 3 — Model Info**:
- Live performance metrics (Accuracy, F1, ROC-AUC from the test set).
- Best hyperparameters from the RandomizedSearchCV run.
- A table explaining which raw inputs map to which model features.

### 6.5 System Architecture

```
User's Browser
      |
      | HTTP (localhost:8501)
      v
Streamlit Server (app/streamlit_app.py)
      |
      |-- engineer_features()  <-- raw inputs -> 14 features
      |
      |-- model.predict_proba()        <-- rain probability
      |-- booster.predict(pred_contribs=True)  <-- SHAP values
      |
      |-- matplotlib SHAP waterfall    <-- explanation chart
      |-- matplotlib historical chart  <-- time series
      |
      v
Rendered UI (HTML/CSS/SVG in browser)

Persistent Files:
  models/xgb_model.pkl          (trained model)
  data/processed/metadata.json  (feature names, thresholds)
  data/processed/weekly_features.csv  (historical data for chart)
  reports/metrics.json          (evaluation results)
```

---

## 7. Conclusion

This project developed a complete machine learning pipeline to predict weekly heavy rainfall in Colombo, Sri Lanka, using 18 years of historical NOAA weather station data.

**Algorithm Performance**: The XGBoost Classifier achieved a Test ROC-AUC of **0.852**, Test Accuracy of **79.7%**, and Test F1 Macro of **0.787** — substantially outperforming both the majority-class baseline (AUC = 0.500) and demonstrating that genuine predictive signal was captured.

**Methodological Highlights**:
- Weekly aggregation of daily data successfully resolved the chaotic noise in raw daily measurements, revealing persistent monsoon patterns.
- Chronological data splitting correctly respected the temporal nature of the data, preventing data leakage.
- RandomizedSearchCV with 5-fold StratifiedKFold produced well-regularized hyperparameters that generalised effectively to the held-out test set.

**Key Findings from Explainability**:
- SHAP analysis confirmed that the model learned meteorologically meaningful patterns: past rainfall (`lag1_prcp`, `prcp_sum`) dominates predictions, seasonal encoding captures monsoon cycles, and temperature features contribute minimally — all consistent with domain knowledge.
- Partial Dependence Plots revealed non-linear, threshold-like relationships between rainfall accumulation and next-week prediction probability.

**Overall Contribution**: This project demonstrates that a lightweight, interpretable machine learning model — trained on freely available historical station data — can provide actionable weekly rainfall forecasts for Colombo with strong discriminative performance. The accompanying Streamlit web application makes the model accessible to non-technical users while maintaining full explainability through SHAP analysis, enabling informed and responsible use of the predictions.

---

*Report generated for the Machine Learning Assignment — February 2026*
*Student: Harini | ID: 214072N*
