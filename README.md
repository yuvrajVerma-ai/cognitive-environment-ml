# ML & DL Based Models to Predict Effects of Light and Noise on Cognitive Tasks

## 1. Problem Statement
Environmental conditions such as light and noise can influence human cognitive
performance during attention/working-memory tasks. This project builds and
compares machine learning and deep learning models that predict reaction time
on an N-back style cognitive task, given the environmental conditions and task
difficulty under which the trial occurred.

## 2. Motivation
Understanding how environmental context affects cognitive performance is
relevant to workplace/classroom design, human-machine interface adaptation,
and general human factors research. This project demonstrates an end-to-end
ML/DL pipeline for this type of contextual, human-performance prediction
problem.

## 3. Objectives
- Build a clean, leakage-free ML/DL pipeline for predicting reaction time
  from environmental and task-difficulty features.
- Compare classical ML models (Linear Regression, Random Forest, XGBoost)
  against a simple ANN.
- Identify which environmental factors matter most using feature importance.
- Package the best model for use in a Streamlit dashboard.

## 4. Dataset -- IMPORTANT DISCLOSURE
This project was originally scoped around the IMPACT dataset (Markov et al.,
2024, "IMPACT: A Dataset for Integrated Measurement of Performance and
Contextual Task-related effects"). That dataset is **request-only** from the
original researchers (Technical University of Varna) and could not be
obtained within the project timeline.

**A synthetic dataset was generated instead** (`src/generate_dataset.py`),
using effect directions drawn from published literature on light and noise
effects on cognitive performance:
- Noise is modeled with a roughly linear reaction-time penalty above a ~45 dB
  comfort threshold.
- Light is modeled with a **U-shaped** reaction-time penalty around an optimal
  ~500 lux.
- Accuracy is modeled to stay relatively stable across conditions (task-related
  adaptation), mirroring a finding explicitly reported in the real IMPACT
  paper's own analysis.

**This is simulated, not real human-subject data.** Model results below
describe how well each algorithm recovered these known, built-in
relationships -- they are NOT a scientific finding about real human cognition.

### Dataset structure
- 24 simulated participants x 4 conditions (Baseline/Light/Noise/Both) x 25
  trials = 2,400 rows, 10 columns, no missing values, no duplicates.
- Columns: participant_id, age, gender, condition, light_lux, noise_db,
  n_back_level, trial_number, reaction_time_ms, correct.

## 5. Features (X)
age, gender, condition, light_lux, noise_db, n_back_level, trial_number.
`participant_id` is used only for group-based splitting. `correct` is
excluded from X to avoid data leakage.

## 6. Target Variable (y)
Primary target: `reaction_time_ms` (regression). Secondary: `correct`.

## 7. Data Preprocessing
Numeric features scaled with `StandardScaler`; categorical features encoded
with `OneHotEncoder`, wrapped in a `ColumnTransformer` fit only on training
data.

## 8. Data Splitting -- Leakage Prevention
Rows grouped by `participant_id` before splitting (`GroupShuffleSplit`, 80/20)
to prevent the same participant appearing in both train and test sets.

## 9. EDA Summary
- Reaction time is fastest at Baseline, slowest under Both.
- Light shows a U-shaped relationship with reaction time; noise shows a
  rising trend.

## 10. ML Methodology
Linear Regression (baseline), Random Forest Regressor, and XGBoost Regressor
trained inside identical preprocessing pipelines.

## 11. DL Methodology
A feed-forward ANN (Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1)) trained
with Adam optimizer, MSE loss, and early stopping.

## 12. Evaluation Metrics & Results

| Model              | MAE   | RMSE  | R2    |
|--------------------|-------|-------|-------|
| Linear Regression  | 60.84 | 77.42 | 0.844 |
| Random Forest      | 62.07 | 76.98 | 0.846 |
| XGBoost            | 66.06 | 83.21 | 0.820 |
| ANN                | 78.09 | 97.81 | 0.751 |

**Best model: Random Forest.**

## 13. Model Interpretability

| Feature         | Importance |
|-----------------|------------|
| light_lux       | 0.818      |
| noise_db        | 0.100      |
| age             | 0.030      |
| trial_number    | 0.024      |
| n_back_level    | 0.020      |

## 14. Limitations
Dataset is synthetic, not real experimental data. Sample size is modest.

## 15. Future Scope
Replace with real IMPACT dataset if access is granted. Extend to
classification. Add SHAP interpretability.

## 16. Conclusion
This project demonstrates a complete, leakage-aware ML/DL pipeline for
predicting cognitive-task reaction time from environmental conditions,
deployed as a live Streamlit dashboard.

**Live demo:** https://cognitive-environment-ml.streamlit.app
