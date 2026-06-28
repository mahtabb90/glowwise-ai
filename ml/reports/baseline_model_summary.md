# GlowWise AI - Baseline Model Summary Report 🤖

This report documents the baseline machine learning models established for predicting skincare review satisfaction (`high_satisfaction`) on the `feat/baseline-model` branch. 

---

## 📊 Dataset & Target Variable

- **Dataset Used**: `data/processed/glowwise_reviews_sample_100k.csv` (100,000 preprocessed reviews, stratified train/test split of 80,000 / 20,000).
- **Features Used**: `combined_text` (concatenated review title and body).
- **Target Variable**: `high_satisfaction`
  - **`1` (High Satisfaction)**: Review rating of 4 or 5 stars (**82.08%** of dataset).
  - **`0` (Low/Medium Satisfaction)**: Review rating of 1, 2, or 3 stars (**17.92%** of dataset).

> [!WARNING]
> ### Class Imbalance Warning
> Skincare reviews exhibit a heavy positive response bias (82% vs 18%). Because of this skew, model accuracy will be artificially inflated: predicting only the majority class yields **82.08% accuracy** while completely failing to flag any low satisfaction reviews.

---

## 🧪 Baseline Models Tested

We trained and evaluated two baseline models on the 20,000-review test set:

1. **Dummy Classifier (Heuristic Baseline)**:
   - Strategy: `most_frequent` (always predicts `1`).
   - Serves as the raw, no-intelligence threshold for metrics.
2. **TF-IDF + Logistic Regression (Linear baseline)**:
   - Text Representation: `TfidfVectorizer` (unigrams & bigrams, max 10,000 features).
   - Estimator: `LogisticRegression` with `class_weight="balanced"` to control for class imbalance.

---

## 📈 Evaluation Metrics Summary

Below is the comparison of evaluation metrics on the 20,000-record test set:

| Evaluation Metric | Dummy Classifier | TF-IDF + Logistic Regression |
| :--- | :---: | :---: |
| **Accuracy** | 82.08% | **91.85%** |
| **Macro Precision** | 41.04% | **84.63%** |
| **Macro Recall** | 50.00% | **91.63%** |
| **Macro F1-Score** | 45.08% | **87.47%** |
| **Weighted F1-Score** | 74.01% | **92.22%** |
| **Class 0 (Low/Med Sat.) Precision** | 0.00% | **71.28%** |
| **Class 0 (Low/Med Sat.) Recall** | 0.00% | **91.29%** |
| **Class 0 (Low/Med Sat.) F1-Score** | 0.00% | **80.05%** |
| **Class 1 (High Sat.) Precision** | 82.08% | **97.98%** |
| **Class 1 (High Sat.) Recall** | 100.00% | **91.97%** |
| **Class 1 (High Sat.) F1-Score** | 90.16% | **94.88%** |

---

## 🔍 Why Accuracy is Not Enough

- The **Dummy Classifier** achieves **82.08% accuracy** but has a **Recall of 0.00%** on critical reviews. It fails to identify a single dissatisfied customer. If we relied only on accuracy, this model would seem highly successful while being completely useless for business operations.
- The **Logistic Regression** model achieves **91.85% accuracy** (+9.77% improvement) but, more importantly, raises the **Macro F1-Score to 87.47%** and the **Recall for dissatisfied reviews (Class 0) to 91.29%**. This ensures that 91.3% of dissatisfied reviews are successfully flagged.

---

## ⚠️ Model Limitations & Interpretability

- **Negation and Semantics**: As an n-gram TF-IDF model, Logistic Regression struggles with complex negation (e.g. "not greasy at all but doesn't moisturize well").
- **Missing Demographics Context**: The baseline model relies solely on text content. It does not utilize categorical metadata features like `skin_type`, `skin_tone`, or `ingredients`.
- **Interpretability (Coefficients)**:
  - Top indicators for **High Satisfaction (Class 1)**: `"love"`, `"amazing"`, `"great"`, `"best"`, `"soft"`, `"easy"`, `"clean"`, `"every day"`.
  - Top indicators for **Low/Medium Satisfaction (Class 0)**: `"disappointed"`, `"waste"`, `"returned"`, `"worst"`, `"nothing"`, `"acne"`, `"break out"`, `"greasy"`, `"dry"`.

---

## 📦 Model Artifacts & GitHub Ignored Files

The trained baseline model pipeline has been saved locally:
- **Logistic Regression Pipeline**: `ml/models/baseline_logistic_regression.joblib`
- **Baseline Evaluation Metrics**: [baseline_metrics.json](file:///c:/Users/mahta/my_projects/glowwise-ai/ml/reports/baseline_metrics.json)
- **Visual Plots**: Saved locally inside [ml/reports/figures/](file:///c:/Users/mahta/my_projects/glowwise-ai/ml/reports/figures/).

*Note: The model `.joblib` file size is large and is excluded from version control via `.gitignore` (as configured under `ml/models/*`). All summaries and configurations are preserved in version control.*

---

## 🚀 Recommendation for Next Branch

We recommend checking out the branch **`feat/model-comparison`** next. 
In that branch, we will compare this TF-IDF baseline against:
1. **Tree-Based Ensemble Models** (e.g., LightGBM, Random Forest) trained on combined customer profiles (`skin_type`, `skin_tone`, etc.) and engineered features.
2. **Dense Vector Representations** (e.g. word embeddings or transformer embeddings) to resolve syntax/negation limitations.
