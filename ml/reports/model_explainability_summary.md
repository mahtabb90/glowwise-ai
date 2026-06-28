# GlowWise AI - Model Explainability & Error Analysis Report 🔍

This report documents the explainability and diagnostics workflow executed for the selected best model (Tuned Logistic Regression text-only pipeline) on the `feat/model-explainability` branch.

---

## 🤖 Selected Model Summary

- **Model Architecture**: TF-IDF Vectorizer + Logistic Regression (with balanced class weights).
- **Parameters**: `{'lr__C': 2.0, 'tfidf__max_features': 20000, 'tfidf__ngram_range': (1, 2)}`.
- **Primary Metrics**:
  - Accuracy: **92.96%**
  - Macro F1-Score: **88.91%**
  - Class 0 (Low/Med Sat.) Recall: **90.93%**
  - Class 0 (Low/Med Sat.) Precision: **75.03%**

---

## 🧠 Explainability Method

Since the best model is a linear classifier (Logistic Regression) operating on a sparse term-frequency matrix (TF-IDF), we extract explainability directly from **model coefficients** mapped to **vocabulary terms**:
* **Positive Coefficients**: Terms that increase the log-odds of a prediction being `high_satisfaction` (Class 1).
* **Negative Coefficients**: Terms that decrease the log-odds of a prediction being `high_satisfaction` (yielding a `low_or_medium_satisfaction` prediction, Class 0).

---

## 📈 Top 15 Satisfaction & Dissatisfaction Drivers

Below are the top 15 terms for each class extracted from the fitted model weights:

### 1. High Satisfaction Drivers (Class 1 - Positive)
1. **`love`** (Coefficient: `+13.43`)
2. **`amazing`** (Coefficient: `+11.96`)
3. **`great`** (Coefficient: `+10.98`)
4. **`best`** (Coefficient: `+10.60`)
5. **`perfect`** (Coefficient: `+9.00`)
6. **`awesome`** (Coefficient: `+8.27`)
7. **`holy`** (Coefficient: `+8.09`)
8. **`holy grail`** (Coefficient: `+8.04`)
9. **`favorite`** (Coefficient: `+7.96`)
10. **`excellent`** (Coefficient: `+7.51`)
11. **`smooth`** (Coefficient: `+7.23`)
12. **`highly`** (Coefficient: `+6.84`)
13. **`nice`** (Coefficient: `+6.82`)
14. **`beautiful`** (Coefficient: `+6.71`)
15. **`wonderful`** (Coefficient: `+6.53`)

### 2. Low/Medium Satisfaction Drivers (Class 0 - Negative)
1. **`disappointed`** (Coefficient: `-14.07`)
2. **`waste`** (Coefficient: `-11.75`)
3. **`returned`** (Coefficient: `-10.95`)
4. **`worst`** (Coefficient: `-10.23`)
5. **`horrible`** (Coefficient: `-9.85`)
6. **`nothing`** (Coefficient: `-9.71`)
7. **`ok`** (Coefficient: `-8.95`)
8. **`terrible`** (Coefficient: `-8.90`)
9. **`disappointing`** (Coefficient: `-8.78`)
10. **`okay`** (Coefficient: `-8.49`)
11. **`awful`** (Coefficient: `-8.42`)
12. **`wanted`** (Coefficient: `-8.41`)
13. **`useless`** (Coefficient: `-8.38`)
14. **`broke out`** (Coefficient: `-8.31`)
15. **`breakout`** (Coefficient: `-8.18`)

*Full 25-driver reports are saved as CSV files:*
- [top_positive_terms.csv](file:///c:/Users/mahta/my_projects/glowwise-ai/ml/reports/top_positive_terms.csv)
- [top_negative_terms.csv](file:///c:/Users/mahta/my_projects/glowwise-ai/ml/reports/top_negative_terms.csv)

### Human-Readable Interpretation:
Skincare review language is highly polarized:
- **Efficacy and Emotion**: Satisfied customers express strong, enthusiastic emotional praise (`love`, `amazing`, `best`, `holy grail`) combined with positive skin feel descriptors (`smooth`, `soft`, `clean`).
- **Functional Failures & Skin Reactions**: Dissatisfied customers focus heavily on product returns and financial regret (`disappointed`, `waste`, `returned`, `useless`) as well as explicit physiological dermatological complaints (`broke out`, `breakout`, `acne`, `greasy`, `burning`, `dry`).

---

## 🔍 Prediction Confidence Analysis

We evaluated prediction probabilities ($max(P(y=0), P(y=1))$) on the 20,000-record test set:
- **Correct Predictions**: Highly concentrated, with a **mean confidence of 0.96**.
- **Incorrect Predictions**: Flatter distribution, with a **mean confidence of 0.84**.
- **Error Categories**:
  - **False Positives (FP)**: Lowest mean confidence (**0.82**). These represent reviews that contain highly conflicting or mixed sentiment (e.g. positive sensory words alongside critical side effects), making the model uncertain.
  - **False Negatives (FN)**: Mean confidence of **0.86**. These represent reviews containing skin-problem keywords (like `acne`, `dryness`) which carry negative weights, even when the overall review is positive (e.g. "this cleared my acne!").

---

## ⚠️ Model Diagnostics & Limitations

1. **Bag-of-Words Limitation**: TF-IDF models count vocabulary term frequencies but ignore word order. 
2. **Negation Failures**: The model struggles with negation structures (e.g. "not greasy" vs "greasy"). Since `"greasy"` carries a strong negative weight, "not greasy" can still push the prediction towards dissatisfied class 0.
3. **Complex/Mixed Reviews**: Reviews that contain "constructive criticism" (e.g. praising texture but complaining about a pump dispenser, or praising fragrance but complaining about a skin breakout) contain terms from both classes, leading to misclassifications.
4. **Correlation vs Causation**: The model coefficients represent **statistical correlation** with satisfaction ratings in review text. They do not represent physiological or biological **causation** regarding skincare ingredients or formulation efficacy.

---

## 💼 Value of Explainability in ML Portfolios

Explainability transforms a machine learning project from a simple "black box" into a business-driven decision engine:
- **Validates Model Logic**: Confirms the model has learned meaningful English and skincare vocabulary (e.g., matching dermatological issues with critical scores) rather than fitting on dataset noise or hidden leaks.
- **Enables Actionable Insights**: Allows businesses to inspect precisely why customers reject certain formulations, turning raw text classifications into specific formulation recommendations (e.g., resolving "breakouts" or "dryness" complaints for specific product categories).

---

## 🚀 Recommended Next Branch

We recommend checking out the branch:
**`feat/clustering-insights`**

In that branch, we will:
1. Filter dataset to critical reviews (Class 0: dissatisfied).
2. Apply unsupervised clustering (Topic Modeling or K-Means) to group reviews into distinct complaint categories (e.g., skin breakouts, packaging failures, scent complaints, skin dryness).
3. Connect complaint topics with specific brands and product formulations to deliver targeted product improvement insights.
