# GlowWise AI - Model Comparison & Optimization Summary Report 📊

This report documents the machine learning model comparison and optimization results established on the `feat/model-comparison` branch. We evaluated seven candidate architectures to predict skincare review satisfaction (`high_satisfaction`).

---

## 📊 Dataset & Evaluation Schema

- **Dataset**: `data/processed/glowwise_reviews_sample_100k.csv` (100,000 preprocessed reviews, stratified train/test split of 80,000 / 20,000).
- **Target Variable**: `high_satisfaction` (Class `1` = rating 4–5, Class `0` = rating 1–3).
- **Features Used**: `combined_text` (concatenated review title and body) for text-only models, plus categorical and numerical traits for the metadata-enhanced model.
- **Evaluation Priority**:
  1. **Primary**: Macro F1-score (harmonic mean of precision and recall across classes).
  2. **Secondary**: Class 0 Recall (ability to flag dissatisfied reviews).
  3. **Tertiary**: Weighted F1-score.
  4. **Quaternary**: Model simplicity (ease of deployment).

---

## ⚙️ Hyperparameter Tuning Results (GridSearchCV)

We optimized the **Logistic Regression** pipeline using a 3-fold cross-validation grid search over 12 combinations of vectorizer parameters and regularization strengths.
* **Best Parameter Set**:
  - `tfidf__max_features`: `20,000` (increased from 10,000)
  - `tfidf__ngram_range`: `(1, 2)` (unigrams & bigrams)
  - `lr__C`: `2.0` (weaker regularization, allowing model to learn more complex text weights)
* **Tuning Time**: **59.45 seconds** (executing 36 total fits in parallel via `n_jobs=-1`).

---

## 📈 Model Comparison Metrics Table

The table below summarizes the test performance of the seven evaluated architectures, sorted by **Macro F1-Score** (descending):

| Model Name | Accuracy | Macro F1 | Weighted F1 | Class 0 Recall | Class 0 Precision | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Logistic Reg (Tuned)** | **92.96%** | **88.91%** | **93.21%** | 90.93% | 75.03% | 59.45s |
| 🥈 **Metadata-Enhanced LR** | 92.24% | 87.96% | 92.57% | 91.10% | 72.58% | 12.22s |
| 🥉 **Linear SVC** | 92.37% | 87.94% | 92.63% | 88.61% | 73.96% | 12.05s |
| 4. **Logistic Reg (Baseline)** | 91.85% | 87.47% | 92.22% | 91.29% | 71.28% | 8.16s |
| 5. **SGD Classifier** | 91.35% | 86.88% | 91.79% | **92.02%** | 69.54% | 10.31s |
| 6. **Complement NB** | 89.01% | 83.98% | 89.74% | **92.02%** | 63.29% | 10.28s |
| 7. **Dummy Classifier** | 82.09% | 45.08% | 74.01% | 0.00% | 0.00% | <0.01s |

---

## 🏆 Selected Best Model: Rationale

The **Tuned Logistic Regression** model (`logistic_regression_tuned`) is selected as the best satisfaction prediction model for the GlowWise AI project.

### Selection Rationale:
1. **Primary Metric**: It achieves the highest **Macro F1-Score of 88.91%** (+1.44% improvement over the next best model).
2. **Imbalance Recall (Class 0)**: It achieves a **90.93% recall** on the low/medium satisfaction reviews. This means that 90.9% of dissatisfied customers are correctly identified.
3. **Imbalance Precision (Class 0)**: It achieves a **75.03% precision** (+2.45% improvement over the metadata-enhanced model), significantly reducing false alarms.
4. **Simplicity / Deployment**: It is a text-only pipeline. The metadata-enhanced model (`metadata_enhanced_lr`) performs slightly worse (Macro F1 of 87.96%) and requires collecting customer skin types and product categories. By using the tuned text-only model, we keep deployment simple and reduce dependency on categorical fields, which often suffer from missingness in real-world scenarios.

---

## ⚠️ Model Limitations & Interpretability

- **Context & Negation**: The bag-of-n-grams representation does not capture word sequences beyond bigrams, meaning complex semantic flips or long-distance syntax can still be misclassified.
- **Unseen Categories**: The metadata-enhanced model was configured with `OneHotEncoder(handle_unknown="ignore")` to prevent failures on unseen brands, but it was not selected due to lower performance.

---

## 📦 Model Artifacts & Git Exclusions

The following artifacts have been created locally:
- **Best Model Pipeline**: `ml/models/best_satisfaction_model.joblib` (Tuned Logistic Regression, size: ~10.4MB).
- **Comparison Results Log**: [model_comparison_results.json](file:///c:/Users/mahta/my_projects/glowwise-ai/ml/reports/model_comparison_results.json).
- **Generated Figures**: Saved inside [ml/reports/figures/](file:///c:/Users/mahta/my_projects/glowwise-ai/ml/reports/figures/).

*Note: Large model binaries (.joblib) are saved locally and are ignored by version control via .gitignore to prevent Git bloat.*

---

## 🚀 Recommendation for Next Branch

We recommend checking out the next branch:
**`feat/model-explainability`**

In that branch, the objectives will be:
1. Extract global and local feature importances (visualizing word weights).
2. Run model diagnostics on false negatives and false positives to identify vocabulary patterns.
3. Validate word weights using domain-specific wellness taxonomies.
