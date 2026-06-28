# GlowWise AI - Data Preprocessing Summary Report 🧪

This report documents the dataset preprocessing pipeline applied to the Sephora skincare review dataset, establishing clean, ML-ready datasets for baseline model training.

---

## ⚙️ Preprocessing Steps Applied

The raw review and product metadata datasets are transformed through the following sequential operations:
1. **Deduplication of Metadata**: Duplicate `product_id` rows are dropped from the product metadata dataset to prevent row multiplication during merging.
2. **Merging & Verification**: The reviews and product metadata are left-joined on `product_id` with strict row count validation before and after to ensure no duplicate rows are created.
3. **Missing Rating Clean-up**: Any reviews with missing `rating` values are dropped (none were found in the raw dataset).
4. **Target Variable Engineering**: Constructing targets (`high_satisfaction` and `sentiment_label`) from review rating stars.
5. **Casing & Whitespace Uniformity**: Normalizing `review_title` and `review_text` by converting text to lowercase, removing excessive spaces/tabs/newlines, and stripping outer padding.
6. **Corpus Synthesis (`combined_text`)**: Constructing a unified feature `combined_text` by concatenating the cleaned `review_title` and `review_text`.
7. **Empty Review Drop**: Dropping reviews where `combined_text` is empty after cleaning (dropped `1,444` records).
8. **Text Metadata Engineering**: Generating character length (`text_length`) and word count (`word_count`) features.
9. **Missing Value Imputation**: Applying custom strategies (categorical, text, and numerical) to fill empty features.
10. **Leakage Filtering**: Excluding all targets (other than the selected label) and post-hoc engagement fields from model features.
11. **Stratified Sampling**: Generating a representative 100k review subset for lightweight local model experimentation.

---

## 🎯 Target Definitions

We engineered two machine learning targets from the original `rating` (1–5 scale):

1. **`high_satisfaction` (Binary Class)**:
   - **`1` (High Satisfaction)**: Review rating of **4 or 5 stars**.
   - **`0` (Low/Medium Satisfaction)**: Review rating of **1, 2, or 3 stars**.
   - *Rationale*: Collapsing neutral and critical ratings into a single class reduces sparsity and focuses the classifier on flagging formulation, efficacy, or skin-reaction issues.
2. **`sentiment_label` (3-Class)**:
   - **`positive`**: Rating of **4 or 5 stars**.
   - **`neutral`**: Rating of **3 stars**.
   - **`negative`**: Rating of **1 or 2 stars**.
   - *Rationale*: Allows more granular sentiment analysis.

---

## 🔍 Missing Value Strategy

Missing value handling is tailored per column type:
* **Text Features** (`review_text`, `review_title`, `combined_text`, `ingredients`): Replaced with empty string `""`.
* **User Demographics / Categoricals** (`skin_type`, `skin_tone`, `hair_color`, `eye_color`, categories, etc.): Replaced with `"unknown"` to preserve records while keeping missingness as a distinct category.
* **Numerical Features** (`price_usd`): Safely coerced to numeric values, and any remaining missing values are filled with the median product price ($39.00).

---

## 🚫 Leakage Columns Removed

To prevent optimistic training bias and ensure model generalization, the following columns are excluded from ML features:

| Excluded Column | Reason for Removal |
| :--- | :--- |
| **`rating`** | Direct leak of targets; used only to construct `sentiment_label` and `high_satisfaction` before removal. |
| **`is_recommended`** | Highly collinear with ratings and satisfaction targets (98% correlation). Creates trivial shortcuts. |
| **`helpfulness`** | Accumulates **after** the review has been published. Unavailable when a customer drafts a new review. |
| **`total_feedback_count`** | Post-review engagement metric. Excluded to prevent temporal leakage. |
| **`total_pos_feedback_count`** | Post-review engagement metric. Excluded to prevent temporal leakage. |
| **`total_neg_feedback_count`** | Post-review engagement metric. Excluded to prevent temporal leakage. |

---

## ⚖️ Class Imbalance After Preprocessing

After filtering out rows with empty text, the final dataset contains **`1,092,967`** records. The target distributions are:

### 1. High Satisfaction (Binary)
- **Class `1` (High Satisfaction)**: `897,154` reviews (**82.08%**)
- **Class `0` (Low/Med Satisfaction)**: `195,813` reviews (**17.92%**)

### 2. Sentiment Label (3-Class)
- **`positive`**: `897,154` reviews (**82.08%**)
- **`negative`**: `114,061` reviews (**10.44%**)
- **`neutral`**: `81,752` reviews (**7.48%**)

> [!WARNING]
> Due to the severe positive bias (82% positive), future classification models must not rely on raw Accuracy as the only performance metric. Instead, evaluation must focus on **Macro-F1**, **Balanced Accuracy**, and **Precision-Recall curves**.

---

## 📦 Saved Processed Datasets

All processed data and summaries are located in:
* **Full Processed Dataset**: [glowwise_reviews_processed.csv](file:///c:/Users/mahta/my_projects/glowwise-ai/data/processed/glowwise_reviews_processed.csv) (`1,092,967` rows × `20` columns)
* **Stratified 100k Sample**: [glowwise_reviews_sample_100k.csv](file:///c:/Users/mahta/my_projects/glowwise-ai/data/processed/glowwise_reviews_sample_100k.csv) (`100,000` rows × `20` columns)
* **Metadata Summary JSON**: [preprocessing_metadata.json](file:///c:/Users/mahta/my_projects/glowwise-ai/ml/reports/preprocessing_metadata.json)

*Note: Processed CSV files are saved locally and are ignored by Git (configured in `.gitignore`) to prevent bloating the source control repository.*

---

## 🚀 Next Step: Baseline Model Training

With the clean, ML-ready datasets, we recommend checking out the next branch:
**`feat/baseline-model`**

In this branch, the objectives will be:
1. Feature extraction using a memory-efficient `TfidfVectorizer` (character/word n-grams).
2. Fitting a simple baseline classifier (e.g. `LogisticRegression` or `LinearSVC`) using class weight adjustments to address target skewness.
3. Establishing baseline metrics (Macro-F1, ROC-AUC) on the stratified 100k sample.
