# GlowWise AI - Dataset Audit Summary Report 📊

This report documents the dataset audit and exploratory data analysis (EDA) results for the Sephora skincare review dataset, establishing a foundation for building machine learning pipelines.

---

## 📈 Dataset Overview & Size

* **Total Reviews Count**: `1,094,411` records
* **Total Products Count**: `8,494` unique items (loaded from `product_info.csv`)
* **Merged Schema Dimensions**: `1,094,411` rows × `25` columns

### Key Columns Evaluated:
1. **Review Content**: `review_text`, `review_title` (Text features for NLP)
2. **Review Targets**: `rating` (Stars 1–5), `is_recommended` (Binary recommendation flag)
3. **User Demographics**: `skin_type`, `skin_tone`, `eye_color`, `hair_color` (Categorical metadata)
4. **Product Metadata**: `brand_name`, `primary_category`, `ingredients`, `highlights` (Merged formulation features)
5. **Engagement Metrics**: `helpfulness`, `total_pos_feedback_count`, `total_neg_feedback_count`, `total_feedback_count`

---

## 🔍 Missing Value Findings

We performed a missing values audit to evaluate data completeness:
* **Review Text**: Extremely high completion rate (**only 0.08% missing**), confirming that NLP models can utilize nearly the entire dataset.
* **Customer Demographics**: Show significant missingness:
  * `skin_tone`: **23.5% missing**
  * `hair_color`: **22.5% missing**
  * `eye_color`: **21.8% missing**
  * `skin_type`: **15.1% missing**
* **Recommendation Status (`is_recommended`)**: **12.8% missing**
* **Product Ingredients & Category**: **0% missing** (after filtering to successfully merged product records).

---

## ⚖️ Target Variables & Class Imbalance Findings

We evaluated three candidate targets for machine learning:

### 1. Star Rating (1–5 Stars)
* **5 Stars**: 56.4%
* **4 Stars**: 20.1%
* **3 Stars**: 9.5%
* **2 Stars**: 5.5%
* **1 Star**: 8.5%
* *Observation*: Skincare reviews suffer from a severe **positive response bias**, with **76.5% of reviews scoring 4 or 5 stars**.

### 2. Derived Sentiment Target (3 Classes)
* **Positive** (Rating 4–5): **76.5%**
* **Negative** (Rating 1–2): **14.0%**
* **Neutral** (Rating 3): **9.5%**
* *Observation*: Multi-class sentiment classification is heavily dominated by positive classes, requiring models to adjust weights or sample to learn negative/neutral vocabulary patterns.

### 3. Derived Satisfaction Target (Binary Class)
* **High Satisfaction** (Rating 4–5): **76.5%**
* **Low/Medium Satisfaction** (Rating 1–3): **23.5%**
* *Observation*: This binary split collapses neutral and critical reviews into a single class. It reduces multi-class sparsity and simplifies model targeting for satisfaction scoring.

---

## ⚠️ Key Data Leakage & Structural Risks

To ensure valid model evaluation and prevent optimistic bias (overfitting), the training pipelines must respect the following constraints:

> [!CAUTION]
> ### 1. Rating vs Recommendation Leakage
> * **Do not use `rating` as a feature when predicting `is_recommended`**, and **do not use `is_recommended` when predicting `rating` or sentiment**.
> * These two targets are highly collinear (98% of 5-star reviews recommend the product, while 96% of 1-star reviews do not). Including them together in a model creates trivial leaks that fail to generalize to review text analysis.

> [!WARNING]
> ### 2. Post-Review Engagement Metrics Leakage
> * **Do not include helpfulness or feedback counts** (`helpfulness`, `total_feedback_count`, `total_pos_feedback_count`, `total_neg_feedback_count`) in prediction models.
> * These indicators accumulate **after** the review has been published. Including them as features will leak information about review visibility and aging, and they are unavailable when a customer drafts a new review.

### 3. Duplicate Reviews
* We identified **4.9% duplicate reviews** (where `author_id`, `review_text`, and `submission_time` match exactly).
* *Recommendation*: These duplicates should be deduplicated prior to splitting train/test sets to prevent the same review from appearing in both folds, which leads to artificial performance inflation.

---

## 🎯 Target Variable & Next Step Recommendations

1. **Primary Model Target**: 
   * We recommend using the **binary `satisfaction_target`** (`high_satisfaction` vs `low_or_medium_satisfaction`) as the baseline model target. It aligns directly with the business case of finding critical issues in formulations.
   * If a sentiment analysis engine is required, utilize the 3-class `sentiment_target` but optimize the models using **Macro-F1** and **Balanced Accuracy** (rather than raw accuracy) to control for the 76% positive skew.

2. **Skincare Personalization Expansion**:
   * Leverage the `skin_type` column (15% missing) by treating missingness as a distinct category (e.g., `'unknown'`). This allows personalization classification (e.g., predicting satisfaction specifically for oily vs dry skin profiles).

3. **Next ML Pipeline Step**:
   * **Text Preprocessing**: Lowercase text, clean special characters, remove beauty-specific noise, and tokenize.
   * **Feature Extraction**: Set up a fast, memory-efficient vectorizer (TF-IDF with sublinear term-frequency scaling, max features = 10,000) using a 20% validation sample.
   * **Baseline Training**: Fit a simple, fast classifier (Logistic Regression or LinearSVM) to establish baseline validation metrics before attempting deep learning or gradient boosting.
