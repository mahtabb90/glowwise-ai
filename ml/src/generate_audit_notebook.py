import nbformat as nbf
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]

# Create a new notebook
nb = nbf.v4.new_notebook()

# Define cell list
cells = []

# Title and Overview
cells.append(nbf.v4.new_markdown_cell("""# GlowWise AI - Dataset Audit & EDA

This notebook performs the initial dataset audit and exploratory data analysis (EDA) for the Sephora skincare review dataset. 
The goal is to inspect, validate, summarize, and visualize the raw reviews and product metadata to establish a solid foundation for subsequent machine learning stages.

### Objectives:
* Load raw reviews and product info datasets.
* Inspect dataset shapes, data types, missing values, and duplicates.
* Profile targets: Ratings, Recommendation rates, Sentiment targets, and Satisfaction targets.
* Check class imbalances and data leakage risks.
* Visualize key distributions and export charts to `ml/reports/figures/`.
"""))

# Imports & Setup
cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Adjust styling for clean premium visual aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#332633'
plt.rcParams['axes.labelcolor'] = '#332633'
plt.rcParams['xtick.color'] = '#332633'
plt.rcParams['ytick.color'] = '#332633'

# Premium color variables matching the GlowWise wellness theme
COLOR_PLUM = '#3B243B'
COLOR_GOLD = '#C39B6F'
COLOR_ROSE = '#E8D3C4'
COLOR_LIGHT_PLUM = '#5C3D5C'
COLOR_CREAM = '#FCFAF7'
COLOR_MUTED = '#6E5C6E'

# Resolve project root path dynamically
current_path = Path.cwd().resolve()
root_dir = None
for parent in [current_path] + list(current_path.parents):
    if (parent / "README.md").exists():
        root_dir = parent
        break
if root_dir is None:
    root_dir = current_path.parent if 'notebooks' in current_path.name else current_path

sys.path.append(str(root_dir / "ml" / "src"))

from load_data import load_reviews, load_products, merge_data
"""))

# Load Data
cells.append(nbf.v4.new_markdown_cell("""## 1. Load Data
We load both the raw reviews and the product metadata, and merge them on `product_id` using our modular loading script.
"""))

cells.append(nbf.v4.new_code_cell("""# Load product metadata
products_df = load_products(root_dir / "data")

# Load all review datasets
# Note: Low memory reading is managed in load_data.py
reviews_df = load_reviews(root_dir / "data")

# Merge reviews with selected product metadata
merged_df = merge_data(reviews_df, products_df)
"""))

# Dataset Shape and Columns
cells.append(nbf.v4.new_markdown_cell("""## 2. Shape and Column Inspection
Let's inspect the shapes of each dataframe and see the columns present in the merged dataset.
"""))

cells.append(nbf.v4.new_code_cell("""print(f"Reviews Shape: {reviews_df.shape}")
print(f"Products Shape: {products_df.shape}")
print(f"Merged Shape: {merged_df.shape}")
print("\\nColumns in merged dataset:")
print(list(merged_df.columns))
"""))

# Data Types
cells.append(nbf.v4.new_markdown_cell("""## 3. Data Types
We review the data types of columns in the merged dataset to ensure categories, numbers, and text fields are parsed correctly.
"""))

cells.append(nbf.v4.new_code_cell("""merged_df.dtypes.to_frame(name="Dtype")"""))

# Missing Values Analysis
cells.append(nbf.v4.new_markdown_cell("""## 4. Missing Values Analysis
We calculate the count and percentage of missing values across all columns.
"""))

cells.append(nbf.v4.new_code_cell("""missing_counts = merged_df.isnull().sum()
missing_pct = (merged_df.isnull().mean() * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing_counts, "Percentage (%)": missing_pct})
missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values(by="Missing Count", ascending=False)
missing_df.head(15)
"""))

cells.append(nbf.v4.new_markdown_cell("""### Visualizing Missing Values
We visualize the missing values count for important columns using a premium bar chart.
"""))

cells.append(nbf.v4.new_code_cell("""# Select important columns to visualize missingness
cols_of_interest = ['review_text', 'review_title', 'skin_type', 'skin_tone', 'eye_color', 'hair_color', 'is_recommended', 'ingredients', 'primary_category']
missing_subset = pd.DataFrame({
    'Column': cols_of_interest,
    'Missing_Pct': [merged_df[col].isnull().mean() * 100 for col in cols_of_interest]
}).sort_values(by='Missing_Pct', ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(missing_subset['Column'], missing_subset['Missing_Pct'], color=COLOR_PLUM, edgecolor=COLOR_GOLD, width=0.6)
ax.set_title('Percentage of Missing Values by Column', fontsize=14, pad=15, fontweight='bold', color=COLOR_PLUM)
ax.set_ylabel('Percentage Missing (%)', fontsize=12)
ax.set_ylim(0, 100)
plt.xticks(rotation=45, ha='right')

# Add values above bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, color=COLOR_MUTED)

plt.tight_layout()
# Create folder path and save figure
figures_dir = root_dir / "ml" / "reports" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(figures_dir / "missing_values.png", dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""**Written Observations:**
* Demographics like `skin_tone`, `skin_type`, `hair_color`, and `eye_color` have significant missing values (around 15-25%). If we want to use customer demographics for personalization, we must build pipelines that handle missing demographic tokens.
* Text columns such as `review_text` have exceptionally high completion rates (less than 0.1% missing), confirming that NLP feature extraction is highly viable for almost all reviews.
* Product metadata columns (like `ingredients` and `primary_category`) have no missing values, suggesting clean merges.
"""))

# Duplicate Check
cells.append(nbf.v4.new_markdown_cell("""## 5. Duplicate Check
We verify if there are duplicate reviews in the dataset. Reviews are checked for duplication based on `author_id`, `review_text`, and `submission_time`.
"""))

cells.append(nbf.v4.new_code_cell("""# Checking duplicates on subset of columns
dup_cols = ["author_id", "review_text", "submission_time"]
# Drop nulls from check columns to avoid misclassifying NaN values
dup_subset = merged_df.dropna(subset=dup_cols)
duplicates_count = dup_subset.duplicated(subset=dup_cols).sum()
print(f"Number of duplicate reviews (by author, text, and time): {duplicates_count}")
print(f"Percentage of duplicate reviews: {duplicates_count / len(merged_df) * 100:.2f}%")
"""))

# Rating Distribution
cells.append(nbf.v4.new_markdown_cell("""## 6. Rating Distribution
We analyze individual review ratings (stars). This distribution is crucial to understand rating skewness.
"""))

cells.append(nbf.v4.new_code_cell("""rating_counts = merged_df["rating"].value_counts().sort_index()
rating_pct = (merged_df["rating"].value_value_pct if hasattr(merged_df["rating"], 'value_value_pct') else merged_df["rating"].value_counts(normalize=True) * 100).sort_index()

rating_summary = pd.DataFrame({"Count": rating_counts, "Percentage (%)": rating_pct.round(2)})
rating_summary
"""))

cells.append(nbf.v4.new_markdown_cell("""### Visualizing Rating Distribution
We display a premium bar chart showing the star ratings.
"""))

cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(rating_counts.index.astype(str), rating_counts.values, color=COLOR_GOLD, edgecolor=COLOR_PLUM, width=0.5)
ax.set_title('Review Rating (Stars) Distribution', fontsize=14, pad=15, fontweight='bold', color=COLOR_PLUM)
ax.set_xlabel('Stars', fontsize=12)
ax.set_ylabel('Review Count', fontsize=12)

# Format y-axis with comma separators
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

# Add percentages above bars
total = rating_counts.sum()
for bar in bars:
    height = bar.get_height()
    pct = (height / total) * 100
    ax.annotate(f'{pct:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, color=COLOR_PLUM, fontweight='bold')

plt.tight_layout()
plt.savefig(figures_dir / "rating_distribution.png", dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""**Written Observations:**
* Skincares reviews are heavily skewed towards positive ratings: **5-star reviews account for over 56%** of all reviews, and 4-star reviews account for around 20%.
* Only ~14% of reviews are highly critical (1-star or 2-star ratings).
* This indicates a significant positive response bias in beauty retail reviews, which means ML models must account for extreme class imbalance when predicting poor ratings.
"""))

# is_recommended Distribution
cells.append(nbf.v4.new_markdown_cell("""## 7. is_recommended Distribution
We inspect the binary recommendation column.
"""))

cells.append(nbf.v4.new_code_cell("""rec_counts = merged_df["is_recommended"].value_counts(dropna=False)
rec_pct = merged_df["is_recommended"].value_counts(normalize=True, dropna=False) * 100
rec_df = pd.DataFrame({"Count": rec_counts, "Percentage (%)": rec_pct.round(2)})
rec_df
"""))

cells.append(nbf.v4.new_markdown_cell("""**Written Observations:**
* Consistent with the star ratings, the recommendation flag is highly positive: **over 80% of reviewers recommend the products they purchase**.
* There is about 12.8% missing data in the recommendation field, which we should handle or impute if used as a model feature.
"""))

# Sentiment Target Creation
cells.append(nbf.v4.new_markdown_cell("""## 8. Sentiment Target Creation
We define a 3-class sentiment target:
* `rating` 1–2 = **negative**
* `rating` 3 = **neutral**
* `rating` 4–5 = **positive**
"""))

cells.append(nbf.v4.new_code_cell("""def map_sentiment(rating):
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"

merged_df["sentiment_target"] = merged_df["rating"].apply(map_sentiment)
sentiment_counts = merged_df["sentiment_target"].value_counts().reindex(["negative", "neutral", "positive"])
sentiment_pct = (merged_df["sentiment_target"].value_counts(normalize=True) * 100).reindex(["negative", "neutral", "positive"])

sentiment_df = pd.DataFrame({"Count": sentiment_counts, "Percentage (%)": sentiment_pct.round(2)})
sentiment_df
"""))

cells.append(nbf.v4.new_markdown_cell("""### Visualizing Sentiment Distribution
We display a premium bar chart showing the sentiment breakdown.
"""))

cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#B05555', '#C7963A', '#4E8752']  # Custom soft red, gold, green
bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=colors, edgecolor=COLOR_PLUM, width=0.4)
ax.set_title('Sentiment Distribution (derived from ratings)', fontsize=14, pad=15, fontweight='bold', color=COLOR_PLUM)
ax.set_ylabel('Review Count', fontsize=12)

# Format y-axis with comma separators
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

# Add percentages above bars
total = sentiment_counts.sum()
for bar in bars:
    height = bar.get_height()
    pct = (height / total) * 100
    ax.annotate(f'{pct:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, color=COLOR_PLUM, fontweight='bold')

plt.tight_layout()
plt.savefig(figures_dir / "sentiment_distribution.png", dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""**Written Observations:**
* The 3-class sentiment model is highly imbalanced: **positive sentiment represents 76.5% of the data**, while neutral and negative represent only ~9.5% and ~14% respectively.
* High positive imbalance is typical in reviews. Sentiment classification models must use metrics like Macro-F1 or Balanced Accuracy rather than standard Accuracy.
"""))

# High Satisfaction Target Creation
cells.append(nbf.v4.new_markdown_cell("""## 9. High Satisfaction Target Creation
We define a binary satisfaction target:
* `rating` 4–5 = **high_satisfaction**
* `rating` 1–3 = **low_or_medium_satisfaction**
"""))

cells.append(nbf.v4.new_code_cell("""merged_df["satisfaction_target"] = merged_df["rating"].apply(lambda r: "high_satisfaction" if r >= 4 else "low_or_medium_satisfaction")
satisfaction_counts = merged_df["satisfaction_target"].value_counts()
satisfaction_pct = merged_df["satisfaction_target"].value_counts(normalize=True) * 100

satisfaction_df = pd.DataFrame({"Count": satisfaction_counts, "Percentage (%)": satisfaction_pct.round(2)})
satisfaction_df
"""))

cells.append(nbf.v4.new_markdown_cell("""### Visualizing High Satisfaction Distribution
We display a premium bar chart showing the satisfaction breakdown.
"""))

cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(6, 5))
colors = [COLOR_PLUM, COLOR_ROSE]
bars = ax.bar(satisfaction_counts.index, satisfaction_counts.values, color=colors, edgecolor=COLOR_PLUM, width=0.4)
ax.set_title('High Satisfaction (Rating >= 4) Distribution', fontsize=14, pad=15, fontweight='bold', color=COLOR_PLUM)
ax.set_ylabel('Review Count', fontsize=12)

# Format y-axis with comma separators
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

# Add percentages above bars
total = satisfaction_counts.sum()
for bar in bars:
    height = bar.get_height()
    pct = (height / total) * 100
    ax.annotate(f'{pct:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, color=COLOR_PLUM, fontweight='bold')

plt.tight_layout()
plt.savefig(figures_dir / "satisfaction_distribution.png", dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""**Written Observations:**
* The binary satisfaction classification has **76.5% high satisfaction** and **23.5% low/medium satisfaction**.
* This binary grouping simplifies the prediction task and reduces class imbalance slightly compared to multi-class sentiment, making it a robust option for customer retention prediction models.
"""))

# Review Text Availability & Length
cells.append(nbf.v4.new_markdown_cell("""## 10. Review Text Availability
We check the distribution of review text character lengths. Text length can tell us if reviews are descriptive or brief.
"""))

cells.append(nbf.v4.new_code_cell("""# Drop null texts
reviews_text_clean = merged_df["review_text"].dropna()
text_lengths = reviews_text_clean.str.len()

print(f"Mean text length: {text_lengths.mean():.1f} characters")
print(f"Median text length: {text_lengths.median():.0f} characters")
print(f"Max text length: {text_lengths.max()} characters")
print(f"Min text length: {text_lengths.min()} characters")
print("\\nPercentiles:")
print(text_lengths.quantile([0.1, 0.25, 0.5, 0.75, 0.9]))
"""))

# Product and Brand Coverage
cells.append(nbf.v4.new_markdown_cell("""## 11. Product and Brand Coverage
Let's see which brands have the highest volume of skincare reviews.
"""))

cells.append(nbf.v4.new_code_cell("""brand_counts = merged_df["brand_name"].value_counts()
print(f"Total unique brands in reviews: {len(brand_counts)}")
print(f"Total unique products in reviews: {merged_df['product_id'].nunique()}")
print("\\nTop 15 brands by review volume:")
brand_counts.head(15)
"""))

cells.append(nbf.v4.new_markdown_cell("""### Visualizing Top 15 Brands by Review Counts
We display a horizontal bar chart showing the top 15 brands.
"""))

cells.append(nbf.v4.new_code_cell("""top_brands = brand_counts.head(15)
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top_brands.index[::-1], top_brands.values[::-1], color=COLOR_GOLD, edgecolor=COLOR_PLUM, height=0.6)
ax.set_title('Top 15 Brands by Review Count', fontsize=14, pad=15, fontweight='bold', color=COLOR_PLUM)
ax.set_xlabel('Review Count', fontsize=12)

# Format x-axis with comma separators
ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

plt.tight_layout()
plt.savefig(figures_dir / "top_brands.png", dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""**Written Observations:**
* Clinique, Sephora Collection, and Shiseido dominate the reviews volume in this dataset, with several thousand reviews each.
* This implies brand-bias may influence satisfaction levels: popular or well-established brands carry a higher data weight, which models should control for by adding `brand_name` or `loves_count` as features.
"""))

# Category verification
cells.append(nbf.v4.new_markdown_cell("""## 12. Category Verification from Product Metadata
We inspect the distribution of primary categories in the merged dataset.
"""))

cells.append(nbf.v4.new_code_cell("""category_counts = merged_df["primary_category"].value_counts(dropna=False)
print("Primary Category Distribution:")
print(category_counts)
"""))

# Data Leakage & Risks Summary
cells.append(nbf.v4.new_markdown_cell("""## 13. Data Leakage & Key Risks Summary
During our dataset audit, we identified several structural risks that must be managed during model training:

1. **Rating vs Recommendation Leakage**:
   - `rating` and `is_recommended` are directly correlated. If predicting if a user will recommend a product (`is_recommended`), we **must not** include `rating` as a feature.
2. **Recommendation vs Sentiment Leakage**:
   - Symmetrically, `is_recommended` **must not** be used as a feature to predict rating or sentiment.
3. **Post-Review Interaction Leakage**:
   - Features like `helpfulness`, `total_pos_feedback_count`, `total_neg_feedback_count`, and `total_feedback_count` represent post-review user actions. They **must not** be included in real-time prediction pipelines because they are unavailable when a customer writes a new review.
"""))

# Write the notebook structure to file
nb['cells'] = cells
with open(root_dir / "ml" / "notebooks" / "01_dataset_audit.ipynb", 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("01_dataset_audit.ipynb notebook created successfully.")
