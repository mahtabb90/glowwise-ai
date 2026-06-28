"""
GlowWise AI - Data Preprocessing Pipeline
This script handles:
1. Loading the raw Sephora reviews and product metadata using load_data.py.
2. Dropping duplicate product_id rows in product info before merging.
3. Merging and validating that no row multiplication occurs.
4. Filtering out missing rating values.
5. Creating target variables: sentiment_label (3-class) and high_satisfaction (binary).
6. Basic text cleaning (lowercase, whitespace removal).
7. Creating the combined_text feature and dropping rows where it's empty after cleaning.
8. Generating text_length and word_count features.
9. Imputing missing values:
   - Text fields filled with ""
   - Categorical fields filled with "unknown"
   - Numeric field (price_usd) converted safely to numeric and filled with median.
10. Dropping data leakage columns.
11. Saving the full processed dataset and a stratified 100k sample.
12. Generating a metadata summary report JSON.
"""

import json
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add current directory to path to ensure load_data can be imported when running from various Cwds
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

from load_data import load_reviews, load_products, merge_data, get_data_dir

def clean_text(text: str) -> str:
    """
    Apply basic text cleaning:
    - Convert to lowercase
    - Remove excessive whitespace (newlines, tabs, multiple spaces)
    """
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Replace any sequence of whitespace characters with a single space and strip
    text = " ".join(text.split())
    return text

def preprocess_pipeline():
    print("=== Starting GlowWise AI Data Preprocessing Pipeline ===")
    
    # Resolve directories
    data_dir = get_data_dir()
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    reports_dir = data_dir.parent / "ml" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print("Loading raw datasets...")
    reviews_df = load_reviews(data_dir=data_dir)
    products_df = load_products(data_dir=data_dir)
    
    # Keep track of original reviews count to validate no row multiplication
    reviews_count_before = len(reviews_df)
    print(f"Initial reviews count: {reviews_count_before:,}")
    
    # 2. Deduplicate products metadata on product_id
    print("Deduplicating products metadata...")
    products_df_clean = products_df.drop_duplicates(subset=["product_id"])
    print(f"Products count: original {len(products_df):,}, unique product_ids {len(products_df_clean):,}")
    
    # 3. Merge Reviews and Products with validation
    # Merge only selected safe product metadata columns
    columns_to_merge = ["product_id", "primary_category", "secondary_category", "tertiary_category", "ingredients"]
    merged_df = merge_data(reviews_df, products_df_clean, columns_to_merge=columns_to_merge)
    
    # Validate row count
    reviews_count_after = len(merged_df)
    print(f"Reviews count after merge: {reviews_count_after:,}")
    if reviews_count_before != reviews_count_after:
        raise ValueError(
            f"Row count mismatch after merge! Before: {reviews_count_before:,}, "
            f"After: {reviews_count_after:,}. Row multiplication detected."
        )
    
    # 4. Drop rows where rating is missing
    print("Dropping rows with missing ratings...")
    initial_len = len(merged_df)
    merged_df = merged_df.dropna(subset=["rating"])
    print(f"Dropped {initial_len - len(merged_df):,} rows with missing rating. Current count: {len(merged_df):,}")
    
    # 5. Create targets
    print("Creating target columns...")
    # sentiment_label: rating 1-2 = negative, 3 = neutral, 4-5 = positive
    def rating_to_sentiment(r):
        if r <= 2:
            return "negative"
        elif r == 3:
            return "neutral"
        else:
            return "positive"
            
    merged_df["sentiment_label"] = merged_df["rating"].apply(rating_to_sentiment)
    # high_satisfaction: rating 4-5 = 1, 1-3 = 0
    merged_df["high_satisfaction"] = merged_df["rating"].apply(lambda r: 1 if r >= 4 else 0)
    
    # 6. Basic text cleaning
    print("Cleaning review text fields...")
    merged_df["review_title"] = merged_df["review_title"].fillna("").astype(str).apply(clean_text)
    merged_df["review_text"] = merged_df["review_text"].fillna("").astype(str).apply(clean_text)
    
    # 7. Create combined_text and drop empty reviews
    print("Creating combined_text and dropping empty reviews...")
    merged_df["combined_text"] = (merged_df["review_title"] + " " + merged_df["review_text"]).str.strip()
    
    # Drop rows where combined_text is empty after cleaning
    pre_drop_count = len(merged_df)
    merged_df = merged_df[merged_df["combined_text"] != ""]
    print(f"Dropped {pre_drop_count - len(merged_df):,} rows with empty combined_text. Current count: {len(merged_df):,}")
    
    # 8. Feature engineering: text length and word count
    print("Engineering text length and word count features...")
    merged_df["text_length"] = merged_df["combined_text"].str.len()
    merged_df["word_count"] = merged_df["combined_text"].apply(lambda x: len(x.split()))
    
    # 9. Handle missing values for other columns
    print("Handling missing values...")
    # Text fields
    merged_df["ingredients"] = merged_df["ingredients"].fillna("").astype(str)
    
    # Categorical fields
    cat_cols = [
        "skin_type", "skin_tone", "hair_color", "eye_color", 
        "brand_name", "product_name", "primary_category", 
        "secondary_category", "tertiary_category"
    ]
    for col in cat_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna("unknown").astype(str)
            
    # Numeric fields: price_usd
    # Convert safely to numeric first
    if "price_usd" in merged_df.columns:
        merged_df["price_usd"] = pd.to_numeric(merged_df["price_usd"], errors="coerce")
        price_median = merged_df["price_usd"].median()
        print(f"Imputing missing price_usd values with median: {price_median}")
        merged_df["price_usd"] = merged_df["price_usd"].fillna(price_median)
        
    # 10. Exclude leakage-prone columns and keep safe features/targets
    # Keep useful features, target columns and safe identifiers
    safe_identifiers = ["product_id", "author_id"]
    targets = ["sentiment_label", "high_satisfaction"]
    features = [
        "review_text",
        "review_title",
        "combined_text",
        "text_length",
        "word_count",
        "skin_type",
        "skin_tone",
        "hair_color",
        "eye_color",
        "brand_name",
        "product_name",
        "primary_category",
        "secondary_category",
        "tertiary_category",
        "price_usd",
        "ingredients"
    ]
    
    # Final ML-ready columns
    final_cols = [col for col in (safe_identifiers + features + targets) if col in merged_df.columns]
    print(f"Selecting final columns: {final_cols}")
    final_df = merged_df[final_cols]
    
    # 11. Save processed datasets
    processed_csv_path = processed_dir / "glowwise_reviews_processed.csv"
    sample_csv_path = processed_dir / "glowwise_reviews_sample_100k.csv"
    
    print(f"Saving full processed dataset to {processed_csv_path.name}...")
    final_df.to_csv(processed_csv_path, index=False)
    print("Full dataset saved successfully.")
    
    # Create Stratified 100k Sample
    print("Creating stratified 100k sample based on high_satisfaction...")
    if len(final_df) > 100000:
        _, sample_df = train_test_split(
            final_df,
            test_size=100000,
            stratify=final_df["high_satisfaction"],
            random_state=42
        )
        print(f"Sample created with shape: {sample_df.shape}")
    else:
        sample_df = final_df.copy()
        print(f"Dataset is small ({len(final_df):,} rows). Using full dataset for sample.")
        
    print(f"Saving stratified sample dataset to {sample_csv_path.name}...")
    sample_df.to_csv(sample_csv_path, index=False)
    print("Sample dataset saved successfully.")
    
    # 12. Save metadata summary file
    metadata_path = reports_dir / "preprocessing_metadata.json"
    print(f"Saving metadata summary to {metadata_path}...")
    
    # Target distribution details
    high_sat_counts = final_df["high_satisfaction"].value_counts().to_dict()
    high_sat_pct = final_df["high_satisfaction"].value_counts(normalize=True).to_dict()
    
    sentiment_counts = final_df["sentiment_label"].value_counts().to_dict()
    sentiment_pct = final_df["sentiment_label"].value_counts(normalize=True).to_dict()
    
    # JSON-serializable target distributions
    high_sat_dist = {str(k): int(v) for k, v in high_sat_counts.items()}
    high_sat_dist_pct = {str(k): float(v) for k, v in high_sat_pct.items()}
    sentiment_dist = {str(k): int(v) for k, v in sentiment_counts.items()}
    sentiment_dist_pct = {str(k): float(v) for k, v in sentiment_pct.items()}
    
    metadata = {
        "processed_row_count": int(len(final_df)),
        "final_column_count": int(len(final_df.columns)),
        "target_distributions": {
            "high_satisfaction": {
                "counts": high_sat_dist,
                "percentages": high_sat_dist_pct
            },
            "sentiment_label": {
                "counts": sentiment_dist,
                "percentages": sentiment_dist_pct
            }
        },
        "random_state": 42,
        "generated_file_paths": {
            "full_processed": str(processed_csv_path.relative_to(data_dir.parent)),
            "sample_100k": str(sample_csv_path.relative_to(data_dir.parent))
        },
        "note": "Processed CSV files are saved locally and are excluded from version control via .gitignore."
    }
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print("Metadata summary saved successfully.")
    
    print("=== GlowWise AI Data Preprocessing Completed Successfully! ===")

if __name__ == "__main__":
    preprocess_pipeline()
