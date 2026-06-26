"""
GlowWise AI - Data Loading Utilities
Provides reusable functions to validate, load, and merge raw Sephora review datasets
and product info metadata using Pathlib and Pandas.
"""

from pathlib import Path
import pandas as pd
import numpy as np

def get_data_dir() -> Path:
    """
    Finds the root data directory relative to this script.
    Assumes standard repository layout where ml/src/load_data.py is executed.
    """
    # script file path: glowwise-ai/ml/src/load_data.py
    # parents[2] is glowwise-ai/
    root_dir = Path(__file__).resolve().parents[2]
    return root_dir / "data"

def load_reviews(data_dir: Path = None, sample_fraction: float = None, random_state: int = 42) -> pd.DataFrame:
    """
    Loads all review CSV files matching data/raw/reviews_*.csv and concatenates them.
    
    Parameters:
    - data_dir: Optional path to data/ directory. If None, it is resolved relative to this file.
    - sample_fraction: Optional float representing fraction of dataset to load (for memory efficiency).
    - random_state: Random state for deterministic sampling.
    """
    if data_dir is None:
        data_dir = get_data_dir()
        
    raw_dir = data_dir / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir.absolute()}")
        
    # Find all reviews files
    review_files = list(raw_dir.glob("reviews_*.csv"))
    if not review_files:
        raise FileNotFoundError(f"No reviews_*.csv files found in {raw_dir.absolute()}")
        
    print(f"Found {len(review_files)} review files. Loading...")
    
    dfs = []
    for f in sorted(review_files):
        print(f"  Loading: {f.name}")
        df = pd.read_csv(f, low_memory=False)
        
        # Apply sampling per file if requested to manage memory
        if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
            df = df.sample(frac=sample_fraction, random_state=random_state)
            
        dfs.append(df)
        
    reviews_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(reviews_df)} total reviews.")
    return reviews_df

def load_products(data_dir: Path = None) -> pd.DataFrame:
    """
    Loads product_info.csv from raw data directory.
    """
    if data_dir is None:
        data_dir = get_data_dir()
        
    product_file = data_dir / "raw" / "product_info.csv"
    if not product_file.exists():
        raise FileNotFoundError(f"Product info file not found: {product_file.absolute()}")
        
    print(f"Loading product metadata: {product_file.name}")
    products_df = pd.read_csv(product_file)
    print(f"Loaded {len(products_df)} products.")
    return products_df

def merge_data(
    reviews_df: pd.DataFrame, 
    products_df: pd.DataFrame, 
    columns_to_merge: list = None
) -> pd.DataFrame:
    """
    Merges review data with selected product metadata columns on product_id.
    Handles missing column names gracefully so the function does not crash.
    
    Parameters:
    - reviews_df: Reviews DataFrame
    - products_df: Products DataFrame
    - columns_to_merge: List of columns to merge from products_df. If None, uses defaults.
    """
    if columns_to_merge is None:
        # Default columns of interest for skincare analysis
        columns_to_merge = [
            "product_id", 
            "primary_category", 
            "secondary_category", 
            "tertiary_category", 
            "loves_count", 
            "ingredients", 
            "highlights"
        ]
        
    # Ensure product_id is in columns_to_merge for merging
    if "product_id" not in columns_to_merge:
        columns_to_merge.append("product_id")
        
    # Filter columns to only those actually present in products_df
    available_cols = []
    missing_cols = []
    for col in columns_to_merge:
        if col in products_df.columns:
            available_cols.append(col)
        else:
            missing_cols.append(col)
            
    if missing_cols:
        print(f"Warning: The following requested metadata columns were not found in product_info: {missing_cols}")
        
    # Subset product info
    products_subset = products_df[available_cols]
    
    # Check if reviews_df has product_id
    if "product_id" not in reviews_df.columns:
        raise ValueError("Cannot merge data: 'product_id' column is missing in reviews DataFrame.")
        
    print(f"Merging reviews with selected metadata columns: {available_cols}")
    
    # Avoid overlapping column duplicate names except merge key
    overlapping = set(reviews_df.columns) & set(products_subset.columns) - {"product_id"}
    if overlapping:
        print(f"Note: Resolving overlapping columns {list(overlapping)} by renaming product metadata columns.")
        # Suffix metadata columns that conflict
        rename_dict = {col: f"meta_{col}" for col in overlapping}
        products_subset = products_subset.rename(columns=rename_dict)
        
    # Perform left join
    merged_df = pd.merge(reviews_df, products_subset, on="product_id", how="left")
    print(f"Merge complete. Final shape: {merged_df.shape}")
    return merged_df

if __name__ == "__main__":
    # Test loading and merging
    try:
        reviews = load_reviews(sample_fraction=0.01)
        products = load_products()
        merged = merge_data(reviews, products)
        print("Success! Head of merged data:")
        print(merged.head(2))
    except Exception as e:
        print(f"Error during execution: {e}")
