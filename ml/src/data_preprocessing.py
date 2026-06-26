"""
GlowWise AI - Data Preprocessing Pipeline
This script handles the raw review dataset cleaning, text preprocessing (lowercasing, punctuation removal, lemmatization),
and numerical feature engineering for ML model consumption.
"""

import os
import pandas as pd

def clean_review_text(text: str) -> str:
    """
    Cleans raw review texts.
    - Lowers casing
    - Strip whitespaces
    - Placeholder for NLTK/SpaCy regex-based cleaning, stopword removal, lemmatization
    """
    if not isinstance(text, str):
        return ""
    return text.strip().lower()

def preprocess_pipeline(raw_data_path: str, output_data_path: str):
    """
    Complete dataset preprocessing pipeline.
    Loads raw skincare review files, processes features, and dumps outputs to processed folder.
    """
    print(f"Loading raw data from: {raw_data_path}")
    # Placeholder implementation
    if not os.path.exists(raw_data_path):
        print(f"Warning: Raw data path '{raw_data_path}' not found. Generating mock processing.")
        return
        
    df = pd.read_csv(raw_data_path)
    
    # Process text column (e.g. 'review_text')
    if 'review_text' in df.columns:
        df['cleaned_review'] = df['review_text'].apply(clean_review_text)
        
    print(f"Saving processed dataset to: {output_data_path}")
    df.to_csv(output_data_path, index=False)

if __name__ == "__main__":
    # Example execution paths
    raw_path = "data/raw/sephora_reviews.csv"
    processed_path = "data/processed/preprocessed_reviews.csv"
    preprocess_pipeline(raw_path, processed_path)
