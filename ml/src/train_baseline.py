"""
GlowWise AI - Baseline Model Training Pipeline
Trains a Dummy Classifier and a TF-IDF + Logistic Regression pipeline on the preprocessed 100k reviews sample.
Saves the trained models, metrics, and confusion matrix visualizations.
"""

import os
import sys
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Ensure the script directory is in the path to import evaluate_model
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

from evaluate_model import calculate_metrics, plot_confusion_matrix

def get_project_root() -> Path:
    """
    Returns the absolute path to the project root directory (glowwise-ai/).
    """
    return Path(__file__).resolve().parents[2]

def train_and_evaluate():
    print("=== Starting Baseline Model Training & Evaluation Pipeline ===")
    
    root_dir = get_project_root()
    data_path = root_dir / "data" / "processed" / "glowwise_reviews_sample_100k.csv"
    model_dir = root_dir / "ml" / "models"
    reports_dir = root_dir / "ml" / "reports"
    figures_dir = reports_dir / "figures"
    
    # Ensure directories exist
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Existence check for dataset
    if not data_path.exists():
        print(f"\n❌ Error: The preprocessed dataset sample was not found at: {data_path.absolute()}")
        print("💡 Please run the data preprocessing script first to generate this file:")
        print("   python ml/src/preprocess_data.py\n")
        sys.exit(1)
        
    print(f"Loading preprocessed sample: {data_path.name}")
    df = pd.read_csv(data_path)
    # Fill missing text fields that pandas parsed as NaN back to empty strings
    for col in ["review_title", "review_text", "combined_text", "ingredients"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    print(f"Loaded dataset with shape: {df.shape}")
    
    # 2. Drop rows with missing target or empty combined_text
    print("Performing pre-training checks and cleaning...")
    initial_len = len(df)
    df = df.dropna(subset=["high_satisfaction"])
    df = df[df["combined_text"].fillna("").str.strip() != ""]
    cleaned_len = len(df)
    if initial_len != cleaned_len:
        print(f"Dropped {initial_len - cleaned_len:,} rows with empty combined_text or missing high_satisfaction.")
    
    # Features and target
    X = df["combined_text"].astype(str)
    y = df["high_satisfaction"].astype(int)
    
    # 3. Stratified train/test split
    print("Splitting dataset into stratified Train and Test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train set size: {len(X_train):,}, Test set size: {len(X_test):,}")
    print(f"Train high_satisfaction distribution:\n{y_train.value_counts(normalize=True).to_dict()}")
    print(f"Test high_satisfaction distribution:\n{y_test.value_counts(normalize=True).to_dict()}")
    
    # 4. Train DummyClassifier (majority class baseline)
    print("\nTraining DummyClassifier baseline (strategy='most_frequent')...")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    
    dummy_metrics = calculate_metrics(y_test, y_pred_dummy)
    print("DummyClassifier evaluation complete.")
    
    # Plot Dummy Confusion Matrix
    dummy_cm_path = figures_dir / "dummy_confusion_matrix.png"
    plot_confusion_matrix(
        y_test, y_pred_dummy, 
        output_path=str(dummy_cm_path), 
        title="Dummy Classifier (Most Frequent) Confusion Matrix"
    )
    
    # 5. Train Logistic Regression Pipeline (TF-IDF + LogisticRegression)
    print("\nTraining TF-IDF + LogisticRegression baseline (class_weight='balanced')...")
    lr_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("lr", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000))
    ])
    
    print("Fitting Logistic Regression pipeline (this might take a few seconds)...")
    lr_pipeline.fit(X_train, y_train)
    print("Model fitted. Predicting on test set...")
    y_pred_lr = lr_pipeline.predict(X_test)
    
    lr_metrics = calculate_metrics(y_test, y_pred_lr)
    print("LogisticRegression evaluation complete.")
    
    # Plot Logistic Regression Confusion Matrix
    lr_cm_path = figures_dir / "logistic_regression_confusion_matrix.png"
    plot_confusion_matrix(
        y_test, y_pred_lr, 
        output_path=str(lr_cm_path), 
        title="TF-IDF + Logistic Regression Confusion Matrix"
    )
    
    # 6. Save trained model pipeline
    model_output_path = model_dir / "baseline_logistic_regression.joblib"
    print(f"\nSaving trained Logistic Regression pipeline to: {model_output_path.name}")
    joblib.dump(lr_pipeline, model_output_path)
    print("Model saved successfully.")
    
    # 7. Save evaluation results as JSON
    metrics_summary = {
        "dummy_classifier": dummy_metrics,
        "logistic_regression": lr_metrics
    }
    
    metrics_output_path = reports_dir / "baseline_metrics.json"
    print(f"Saving baseline evaluation metrics to: {metrics_output_path.name}")
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
    print("Metrics saved successfully.")
    
    print("\n=== Baseline Model Training & Evaluation Completed Successfully! ===")

if __name__ == "__main__":
    train_and_evaluate()
