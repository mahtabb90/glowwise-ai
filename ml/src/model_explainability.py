"""
GlowWise AI - Model Explainability & Error Analysis Pipeline
Loads the best satisfaction model pipeline, extracts terms and coefficients robustly by object type.
Saves top satisfaction and dissatisfaction drivers as JSON and CSV files.
Performs test-split prediction confidence analysis and errors sampling (confidently wrong reviews).
Generates custom wellness-themed figures using pure Matplotlib (no Seaborn).
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Set paths
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def run_explainability():
    print("=== Starting GlowWise AI Model Explainability & Error Analysis ===")
    
    root_dir = get_project_root()
    data_path = root_dir / "data" / "processed" / "glowwise_reviews_sample_100k.csv"
    model_path = root_dir / "ml" / "models" / "best_satisfaction_model.joblib"
    reports_dir = root_dir / "ml" / "reports"
    figures_dir = reports_dir / "figures"
    
    # Ensure directories exist
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Existence check for dataset and model
    if not data_path.exists():
        print(f"\n[ERROR] Processed sample dataset not found at: {data_path.absolute()}")
        print("Please run the data preprocessing script first to generate this file:")
        print("   python ml/src/preprocess_data.py\n")
        sys.exit(1)
        
    if not model_path.exists():
        print(f"\n[ERROR] Best model artifact not found at: {model_path.absolute()}")
        print("Please run the model comparison script first to generate this file:")
        print("   python ml/src/model_comparison.py\n")
        sys.exit(1)
        
    print(f"Loading preprocessed dataset: {data_path.name}")
    df = pd.read_csv(data_path)
    
    print(f"Loading best satisfaction model: {model_path.name}")
    pipeline = joblib.load(model_path)
    
    # Impute missing values to ensure 0 NaNs
    print("Imputing text, categorical, and numerical columns...")
    for col in ["review_title", "review_text", "combined_text", "ingredients"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
            
    for col in ["brand_name", "skin_type", "primary_category", "secondary_category"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)
            
    if "price_usd" in df.columns:
        df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
        df["price_usd"] = df["price_usd"].fillna(df["price_usd"].median())
        
    # Drop rows with missing target or empty combined_text
    df = df.dropna(subset=["high_satisfaction"])
    df = df[df["combined_text"].str.strip() != ""]
    
    # 2. Train/Test split (same stratified split as previous branches)
    print("Splitting dataset into stratified Train and Test sets...")
    X = df
    y = df["high_satisfaction"].astype(int)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_test_text = X_test_df["combined_text"].astype(str)
    
    # 3. Robust Step Extraction from Pipeline
    print("Extracting TfidfVectorizer and LogisticRegression steps robustly by type...")
    tfidf_vectorizer = None
    lr_classifier = None
    
    if hasattr(pipeline, "steps"):
        for step_name, step_obj in pipeline.steps:
            if isinstance(step_obj, TfidfVectorizer):
                tfidf_vectorizer = step_obj
            elif isinstance(step_obj, LogisticRegression):
                lr_classifier = step_obj
    else:
        raise ValueError("Loaded model is not a pipeline containing named steps.")
        
    if tfidf_vectorizer is None or lr_classifier is None:
        raise ValueError("Could not robustly extract TfidfVectorizer or LogisticRegression from loaded pipeline.")
        
    # 4. Extract coefficients and terms
    print("Extracting vocabulary terms and term coefficients...")
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
    coefs = lr_classifier.coef_[0]
    
    # Sort terms by coefficient weight
    sorted_indices = np.argsort(coefs)
    
    # Top 25 Positive Terms (Satisfaction)
    top_pos_indices = sorted_indices[-25:]
    top_pos_words = feature_names[top_pos_indices]
    top_pos_coefs = coefs[top_pos_indices]
    
    # Top 25 Negative Terms (Dissatisfaction)
    top_neg_indices = sorted_indices[:25]
    top_neg_words = feature_names[top_neg_indices]
    top_neg_coefs = coefs[top_neg_indices]
    
    # Save term result lists to JSON
    explainability_results = {
        "top_positive_terms": [
            {"term": str(w), "coefficient": float(c), "direction": "positive", "rank": int(25 - idx)} 
            for idx, (w, c) in enumerate(zip(top_pos_words[::-1], top_pos_coefs[::-1]))
        ],
        "top_negative_terms": [
            {"term": str(w), "coefficient": float(c), "direction": "negative", "rank": int(idx + 1)}
            for idx, (w, c) in enumerate(zip(top_neg_words, top_neg_coefs))
        ]
    }
    
    json_output_path = reports_dir / "model_explainability_results.json"
    print(f"Saving term weight results to JSON: {json_output_path.name}")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(explainability_results, f, indent=4)
        
    # Save terms to CSV files
    pos_csv_path = reports_dir / "top_positive_terms.csv"
    neg_csv_path = reports_dir / "top_negative_terms.csv"
    
    print(f"Saving top positive terms to CSV: {pos_csv_path.name}")
    pd.DataFrame(explainability_results["top_positive_terms"]).to_csv(pos_csv_path, index=False)
    print(f"Saving top negative terms to CSV: {neg_csv_path.name}")
    pd.DataFrame(explainability_results["top_negative_terms"]).to_csv(neg_csv_path, index=False)
    
    # 5. Prediction confidence analysis
    print("Computing prediction confidences on test set...")
    probs = pipeline.predict_proba(X_test_text)
    preds = pipeline.predict(X_test_text)
    confidences = np.max(probs, axis=1)
    
    eval_df = X_test_df.copy()
    eval_df["true_label"] = y_test
    eval_df["predicted_label"] = preds
    eval_df["confidence"] = confidences
    eval_df["is_correct"] = (eval_df["true_label"] == eval_df["predicted_label"])
    
    def get_error_type(row):
        t = row["true_label"]
        p = row["predicted_label"]
        if t == 1 and p == 1:
            return "TP"
        elif t == 0 and p == 0:
            return "TN"
        elif t == 0 and p == 1:
            return "FP"
        else:
            return "FN"
            
    eval_df["error_type"] = eval_df.apply(get_error_type, axis=1)
    
    print(f"Prediction breakdown on test set:")
    print(eval_df["error_type"].value_counts())
    
    # 6. Error Analysis: Sample informative cases (sort FP/FN by confidence descending)
    print("Sampling most informative error and success cases (confidently wrong)...")
    
    # Sort FP and FN by confidence descending (confidently wrong)
    fps_sample = eval_df[eval_df["error_type"] == "FP"].sort_values(by="confidence", ascending=False).head(10)
    fns_sample = eval_df[eval_df["error_type"] == "FN"].sort_values(by="confidence", ascending=False).head(10)
    
    # Sort TP and TN by confidence descending (confidently right)
    tps_sample = eval_df[eval_df["error_type"] == "TP"].sort_values(by="confidence", ascending=False).head(10)
    tns_sample = eval_df[eval_df["error_type"] == "TN"].sort_values(by="confidence", ascending=False).head(10)
    
    examples_df = pd.concat([tps_sample, tns_sample, fps_sample, fns_sample])
    
    # Truncate text to a readable length (200 characters)
    examples_df["combined_text"] = examples_df["combined_text"].apply(
        lambda x: x[:200] + "..." if len(str(x)) > 200 else x
    )
    
    columns_to_keep = [
        "product_id", "product_name", "brand_name", 
        "combined_text", "true_label", "predicted_label", 
        "confidence", "error_type"
    ]
    
    error_csv_path = reports_dir / "error_analysis_examples.csv"
    print(f"Saving error analysis examples table to CSV: {error_csv_path.name}")
    examples_df[columns_to_keep].to_csv(error_csv_path, index=False)
    
    # 7. Generate Visualizations (No Seaborn, clean wellness colors)
    print("Generating Matplotlib plots...")
    
    # GlowWise Color Palettes
    COLOR_PLUM = '#3B243B'
    COLOR_GOLD = '#C39B6F'
    COLOR_ROSE = '#E8D3C4'
    COLOR_CREAM = '#FCFAF7'
    COLOR_MUTED = '#6E5C6E'
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['text.color'] = '#332633'
    plt.rcParams['axes.labelcolor'] = '#332633'
    plt.rcParams['xtick.color'] = '#332633'
    plt.rcParams['ytick.color'] = '#332633'
    
    # Plot 1: Top 25 Positive Terms (Satisfaction drivers)
    fig, ax = plt.subplots(figsize=(8, 9), facecolor=COLOR_CREAM)
    ax.set_facecolor(COLOR_CREAM)
    bars = ax.barh(top_pos_words, top_pos_coefs, color=COLOR_PLUM, edgecolor=COLOR_PLUM, height=0.6)
    ax.set_title('Top 25 Drivers of High Satisfaction (Class 1)', fontsize=12, pad=20, fontweight='bold', color=COLOR_PLUM)
    ax.set_xlabel('Logistic Regression Coefficient Weight', fontsize=10, fontweight='bold', color=COLOR_PLUM)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "top_positive_terms.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: top_positive_terms.png")
    
    # Plot 2: Top 25 Negative Terms (Dissatisfaction drivers)
    fig, ax = plt.subplots(figsize=(8, 9), facecolor=COLOR_CREAM)
    ax.set_facecolor(COLOR_CREAM)
    # Sort negative terms ascending (from most negative to least)
    bars = ax.barh(top_neg_words[::-1], top_neg_coefs[::-1], color=COLOR_GOLD, edgecolor=COLOR_PLUM, height=0.6)
    ax.set_title('Top 25 Drivers of Low/Medium Satisfaction (Class 0)', fontsize=12, pad=20, fontweight='bold', color=COLOR_PLUM)
    ax.set_xlabel('Logistic Regression Coefficient Weight', fontsize=10, fontweight='bold', color=COLOR_PLUM)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "top_negative_terms.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: top_negative_terms.png")
    
    # Plot 3: Prediction Confidence Distribution (Correct vs Incorrect)
    correct_conf = eval_df[eval_df["is_correct"] == True]["confidence"]
    incorrect_conf = eval_df[eval_df["is_correct"] == False]["confidence"]
    
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=COLOR_CREAM)
    ax.set_facecolor(COLOR_CREAM)
    
    ax.hist(correct_conf, bins=30, range=(0.5, 1.0), alpha=0.75, color=COLOR_PLUM, label="Correct Predictions", edgecolor=COLOR_PLUM)
    ax.hist(incorrect_conf, bins=30, range=(0.5, 1.0), alpha=0.75, color=COLOR_GOLD, label="Incorrect Predictions", edgecolor=COLOR_PLUM)
    
    # Annotate mean confidence
    ax.axvline(correct_conf.mean(), color=COLOR_PLUM, linestyle='--', linewidth=1.5, label=f"Mean Correct: {correct_conf.mean():.2f}")
    ax.axvline(incorrect_conf.mean(), color=COLOR_GOLD, linestyle='--', linewidth=1.5, label=f"Mean Incorrect: {incorrect_conf.mean():.2f}")
    
    ax.set_title('Prediction Confidence Distributions (Correct vs Incorrect)', fontsize=12, pad=20, fontweight='bold', color=COLOR_PLUM)
    ax.set_xlabel('Model Prediction Confidence (Max Probability)', fontsize=10, fontweight='bold', color=COLOR_PLUM)
    ax.set_ylabel('Number of Reviews', fontsize=10)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Format y-axis with comma separators
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    
    plt.tight_layout()
    plt.savefig(figures_dir / "prediction_confidence_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: prediction_confidence_distribution.png")
    
    # Plot 4: Error Analysis Breakdown (Mean Confidence by Group)
    categories = ["TP", "TN", "FP", "FN"]
    means = [eval_df[eval_df["error_type"] == cat]["confidence"].mean() for cat in categories]
    counts = [len(eval_df[eval_df["error_type"] == cat]) for cat in categories]
    
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=COLOR_CREAM)
    ax.set_facecolor(COLOR_CREAM)
    
    bars = ax.bar(categories, means, color=[COLOR_PLUM, COLOR_PLUM, COLOR_GOLD, COLOR_GOLD], edgecolor=COLOR_PLUM, width=0.45)
    
    # Annotate bar values (mean confidence & group count)
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        ax.annotate(f'Mean: {height:.2f}\nCount: {counts[idx]:,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=COLOR_PLUM)
                    
    ax.set_title('Mean Model Confidence by Prediction Category', fontsize=12, pad=20, fontweight='bold', color=COLOR_PLUM)
    ax.set_ylabel('Mean Prediction Confidence (Probability)', fontsize=10, fontweight='bold', color=COLOR_PLUM)
    ax.set_ylim(0, 1.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "error_analysis_breakdown.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: error_analysis_breakdown.png")
    
    print("=== GlowWise AI Model Explainability Completed Successfully! ===")

if __name__ == "__main__":
    run_explainability()
