"""
GlowWise AI - Model Comparison and Optimization Pipeline
Trains, tunes, and compares 7 machine learning models for Predicting Skincare Review Satisfaction.
Selects the best model using a structured ranking strategy:
Primary: Macro F1-score
Secondary: Class 0 Recall
Tertiary: Weighted F1-score
Quaternary: Model simplicity
Saves the best model, metrics summary, and comparative visualizations.
"""

import os
import sys
import json
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Ensure the script directory is in path for imports
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

from evaluate_model import calculate_metrics, plot_confusion_matrix

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def run_model_comparison():
    print("=== Starting GlowWise AI Model Comparison & Optimization Pipeline ===")
    
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
        print(f"\n[ERROR] Processed sample dataset not found at: {data_path.absolute()}")
        print("Please run the data preprocessing script first to generate this file:")
        print("   python ml/src/preprocess_data.py\n")
        sys.exit(1)
        
    print(f"Loading preprocessed dataset: {data_path.name}")
    df = pd.read_csv(data_path)
    
    # 2. Impute text, categorical, and numerical features to ensure 0 NaNs
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
    print(f"Dataset ready. Dimensions: {df.shape}")
    
    # 3. Stratified Train/Test split
    print("Splitting dataset into stratified Train (80%) and Test (20%) sets...")
    X = df  # Keep full DataFrame to extract metadata features in pipeline
    y = df["high_satisfaction"].astype(int)
    
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train size: {len(X_train_df):,}, Test size: {len(X_test_df):,}")
    
    # Extract text-only inputs for text-only models
    X_train_text = X_train_df["combined_text"].astype(str)
    X_test_text = X_test_df["combined_text"].astype(str)
    
    # Track results
    results = {}
    fitted_models = {}
    best_params_map = {}
    
    # Define color styles for console logs
    def log_section(title):
        print(f"\n--- [TRAINING] {title} ---")
        
    # Model 1: Dummy Classifier (heuristic baseline)
    log_section("1. DummyClassifier (strategy='most_frequent')")
    t0 = time.time()
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_text, y_train)
    y_pred = dummy.predict(X_test_text)
    t_elapsed = time.time() - t0
    print(f"Completed in {t_elapsed:.2f}s")
    results["dummy_classifier"] = calculate_metrics(y_test, y_pred)
    results["dummy_classifier"]["training_time_seconds"] = t_elapsed
    fitted_models["dummy_classifier"] = dummy
    
    # Model 2: Logistic Regression baseline
    log_section("2. LogisticRegression + TfidfVectorizer (Baseline)")
    t0 = time.time()
    lr_baseline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("lr", LogisticRegression(class_weight="balanced", random_state=42, max_iter=2000))
    ])
    lr_baseline.fit(X_train_text, y_train)
    y_pred = lr_baseline.predict(X_test_text)
    t_elapsed = time.time() - t0
    print(f"Completed in {t_elapsed:.2f}s")
    results["logistic_regression_baseline"] = calculate_metrics(y_test, y_pred)
    results["logistic_regression_baseline"]["training_time_seconds"] = t_elapsed
    fitted_models["logistic_regression_baseline"] = lr_baseline
    
    # Model 3: Tuned Logistic Regression using GridSearchCV
    log_section("3. Tuned LogisticRegression + TfidfVectorizer (GridSearchCV)")
    t0 = time.time()
    # Define pipeline for grid search
    grid_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("lr", LogisticRegression(class_weight="balanced", random_state=42, max_iter=2000))
    ])
    # Parameter grid (keep small for performance)
    param_grid = {
        "tfidf__max_features": [10000, 20000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "lr__C": [0.5, 1.0, 2.0]
    }
    print("Starting GridSearchCV with 3-fold CV...")
    grid_search = GridSearchCV(
        grid_pipeline, param_grid, cv=3, scoring="f1_macro", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train_text, y_train)
    y_pred = grid_search.predict(X_test_text)
    t_elapsed = time.time() - t0
    print(f"Completed in {t_elapsed:.2f}s")
    print(f"Best parameters: {grid_search.best_params_}")
    
    results["logistic_regression_tuned"] = calculate_metrics(y_test, y_pred)
    results["logistic_regression_tuned"]["training_time_seconds"] = t_elapsed
    results["logistic_regression_tuned"]["best_params"] = grid_search.best_params_
    best_params_map["logistic_regression_tuned"] = grid_search.best_params_
    fitted_models["logistic_regression_tuned"] = grid_search.best_estimator_
    
    # Model 4: LinearSVC
    log_section("4. LinearSVC + TfidfVectorizer")
    t0 = time.time()
    svc_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("svc", LinearSVC(class_weight="balanced", random_state=42, max_iter=2000))
    ])
    svc_pipeline.fit(X_train_text, y_train)
    y_pred = svc_pipeline.predict(X_test_text)
    t_elapsed = time.time() - t0
    print(f"Completed in {t_elapsed:.2f}s")
    results["linear_svc"] = calculate_metrics(y_test, y_pred)
    results["linear_svc"]["training_time_seconds"] = t_elapsed
    fitted_models["linear_svc"] = svc_pipeline
    
    # Model 5: Complement Naive Bayes (CNB is robust for imbalanced text classes)
    log_section("5. ComplementNB + TfidfVectorizer")
    t0 = time.time()
    cnb_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("cnb", ComplementNB())
    ])
    cnb_pipeline.fit(X_train_text, y_train)
    y_pred = cnb_pipeline.predict(X_test_text)
    t_elapsed = time.time() - t0
    print(f"Completed in {t_elapsed:.2f}s")
    results["complement_nb"] = calculate_metrics(y_test, y_pred)
    results["complement_nb"]["training_time_seconds"] = t_elapsed
    fitted_models["complement_nb"] = cnb_pipeline
    
    # Model 6: SGDClassifier
    log_section("6. SGDClassifier + TfidfVectorizer")
    t0 = time.time()
    sgd_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("sgd", SGDClassifier(loss="hinge", class_weight="balanced", random_state=42))
    ])
    sgd_pipeline.fit(X_train_text, y_train)
    y_pred = sgd_pipeline.predict(X_test_text)
    t_elapsed = time.time() - t0
    print(f"Completed in {t_elapsed:.2f}s")
    results["sgd_classifier"] = calculate_metrics(y_test, y_pred)
    results["sgd_classifier"]["training_time_seconds"] = t_elapsed
    fitted_models["sgd_classifier"] = sgd_pipeline
    
    # Model 7: Metadata-enhanced model (Logistic Regression with ColumnTransformer)
    log_section("7. LogisticRegression with ColumnTransformer (Metadata-Enhanced)")
    t0 = time.time()
    
    # Define transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=10000, ngram_range=(1, 2)), "combined_text"),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["brand_name", "skin_type", "primary_category", "secondary_category"]),
            ("num", StandardScaler(), ["price_usd", "text_length", "word_count"])
        ]
    )
    
    meta_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("lr", LogisticRegression(class_weight="balanced", random_state=42, max_iter=2000))
    ])
    
    meta_pipeline.fit(X_train_df, y_train)
    y_pred = meta_pipeline.predict(X_test_df)
    t_elapsed = time.time() - t0
    print(f"Completed in {t_elapsed:.2f}s")
    results["metadata_enhanced_lr"] = calculate_metrics(y_test, y_pred)
    results["metadata_enhanced_lr"]["training_time_seconds"] = t_elapsed
    fitted_models["metadata_enhanced_lr"] = meta_pipeline
    
    # 4. Save results to JSON
    results_json_path = reports_dir / "model_comparison_results.json"
    print(f"\nSaving results to JSON: {results_json_path.name}")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    # 5. Model Selection Ranking Strategy
    print("\nRanking models to select the best satisfaction model...")
    ranking = []
    for name, metrics in results.items():
        ranking.append({
            "name": name,
            "macro_f1": metrics["macro"]["f1_score"],
            "class_0_recall": metrics["per_class"]["low_or_medium_satisfaction"]["recall"],
            "weighted_f1": metrics["weighted"]["f1_score"],
            "is_simpler": 1 if name in ["dummy_classifier", "logistic_regression_baseline", "logistic_regression_tuned", "complement_nb", "sgd_classifier"] else 0
        })
        
    # Sort ranking:
    # 1. Primary: Macro F1-score (descending)
    # 2. Secondary: Class 0 recall (descending)
    # 3. Tertiary: Weighted F1-score (descending)
    # 4. Quaternary: Simpler/deployment-friendly model (descending)
    ranking.sort(key=lambda x: (x["macro_f1"], x["class_0_recall"], x["weighted_f1"], x["is_simpler"]), reverse=True)
    
    print("\n--- MODEL RANKING SUMMARY ---")
    for idx, r in enumerate(ranking):
        print(f" {idx+1}. {r['name']:<30} | Macro F1: {r['macro_f1']:.4f} | Class 0 Recall: {r['class_0_recall']:.4f} | Weighted F1: {r['weighted_f1']:.4f}")
        
    best_model_name = ranking[0]["name"]
    best_model = fitted_models[best_model_name]
    print(f"\n[BEST MODEL] Best model selected: {best_model_name}")
    
    # 6. Save best model locally
    best_model_path = model_dir / "best_satisfaction_model.joblib"
    print(f"Saving best model to: {best_model_path.name}")
    joblib.dump(best_model, best_model_path)
    print("Best model successfully serialized.")
    
    # 7. Generate Visualizations
    print("\nGenerating charts...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'sans-serif'
    
    # GlowWise theme colors
    COLOR_PLUM = '#3B243B'
    COLOR_GOLD = '#C39B6F'
    COLOR_ROSE = '#E8D3C4'
    COLOR_MUTED = '#6E5C6E'
    COLOR_CREAM = '#FCFAF7'
    
    model_names_clean = {
        "dummy_classifier": "Dummy Classifier",
        "logistic_regression_baseline": "Logistic Reg (Baseline)",
        "logistic_regression_tuned": "Logistic Reg (Tuned)",
        "linear_svc": "Linear SVC",
        "complement_nb": "Complement NB",
        "sgd_classifier": "SGD Classifier",
        "metadata_enhanced_lr": "Metadata-Enhanced LR"
    }
    
    model_labels = [model_names_clean[r["name"]] for r in ranking[::-1]]
    macro_f1s = [results[r["name"]]["macro"]["f1_score"] for r in ranking[::-1]]
    accuracies = [results[r["name"]]["accuracy"] for r in ranking[::-1]]
    weighted_f1s = [results[r["name"]]["weighted"]["f1_score"] for r in ranking[::-1]]
    
    # Plot 1: Model Metrics Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(model_labels))
    height = 0.25
    
    rects1 = ax.barh(y_pos + height, macro_f1s, height, label='Macro F1-Score', color=COLOR_PLUM, edgecolor=COLOR_PLUM)
    rects2 = ax.barh(y_pos, weighted_f1s, height, label='Weighted F1-Score', color=COLOR_GOLD, edgecolor=COLOR_PLUM)
    rects3 = ax.barh(y_pos - height, accuracies, height, label='Accuracy', color=COLOR_ROSE, edgecolor=COLOR_PLUM)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_labels, fontsize=10, fontweight='bold', color=COLOR_PLUM)
    ax.set_xlabel('Score Value', fontsize=11, fontweight='bold', color=COLOR_PLUM)
    ax.set_title('Skincare Satisfaction Model Performance Metrics', fontsize=13, fontweight='bold', pad=20, color=COLOR_PLUM)
    ax.set_xlim(0, 1.15)
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Annotate metrics scores
    for idx, r in enumerate(ranking[::-1]):
        ax.annotate(f"{macro_f1s[idx]:.2f}", xy=(macro_f1s[idx] + 0.01, idx + height - 0.05), fontsize=8, color=COLOR_PLUM, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Metrics comparison chart saved to: model_comparison_metrics.png")
    
    # Plot 2: Model Comparison Minority Recall & Precision
    class_0_recalls = [results[r["name"]]["per_class"]["low_or_medium_satisfaction"]["recall"] for r in ranking[::-1]]
    class_0_precisions = [results[r["name"]]["per_class"]["low_or_medium_satisfaction"]["precision"] for r in ranking[::-1]]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects_rec = ax.barh(y_pos + height/2, class_0_recalls, height, label='Minority Recall (Class 0)', color=COLOR_PLUM, edgecolor=COLOR_PLUM)
    rects_prec = ax.barh(y_pos - height/2, class_0_precisions, height, label='Minority Precision (Class 0)', color=COLOR_GOLD, edgecolor=COLOR_PLUM)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_labels, fontsize=10, fontweight='bold', color=COLOR_PLUM)
    ax.set_xlabel('Score Value', fontsize=11, fontweight='bold', color=COLOR_PLUM)
    ax.set_title('Minority Class (Low/Medium Satisfaction) Recall & Precision', fontsize=13, fontweight='bold', pad=20, color=COLOR_PLUM)
    ax.set_xlim(0, 1.15)
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Annotate recall scores
    for idx in range(len(model_labels)):
        ax.annotate(f"{class_0_recalls[idx]:.2f}", xy=(class_0_recalls[idx] + 0.01, idx + height/2 - 0.05), fontsize=8, color=COLOR_PLUM, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison_minority_recall.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Minority Recall comparison chart saved to: model_comparison_minority_recall.png")
    
    # Plot 3: Best Model Confusion Matrix
    print("Generating best model confusion matrix plot...")
    best_pred_y = best_model.predict(X_test_df if best_model_name == "metadata_enhanced_lr" else X_test_text)
    plot_confusion_matrix(
        y_test, best_pred_y, 
        output_path=str(figures_dir / "best_model_confusion_matrix.png"),
        title=f"Best Model ({model_names_clean[best_model_name]}) Confusion Matrix"
    )
    
    print("=== GlowWise AI Model Comparison & Optimization Completed Successfully! ===")

if __name__ == "__main__":
    run_model_comparison()
