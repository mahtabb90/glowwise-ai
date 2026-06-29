"""
GlowWise AI - Advanced ML and Deep Learning Experiments
Trains and compares KNN, ANN, and CNN models against reference models on a consistent train/test split.
Handles TensorFlow absence gracefully by skipping the CNN and falling back to scikit-learn MLPClassifier for the ANN.
"""

import os
import sys
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

# Add current directory to path to ensure evaluate_model can be imported
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

from evaluate_model import calculate_metrics, plot_confusion_matrix

# Global styling options for figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#332633'
plt.rcParams['axes.labelcolor'] = '#332633'
plt.rcParams['xtick.color'] = '#332633'
plt.rcParams['ytick.color'] = '#332633'

COLOR_PLUM = '#3B243B'
COLOR_GOLD = '#C39B6F'
COLOR_ROSE = '#E8D3C4'
COLOR_MUTED = '#6E5C6E'
COLOR_CREAM = '#FCFAF7'

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

# Try to import TensorFlow and verify if it's usable
tf_available = False
try:
    import tensorflow as tf
    # Check version to ensure it imported correctly
    print(f"TensorFlow Version: {tf.__version__}")
    # Simple check to see if basic API works
    _ = tf.constant([1.0])
    tf_available = True
    print("TensorFlow is available and functional!")
except Exception as e:
    print(f"\n[INFO] TensorFlow is not available or functional: {e}")
    print("Proceeding with graceful fallbacks (skipping CNN, using scikit-learn MLPClassifier for ANN).\n")

def train_and_evaluate_all():
    print("=== Starting GlowWise AI Deep Learning & Advanced ML Experiments ===")
    
    root_dir = get_project_root()
    data_path = root_dir / "data" / "processed" / "glowwise_reviews_sample_100k.csv"
    reports_dir = root_dir / "ml" / "reports"
    figures_dir = reports_dir / "figures"
    model_dir = root_dir / "ml" / "models"
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_path.exists():
        print(f"[ERROR] Processed dataset not found at: {data_path.absolute()}")
        print("Please run 'python ml/src/preprocess_data.py' first.")
        sys.exit(1)
        
    # 1. Load Data
    print(f"Loading preprocessed sample data: {data_path.name}")
    df = pd.read_csv(data_path)
    
    # Impute missing values
    for col in ["review_title", "review_text", "combined_text", "ingredients"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
            
    df = df.dropna(subset=["high_satisfaction"])
    df = df[df["combined_text"].str.strip() != ""]
    
    X = df
    y = df["high_satisfaction"].astype(int)
    
    # 2. Stratified Train/Test Split
    print("Performing standard 80/20 stratified split...")
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Test set is kept consistent (all 20,000 test reviews)
    X_test_text = X_test_df["combined_text"].astype(str)
    print(f"Full Test Set Size: {len(X_test_df):,}")
    
    # For training, to manage computation time/memory, we subset the training set to a smaller stratified subset (15,000 samples)
    train_subset_size = 15000
    print(f"Subsetting training set to a stratified sample of {train_subset_size:,} reviews for fair comparison...")
    _, X_train_sub, _, y_train_sub = train_test_split(
        X_train_df, y_train, test_size=train_subset_size, stratify=y_train, random_state=42
    )
    X_train_text = X_train_sub["combined_text"].astype(str)
    
    # Compute class weights for training (since dataset is imbalanced)
    classes = np.unique(y_train_sub)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_sub)
    class_weight_dict = {c: w for c, w in zip(classes, weights)}
    print(f"Computed class weights for training: {class_weight_dict}")
    
    results = {}
    curves_data = {}
    
    # ------------------ Production Reference ------------------
    prod_model_path = model_dir / "best_satisfaction_model.joblib"
    if prod_model_path.exists():
        print(f"\n--- Evaluating Production Reference model from {prod_model_path.name} ---")
        try:
            prod_model = joblib.load(prod_model_path)
            # Check if this model is the one with column transformer or text-only
            # In model_comparison, the best model was Tuned LR (text-only)
            y_pred = prod_model.predict(X_test_text)
            
            # Get probability scores for ROC/PR curves
            if hasattr(prod_model, "predict_proba"):
                y_prob = prod_model.predict_proba(X_test_text)[:, 1]
            else:
                y_prob = None
                
            results["production_reference_lr_80k"] = calculate_metrics(y_test, y_pred)
            results["production_reference_lr_80k"]["description"] = "Tuned Logistic Regression (trained on full 80k)"
            
            if y_prob is not None:
                curves_data["production_reference_lr_80k"] = {
                    "y_prob": y_prob.tolist()
                }
            print("Production Reference model evaluated.")
        except Exception as e:
            print(f"Could not load or evaluate production reference: {e}")
    else:
        print("\nProduction reference model not found in ml/models/. Skipping reference evaluation.")

    # ------------------ Model 1: Tuned Logistic Regression (15k) ------------------
    print("\n--- Training Model 1: Tuned Logistic Regression (on 15k subset) ---")
    t0 = time.time()
    lr_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
        ("lr", LogisticRegression(class_weight="balanced", C=2.0, max_iter=2000, random_state=42))
    ])
    lr_pipeline.fit(X_train_text, y_train_sub)
    y_pred = lr_pipeline.predict(X_test_text)
    y_prob = lr_pipeline.predict_proba(X_test_text)[:, 1]
    elapsed = time.time() - t0
    
    results["tuned_logistic_regression"] = calculate_metrics(y_test, y_pred)
    results["tuned_logistic_regression"]["training_time_seconds"] = elapsed
    results["tuned_logistic_regression"]["description"] = f"Tuned Logistic Regression (trained on {train_subset_size} subset)"
    curves_data["tuned_logistic_regression"] = {"y_prob": y_prob.tolist()}
    print(f"Completed in {elapsed:.2f}s | Macro F1: {results['tuned_logistic_regression']['macro']['f1_score']:.4f}")
    
    # ------------------ Model 2: LinearSVC (15k) ------------------
    print("\n--- Training Model 2: LinearSVC (on 15k subset) ---")
    t0 = time.time()
    svc_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("svc", LinearSVC(class_weight="balanced", random_state=42, max_iter=2000))
    ])
    svc_pipeline.fit(X_train_text, y_train_sub)
    y_pred = svc_pipeline.predict(X_test_text)
    y_decision = svc_pipeline.decision_function(X_test_text)
    elapsed = time.time() - t0
    
    results["linear_svc"] = calculate_metrics(y_test, y_pred)
    results["linear_svc"]["training_time_seconds"] = elapsed
    results["linear_svc"]["description"] = f"LinearSVC (trained on {train_subset_size} subset)"
    curves_data["linear_svc"] = {"y_prob": y_decision.tolist()} # Save decision function for SVC
    print(f"Completed in {elapsed:.2f}s | Macro F1: {results['linear_svc']['macro']['f1_score']:.4f}")

    # ------------------ Dimensionality Reduction for KNN & ANN ------------------
    print("\n--- Dimensionality Reduction (TF-IDF + SVD) for KNN and ANN ---")
    t0 = time.time()
    # 1. TF-IDF
    tfidf_dim = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_dim.fit_transform(X_train_text)
    X_test_tfidf = tfidf_dim.transform(X_test_text)
    
    # 2. TruncatedSVD
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_train_svd = svd.fit_transform(X_train_tfidf)
    X_test_svd = svd.transform(X_test_tfidf)
    elapsed = time.time() - t0
    print(f"SVD reduction completed in {elapsed:.2f}s. Input features shape: {X_train_svd.shape}")

    # ------------------ Model 3: KNN (15k) ------------------
    print("\n--- Training Model 3: KNN (on 15k subset) ---")
    t0 = time.time()
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
    knn.fit(X_train_svd, y_train_sub)
    y_pred = knn.predict(X_test_svd)
    y_prob = knn.predict_proba(X_test_svd)[:, 1]
    elapsed = time.time() - t0
    
    results["knn"] = calculate_metrics(y_test, y_pred)
    results["knn"]["training_time_seconds"] = elapsed
    results["knn"]["description"] = f"KNN using TF-IDF + SVD (n_components=100) on {train_subset_size} subset"
    curves_data["knn"] = {"y_prob": y_prob.tolist()}
    print(f"Completed in {elapsed:.2f}s | Macro F1: {results['knn']['macro']['f1_score']:.4f}")
    
    # ------------------ Model 4: ANN (15k) ------------------
    print("\n--- Training Model 4: ANN (on 15k subset) ---")
    t0 = time.time()
    
    # Balance the training set manually for MLPClassifier since it doesn't support class_weights
    df_train_sub = pd.DataFrame(X_train_svd)
    df_train_sub["high_satisfaction"] = y_train_sub.values
    
    df_class_0 = df_train_sub[df_train_sub["high_satisfaction"] == 0]
    df_class_1 = df_train_sub[df_train_sub["high_satisfaction"] == 1]
    
    # Oversample minority class to match majority class size
    df_class_0_over = df_class_0.sample(len(df_class_1), replace=True, random_state=42)
    df_train_balanced = pd.concat([df_class_0_over, df_class_1], axis=0).sample(frac=1.0, random_state=42)
    
    X_train_svd_balanced = df_train_balanced.drop(columns=["high_satisfaction"]).values
    y_train_balanced = df_train_balanced["high_satisfaction"].values
    
    if tf_available:
        # Build Dense Neural Network with Keras
        print("Training Keras ANN Model...")
        ann_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(100,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        ann_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True
        )
        
        # Fit Keras ANN model
        history = ann_model.fit(
            X_train_svd, y_train_sub,
            epochs=20,
            batch_size=64,
            validation_split=0.2,
            class_weight=class_weight_dict,
            callbacks=[early_stopping],
            verbose=1
        )
        
        y_prob = ann_model.predict(X_test_svd).flatten()
        y_pred = (y_prob >= 0.5).astype(int)
        
        ann_history = {
            "loss": history.history["loss"],
            "val_loss": history.history["val_loss"],
            "accuracy": history.history["accuracy"],
            "val_accuracy": history.history["val_accuracy"]
        }
        
    else:
        # Fallback to MLPClassifier with early stopping
        print("TensorFlow not available. Training scikit-learn MLPClassifier...")
        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=100,
            alpha=0.0001,
            solver='adam',
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2
        )
        # Train on balanced dataset to handle class imbalance
        mlp.fit(X_train_svd_balanced, y_train_balanced)
        
        y_pred = mlp.predict(X_test_svd)
        y_prob = mlp.predict_proba(X_test_svd)[:, 1]
        
        # Scikit-learn MLP doesn't provide validation loss directly, but does have validation_scores_ (which is validation accuracy) and loss_curve_ (training loss)
        ann_history = {
            "loss": mlp.loss_curve_,
            "val_accuracy": mlp.validation_scores_ if mlp.validation_scores_ is not None else []
        }
        
    elapsed = time.time() - t0
    results["ann"] = calculate_metrics(y_test, y_pred)
    results["ann"]["training_time_seconds"] = elapsed
    results["ann"]["description"] = "Dense Neural Network (ANN) trained on TF-IDF + SVD features"
    curves_data["ann"] = {"y_prob": y_prob.tolist()}
    print(f"Completed in {elapsed:.2f}s | Macro F1: {results['ann']['macro']['f1_score']:.4f}")
    
    # Plot ANN Training curves (side-by-side layout)
    print("Generating ANN training curve plots (side-by-side)...")
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Title selection based on TensorFlow availability
    main_title = "ANN Model Training & Validation Curves" if tf_available else "ANN Fallback Training Diagnostics"
    
    # Left plot: Loss
    ax_loss.plot(ann_history["loss"], label="Training Loss", color=COLOR_PLUM, linewidth=2)
    if "val_loss" in ann_history:
        ax_loss.plot(ann_history["val_loss"], label="Validation Loss", color=COLOR_PLUM, linestyle="--", linewidth=2)
    ax_loss.set_xlabel("Epochs", fontweight="bold")
    ax_loss.set_ylabel("Loss", fontweight="bold")
    ax_loss.set_title("Loss Curve", fontweight="bold")
    ax_loss.legend(loc="upper right", frameon=True)
    ax_loss.grid(True, linestyle="--", alpha=0.5)
    
    # Right plot: Accuracy
    has_acc = False
    if "accuracy" in ann_history:
        ax_acc.plot(ann_history["accuracy"], label="Training Accuracy", color=COLOR_GOLD, linewidth=2)
        has_acc = True
    if "val_accuracy" in ann_history and len(ann_history["val_accuracy"]) > 0:
        ax_acc.plot(ann_history["val_accuracy"], label="Validation Accuracy", color=COLOR_GOLD, linestyle="--", linewidth=2)
        has_acc = True
        
    ax_acc.set_xlabel("Epochs", fontweight="bold")
    ax_acc.set_ylabel("Accuracy", fontweight="bold")
    ax_acc.set_title("Accuracy Curve", fontweight="bold")
    if has_acc:
        ax_acc.legend(loc="lower right", frameon=True)
    ax_acc.grid(True, linestyle="--", alpha=0.5)
    
    plt.suptitle(main_title, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(figures_dir / "ann_training_curves.png", dpi=150)
    plt.close()
    
    # ------------------ Model 5: Text CNN (15k) ------------------
    cnn_skipped = True
    if tf_available:
        print("\n--- Training Model 5: Text CNN (on 15k subset) ---")
        t0 = time.time()
        
        # We need a proper tokenizer for Keras
        from tensorflow.keras.layers import TextVectorization
        
        max_vocab_size = 15000
        max_seq_len = 150
        
        vectorizer = TextVectorization(
            max_tokens=max_vocab_size,
            output_sequence_length=max_seq_len,
            output_mode='int'
        )
        
        # Fit vectorizer only on training data
        vectorizer.adapt(X_train_text.values)
        
        # Vectorize inputs
        X_train_seq = vectorizer(X_train_text.values)
        X_test_seq = vectorizer(X_test_text.values)
        
        # Build 1D CNN
        cnn_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(max_seq_len,)),
            tf.keras.layers.Embedding(input_dim=max_vocab_size, output_dim=128, input_length=max_seq_len),
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        cnn_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        early_stopping_cnn = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True
        )
        
        history_cnn = cnn_model.fit(
            X_train_seq, y_train_sub,
            epochs=15,
            batch_size=64,
            validation_split=0.2,
            class_weight=class_weight_dict,
            callbacks=[early_stopping_cnn],
            verbose=1
        )
        
        y_prob = cnn_model.predict(X_test_seq).flatten()
        y_pred = (y_prob >= 0.5).astype(int)
        elapsed = time.time() - t0
        
        results["cnn"] = calculate_metrics(y_test, y_pred)
        results["cnn"]["training_time_seconds"] = elapsed
        results["cnn"]["description"] = "1D Convolutional Neural Network (CNN) for text classification"
        curves_data["cnn"] = {"y_prob": y_prob.tolist()}
        cnn_skipped = False
        print(f"Completed in {elapsed:.2f}s | Macro F1: {results['cnn']['macro']['f1_score']:.4f}")
        
        # Plot CNN curves
        print("Generating CNN training curve plots...")
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(history_cnn.history["loss"], label="Training Loss", color=COLOR_PLUM, linewidth=2)
        ax1.plot(history_cnn.history["val_loss"], label="Validation Loss", color=COLOR_PLUM, linestyle="--", linewidth=2)
        ax1.set_xlabel("Epochs", fontweight="bold")
        ax1.set_ylabel("Loss", color=COLOR_PLUM, fontweight="bold")
        ax1.tick_params(axis='y', labelcolor=COLOR_PLUM)
        
        ax2 = ax1.twinx()
        ax2.plot(history_cnn.history["accuracy"], label="Training Acc", color=COLOR_GOLD, linewidth=2)
        ax2.plot(history_cnn.history["val_accuracy"], label="Validation Acc", color=COLOR_GOLD, linestyle="--", linewidth=2)
        ax2.set_ylabel("Accuracy", color=COLOR_GOLD, fontweight="bold")
        ax2.tick_params(axis='y', labelcolor=COLOR_GOLD)
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)
        plt.title("CNN Model Training & Validation Curves", fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(figures_dir / "cnn_training_curves.png", dpi=150)
        plt.close()
    else:
        # Check if Colab results exist to merge them
        colab_results_path = reports_dir / "cnn_colab_results.json"
        if colab_results_path.exists():
            print("\n--- Integrating Google Colab CNN Results ---")
            try:
                with open(colab_results_path, "r", encoding="utf-8") as f:
                    colab_data = json.load(f)
                results["cnn"] = {
                    "accuracy": colab_data["accuracy"],
                    "macro": colab_data["macro"],
                    "weighted": colab_data["weighted"],
                    "per_class": colab_data["per_class"],
                    "training_time_seconds": colab_data["training_time_seconds"],
                    "description": colab_data["description"],
                    "training_size": colab_data["training_size"],
                    "status": "Completed (Colab GPU)",
                    "roc_auc": colab_data["roc_auc"],
                    "pr_auc": colab_data["pr_auc"],
                    "skipped": False
                }
                cnn_skipped = False
            except Exception as e:
                print(f"Error loading Colab results: {e}")
                results["cnn"] = {
                    "skipped": True,
                    "status": "Not executed – TensorFlow/Keras not available in this environment",
                    "reason": f"TensorFlow/Keras was not available, and error reading Colab results: {e}",
                    "training_size": 15000
                }
        else:
            print("\n--- Skipping Model 5: CNN (TensorFlow not available and no Colab results found) ---")
            results["cnn"] = {
                "skipped": True,
                "status": "Not executed – TensorFlow/Keras not available in this environment",
                "reason": "TensorFlow/Keras was not available in this environment.",
                "training_size": 15000
            }

    # Add training_size, status, roc_auc, and pr_auc to results JSON
    for m in results:
        if results[m].get("skipped", False):
            continue
        results[m]["training_size"] = 80000 if m == "production_reference_lr_80k" else train_subset_size
        results[m]["status"] = "Completed"
        
        if m in curves_data:
            scores = np.array(curves_data[m]["y_prob"])
            fpr, tpr, _ = roc_curve(y_test, scores)
            results[m]["roc_auc"] = float(auc(fpr, tpr))
            
            precision, recall, _ = precision_recall_curve(y_test, scores)
            results[m]["pr_auc"] = float(auc(recall, precision))
        else:
            results[m]["roc_auc"] = None
            results[m]["pr_auc"] = None

    # ------------------ Save Results JSON ------------------
    results_json_path = reports_dir / "deep_learning_results.json"
    print(f"\nSaving results to JSON: {results_json_path.name}")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    # ------------------ Plotting ROC and Precision-Recall Curves ------------------
    print("\n--- Generating Curve Comparisons ---")
    
    # 1. ROC Curves
    plt.figure(figsize=(8, 6))
    
    # Available models for plotting
    plot_models = ["tuned_logistic_regression", "linear_svc", "knn", "ann"]
    if not cnn_skipped:
        plot_models.append("cnn")
    if "production_reference_lr_80k" in curves_data:
        plot_models.append("production_reference_lr_80k")
        
    model_labels = {
        "production_reference_lr_80k": "Prod Reference: Tuned LR (80k)",
        "tuned_logistic_regression": "Tuned Logistic Reg (15k)",
        "linear_svc": "LinearSVC (15k)",
        "knn": "KNN (15k)",
        "ann": "Dense ANN (15k)",
        "cnn": "Text CNN (15k)"
    }
    
    colors_map = {
        "production_reference_lr_80k": "#1f77b4", # blue
        "tuned_logistic_regression": "#2ca02c", # green
        "linear_svc": "#9467bd", # purple
        "knn": "#ff7f0e", # orange
        "ann": COLOR_PLUM,
        "cnn": COLOR_GOLD
    }
    
    for m in plot_models:
        if m not in curves_data:
            continue
        scores = np.array(curves_data[m]["y_prob"])
        
        # Compute ROC
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc_score = auc(fpr, tpr)
        
        plt.plot(
            fpr, tpr, 
            label=f"{model_labels[m]} (AUC = {auc_score:.4f})", 
            color=colors_map[m], 
            linewidth=2
        )
        
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=10)
    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=10)
    plt.title('ROC Curve Comparison', fontweight='bold', fontsize=12, pad=15)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curve_comparison.png", dpi=150)
    plt.close()
    
    # 2. Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    
    for m in plot_models:
        if m not in curves_data:
            continue
        scores = np.array(curves_data[m]["y_prob"])
        
        # Compute PR curve
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr_auc = auc(recall, precision)
        
        plt.plot(
            recall, precision, 
            label=f"{model_labels[m]} (PR-AUC = {pr_auc:.4f})", 
            color=colors_map[m], 
            linewidth=2
        )
        
    # Baseline line represents the ratio of positive class (high satisfaction) in test set
    pos_ratio = y_test.sum() / len(y_test)
    plt.plot([0, 1], [pos_ratio, pos_ratio], 'k--', alpha=0.5, label=f"No Skill (ratio = {pos_ratio:.4f})")
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([pos_ratio - 0.05, 1.02])
    plt.xlabel('Recall', fontweight='bold', fontsize=10)
    plt.ylabel('Precision', fontweight='bold', fontsize=10)
    plt.title('Precision-Recall Curve Comparison', fontweight='bold', fontsize=12, pad=15)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(figures_dir / "precision_recall_curve_comparison.png", dpi=150)
    plt.close()
    
    # 3. Model Comparison Bar Chart
    print("Generating model comparison metrics bar chart...")
    
    metrics_to_compare = {
        "Accuracy": lambda r: r["accuracy"],
        "Macro F1-Score": lambda r: r["macro"]["f1_score"],
        "Weighted F1-Score": lambda r: r["weighted"]["f1_score"],
        "Class 0 Recall": lambda r: r["per_class"]["low_or_medium_satisfaction"]["recall"],
        "Class 0 Precision": lambda r: r["per_class"]["low_or_medium_satisfaction"]["precision"]
    }
    
    # Exclude models that didn't run (e.g. CNN if skipped)
    valid_models_for_chart = [m for m in ["tuned_logistic_regression", "linear_svc", "knn", "ann"] if m in results]
    if "production_reference_lr_80k" in results:
        valid_models_for_chart.insert(0, "production_reference_lr_80k")
    if not cnn_skipped:
        valid_models_for_chart.append("cnn")
        
    chart_model_labels = [model_labels[m] for m in valid_models_for_chart]
    
    # Create comparison figures
    x = np.arange(len(chart_model_labels))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(12, 6.5))
    
    colors_metrics = {
        "Accuracy": COLOR_ROSE,
        "Macro F1-Score": COLOR_PLUM,
        "Weighted F1-Score": COLOR_GOLD,
        "Class 0 Recall": COLOR_MUTED,
        "Class 0 Precision": "#D98880"
    }
    
    for idx, (metric_name, getter) in enumerate(metrics_to_compare.items()):
        scores = [getter(results[m]) for m in valid_models_for_chart]
        ax.bar(x + (idx - 2) * width, scores, width, label=metric_name, color=colors_metrics[metric_name], edgecolor=COLOR_PLUM, linewidth=0.5)
        
    ax.set_ylabel('Score Value', fontsize=11, fontweight='bold', color=COLOR_PLUM)
    ax.set_title('Advanced ML & Deep Learning Model Performance Comparison', fontsize=13, fontweight='bold', pad=20, color=COLOR_PLUM)
    ax.set_xticks(x)
    ax.set_xticklabels(chart_model_labels, rotation=15, ha="right", fontsize=10, fontweight='bold', color=COLOR_PLUM)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='lower left', frameon=True, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add values on top of bars for Macro F1-score specifically to make it highly legible
    macro_scores = [getter(results[m]) for getter in [metrics_to_compare["Macro F1-Score"]] for m in valid_models_for_chart]
    for i, score in enumerate(macro_scores):
        ax.annotate(f"{score:.2f}", xy=(i - width*0.5, score + 0.02), fontsize=9, fontweight='bold', color=COLOR_PLUM)
        
    plt.tight_layout()
    plt.savefig(figures_dir / "deep_learning_model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Presentation-Ready Model Comparison Chart (Horizontal Bar Chart)
    print("Generating presentation-ready model comparison chart...")
    pres_metrics = {
        "Macro F1": lambda r: r["macro"]["f1_score"],
        "Class 0 Recall": lambda r: r["per_class"]["low_or_medium_satisfaction"]["recall"],
        "Accuracy": lambda r: r["accuracy"]
    }
    ref_model = "production_reference_lr_80k"
    exp_models = ["tuned_logistic_regression", "linear_svc", "ann", "knn"]
    if not cnn_skipped:
        exp_models.append("cnn")
    exp_models = [m for m in exp_models if m in results]
    
    fig, (ax_ref, ax_exp) = plt.subplots(
        2, 1, 
        figsize=(11, 7.5), 
        gridspec_kw={'height_ratios': [1, len(exp_models)]},
        sharex=True
    )
    metrics_list = list(pres_metrics.keys())
    y_pos_ref = np.array([0])
    y_pos_exp = np.arange(len(exp_models))
    height = 0.22
    colors_pres = {
        "Macro F1": COLOR_PLUM,
        "Class 0 Recall": COLOR_GOLD,
        "Accuracy": COLOR_ROSE
    }
    
    if ref_model in results:
        ref_metrics = results[ref_model]
        for idx, metric_name in enumerate(metrics_list):
            val = pres_metrics[metric_name](ref_metrics)
            ax_ref.barh(y_pos_ref + (idx - 1) * height, [val], height, label=metric_name, color=colors_pres[metric_name], edgecolor=COLOR_PLUM, linewidth=0.5)
            ax_ref.annotate(f"{val:.2%}", xy=(val + 0.01, (idx - 1) * height - 0.05), fontsize=9, fontweight='bold', color=COLOR_PLUM)
        ax_ref.set_yticks(y_pos_ref)
        ax_ref.set_yticklabels(["Prod Reference:\nTuned LR (80k)"], fontsize=10, fontweight='bold', color=COLOR_PLUM)
        ax_ref.set_title("Production Reference (Trained on full 80k reviews)", fontsize=11, fontweight='bold', color=COLOR_PLUM, loc='left')
        ax_ref.grid(axis='x', linestyle='--', alpha=0.5)
        ax_ref.spines['top'].set_visible(False)
        ax_ref.spines['right'].set_visible(False)
        ax_ref.spines['left'].set_color(COLOR_MUTED)
        ax_ref.spines['bottom'].set_visible(False)
    else:
        ax_ref.set_visible(False)
        
    for idx, metric_name in enumerate(metrics_list):
        scores = [pres_metrics[metric_name](results[m]) for m in exp_models]
        ax_exp.barh(y_pos_exp + (idx - 1) * height, scores, height, label=metric_name, color=colors_pres[metric_name], edgecolor=COLOR_PLUM, linewidth=0.5)
        for i, val in enumerate(scores):
            ax_exp.annotate(f"{val:.2%}", xy=(val + 0.01, i + (idx - 1) * height - 0.05), fontsize=9, fontweight='bold', color=COLOR_PLUM)
            
    exp_labels = [model_labels[m] for m in exp_models]
    ax_exp.set_yticks(y_pos_exp)
    ax_exp.set_yticklabels(exp_labels, fontsize=10, fontweight='bold', color=COLOR_PLUM)
    ax_exp.set_title("Fair 15k Experimental Comparison (Trained on identical 15k training subset)", fontsize=11, fontweight='bold', color=COLOR_PLUM, loc='left')
    ax_exp.set_xlabel("Score Value", fontsize=11, fontweight='bold', color=COLOR_PLUM)
    ax_exp.grid(axis='x', linestyle='--', alpha=0.5)
    ax_exp.set_xlim(0, 1.12)
    ax_exp.spines['top'].set_visible(False)
    ax_exp.spines['right'].set_visible(False)
    ax_exp.spines['left'].set_color(COLOR_MUTED)
    ax_exp.spines['bottom'].set_color(COLOR_MUTED)
    ax_exp.legend(loc="lower right", frameon=True, fontsize=10)
    
    plt.suptitle("GlowWise AI - Model Performance Presentation Chart", fontsize=13, fontweight='bold', y=0.98, color=COLOR_PLUM)
    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison_presentation.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot Confusion Matrices for KNN and ANN (and CNN if run)
    print("Generating confusion matrices for new models...")
    for m in ["knn", "ann"]:
        if m in results:
            if m == "knn":
                y_pred_m = knn.predict(X_test_svd)
            elif m == "ann":
                if tf_available:
                    y_prob_m = ann_model.predict(X_test_svd).flatten()
                    y_pred_m = (y_prob_m >= 0.5).astype(int)
                else:
                    y_pred_m = mlp.predict(X_test_svd)
            plot_confusion_matrix(
                y_test, y_pred_m,
                output_path=str(figures_dir / f"{m}_confusion_matrix.png"),
                title=f"{model_labels[m]} Confusion Matrix",
                class_names=["Low/Medium", "High"]
            )
            
    if tf_available and not cnn_skipped:
        y_prob_cnn = cnn_model.predict(X_test_seq).flatten()
        y_pred_cnn = (y_prob_cnn >= 0.5).astype(int)
        plot_confusion_matrix(
            y_test, y_pred_cnn,
            output_path=str(figures_dir / "cnn_confusion_matrix.png"),
            title="Text CNN Confusion Matrix",
            class_names=["Low/Medium", "High"]
        )
        
    print("=== Advanced ML & Deep Learning Experiments Completed Successfully! ===")

if __name__ == "__main__":
    train_and_evaluate_all()
