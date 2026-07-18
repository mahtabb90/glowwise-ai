"""
GlowWise AI - Transformer Transfer Learning Experiment
Trains a classifier on top of dense sentence embeddings extracted from reviews using a pre-trained transformer model.
Gracefully handles absence of torch/sentence-transformers by generating a Colab-ready notebook.
"""

import os
import sys
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure the script directory is in the path to import evaluate_model
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

from evaluate_model import calculate_metrics, plot_confusion_matrix

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

# Try to import torch and sentence_transformers
torch_available = False
st_available = False
try:
    import torch
    torch_available = True
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    st_available = True
except Exception as e:
    print(f"[INFO] Transformer libraries not available locally: {e}")
    print("Graceful fallback: notebook will be generated as Colab-ready.")

def run_experiment(subset_size=10000):
    print("=== Starting GlowWise AI Transformer NLP Experiment ===")
    
    root_dir = get_project_root()
    data_path = root_dir / "data" / "processed" / "glowwise_reviews_sample_100k.csv"
    reports_dir = root_dir / "ml" / "reports"
    figures_dir = reports_dir / "figures"
    cache_dir = root_dir / "ml" / "cache"
    notebooks_dir = root_dir / "ml" / "notebooks"
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Existence check for dataset
    if not data_path.exists():
        print(f"[ERROR] Processed dataset not found at: {data_path.absolute()}")
        print("Please run data preprocessing first.")
        sys.exit(1)
        
    print(f"Loading preprocessed sample: {data_path.name}")
    df = pd.read_csv(data_path)
    
    for col in ["review_title", "review_text", "combined_text"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
            
    df = df.dropna(subset=["high_satisfaction"])
    df = df[df["combined_text"].str.strip() != ""]
    
    print(f"Loaded dataset with shape: {df.shape}")
    
    # Stratified split: 80% train / 20% test
    X = df
    y = df["high_satisfaction"].astype(int)
    
    from sklearn.model_selection import train_test_split
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Stratified Train size: {len(X_train_df):,}, Test size: {len(X_test_df):,}")
    
    # Check if we should execute locally
    local_run = torch_available and st_available
    
    if local_run:
        print("\nDependencies available. Running experiment locally...")
        
        # Take a stratified training subset of requested size
        if len(X_train_df) > subset_size:
            _, X_train_sub, _, y_train_sub = train_test_split(
                X_train_df, y_train, test_size=subset_size, stratify=y_train, random_state=42
            )
        else:
            X_train_sub = X_train_df
            y_train_sub = y_train
            
        # Extract a stratified test subset for speed in local CPU environment (e.g. 5,000 samples)
        test_subset_size = min(5000, len(X_test_df))
        if len(X_test_df) > test_subset_size:
            _, X_test_sub, _, y_test_sub = train_test_split(
                X_test_df, y_test, test_size=test_subset_size, stratify=y_test, random_state=42
            )
        else:
            X_test_sub = X_test_df
            y_test_sub = y_test
            
        print(f"Local training subset size: {len(X_train_sub):,}")
        print(f"Local test subset size: {len(X_test_sub):,}")
        
        train_texts = X_train_sub["combined_text"].astype(str).tolist()
        test_texts = X_test_sub["combined_text"].astype(str).tolist()
        
        # Embedding model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Loading SentenceTransformer model: {model_name}...")
        st_model = SentenceTransformer(model_name)
        
        # Local caching to ml/cache/
        train_cache_file = cache_dir / f"train_embeddings_{len(X_train_sub)}.npy"
        test_cache_file = cache_dir / f"test_embeddings_{len(X_test_sub)}.npy"
        
        # Train embeddings
        if train_cache_file.exists():
            print("Loading train embeddings from cache...")
            X_train_emb = np.load(train_cache_file)
        else:
            print("Extracting train embeddings (this may take a minute)...")
            t0 = time.time()
            X_train_emb = st_model.encode(train_texts, show_progress_bar=True, batch_size=32)
            np.save(train_cache_file, X_train_emb)
            print(f"Train embeddings extracted in {time.time() - t0:.2f} seconds.")
            
        # Test embeddings
        if test_cache_file.exists():
            print("Loading test embeddings from cache...")
            X_test_emb = np.load(test_cache_file)
        else:
            print("Extracting test embeddings...")
            t0 = time.time()
            X_test_emb = st_model.encode(test_texts, show_progress_bar=True, batch_size=32)
            np.save(test_cache_file, X_test_emb)
            print(f"Test embeddings extracted in {time.time() - t0:.2f} seconds.")
            
        # Train Logistic Regression Classifier on top of embeddings
        from sklearn.linear_model import LogisticRegression
        print("Training Logistic Regression classifier on top of sentence embeddings...")
        clf = LogisticRegression(class_weight="balanced", C=1.0, random_state=42, max_iter=1000)
        t0 = time.time()
        clf.fit(X_train_emb, y_train_sub)
        train_time = time.time() - t0
        print(f"Classifier trained in {train_time:.4f} seconds.")
        
        # Predict
        y_pred = clf.predict(X_test_emb)
        y_prob = clf.predict_proba(X_test_emb)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import roc_curve, precision_recall_curve, auc
        metrics = calculate_metrics(y_test_sub, y_pred)
        
        # Calculate ROC-AUC & PR-AUC
        fpr, tpr, _ = roc_curve(y_test_sub, y_prob)
        roc_auc = auc(fpr, tpr)
        
        precision_vals, recall_vals, _ = precision_recall_curve(y_test_sub, y_prob)
        pr_auc = auc(recall_vals, precision_vals)
        
        # Update metrics dictionary
        metrics["roc_auc"] = float(roc_auc)
        metrics["pr_auc"] = float(pr_auc)
        metrics["training_time_seconds"] = float(train_time)
        metrics["training_size"] = len(X_train_sub)
        metrics["status"] = "Completed"
        metrics["description"] = f"Logistic Regression trained on sentence-transformers/all-MiniLM-L6-v2 embeddings ({len(X_train_sub)} train subset)"
        metrics["confusion_matrix"] = {
            "true_negative": int(np.sum((y_test_sub == 0) & (y_pred == 0))),
            "false_positive": int(np.sum((y_test_sub == 0) & (y_pred == 1))),
            "false_negative": int(np.sum((y_test_sub == 1) & (y_pred == 0))),
            "true_positive": int(np.sum((y_test_sub == 1) & (y_pred == 1)))
        }
        
        print("\n=== Transformer Classifier Metrics ===")
        print(json.dumps(metrics, indent=4))
        
        # Save metrics to transformer_results.json
        results_output_path = reports_dir / "transformer_results.json"
        with open(results_output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved results to: {results_output_path.name}")
        
        # Save confusion matrix
        plot_confusion_matrix(
            y_test_sub, y_pred,
            output_path=str(figures_dir / "transformer_confusion_matrix.png"),
            title="Transformer + LR Confusion Matrix"
        )
        
        # Generate baseline comparison plot
        generate_comparison_plot(metrics, reports_dir, figures_dir)
        
    else:
        print("\nRunning in Colab-Ready Preparation Mode...")
        # Write skipped results
        metrics = {
            "accuracy": None,
            "macro": {
                "precision": None,
                "recall": None,
                "f1_score": None
            },
            "weighted": {
                "precision": None,
                "recall": None,
                "f1_score": None
            },
            "per_class": {
                "low_or_medium_satisfaction": {
                    "precision": None,
                    "recall": None,
                    "f1_score": None,
                    "support": None
                },
                "high_satisfaction": {
                    "precision": None,
                    "recall": None,
                    "f1_score": None,
                    "support": None
                }
            },
            "training_size": subset_size,
            "status": "Prepared for Google Colab",
            "description": "Sentence Transformer experiment set up for Google Colab run. Excluded from local execution due to missing dependencies.",
            "skipped": True
        }
        results_output_path = reports_dir / "transformer_results.json"
        with open(results_output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved preparation config to: {results_output_path.name}")
        
        # Generate placeholder comparison plot containing baselines
        generate_comparison_plot(None, reports_dir, figures_dir)
        
    # Generate the Notebook
    generate_notebook(notebooks_dir, local_run, subset_size)
    
def generate_comparison_plot(transformer_metrics, reports_dir, figures_dir):
    """
    Loads baseline models and compares them against the Transformer model in a clean bar chart.
    """
    # GlowWise Color Theme
    COLOR_PLUM = '#3B243B'
    COLOR_GOLD = '#C39B6F'
    COLOR_ROSE = '#E8D3C4'
    COLOR_MUTED = '#6E5C6E'
    COLOR_CREAM = '#FCFAF7'
    
    # 1. Load Baseline Results
    baseline_lr = None
    text_cnn = None
    
    # Try to load deep_learning_results.json which holds reference models
    dl_results_path = reports_dir / "deep_learning_results.json"
    if dl_results_path.exists():
        try:
            with open(dl_results_path, "r", encoding="utf-8") as f:
                dl_results = json.load(f)
                
            # Production Tuned LR
            if "production_reference_lr_80k" in dl_results:
                m = dl_results["production_reference_lr_80k"]
                baseline_lr = {
                    "name": "Production Reference (LR 80k)",
                    "accuracy": m["accuracy"],
                    "macro_f1": m["macro"]["f1_score"],
                    "class_0_recall": m["per_class"]["low_or_medium_satisfaction"]["recall"]
                }
            elif "tuned_logistic_regression" in dl_results:
                m = dl_results["tuned_logistic_regression"]
                baseline_lr = {
                    "name": "Tuned Logistic Regression (15k)",
                    "accuracy": m["accuracy"],
                    "macro_f1": m["macro"]["f1_score"],
                    "class_0_recall": m["per_class"]["low_or_medium_satisfaction"]["recall"]
                }
                
            # Text CNN
            if "cnn" in dl_results and not dl_results["cnn"].get("skipped", False):
                m = dl_results["cnn"]
                text_cnn = {
                    "name": "Text CNN (Colab)",
                    "accuracy": m["accuracy"],
                    "macro_f1": m["macro"]["f1_score"],
                    "class_0_recall": m["per_class"]["low_or_medium_satisfaction"]["recall"]
                }
        except Exception as e:
            print(f"Warning: Could not parse deep_learning_results.json: {e}")
            
    # Try baseline_metrics.json if baseline_lr is still None
    if baseline_lr is None:
        baseline_path = reports_dir / "baseline_metrics.json"
        if baseline_path.exists():
            try:
                with open(baseline_path, "r", encoding="utf-8") as f:
                    base_res = json.load(f)
                if "logistic_regression" in base_res:
                    m = base_res["logistic_regression"]
                    baseline_lr = {
                        "name": "Logistic Regression (TF-IDF)",
                        "accuracy": m["accuracy"],
                        "macro_f1": m["macro"]["f1_score"],
                        "class_0_recall": m["per_class"]["low_or_medium_satisfaction"]["recall"]
                    }
            except Exception as e:
                print(f"Warning: Could not parse baseline_metrics.json: {e}")
                
    # 2. Extract Transformer Metrics
    trans_data = None
    if transformer_metrics and transformer_metrics.get("status") == "Completed":
        trans_data = {
            "name": "Transformer (MiniLM + LR)",
            "accuracy": transformer_metrics["accuracy"],
            "macro_f1": transformer_metrics["macro"]["f1_score"],
            "class_0_recall": transformer_metrics["per_class"]["low_or_medium_satisfaction"]["recall"]
        }
    
    # Assemble models list
    models_to_plot = []
    if baseline_lr:
        models_to_plot.append(baseline_lr)
    if text_cnn:
        models_to_plot.append(text_cnn)
    if trans_data:
        models_to_plot.append(trans_data)
    else:
        # If transformer was skipped, put a placeholder in chart
        models_to_plot.append({
            "name": "Transformer (Colab Ready)",
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "class_0_recall": 0.0,
            "is_placeholder": True
        })
        
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    metrics_names = ["Accuracy", "Macro F1-score", "Class 0 Recall (Minority)"]
    x = np.arange(len(metrics_names))
    width = 0.25
    
    # Styled Seaborn look
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Model colors
    colors = [COLOR_PLUM, COLOR_GOLD, COLOR_ROSE]
    
    for i, model in enumerate(models_to_plot):
        vals = [model["accuracy"], model["macro_f1"], model["class_0_recall"]]
        rects_offset = x + (i - len(models_to_plot)/2 + 0.5) * width
        
        if model.get("is_placeholder", False):
            # Draw placeholder bar with dashed pattern
            rects = ax.bar(rects_offset, [0.05, 0.05, 0.05], width, label=model["name"],
                           color='none', edgecolor=COLOR_MUTED, linestyle='--', linewidth=1.5)
            # Add text annotation
            for rect in rects:
                height = rect.get_height()
                ax.annotate('Pending Colab',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, color=COLOR_MUTED, fontweight='bold')
        else:
            rects = ax.bar(rects_offset, vals, width, label=model["name"], color=colors[i % len(colors)], edgecolor='white', linewidth=1)
            # Annotate values
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1%}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold', color='#332633')
                            
    ax.set_ylabel('Score / Percentage', fontsize=11, fontweight='bold', color=COLOR_PLUM, labelpad=10)
    ax.set_title('Skincare Satisfaction Classifier Comparison: TF-IDF vs. CNN vs. Transformer', fontsize=13, fontweight='bold', pad=20, color=COLOR_PLUM)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=10, color='#332633')
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=True, facecolor=COLOR_CREAM, edgecolor='#EAE6DF', loc='lower left', fontsize=9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#EAE6DF')
    ax.spines['bottom'].set_color('#EAE6DF')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    output_path = figures_dir / "transformer_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot successfully saved to: {output_path.name}")

def generate_notebook(notebooks_dir, local_run, subset_size):
    """
    Generates the Jupyter notebook structure using nbformat and saves it.
    """
    import nbformat as nbf
    
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Cell 1: Header Markdown
    cells.append(nbf.v4.new_markdown_cell(f"""# GlowWise AI - Skincare Review Satisfaction Classification 🧴✨
## NLP Transfer Learning & Sentence Embeddings Experiment

This notebook details our experiment leveraging **Transfer Learning** and **Transformer Sentence Representations** to predict review satisfaction (`high_satisfaction`). 

* **Target Variable**: `high_satisfaction` (Rating 4-5 = 1, Rating 1-3 = 0)
* **Execution Environment**: {"Executed Locally on CPU" if local_run else "Prepared for Google Colab GPU / High RAM CPU"}
* **Pre-trained Representation Model**: `sentence-transformers/all-MiniLM-L6-v2` (Sentence Embeddings)

---

### What is Transfer Learning in this Project?
In classical natural language processing, text is transformed into numerical vectors using statistics like word frequencies (e.g., TF-IDF). While highly effective and extremely fast, these representations treat vocabulary words as isolated tokens (Bag-of-Words) and completely discard grammar, word order, and context.

**Transfer Learning** shifts the paradigm. Instead of learning representations from scratch, we use a deep neural network (a Transformer model) that has been pre-trained on billions of sentences to understand generic English semantics. By feeding our review texts through this network, we extract dense contextual representations (**embeddings**). We then *transfer* this pre-trained language understanding to our specific task (skincare review classification) by training a simple linear classifier (like Logistic Regression or LinearSVC) on top of the dense embeddings.

### Why Transformer NLP for Skincare Reviews?
Reviews are rich in semantic nuances, context-shifting qualifiers, and multi-word idioms. Simple Bag-of-Words vectors fail on context. Consider:
1. **Negation & Transitions**: *"not dry at all"* vs *"dry and not good"*. TF-IDF views "dry", "not", and "good" as independent weights. Transformers understand that "not" modifies "dry" based on attention mappings.
2. **Skincare Idioms**: *"broke me out"* is a strong negative signal in skincare, but split into individual words "broke", "me", "out", its meaning is lost. A transformer embedding maps the sequence *"broke me out"* into a vector region close to synonyms like "allergic reaction" or "irritating".
3. **Robustness**: Dense representations generalize much better to spelling variations and brand/product names not seen in training.
"""))

    # Cell 2: Setup, packages, and GPU check code
    cells.append(nbf.v4.new_code_cell("""# 1. Install & Import Dependencies
# If running in Google Colab, uncomment the next line to install sentence-transformers:
# !pip install sentence-transformers torch

import os
import sys
import json
import time
import random
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print("GPU Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using GPU Device:", torch.cuda.get_device_name(0))
except ImportError:
    print("WARNING: PyTorch not found. Please install torch.")

try:
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    print(f"SentenceTransformers version: {sentence_transformers.__version__}")
except ImportError:
    print("WARNING: sentence-transformers not found. Please install sentence-transformers.")
"""))

    # Cell 3: Data loading markdown
    cells.append(nbf.v4.new_markdown_cell("""## 2. Load Processed Reviews Data

We load the processed reviews dataset sample (`glowwise_reviews_sample_100k.csv`).
If running in Google Colab:
1. Upload the CSV file manually to Colab's file storage (`/content/`).
2. Alternatively, mount your Google Drive and load it.
"""))

    # Cell 4: Data loading code
    cells.append(nbf.v4.new_code_cell("""# Locate and load the preprocessed reviews sample
data_paths = [
    "glowwise_reviews_sample_100k.csv",
    "/content/glowwise_reviews_sample_100k.csv",
    "/content/drive/MyDrive/glowwise_reviews_sample_100k.csv",
    "../../data/processed/glowwise_reviews_sample_100k.csv",
    "data/processed/glowwise_reviews_sample_100k.csv"
]

data_path = None
for path in data_paths:
    if os.path.exists(path):
        data_path = path
        print(f"Found dataset at: {path}")
        break

if data_path is None:
    raise FileNotFoundError("Dataset sample CSV not found. Please place or upload 'glowwise_reviews_sample_100k.csv'.")

df = pd.read_csv(data_path)

# Clean text features
for col in ["review_title", "review_text", "combined_text"]:
    if col in df.columns:
        df[col] = df[col].fillna("")
        
df = df.dropna(subset=["high_satisfaction"])
df = df[df["combined_text"].str.strip() != ""]
print(f"Total reviews loaded and verified: {len(df):,}")
"""))

    # Cell 5: Train/Test Split markdown
    cells.append(nbf.v4.new_markdown_cell("""## 3. Stratified Train / Test Split & Down-Sampling

To maintain complete reproducibility, we split our data using `random_state=42`.
Because extracting transformer embeddings on CPU can be slow, we extract a stratified training subset of **15,000 reviews** and evaluate on a test subset of **5,000 reviews**. In Google Colab with GPU acceleration, you can easily scale these numbers to the full dataset!
"""))

    # Cell 6: Train/Test Split code
    cells.append(nbf.v4.new_code_cell(f"""from sklearn.model_selection import train_test_split

X = df
y = df["high_satisfaction"].astype(int)

# First split into 80/20 train/test
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Extract a smaller stratified training subset for fast embedding extraction
train_subset_size = {subset_size}
if len(X_train_df) > train_subset_size:
    _, X_train_sub, _, y_train_sub = train_test_split(
        X_train_df, y_train, test_size=train_subset_size, stratify=y_train, random_state=42
    )
else:
    X_train_sub = X_train_df
    y_train_sub = y_train

# Extract a stratified test subset for evaluation speed
test_subset_size = 5000
if len(X_test_df) > test_subset_size:
    _, X_test_sub, _, y_test_sub = train_test_split(
        X_test_df, y_test, test_size=test_subset_size, stratify=y_test, random_state=42
    )
else:
    X_test_sub = X_test_df
    y_test_sub = y_test

train_texts = X_train_sub["combined_text"].astype(str).tolist()
test_texts = X_test_sub["combined_text"].astype(str).tolist()

print(f"Train subset size: {{len(X_train_sub):,}}")
print(f"Test subset size: {{len(X_test_sub):,}}")
"""))

    # Cell 7: Embedding extraction markdown
    cells.append(nbf.v4.new_markdown_cell("""## 4. Dense Embedding Extraction using all-MiniLM-L6-v2

We load the lightweight sentence representation model `sentence-transformers/all-MiniLM-L6-v2`. This model maps reviews to 384-dimensional dense vectors.
We also set up a local cache directory `ml/cache/` to store the numpy arrays so we don't re-embed reviews on consecutive runs.
"""))

    # Cell 8: Embedding extraction code
    cells.append(nbf.v4.new_code_cell("""# Set up caching directory
cache_dir = Path("ml/cache") if os.path.exists("ml") else Path("cache")
cache_dir.mkdir(exist_ok=True)

train_cache_path = cache_dir / f"train_embeddings_{len(X_train_sub)}.npy"
test_cache_path = cache_dir / f"test_embeddings_{len(X_test_sub)}.npy"

# Load pre-trained SentenceTransformer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading SentenceTransformer: {model_name} on '{device}'...")
st_model = SentenceTransformer(model_name, device=device)

# Encode training reviews
if train_cache_path.exists():
    print("Loading train embeddings from cache...")
    X_train_emb = np.load(train_cache_path)
else:
    print("Extracting train embeddings (encoding reviews)...")
    t0 = time.time()
    X_train_emb = st_model.encode(train_texts, show_progress_bar=True, batch_size=64)
    np.save(train_cache_path, X_train_emb)
    print(f"Train embeddings shape: {X_train_emb.shape} (extracted in {time.time() - t0:.2f} seconds)")

# Encode testing reviews
if test_cache_path.exists():
    print("Loading test embeddings from cache...")
    X_test_emb = np.load(test_cache_path)
else:
    print("Extracting test embeddings...")
    t0 = time.time()
    X_test_emb = st_model.encode(test_texts, show_progress_bar=True, batch_size=64)
    np.save(test_cache_path, X_test_emb)
    print(f"Test embeddings shape: {X_test_emb.shape} (extracted in {time.time() - t0:.2f} seconds)")
"""))

    # Cell 9: Classifier training markdown
    cells.append(nbf.v4.new_markdown_cell("""## 5. Classifier Training & Hyperparameters

We train a **Logistic Regression** classifier (with `class_weight='balanced'`) on top of the dense embeddings. Since embeddings are already low-dimensional and regularized, a simple linear decision boundary works incredibly well, has sub-millisecond inference time, and avoids overfitting.
"""))

    # Cell 10: Classifier training code
    cells.append(nbf.v4.new_code_cell("""from sklearn.linear_model import LogisticRegression

print("Training Logistic Regression classifier on embeddings...")
clf = LogisticRegression(class_weight="balanced", C=1.0, random_state=42, max_iter=1000)

t0 = time.time()
clf.fit(X_train_emb, y_train_sub)
train_time = time.time() - t0

print(f"Model training complete in {train_time:.4f} seconds!")
"""))

    # Cell 11: Evaluation markdown
    cells.append(nbf.v4.new_markdown_cell("""## 6. Evaluation on Test Subset

We evaluate the trained model on our test subset. We compute:
* Accuracy
* Macro and Weighted F1-score
* Class-specific Precision and Recall
* ROC-AUC and PR-AUC
"""))

    # Cell 12: Evaluation code
    cells.append(nbf.v4.new_code_cell("""from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, precision_recall_curve, auc

# Get predictions
y_pred = clf.predict(X_test_emb)
y_prob = clf.predict_proba(X_test_emb)[:, 1]

# Calculate classification metrics
accuracy = accuracy_score(y_test_sub, y_pred)
p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_test_sub, y_pred, average="macro", zero_division=0)
p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(y_test_sub, y_pred, average="weighted", zero_division=0)
p_classes, r_classes, f_classes, support = precision_recall_fscore_support(y_test_sub, y_pred, average=None, zero_division=0)

# Curves
fpr, tpr, _ = roc_curve(y_test_sub, y_prob)
roc_auc = auc(fpr, tpr)

precision_vals, recall_vals, _ = precision_recall_curve(y_test_sub, y_prob)
pr_auc = auc(recall_vals, precision_vals)

transformer_metrics = {
    "accuracy": float(accuracy),
    "macro": {
        "precision": float(p_macro),
        "recall": float(r_macro),
        "f1_score": float(f_macro)
    },
    "weighted": {
        "precision": float(p_weighted),
        "recall": float(r_weighted),
        "f1_score": float(f_weighted)
    },
    "per_class": {
        "low_or_medium_satisfaction": {
            "precision": float(p_classes[0]),
            "recall": float(r_classes[0]),
            "f1_score": float(f_classes[0]),
            "support": int(support[0])
        },
        "high_satisfaction": {
            "precision": float(p_classes[1]),
            "recall": float(r_classes[1]),
            "f1_score": float(f_classes[1]),
            "support": int(support[1])
        }
    },
    "training_time_seconds": float(train_time),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "training_size": len(X_train_sub),
    "status": "Completed",
    "description": "Logistic Regression trained on sentence-transformers/all-MiniLM-L6-v2 embeddings"
}

print("=== Transformer Performance Metrics ===")
print(json.dumps(transformer_metrics, indent=4))
"""))

    # Cell 13: Plot confusion matrix and curves code
    cells.append(nbf.v4.new_code_cell("""# 1. Plot Confusion Matrix
cm = confusion_matrix(y_test_sub, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(6.5, 5.5), layout="constrained")
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("glowwise_cmap", ['#FCFAF7', '#3B243B'])
plt.imshow(cm_norm, interpolation='nearest', cmap=cmap)
plt.title('Transformer + LR Confusion Matrix', fontweight='bold', pad=25, fontsize=11, color='#3B243B')
plt.colorbar(pad=0.08, format='%.2f')
classes = ['Low/Medium', 'High']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted Label', fontweight='bold', labelpad=10, color='#3B243B')
plt.ylabel('True Label', fontweight='bold', labelpad=10, color='#3B243B')

thresh = cm_norm.max() / 2.
for i, j in np.ndindex(cm.shape):
    val_str = f"{cm[i, j]:,}\\n({cm_norm[i, j]:.1%})"
    plt.text(j, i, val_str,
             ha="center", va="center",
             color="white" if cm_norm[i, j] > thresh else "#3B243B",
             fontweight='bold')
plt.savefig('transformer_confusion_matrix.png', dpi=150)
plt.show()

# 2. Plot ROC and PR Curves
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 5.5))

ax_roc.plot(fpr, tpr, color='#3B243B', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax_roc.set_xlabel('False Positive Rate', fontweight='bold')
ax_roc.set_ylabel('True Positive Rate', fontweight='bold')
ax_roc.set_title('Receiver Operating Characteristic (ROC)', fontweight='bold')
ax_roc.legend()
ax_roc.grid(True, linestyle='--', alpha=0.5)

ax_pr.plot(recall_vals, precision_vals, color='#C39B6F', linewidth=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
ax_pr.set_xlabel('Recall', fontweight='bold')
ax_pr.set_ylabel('Precision', fontweight='bold')
ax_pr.set_title('Precision-Recall Curve', fontweight='bold')
ax_pr.legend()
ax_pr.grid(True, linestyle='--', alpha=0.5)

plt.suptitle('Transformer Experiment ROC & PR Evaluation Curves', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('transformer_curves.png', dpi=150)
plt.show()
"""))

    # Cell 14: Save results & download code for Colab users
    cells.append(nbf.v4.new_code_cell("""# Save outputs locally (or in Google Colab environment)
with open('transformer_results.json', 'w', encoding='utf-8') as f:
    json.dump(transformer_metrics, f, indent=4)
print("Saved transformer_results.json successfully!")

# Colab automatic downloads helper
try:
    from google.colab import files
    print("Downloading transformer_results.json...")
    files.download('transformer_results.json')
    files.download('transformer_confusion_matrix.png')
    files.download('transformer_curves.png')
except Exception as e:
    print("Local notebook run detected; skip automatic file download browser prompts.")
"""))

    # Cell 15: Conclusions & production rationale markdown
    cells.append(nbf.v4.new_markdown_cell("""## 7. Rationale for Production Choice and Future Work

### Performance & Compute Cost Trade-Offs
* **Pre-trained Embeddings**: Dense language vectors represent a major architectural leap over token counters (TF-IDF), providing superior contextual mapping of review syntax.
* **Why Logistic Regression remains the production choice**:
  * **Latency**: Simple TF-IDF + Logistic Regression returns predictions in ~1-5 milliseconds on CPU, whereas downloading, loading, and encoding text using deep transformer architectures adds hundreds of milliseconds of compute overhead.
  * **Deployability**: Serves easily in lightweight REST frameworks (FastAPI) without needing GPU acceleration, massive environment dependencies (`torch`), or virtual memory capacity.
  * **Explainability**: Model weights map directly back to individual keywords, allowing products to explain exactly which terms drive positive or negative classification.

### Future Direction: Fine-Tuning
Our current experiment uses **Feature Extraction** (extracting representations from a frozen model). A next logical iteration is **fine-tuning**—updating the weights of a model like `DistilBERT` or `BERT-Base` directly on Sephora skincare reviews using a GPU. Fine-tuning allows the transformer attention weights to adapt directly to domain-specific vocabularies (skin types, ingredient efficacy, skincare product names), yielding higher accuracy on low-support satisfaction labels.
"""))

    nb['cells'] = cells
    
    # Save the notebook
    notebook_path = notebooks_dir / "09_transformer_transfer_learning.ipynb"
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Jupyter notebook written to: {notebook_path.name}")
    
    # Execute notebook if local run is enabled
    if local_run:
        print("Executing notebook to populate cell outputs...")
        try:
            from nbconvert.preprocessors import ExecutePreprocessor
            ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
            ep.preprocess(nb, {'metadata': {'path': str(notebooks_dir)}})
            
            # Save again with outputs populated
            with open(notebook_path, 'w', encoding='utf-8') as f:
                nbf.write(nb, f)
            print("Notebook execution succeeded and outputs saved.")
        except Exception as e:
            print(f"Warning during notebook execution: {e}")
            print("This is normal if jupyter execute preprocessors are not fully set up. The notebook structure is still saved.")

if __name__ == "__main__":
    # Check for CLI arguments for subset size
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-size", type=int, default=10000, help="Stratified training subset size")
    args, unknown = parser.parse_known_args()
    
    run_experiment(subset_size=args.subset_size)
