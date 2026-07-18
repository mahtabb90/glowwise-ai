"""
GlowWise AI - Autoencoder Anomaly Detection Experiment
Generates the Colab-ready notebook for unsupervised anomaly detection.
Saves local results JSON in prepared/skipped state since TensorFlow/Keras is not available locally.
"""

import os
import sys
import json
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def run_preparation():
    print("=== Starting GlowWise AI Autoencoder Anomaly Detection Preparation ===")
    
    root_dir = get_project_root()
    data_path = root_dir / "data" / "processed" / "glowwise_reviews_sample_100k.csv"
    reports_dir = root_dir / "ml" / "reports"
    notebooks_dir = root_dir / "ml" / "notebooks"
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Check dataset existence
    if not data_path.exists():
        print(f"[ERROR] Processed dataset not found at: {data_path.absolute()}")
        print("Please run data preprocessing first.")
        sys.exit(1)
        
    print(f"Verified dataset sample source: {data_path.name}")
    
    # 2. Write autoencoder_anomaly_results.json in prepared_for_colab state
    results_path = reports_dir / "autoencoder_anomaly_results.json"
    results_data = {
        "status": "prepared_for_colab",
        "local_execution": "skipped",
        "reason": "TensorFlow/Keras not available locally"
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4)
    print(f"Saved preparation status log: {results_path.name}")
    
    # 3. Generate the unexecuted Colab-ready notebook
    generate_notebook(notebooks_dir)

def generate_notebook(notebooks_dir: Path):
    import nbformat as nbf
    
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Cell 1: Header Markdown
    cells.append(nbf.v4.new_markdown_cell("""# GlowWise AI - Skincare Review Anomaly Detection 🧴✨
## Unsupervised Feature Learning & Reconstruction Error Evaluation

This notebook details our experiment leveraging **Autoencoder Neural Networks** to identify unusual or anomalous skincare reviews.

* **Objective**: Identify reviews that deviate significantly from typical vocabularies and syntax.
* **Input Features**: TF-IDF vectors (max_features=5000) reduced to dense vectors using TruncatedSVD (n_components=100).
* **Model Architecture**: Keras Symmetric Dense Autoencoder (`100 -> 64 -> 32 -> 16 -> 32 -> 64 -> 100`).
* **Evaluation Metric**: Mean Squared Error (MSE) reconstruction error.
* **Anomaly Threshold**: 95th percentile of reconstruction error.

---

### What is Anomaly Detection with an Autoencoder?
Unlike supervised classification which learns to map reviews to target labels (e.g. `high_satisfaction`), **unsupervised anomaly detection** learns to reconstruct the input. 

An **Autoencoder** squeezes the input vectors through a low-dimensional bottleneck (16 dimensions) and then reconstructs them. 
* Because standard skincare reviews follow typical linguistic styles, product sentiments, and topics, the network learns to reconstruct them with high fidelity (yielding **low reconstruction error**).
* Unusual reviews—such as spam, copy-paste errors, promotional codes, gibberish, or highly atypical product issues—contain features that the autoencoder has not learned to generalize, yielding **high reconstruction error**.
"""))

    # Cell 2: Setup, GPU check, and dependency installation code
    cells.append(nbf.v4.new_code_cell("""# 1. Install & Import Dependencies
# If running in Google Colab, uncomment and run the following line:
# !pip install tensorflow scikit-learn pandas numpy matplotlib

import os
import json
import time
import random
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for complete reproducibility
random.seed(42)
np.random.seed(42)

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.callbacks import EarlyStopping
    print(f"TensorFlow version: {tf.__version__}")
    # Set seed for TensorFlow
    tf.random.set_seed(42)
except ImportError:
    print("WARNING: TensorFlow not found. Please install tensorflow.")
"""))

    # Cell 3: Data loading markdown
    cells.append(nbf.v4.new_markdown_cell("""## 2. Load Processed Reviews Data

We search for the processed reviews dataset. If you are running in Google Colab:
1. Upload `glowwise_reviews_sample_100k.csv` directly to the left panel (Colab's folder `/content/`).
2. Run the cell below to load the dataset.
"""))

    # Cell 4: Data loading code
    cells.append(nbf.v4.new_code_cell("""# Search locations for the preprocessed dataset
data_paths = [
    "glowwise_reviews_sample_100k.csv",
    "/content/glowwise_reviews_sample_100k.csv",
    "../../data/processed/glowwise_reviews_sample_100k.csv",
    "data/processed/glowwise_reviews_sample_100k.csv"
]

data_path = None
for path in data_paths:
    if os.path.exists(path):
        data_path = path
        print(f"Found reviews dataset at: {path}")
        break

if data_path is None:
    raise FileNotFoundError("Dataset sample CSV not found. Please upload 'glowwise_reviews_sample_100k.csv' to the files tab.")

df = pd.read_csv(data_path)

# Fill text null values
for col in ["review_title", "review_text", "combined_text"]:
    if col in df.columns:
        df[col] = df[col].fillna("")

# Ensure we have a valid text column
if "combined_text" not in df.columns or df["combined_text"].str.strip().eq("").all():
    print("Generating combined_text feature...")
    df["combined_text"] = df["review_title"].astype(str) + " " + df["review_text"].astype(str)

df = df.dropna(subset=["high_satisfaction"])
df = df[df["combined_text"].str.strip() != ""]
print(f"Total reviews verified: {len(df):,}")
"""))

    # Cell 5: Down-sampling and Text Vectorization markdown
    cells.append(nbf.v4.new_markdown_cell("""## 3. Manageable Sampling & Dense Vectorization

To balance execution speed and data density, we sample **20,000 reviews** randomly (using `random_state=42`). 

To prepare text for a neural network autoencoder:
1. **TF-IDF Vectorization**: Extract word frequencies and filter for the top 5,000 features.
2. **Truncated SVD (Dimensionality Reduction)**: Reduce the 5,000-dimensional sparse representations into **100-dimensional dense vectors**. This captures semantic features while removing sparsity, which is ideal for standard fully-connected autoencoders.
"""))

    # Cell 6: Sampling & Vectorization code
    cells.append(nbf.v4.new_code_cell("""from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Take a random sample of 20,000 reviews
sample_size = min(20000, len(df))
df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
print(f"Subsampled dataset to {len(df_sample):,} reviews.")

# Fit TF-IDF Vectorizer
print("Extracting TF-IDF features (max_features=5000)...")
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X_tfidf = tfidf.fit_transform(df_sample["combined_text"])

# Reduce dimensions to 100 components using SVD
print("Reducing dimensions to 100 components using TruncatedSVD...")
svd = TruncatedSVD(n_components=100, random_state=42)
X_dense = svd.fit_transform(X_tfidf)

print(f"Dense vector representation shape: {X_dense.shape}")

# Split into train and validation sets (90% train / 10% validation)
X_train, X_val = train_test_split(X_dense, test_size=0.1, random_state=42)
print(f"Train subset size: {X_train.shape[0]:,}, Validation subset size: {X_val.shape[0]:,}")
"""))

    # Cell 7: Autoencoder network definition markdown
    cells.append(nbf.v4.new_markdown_cell("""## 4. Build and Train the Autoencoder

We build a symmetric deep fully-connected autoencoder:
* **Input Layer**: 100 dense SVD features.
* **Encoder**:
  * Dense (64 units, ReLU activation)
  * Dense (32 units, ReLU activation)
* **Bottleneck**:
  * Dense (16 units, ReLU activation)
* **Decoder**:
  * Dense (32 units, ReLU activation)
  * Dense (64 units, ReLU activation)
* **Output Layer**:
  * Dense (100 units, Linear activation) - reconstructs the input vector.

We train the model using **Mean Squared Error (MSE) loss** and the **Adam optimizer**, with an `EarlyStopping` callback to prevent overfitting.
"""))

    # Cell 8: Model Definition & Training code
    cells.append(nbf.v4.new_code_cell("""# Define Autoencoder model
input_dim = 100
bottleneck_dim = 16

model = Sequential([
    # Encoder
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dense(32, activation='relu'),
    
    # Bottleneck representation
    Dense(bottleneck_dim, activation='relu', name="bottleneck"),
    
    # Decoder
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    
    # Reconstructed Output
    Dense(input_dim, activation='linear')
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Setup early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Train the model (reconstructing the input X_train)
t0 = time.time()
history = model.fit(
    X_train, X_train,
    validation_data=(X_val, X_val),
    epochs=50,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)
training_time = time.time() - t0
print(f"Model trained successfully in {training_time:.2f} seconds!")
"""))

    # Cell 9: Training curves visualization code
    cells.append(nbf.v4.new_code_cell("""# 1. Plot Training History (Loss Curves)
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss', color='#3B243B', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='#C39B6F', linewidth=2, linestyle='--')
plt.title('Autoencoder Loss Convergence', fontsize=12, fontweight='bold', pad=15, color='#3B243B')
plt.xlabel('Epochs', fontsize=10, fontweight='bold', labelpad=10)
plt.ylabel('Mean Squared Error (MSE)', fontsize=10, fontweight='bold', labelpad=10)
plt.legend(frameon=True, facecolor='#FCFAF7')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Save locally to figures folder
figures_dir = Path("ml/reports/figures") if os.path.exists("ml") else Path("reports/figures")
figures_dir.mkdir(parents=True, exist_ok=True)
curves_output_path = figures_dir / "autoencoder_training_curves.png"
plt.savefig(curves_output_path, dpi=150)
plt.show()
print(f"Training curves saved to: {curves_output_path}")
"""))

    # Cell 10: Reconstruction error calculation markdown
    cells.append(nbf.v4.new_markdown_cell("""## 5. Calculate Reconstruction Error and Flag Threshold

We calculate the reconstruction error for all 20,000 reviews in our sample.
* **Reconstruction Error** is computed as the MSE between the original dense vector and its reconstructed vector output.
* **Anomaly Threshold**: Following standard practices, we select the **95th percentile** of reconstruction error. Any review yielding an MSE above this value is flagged as an anomaly.
"""))

    # Cell 11: Reconstruction error code
    cells.append(nbf.v4.new_code_cell("""# Predict reconstructed representations
X_reconstructed = model.predict(X_dense)

# Calculate MSE error for each review
reconstruction_errors = np.mean(np.square(X_dense - X_reconstructed), axis=1)

# Add errors to our sample DataFrame
df_sample["reconstruction_error"] = reconstruction_errors

# Define threshold at 95th percentile
threshold = np.percentile(reconstruction_errors, 95)
df_sample["is_anomaly"] = df_sample["reconstruction_error"] > threshold

print(f"Reconstruction Error statistics:")
print(f" - Min Error:  {reconstruction_errors.min():.6f}")
print(f" - Mean Error: {reconstruction_errors.mean():.6f}")
print(f" - Max Error:  {reconstruction_errors.max():.6f}")
print(f" - Anomaly Threshold (95th percentile): {threshold:.6f}")
print(f" - Total Anomalies Flagged: {df_sample['is_anomaly'].sum():,} out of {len(df_sample):,}")
"""))

    # Cell 12: Distribution plot code
    cells.append(nbf.v4.new_code_cell("""# 2. Plot Reconstruction Error Distribution Histogram
plt.figure(figsize=(9, 5.5), layout="constrained")
plt.hist(reconstruction_errors, bins=50, color='#E8D3C4', edgecolor='#3B243B', alpha=0.85, label='Reviews')
plt.axvline(threshold, color='#3B243B', linestyle='--', linewidth=2, label=f'Anomaly Threshold (95th% = {threshold:.5f})')
plt.title('Distribution of Autoencoder Reconstruction Errors', fontsize=12, fontweight='bold', pad=25, color='#3B243B')
plt.xlabel('Reconstruction Mean Squared Error (MSE)', fontsize=10, fontweight='bold', labelpad=10, color='#3B243B')
plt.ylabel('Number of Reviews', fontsize=10, fontweight='bold', labelpad=10, color='#3B243B')
plt.legend(frameon=True, facecolor='#FCFAF7')
plt.grid(True, linestyle='--', alpha=0.3)

dist_output_path = figures_dir / "autoencoder_reconstruction_error_distribution.png"
plt.savefig(dist_output_path, dpi=150)
plt.show()
print(f"Distribution plot saved to: {dist_output_path}")
"""))

    # Cell 13: Extract top anomalies markdown
    cells.append(nbf.v4.new_markdown_cell("""## 6. Extract Top Anomalous Reviews

We sort flagged anomalies descending by reconstruction error to isolate the most unusual reviews in the dataset.
We then plot their MSE values in a horizontal bar chart.
"""))

    # Cell 14: Top anomalies extraction and plotting code
    cells.append(nbf.v4.new_code_cell("""# Isolate anomalies and sort
anomalies_df = df_sample[df_sample["is_anomaly"]].sort_values(by="reconstruction_error", ascending=False).reset_index(drop=True)

# Select top 10 anomalies for visualization
top_10_anomalies = anomalies_df.head(10)

# Create horizontal bar chart of top anomalies
plt.figure(figsize=(10, 6), layout="constrained")
y_ticks = np.arange(10)

# Truncate text snippets to use as labels
labels = []
for idx, row in top_10_anomalies.iterrows():
    title = str(row['review_title']).strip()
    text = str(row['review_text']).strip()[:40] + "..."
    lbl = f"[{row['brand_name'] if 'brand_name' in row and pd.notna(row['brand_name']) else 'Index ' + str(idx)}] "
    lbl += f"'{title}'" if title else text
    labels.append(lbl)

plt.barh(y_ticks, top_10_anomalies["reconstruction_error"][::-1], color='#C39B6F', edgecolor='#3B243B', height=0.6)
plt.yticks(y_ticks, labels[::-1], fontsize=9)
plt.xlabel('Reconstruction Mean Squared Error (MSE)', fontsize=10, fontweight='bold', labelpad=10, color='#3B243B')
plt.title('Top 10 Most Anomalous Skincare Reviews', fontsize=12, fontweight='bold', pad=25, color='#3B243B')
plt.grid(axis='x', linestyle='--', alpha=0.4)

anom_plot_path = figures_dir / "autoencoder_top_anomalies.png"
plt.savefig(anom_plot_path, dpi=150)
plt.show()
print(f"Top anomalies plot saved to: {anom_plot_path}")
"""))

    # Cell 15: Save output metrics, anomalous reviews list code
    cells.append(nbf.v4.new_code_cell("""# 1. Save top anomalous reviews to CSV
reports_dir = Path("ml/reports") if os.path.exists("ml") else Path("reports")
reports_dir.mkdir(parents=True, exist_ok=True)

csv_columns = ["review_title", "review_text", "high_satisfaction", "reconstruction_error"]
for col in ["product_name", "brand_name"]:
    if col in anomalies_df.columns:
        csv_columns.append(col)

anom_csv_path = reports_dir / "autoencoder_top_anomalous_reviews.csv"
anomalies_df[csv_columns].to_csv(anom_csv_path, index=False, encoding="utf-8")
print(f"Saved anomalies list to: {anom_csv_path}")

# 2. Save final experiment results JSON
results_json_path = reports_dir / "autoencoder_anomaly_results.json"
results_metrics = {
    "status": "completed",
    "local_execution": "completed_via_colab",
    "sample_size": int(sample_size),
    "epochs_trained": len(history.history['loss']),
    "final_training_loss_mse": float(history.history['loss'][-1]),
    "final_validation_loss_mse": float(history.history['val_loss'][-1]),
    "anomaly_threshold_95th_percentile": float(threshold),
    "total_anomalies_count": int(df_sample['is_anomaly'].sum()),
    "reconstruction_error_mean": float(reconstruction_errors.mean()),
    "reconstruction_error_std": float(reconstruction_errors.std()),
    "training_time_seconds": float(training_time)
}

with open(results_json_path, "w", encoding="utf-8") as f:
    json.dump(results_metrics, f, indent=4)
print(f"Saved final results metrics: {results_json_path}")
"""))

    # Cell 16: Colab downloads helper code
    cells.append(nbf.v4.new_code_cell("""# Download files from Google Colab environment
try:
    from google.colab import files
    print("Preparing downloads...")
    files.download('reports/autoencoder_anomaly_results.json')
    files.download('reports/autoencoder_top_anomalous_reviews.csv')
    files.download('reports/figures/autoencoder_training_curves.png')
    files.download('reports/figures/autoencoder_reconstruction_error_distribution.png')
    files.download('reports/figures/autoencoder_top_anomalies.png')
except Exception as e:
    print("Not running in Colab; skipping automatic file downloads.")
"""))

    # Cell 17: Colab copy instructions markdown
    cells.append(nbf.v4.new_markdown_cell("""## 7. Instructions for Local Integration

Once you have executed all cells in this notebook and downloaded the resulting files, follow these steps to copy them back into your local git repository:

1. **Move Results JSON**:
   Copy `autoencoder_anomaly_results.json` into:
   `ml/reports/`

2. **Move Anomalous Reviews CSV**:
   Copy `autoencoder_top_anomalous_reviews.csv` into:
   `ml/reports/`

3. **Move Figures**:
   Copy the three PNG charts into:
   `ml/reports/figures/`
   * `autoencoder_training_curves.png`
   * `autoencoder_reconstruction_error_distribution.png`
   * `autoencoder_top_anomalies.png`

4. **Verify Git Diff**:
   Run `git status` locally. You should see the generated reports, plots, and updated results JSON added under the `ml/` experimental directory.
"""))

    nb['cells'] = cells
    
    # Save the notebook
    notebook_path = notebooks_dir / "10_autoencoder_anomaly_detection.ipynb"
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Jupyter notebook written to: {notebook_path.name}")

if __name__ == "__main__":
    run_preparation()
