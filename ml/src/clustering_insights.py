"""
GlowWise AI - Unsupervised Clustering & Skincare Review Insights
Processes 100k skincare reviews using TF-IDF and TruncatedSVD.
Evaluates cluster configurations for k=3 to 8 using MiniBatchKMeans and sampled Silhouette scores.
Profiles and labels 5 final clusters based on actual text, brands, satisfaction, and sentiment.
Extracts representative reviews closest to cluster centroids in SVD space.
Saves metrics, profiles, examples, and wellness-themed Matplotlib plots.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

# Add script directory to path
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def run_clustering():
    print("=== Starting GlowWise AI Unsupervised Clustering Insights Pipeline ===")
    
    root_dir = get_project_root()
    data_path = root_dir / "data" / "processed" / "glowwise_reviews_sample_100k.csv"
    reports_dir = root_dir / "ml" / "reports"
    figures_dir = reports_dir / "figures"
    
    # Ensure directories exist
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Existence check for dataset
    if not data_path.exists():
        print(f"\n[ERROR] Processed sample dataset not found at: {data_path.absolute()}")
        print("Please run the data preprocessing script first to generate this file:")
        print("   python ml/src/preprocess_data.py\n")
        sys.exit(1)
        
    print(f"Loading preprocessed sample: {data_path.name}")
    df = pd.read_csv(data_path)
    
    # Safe imputation for columns on load
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
        
    # Drop rows with missing targets or empty combined_text
    df = df.dropna(subset=["high_satisfaction"])
    df = df[df["combined_text"].str.strip() != ""].reset_index(drop=True)
    print(f"Dataset shape: {df.shape}")
    
    # 2. Unsupervised Feature Extraction (TF-IDF + TruncatedSVD)
    print("Fitting TF-IDF Vectorizer on combined_text (unsupervised)...")
    tfidf = TfidfVectorizer(
        stop_words="english",
        min_df=5,
        max_df=0.85,
        max_features=10000,
        ngram_range=(1, 2)
    )
    X_tfidf = tfidf.fit_transform(df["combined_text"])
    print(f"TF-IDF Matrix shape: {X_tfidf.shape}")
    
    print("Reducing sparse TF-IDF to 50 dense components using TruncatedSVD...")
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_svd = svd.fit_transform(X_tfidf)
    print(f"TruncatedSVD Matrix shape: {X_svd.shape}")
    
    # 3. Parameter Search for K (3 to 8)
    print("\n--- Evaluate K Values from 3 to 8 ---")
    k_values = list(range(3, 9))
    inertias = []
    silhouettes = []
    
    for k in k_values:
        t0 = time.time()
        kmeans_temp = MiniBatchKMeans(n_clusters=k, batch_size=2048, random_state=42)
        labels_temp = kmeans_temp.fit_predict(X_svd)
        
        # Inertia
        inertia = kmeans_temp.inertia_
        inertias.append(inertia)
        
        # Silhouette Score (subsampled to 10k to prevent OOM)
        sil_score = silhouette_score(X_svd, labels_temp, sample_size=10000, random_state=42)
        silhouettes.append(sil_score)
        
        print(f"  K = {k} | Inertia: {inertia:,.2f} | Silhouette Score: {sil_score:.4f} | Time: {time.time()-t0:.2f}s")
        
    # Plot 1: K Selection metrics
    fig, ax1 = plt.subplots(figsize=(8, 5), facecolor='#FCFAF7')
    ax1.set_facecolor('#FCFAF7')
    
    COLOR_PLUM = '#3B243B'
    COLOR_GOLD = '#C39B6F'
    COLOR_ROSE = '#E8D3C4'
    COLOR_MUTED = '#6E5C6E'
    COLOR_CREAM = '#FCFAF7'
    
    # Plot Inertia (Elbow method)
    ax1.plot(k_values, inertias, marker='o', color=COLOR_PLUM, linewidth=2, label="Inertia")
    ax1.set_xlabel('Number of Clusters (k)', fontsize=10, fontweight='bold', color=COLOR_PLUM)
    ax1.set_ylabel('Inertia (Elbow Metric)', fontsize=10, fontweight='bold', color=COLOR_PLUM)
    ax1.tick_params(axis='y', labelcolor=COLOR_PLUM)
    ax1.spines['top'].set_visible(False)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Dual axis for Silhouette score
    ax2 = ax1.twinx()
    ax2.plot(k_values, silhouettes, marker='s', color=COLOR_GOLD, linewidth=2, label="Silhouette")
    ax2.set_ylabel('Silhouette Score (Sample size = 10k)', fontsize=10, fontweight='bold', color=COLOR_PLUM)
    ax2.tick_params(axis='y', labelcolor=COLOR_GOLD)
    ax2.spines['top'].set_visible(False)
    
    plt.title('Clustering Parameter Selection (K-Means)', fontsize=12, pad=20, fontweight='bold', color=COLOR_PLUM)
    fig.tight_layout()
    plt.savefig(figures_dir / "clustering_k_selection.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: clustering_k_selection.png")
    
    # 4. Fit Final Model (k=5 based on high business interpretability)
    print("\nFitting final MiniBatchKMeans with k=5 for profiling...")
    final_k = 5
    kmeans = MiniBatchKMeans(n_clusters=final_k, batch_size=2048, random_state=42)
    labels = kmeans.fit_predict(X_svd)
    centroids = kmeans.cluster_centers_
    
    df["cluster"] = labels
    
    # Human-friendly cluster personas assigned based on top terms and satisfaction rates
    persona_mapping = {
        0: "Makeup & General Enthusiasts",
        1: "Daily Skincare Users",
        2: "Moisture & Texture Fans",
        3: "Acne & Blemish Care",
        4: "Lip Care Seekers"
    }
    df["cluster_name"] = df["cluster"].map(persona_mapping)
    
    # Calculate profiles & top brands/terms
    print("\nProfiling final clusters...")
    cluster_profiles = []
    cluster_examples = []
    feature_names = np.array(tfidf.get_feature_names_out())
    
    for c in range(final_k):
        c_df = df[df["cluster"] == c]
        c_size = len(c_df)
        c_pct = c_size / len(df)
        
        # Satisfaction rate
        sat_rate = c_df["high_satisfaction"].mean()
        
        # Sentiment distribution
        sent_counts = c_df["sentiment_label"].value_counts(normalize=True).to_dict()
        sent_pos = sent_counts.get("positive", 0.0)
        sent_neu = sent_counts.get("neutral", 0.0)
        sent_neg = sent_counts.get("negative", 0.0)
        
        # Text details
        avg_len = c_df["text_length"].mean()
        avg_word = c_df["word_count"].mean()
        
        # Top brand
        top_brands = c_df["brand_name"].value_counts().head(3).index.tolist()
        top_brands_str = ", ".join(top_brands)
        
        # Top TF-IDF words
        c_tfidf = X_tfidf[labels == c]
        mean_tfidf = np.asarray(c_tfidf.mean(axis=0)).flatten()
        top_words_indices = np.argsort(mean_tfidf)[-10:]
        top_words = feature_names[top_words_indices][::-1].tolist()
        top_words_str = ", ".join(top_words)
        
        # Append to profiles list
        cluster_profiles.append({
            "cluster_id": c,
            "persona_name": persona_mapping[c],
            "size": c_size,
            "percentage": c_pct,
            "high_satisfaction_rate": sat_rate,
            "sentiment_positive": sent_pos,
            "sentiment_neutral": sent_neu,
            "sentiment_negative": sent_neg,
            "avg_text_length": avg_len,
            "avg_word_count": avg_word,
            "top_brands": top_brands_str,
            "top_terms": top_words_str
        })
        
        # Extract representative reviews closest to cluster centroid
        c_svd = X_svd[labels == c]
        c_indices = np.where(labels == c)[0]
        centroid = centroids[c]
        
        # Compute Euclidean distance from SVD components to centroid
        distances = np.linalg.norm(c_svd - centroid, axis=1)
        closest_rel_indices = np.argsort(distances)[:10]
        closest_abs_indices = c_indices[closest_rel_indices]
        
        for abs_idx in closest_abs_indices:
            row = df.iloc[abs_idx]
            cluster_examples.append({
                "cluster_id": c,
                "persona_name": persona_mapping[c],
                "product_id": row["product_id"],
                "product_name": row["product_name"],
                "brand_name": row["brand_name"],
                "combined_text": row["combined_text"][:200] + "..." if len(str(row["combined_text"])) > 200 else row["combined_text"],
                "high_satisfaction": row["high_satisfaction"],
                "sentiment_label": row["sentiment_label"]
            })
            
    # Export profiles and examples
    profiles_df = pd.DataFrame(cluster_profiles)
    examples_df = pd.DataFrame(cluster_examples)
    
    profiles_df.to_csv(reports_dir / "cluster_profiles.csv", index=False)
    examples_df.to_csv(reports_dir / "cluster_examples.csv", index=False)
    print("Saved: cluster_profiles.csv and cluster_examples.csv")
    
    # Save results to JSON
    k_search_metrics = []
    for idx, k in enumerate(k_values):
        k_search_metrics.append({
            "k": k,
            "inertia": float(inertias[idx]),
            "silhouette_score": float(silhouettes[idx])
        })
        
    clustering_results = {
        "k_search_history": k_search_metrics,
        "selected_k": final_k,
        "cluster_profiles": cluster_profiles,
        "note": "high_satisfaction and sentiment_label targets were strictly excluded from feature inputs, and utilized only for post-hoc cluster profiling."
    }
    
    with open(reports_dir / "clustering_results.json", "w", encoding="utf-8") as f:
        json.dump(clustering_results, f, indent=4)
    print("Saved: clustering_results.json")
    
    # Plot 2: Cluster Size Distribution
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='#FCFAF7')
    ax.set_facecolor('#FCFAF7')
    
    c_names = [p["persona_name"] for p in cluster_profiles]
    c_sizes = [p["size"] for p in cluster_profiles]
    c_pcts = [p["percentage"] for p in cluster_profiles]
    
    bars = ax.bar(c_names, c_sizes, color=COLOR_PLUM, edgecolor=COLOR_PLUM, width=0.45)
    
    # Annotate sizes
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        ax.annotate(f'{height:,}\n({c_pcts[idx]:.1%})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=COLOR_PLUM)
                    
    ax.set_title('Discovered Cluster Size Distribution', fontsize=12, pad=20, fontweight='bold', color=COLOR_PLUM)
    ax.set_ylabel('Number of Reviews', fontsize=10, fontweight='bold', color=COLOR_PLUM)
    ax.set_xticklabels(c_names, rotation=15, ha='right', fontsize=9)
    ax.set_ylim(0, max(c_sizes) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Format y-axis with comma separators
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    
    plt.tight_layout()
    plt.savefig(figures_dir / "cluster_size_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: cluster_size_distribution.png")
    
    # Plot 3: Cluster Satisfaction Rates
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='#FCFAF7')
    ax.set_facecolor('#FCFAF7')
    
    sat_rates = [p["high_satisfaction_rate"] for p in cluster_profiles]
    bars = ax.bar(c_names, sat_rates, color=COLOR_GOLD, edgecolor=COLOR_PLUM, width=0.45)
    
    # Annotate rates
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        ax.annotate(f'{height:.1%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=COLOR_PLUM)
                    
    ax.set_title('High Satisfaction Rate per Cluster Segment', fontsize=12, pad=20, fontweight='bold', color=COLOR_PLUM)
    ax.set_ylabel('Satisfaction Percentage (Rating 4-5)', fontsize=10, fontweight='bold', color=COLOR_PLUM)
    ax.set_xticklabels(c_names, rotation=15, ha='right', fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "cluster_satisfaction_rates.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: cluster_satisfaction_rates.png")
    
    # Plot 4: 2D Cluster Projection (PCA) - Sample 20,000 max points
    print("Generating 2D Projection plot (PCA)...")
    subsample_size = min(20000, len(df))
    np.random.seed(42)
    sample_indices = np.random.choice(len(df), size=subsample_size, replace=False)
    
    X_svd_sample = X_svd[sample_indices]
    labels_sample = labels[sample_indices]
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_svd_sample)
    
    fig, ax = plt.subplots(figsize=(9, 7), facecolor='#FCFAF7')
    ax.set_facecolor('#FCFAF7')
    
    # Plot distinct colors matching brand
    colors = [COLOR_PLUM, COLOR_GOLD, COLOR_ROSE, COLOR_MUTED, '#A57C95']
    for c in range(final_k):
        c_mask = (labels_sample == c)
        ax.scatter(
            X_pca[c_mask, 0], X_pca[c_mask, 1],
            s=4, alpha=0.5,
            color=colors[c], label=persona_mapping[c]
        )
        
    ax.set_title('2D PCA Projection of Skincare Review Clusters', fontsize=12, pad=20, fontweight='bold', color=COLOR_PLUM)
    ax.set_xlabel('Principal Component 1', fontsize=10, fontweight='bold', color=COLOR_PLUM)
    ax.set_ylabel('Principal Component 2', fontsize=10, fontweight='bold', color=COLOR_PLUM)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "cluster_2d_projection.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: cluster_2d_projection.png")
    
    # Plot 5: Cluster Top Terms Horizontal subplots
    print("Generating cluster top terms chart...")
    fig, axes = plt.subplots(final_k, 1, figsize=(8, 12), facecolor='#FCFAF7')
    
    for c in range(final_k):
        ax = axes[c]
        ax.set_facecolor('#FCFAF7')
        
        # Get terms
        p = cluster_profiles[c]
        terms = [t.strip() for t in p["top_terms"].split(",")][:8]
        # Use simple decreasing counts for plotting
        x_vals = np.arange(len(terms))[::-1]
        
        ax.barh(x_vals, np.arange(1, len(terms) + 1), color=colors[c], edgecolor=COLOR_PLUM, height=0.5)
        ax.set_yticks(x_vals)
        ax.set_yticklabels(terms, fontsize=9, fontweight='bold', color=COLOR_PLUM)
        ax.set_title(f"Cluster {c}: {p['persona_name']}", fontsize=10, pad=10, fontweight='bold', color=COLOR_PLUM)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.grid(False)
        
    plt.suptitle('Top Discovered Skincare Vocabulary per Cluster', fontsize=12, fontweight='bold', color=COLOR_PLUM, y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "cluster_top_terms.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: cluster_top_terms.png")
    
    print("=== GlowWise AI Unsupervised Clustering Insights Completed Successfully! ===")

if __name__ == "__main__":
    run_clustering()
