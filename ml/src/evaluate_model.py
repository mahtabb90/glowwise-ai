"""
GlowWise AI - Model Evaluation Module
Provides reusable utilities to evaluate classification models, calculate metrics,
and plot premium styled confusion matrices.
"""

import os
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

def calculate_metrics(y_true, y_pred) -> dict:
    """
    Calculate classification metrics: accuracy, macro/weighted precision/recall/f1,
    and per-class precision/recall/f1.
    Uses zero_division=0 to prevent warnings when classes are not predicted.
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate Macro metrics
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    # Calculate Weighted metrics
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    # Calculate per-class metrics
    # Sorted order of classes (0, then 1)
    p_classes, r_classes, f_classes, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    class_labels = ["low_or_medium_satisfaction", "high_satisfaction"]
    per_class_metrics = {}
    for i, label in enumerate(class_labels):
        if i < len(p_classes):
            per_class_metrics[label] = {
                "precision": float(p_classes[i]),
                "recall": float(r_classes[i]),
                "f1_score": float(f_classes[i]),
                "support": int(support[i])
            }
            
    return {
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
        "per_class": per_class_metrics
    }

def plot_confusion_matrix(y_true, y_pred, output_path: str, title: str = "Confusion Matrix", class_names: list = None):
    """
    Generates and saves a premium styled confusion matrix using matplotlib.
    Uses class labels: 'low_or_medium_satisfaction' (0) and 'high_satisfaction' (1) by default.
    Styled to match the GlowWise theme (Deep Plum and Rose accents).
    """
    if class_names is None:
        class_names = ["low_or_medium_satisfaction", "high_satisfaction"]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix by row (true class support)
    # Handle rows with 0 sum gracefully
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm.astype('float')), where=row_sums!=0)
    
    # Visual Aesthetics Setup
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    
    # GlowWise Brand Colors
    COLOR_PLUM = '#3B243B'
    COLOR_CREAM = '#FCFAF7'
    
    # Create colormap from Cream to Deep Plum
    cmap = LinearSegmentedColormap.from_list("glowwise_cmap", [COLOR_CREAM, COLOR_PLUM])
    
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap)
    
    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, format='%.2f')
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_visible(False)
    
    # Setup Ticks
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=15, ha="right", fontsize=9, color='#332633')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=9, color='#332633')
    
    # Labels & Title
    ax.set_title(title, fontsize=12, pad=20, fontweight='bold', color=COLOR_PLUM)
    ax.set_xlabel('Predicted Label', fontsize=10, labelpad=10, fontweight='bold', color=COLOR_PLUM)
    ax.set_ylabel('True Label', fontsize=10, labelpad=10, fontweight='bold', color=COLOR_PLUM)
    
    # Clear grid
    ax.grid(False)
    
    # Draw values inside cells
    thresh = cm_norm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        val_str = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})"
        ax.text(j, i, val_str,
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm_norm[i, j] > thresh else COLOR_PLUM,
                fontsize=10, fontweight='bold')
                
    plt.tight_layout()
    
    # Save file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix plot successfully saved to: {output_path}")

if __name__ == "__main__":
    # Test execution
    y_t = [1, 1, 0, 0, 1, 0, 1, 1, 1, 0]
    y_p = [1, 1, 1, 0, 1, 0, 0, 1, 1, 1]
    
    metrics = calculate_metrics(y_t, y_p)
    print("Test calculate_metrics output:")
    print(json.dumps(metrics, indent=4))
