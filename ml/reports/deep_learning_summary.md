# GlowWise AI - Advanced ML & Deep Learning Experiments Summary 🧠📊

This report presents the findings from our academic portfolio and course project experiments exploring K-Nearest Neighbors (KNN), Artificial Neural Networks (ANN/MLP), and 1D Convolutional Neural Networks (CNN) for skincare review satisfaction prediction (`high_satisfaction`).

---

## 📊 Performance Comparison & Evaluation Schema

The fair experimental comparison is between models trained on the **same stratified training subset of 15,000 reviews** and evaluated on the **same test set of 20,000 reviews** under identical split parameters (`random_state=42`). 

The production model (trained on the full 80,000 training set) is included at the top as a reference point but is separated from the comparative ranking to maintain a rigorous evaluation methodology.

### 🏆 Production Reference Model
*This model was trained on the full 80,000 training reviews (80% of the 100k sample) and represents our baseline production pipeline.*

| Model | Training Size | Status | Accuracy | Macro F1 | Weighted F1 | Class 0 Recall | Class 0 Precision | ROC-AUC | PR-AUC | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Production Reference: Tuned LR** | 80,000 | Completed | 92.96% | 88.91% | 93.21% | 90.93% | 75.03% | 97.57% | 99.43% | Tuned Logistic Regression model currently in production |

---

### ⚖️ Fair 15K Experimental Comparison
*These models were trained on the same 15,000 training subset and evaluated on the same 20,000 test set. Ranked by **Macro F1-Score** (descending):*

| Rank | Model | Training Size | Status | Accuracy | Macro F1 | Weighted F1 | Class 0 Recall | Class 0 Precision | ROC-AUC | PR-AUC | Notes |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **Tuned Logistic Regression** | 15,000 | Completed | **92.05%** | **87.34%** | **92.29%** | **86.77%** | 73.57% | **96.60%** | **99.18%** | Tuned LR baseline for experimental split |
| 2 | **LinearSVC** | 15,000 | Completed | 92.12% | 87.04% | 92.24% | 82.42% | **75.72%** | 96.17% | 99.06% | Linear Support Vector Classification |
| 3 | **Dense ANN / MLP Fallback** | 15,000 | Completed | 86.06% | 76.57% | 86.14% | 62.55% | 60.78% | 89.01% | 97.26% | scikit-learn MLPClassifier early stopping fallback |
| 4 | **KNN + SVD** | 15,000 | Completed | 83.20% | 70.37% | 82.88% | 48.53% | 53.43% | 80.39% | 94.65% | KNeighborsClassifier on 100-dim SVD features |
| — | **Text CNN** | 15,000 | **Not executed – TensorFlow/Keras not available in this environment** | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | Skipped because TensorFlow was missing in Python 3.13 |

*Note: Since the dataset is highly imbalanced (~82% high satisfaction), overall accuracy is misleading. We prioritize **Macro F1-Score** and **Class 0 Recall** (low/medium satisfaction flagging).*

---

## 🔍 Model Analyses & Interpretations

### 1. Classical Linear Text Models (Logistic Regression & LinearSVC)
* **Tuned Logistic Regression** remained the strongest model among the fair 15k experiments, achieving **87.34% Macro F1** and **86.77% Class 0 Recall**. 
* **LinearSVC** performed very close to Logistic Regression, achieving **87.04% Macro F1** and **82.42% Class 0 Recall**. This confirms that classical linear text models utilizing high-dimensional TF-IDF vectors are extremely robust for text classification because review sentiment relies heavily on individual keyword frequencies (e.g., "broke", "redness", "amazing").

### 2. K-Nearest Neighbors (KNN + SVD)
* **Why it was tested**: KNN is an instance-based model representing standard coursework distance metrics.
* **Why it is not ideal**: Raw TF-IDF matrices suffer from the "Curse of Dimensionality" in high dimensions, making distances between all points converge and computing pairwise distances extremely slow. Even after reducing features to 100 dense dimensions using **TruncatedSVD**, KNN achieved the weakest performance (**70.37% Macro F1**, **48.53% Class 0 Recall**), showing distance clustering is poor at mapping fine sentiment boundaries.

### 3. Dense ANN / MLP Fallback
* **Why it was tested**: To demonstrate feedforward deep learning capabilities.
* **Interpretation of Fallback Diagnostics**: Due to TensorFlow's absence, the model was executed using scikit-learn's `MLPClassifier` with early stopping and manual balanced oversampling. The fallback training diagnostics show training loss and validation accuracy. The ANN achieved **76.57% Macro F1** and **62.55% Class 0 Recall**. While it outperformed KNN, it fell short of the linear models because feedforward neural networks on static dense SVD representations do not capture lexical clues as effectively as sparse TF-IDF n-grams.

### 4. Text CNN (Planned but Skipped)
* **Why it was planned**: 1D CNNs capture local word patterns (receptive fields) like *"broke me out"*, *"not worth it"*, or *"holy grail"* by sliding a convolutional filter over word sequences.
* **Skip Rationale**: Since TensorFlow/Keras was not available, the CNN model was **honestly skipped** to maintain 100% data integrity.

---

## 💡 Key Takeaway: Simplicity vs. Complexity
This project shows that **more complex models do not automatically perform better**. High-dimensional neural models and distance-based classifiers require significantly more parameter tuning, training data, and compute resources.

For text classification datasets where keyword indicators carry the bulk of the predictive signal, a **Tuned Logistic Regression** model currently offers the best solution. It is:
1. **Highly Interpretable**: Model coefficients map directly back to words for explainability.
2. **Computationaly Light**: It trains in under 10 seconds and evaluates in sub-milliseconds on CPU.
3. **Easy to Deploy**: It compiles into a 10MB joblib package, avoiding heavy dependency loads.

---

## 🎯 Presentation-Ready Conclusion

> “This experiment showed that advanced models such as ANN and KNN can be tested as part of a complete ML workflow, but for this dataset, the tuned Logistic Regression model gave the best balance between performance, interpretability, speed, and deployment simplicity.”

---

## 🚀 Limitations & Next Steps
1. **Semantic Negations**: TF-IDF models ignore word order, missing negation swings (e.g., "not bad" vs. "bad").
2. **Pre-trained Embeddings**: Use GloVe or Word2Vec features to supply semantic word vectors to the ANN.
3. **Transformer Fine-tuning**: Future work could fine-tune a pre-trained Transformer (e.g. DistilBERT) on a GPU server.

---

## 🖼️ Figures & Visualizations
The following plots have been generated and saved inside [ml/reports/figures/](file:///c:/Users/mahta/my_projects/glowwise-ai/ml/reports/figures/):
* **Presentation Comparison Chart**: [model_comparison_presentation.png](file:///c:/Users/mahta/my_projects/glowwise-ai/ml/reports/figures/model_comparison_presentation.png) (Visually separates the 80k Production Reference from the 15k models, focusing on Macro F1, Class 0 Recall, and Accuracy).
* **ANN Fallback Diagnostics**: [ann_training_curves.png](file:///c:/Users/mahta/my_projects/glowwise-ai/ml/reports/figures/ann_training_curves.png) (Side-by-side loss and accuracy training curves).
* **ROC Curve Comparison**: [roc_curve_comparison.png](file:///c:/Users/mahta/my_projects/glowwise-ai/ml/reports/figures/roc_curve_comparison.png) (Shows True Positive vs False Positive rates).
* **Precision-Recall Curve Comparison**: [precision_recall_curve_comparison.png](file:///c:/Users/mahta/my_projects/glowwise-ai/ml/reports/figures/precision_recall_curve_comparison.png) (Evaluates model tradeoffs under label imbalance).
* **Confusion Matrices**: Saved for [ann_confusion_matrix.png](file:///c:/Users/mahta/my_projects/glowwise-ai/ml/reports/figures/ann_confusion_matrix.png) and [knn_confusion_matrix.png](file:///c:/Users/mahta/my_projects/glowwise-ai/ml/reports/figures/knn_confusion_matrix.png) with simplified display labels.
