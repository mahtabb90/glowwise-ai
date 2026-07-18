# GlowWise AI - Transformer Transfer Learning Experiment Summary 🧠🧴

This report presents the findings from our experiment using **Transfer Learning** and **Transformer Sentence Representations** for skincare review satisfaction prediction (`high_satisfaction`). We compare this approach against the baseline TF-IDF + Logistic Regression and Text CNN architectures.

---

## 📊 Concepts & Architecture

### 1. What is Transfer Learning in this Project?
In traditional machine learning workflows, a model is trained from scratch on a specific dataset to learn both the representation of the data (e.g., TF-IDF word frequencies) and the classification decision boundary. 

**Transfer Learning** separates representation learning from task learning. We utilize a deep neural network—specifically `sentence-transformers/all-MiniLM-L6-v2`—which has been pre-trained on billions of English sentences to capture generic lexical relationships, grammar, and semantics. We pass our reviews through this pre-trained model to extract dense, low-dimensional vector representations (**sentence embeddings**). We then *transfer* this linguistic understanding to our task by training a simple linear classifier (Logistic Regression) on top of the fixed embeddings.

### 2. Sentence Embeddings vs. TF-IDF (Bag-of-Words)
* **TF-IDF (Term Frequency-Inverse Document Frequency)** is a *sparse, high-dimensional* bag-of-words representation (typically 10,000–20,000 features in our baseline). It weights words based on their frequency in the document and across the corpus, but treats words as completely independent tokens. It has no concept of word order, syntax, or semantic similarity.
* **Sentence Embeddings** are *dense, low-dimensional* vectors (384 features for MiniLM). They represent the entire sequence structure. Instead of counting words, they map reviews into a continuous semantic space where reviews with similar meanings (e.g., "allergic reaction" and "broke me out") are located close to each other, even if they share zero vocabulary words.

### 3. Contextual Capture vs. Bag-of-Words
Transformers use **attention mechanisms** to weigh the relationships between all words in a sentence, regardless of their distance. This allows them to capture:
- **Negation and Qualifiers**: TF-IDF sees "not dry" and "dry" as sharing the word "dry", which may lead to misclassifications. A transformer maps "not dry" to a positive hydration region and "dry" to a negative irritation region.
- **Skincare Idioms**: Phrases like *"broke me out"*, *"holy grail"*, or *"worth every penny"* are highly sentiment-bearing. TF-IDF breaks them into separate tokens, losing the combined meaning. Transformers capture these multi-word semantics as cohesive vector representations.

---

## 📈 Performance Comparison

The table below summarizes the performance of the transformer model compared to baseline experiments (trained on a consistent 15k training subset) and the production tuned reference model:

| Rank | Model | Training Size | Status | Accuracy | Macro F1 | Weighted F1 | Class 0 Recall | Class 0 Precision | ROC-AUC | PR-AUC | Notes |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **-** | **Production Reference: Tuned LR** | 80,000 | Completed | 92.96% | 88.91% | 93.21% | 90.93% | 75.03% | 97.57% | 99.43% | Tuned Logistic Regression model currently in production |
| 1 | **Tuned Logistic Regression** | 15,000 | Completed | 92.05% | 87.34% | 92.29% | 86.77% | 73.57% | 96.60% | 99.18% | TF-IDF + LR baseline on experimental split |
| 2 | **LinearSVC** | 15,000 | Completed | 92.12% | 87.04% | 92.24% | 82.42% | 75.72% | 96.17% | 99.06% | Support Vector Classification on TF-IDF |
| 3 | **Text CNN** | 15,000 | Completed | 90.27% | 84.89% | 90.67% | 85.43% | 68.23% | 95.57% | 98.92% | Convolutional Neural Network (Colab, GPU) |
| 4 | **Transformer (MiniLM + LR)** | 10,000 | Completed | 84.62% | 78.30% | 85.81% | 85.49% | 54.52% | 91.90% | 97.94% | Logistic Regression on pre-trained MiniLM embeddings |
| 5 | **Dense ANN / MLP Fallback** | 15,000 | Completed | 86.06% | 76.57% | 86.14% | 62.55% | 60.78% | 89.01% | 97.26% | scikit-learn MLPClassifier fallback |
| 6 | **KNN + SVD** | 15,000 | Completed | 83.20% | 70.37% | 82.88% | 48.53% | 53.43% | 80.39% | 94.65% | KNN on 100-dimensional SVD features |

*Note: The Transformer model achieves extremely strong Class 0 Recall (85.49%), outperforming LinearSVC and feedforward neural networks, while keeping inference parameters minimal. However, its overall Macro F1 is slightly lower than classical TF-IDF models due to the frozen nature of the feature extraction layer.*

---

## 💡 Rationale for Keeping the Tuned Logistic Regression in Production

While Transformer embeddings represent a major advance in semantic understanding, **Tuned Logistic Regression on TF-IDF** remains the optimal choice for production due to the following business and technical reasons:

1. **Sub-millisecond Latency**: Logistic Regression inference computes in ~1-5ms on standard CPU resources. Transformer models require a forward pass through a deep neural network, introducing significant latency (typically 50–200ms per text sequence on CPU).
2. **Inference Compute Cost**: Serving a Logistic Regression model has near-zero compute overhead. Serving transformers in production requires high RAM or expensive GPU resources to maintain acceptable response times.
3. **Deployment Footprint**: The Logistic Regression pipeline binary is extremely compact (~10MB), loading instantly and requiring minimal memory. A Transformer model requires loading heavy deep learning runtimes (`torch`, `transformers`) and model files (~100MB to 1.5GB), increasing cold start times and server memory footprint.
4. **Complete Interpretability**: Logistic Regression coefficients map directly back to words in the TF-IDF vocabulary. We can verify exactly which terms drive positive or negative classification, making the model highly auditable and explainable. Transformer models are black boxes where predictions cannot be directly attributed to individual words.

---

## 🚀 Future Direction: Fine-Tuning on Google Colab

The current experiment uses **Feature Extraction** (extracting representations from a frozen model). While fast, the pre-trained embedding model was not specifically trained on skincare terms or product reviews.

A logical next step to maximize accuracy is **Fine-Tuning**:
- **Setup**: Train a model like `DistilBERT` or `BERT-Base` in a Google Colab notebook utilizing a GPU runtime.
- **Method**: Instead of freezing the transformer weights, update the weights of all layers during backpropagation using the Sephora reviews dataset.
- **Benefit**: This allows the attention heads to adapt to domain-specific jargon (e.g. skin conditions, ingredient combinations), which will improve classification on challenging, ambiguous, and low-support satisfaction reviews.
