# GlowWise AI - Presentation Notes 🎙️

These notes serve as a guide for presenting GlowWise AI to stakeholders, recruiters, or during technical portfolio reviews.

---

## 📢 The Pitch (1-Minute Elevator Pitch)
> "Skincare is highly personal, but online reviews are overwhelming. GlowWise AI is a machine learning platform that translates Sephora product feedback into structured business intelligence. It classifies review sentiments, predicts customer recommendation propensity, and groups products by formulation and concern similarity. With this tool, product developers can pinpoint formulation issues, and brands can understand their audience with clinical precision, presented through a luxury beauty-themed analytics dashboard."

---

## 🎯 Key Talk Tracks

### 1. The Problem Space
* **Volume**: Thousands of reviews for a single moisturizer, making it impossible for product managers or researchers to digest.
* **Ambiguity**: Star ratings don't tell the full story. A 3-star rating could mean "great product but terrible pump container," or "it broke me out."
* **Clutter**: Sephora products target highly specific skin concerns (acne, aging, redness) which standard NLP sentiment tools ignore because they lack domain knowledge.

### 2. The Solution: GlowWise AI
* **Domain-Specific Preprocessing**: Skincare features have specific terms (e.g., *hyaluronic acid, purge, pilling, non-comedogenic*).
* **Multi-Aspect Sentiment Analysis**: Categorizing user feedback into dimensions like **Efficacy**, **Texture/Feel**, **Packaging**, and **Scent**.
* **Clustering Engine**: Using unsupervised models (K-Means) to discover product clusters based on ingredients and customer demographic satisfaction patterns (e.g., dry skin vs oily skin satisfaction).

### 3. Technical Highlights
* **FastAPI Backend**: Built for low-latency JSON model inferences. Separated service logic allows seamless swap-out of model checkpoints (e.g., from TF-IDF models to fine-tuned Hugging Face Transformers).
* **React + TypeScript SPA**: Implements a sleek, premium, wellness-oriented design system. Designed to emulate luxury skincare branding (Glossier/Drunk Elephant style) to showcase that data science dashboards can be visually stunning.

---

## 📊 Future Demo Walkthrough Script

1. **The Landing Page**: Introduce GlowWise AI, highlighting its value proposition.
2. **Interactive Predictor**: Input a custom text review (e.g., *"This serum felt amazing on my skin and cleared my acne in two weeks, but the bottle leaks."*).
   * Show predicted rating (e.g., `4.2 / 5`).
   * Show sentiment breakdown (Efficacy: `Positive`, Packaging: `Negative`, Texture: `Positive`).
3. **Product Landscape Map**: Show a 2D UMAP scatterplot visualizing the product clusters, revealing product formulation groups.
4. **Insights Dashboard**: Display summary statistics (most mentioned positive ingredients, most common complaints by skin type).
