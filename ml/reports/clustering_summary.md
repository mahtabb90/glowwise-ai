# GlowWise AI - Unsupervised Clustering & Skincare Insights Summary Report 📊

This report documents the unsupervised learning and customer review segmentation workflow implemented on the `feat/clustering-insights` branch. We discover consumer segments and product usage themes in Sephora skincare reviews.

---

## 🎯 Purpose of Clustering

In skincare marketing and formulation science, customer feedback is highly diverse. Supervised satisfaction prediction tells us *whether* a user is happy, but **unsupervised clustering** discovers *what topics and segments* define their conversations. This reveals:
- Distinct consumer categories based on skin care habits.
- Actionable formulations feedback (e.g., texture vs acne vs lip care).
- Cohorts that can guide product recommendations in the GlowWise AI web application.

---

## ⚙️ Unsupervised Feature Engineering Strategy

To cluster the text reviews without supervised bias:
1. **Target Exclusion**: Target variables (`high_satisfaction` and `sentiment_label`) were **strictly excluded** from the clustering input features. They were used only *after* clustering to profile and interpret the segments.
2. **Text Representation**: We constructed an n-gram tf-idf matrix on `combined_text` (review title + review body) with parameters:
   - `stop_words="english"`
   - `min_df=5`
   - `max_df=0.85`
   - `max_features=10,000`
   - `ngram_range=(1, 2)`
3. **Dimensionality Reduction**: We applied **TruncatedSVD (LSA)** to reduce the 10,000-dimensional sparse TF-IDF representation to **50 dense components**. This speeds up distance computations, removes collinearity, and prevents noise from sparse rare words.

---

## 📈 Parameter Search & K Selection

We tested cluster sizes $k$ from 3 to 8 using `MiniBatchKMeans` with `batch_size=2048`.
* **Inertia (Elbow Method)**: Decreased smoothly from 10,522.18 ($k=3$) to 9,636.59 ($k=8$).
* **Silhouette Score**: Evaluated on a **random sample of 10,000 reviews** to prevent Out-Of-Memory (OOM) crashes. It showed a peak at $k=7$ ($0.0360$) and $k=5$ ($0.0280$).
* **Chosen K**: **$k=5$** was selected for final profiling because it yields distinct, interpretable consumer cohorts with clear product categories (e.g. Lip Care, Acne, Moisturizers) and avoids segment overlap.

---

## 🏆 Final Discovered Cluster Profiles

The table below summarizes the five discovered customer segments from the final $k=5$ configuration:

| Cluster ID | Persona Name | Size | Size % | High Sat % | Avg Word Count | Top Brand Counts | Top Discovered TF-IDF Terms |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **0** | **General Beauty Enthusiasts** | 27,608 | 27.6% | 84.4% | 23.3 | Sephora, Clinique, Lancome | love, face, makeup, product, great, skin |
| **1** | **Daily Skincare Users** | 40,467 | 40.5% | 76.9% | 45.4 | Sephora, Clinique, Kiehl's | skin, product, using, use, ve, really |
| **2** | **Moisture & Texture Fans** | 17,495 | 17.5% | 87.9% | 42.1 | Clinique, Sephora, Kiehl's | skin, moisturizer, feels, oily, dry, cream |
| **3** | **Acne & Blemish Care** | 9,926 | 9.9% | 85.9% | 47.9 | Clinique, Sephora, Murad | acne, serum, skin, prone, acne prone, product |
| **4** | **Lip Care Seekers** | 4,504 | 4.5% | 83.9% | 22.8 | Laneige, Sephora, Fresh | lips, lip, balm, lip balm, love, product |

---

## 💡 Key Business & Skincare Insights Discovered

1. **Daily Routine Efficacy Gap**: The largest cluster, **Daily Skincare Users (40.5%)**, contains reviews detailing standard skin routines. This cluster exhibits the lowest satisfaction rate (**76.9%**). This highlights an efficacy gap where consumers write extensive reviews detailing routine steps but express disappointment in visual results, suggesting a market need for high-efficacy formulation lines.
2. **Sensory & Moisture Satisfaction**: The **Moisture & Texture Fans (17.5%)** cluster shows the highest satisfaction rate (**87.9%**). This indicates that consumers are highly pleased by positive sensory attributes (e.g., "feels nice", "lightweight", "hydrating cream") and texture satisfaction directly correlates with ratings.
3. **Acne & Lip Care Isolation**: The **Acne & Blemish Care (9.9%)** and **Lip Care Seekers (4.5%)** segments form highly distinct branches. This indicates that acne-prone users and lip-treatment seekers have specific, localized product vocabularies, making them ideal targets for specialized, isolated product filters.

---

## ⚠️ Limitations of Clustering Skincare Text

- **Hard Boundaries**: K-Means forces each review into a single cluster. In reality, a review might discuss both acne breakout problems (Cluster 3) and moisturizing cream textures (Cluster 2).
- **Post-Hoc Labels**: Labels like `high_satisfaction` and `sentiment_label` were strictly used for post-hoc validation. Unsupervised clustering itself does not cluster reviews by satisfaction level, meaning positive and negative reviews are mixed together in the same clusters if they share vocabulary themes.

---

## 🚀 How Clustering Can Be Used in the GlowWise AI Web App

1. **Smart Filtering**: Enable users to filter product reviews by their cohort (e.g., "Show me reviews from other Acne & Blemish Care users").
2. **Targeted Recommendation Engine**: If a user is identified as a "Moisture & Texture Fan" based on their past review behavior, the app can prioritize products described as "creamy", "lightweight", or "non-greasy".
3. **Product Complaint Summarizer**: The backend can group negative reviews of a single product to show brands their main complaints (e.g., showing a brand that 80% of their critical feedback belongs to "Acne Breakouts" vs "Sensory Texture issues").

---

## 🚀 Recommended Next Branch

We recommend checking out the branch:
**`feat/backend-model-api`**

In that branch, we will build a FastAPI microservice that serves our Tuned Logistic Regression model `/predict` endpoint and integrates clustering definitions to support dynamic API integrations.
