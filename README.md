# GlowWise AI 🌟

> **Skincare Review Intelligence powered by Machine Learning**

GlowWise AI is a professional full-stack machine learning portfolio project designed to unlock deep insights from skincare product reviews and metadata (inspired by Sephora datasets). By combining state-of-the-art sentiment analysis, customer satisfaction prediction, and intelligent product clustering, GlowWise AI translates raw consumer feedback into actionable formulation, marketing, and personalization intelligence through a premium web dashboard.

---

## ✨ Project Vision & Key Features

Modern beauty consumers and brands face information overload. GlowWise AI leverages NLP and Machine Learning to process large-scale text reviews and metadata to offer:
* **🔮 Customer Satisfaction Predictor**: Predict rating levels and customer recommendation probability based on review texts and product variables.
* **🎭 Sentiment Analysis Engine**: Detect fine-grained sentiment (e.g., product efficacy, texture, scent, packaging) in review narratives.
* **🧩 Product Clustering & Similarity Search**: Group products dynamically based on ingredient similarity, user concern alignment (e.g., acne-prone, hydration), and customer feedback profile.
* **📊 Premium Analytics Dashboard**: A high-end beauty/wellness themed executive dashboard with interactive charts, review filters, and real-time model inferences.

---

## 🛠️ Tech Stack

* **Frontend**: React (v18+) + TypeScript + Vite
  * *Styling*: Premium custom Vanilla CSS (deep plum, champagne gold, soft rose, cream, smooth animations, glassmorphism cards).
* **Backend**: FastAPI (Python 3.10+)
  * *Features*: Clean REST endpoints, CORS middleware support, high-performance async requests.
* **Machine Learning**: Python, pandas, scikit-learn, NLTK/SpaCy
* **Environment/Config**: dotenv management (`.env.example` configurations provided)

---

## 🧠 Machine Learning Goals

1. **Text Preprocessing**: Robust pipeline for cleaning skincare text reviews (removing noise, tokenizing, lemmatizing, and handling beauty-specific domain vocabulary).
2. **Sentiment & Rating Prediction**: Build classification/regression models (TF-IDF/embeddings + Logistic Regression/Gradient Boosting) to predict review rating scores and overall sentiment.
3. **Product Clustering**: Implement unsupervised clustering (K-Means / Hierarchical Clustering) combined with dimensionality reduction (PCA/UMAP) to map the product space by ingredients and target concerns.
4. **Explainability**: Integrate feature importance or review phrase highlighting to show why a model predicts a particular satisfaction score.

---

## 🏗️ Planned Architecture

```
                       +-------------------------+
                       |   React SPA Frontend    |
                       | (TypeScript, Vite, CSS) |
                       +------------+------------+
                                    |
                             HTTP REST Requests
                                    v
                       +-------------------------+
                       |     FastAPI Backend     |
                       +------------+------------+
                                    |
                            Model Inferences
                                    v
                       +-------------------------+
                       |  Serialized ML Models   |
                       |    (.joblib / .pkl)     |
                       +------------+------------+
                                    ^
                             Trained Models
                                    |
                       +-------------------------+
                       |   ML Training Pipeline  |
                       |  (pandas, scikit-learn) |
                       +-------------------------+
```

---

## 📁 Repository Structure

```
glowwise-ai/
├── backend/                  # FastAPI Application
│   ├── app/
│   │   ├── api/              # API router endpoints
│   │   ├── core/             # Configuration & security
│   │   ├── models/           # Pydantic schemas / DB models
│   │   ├── services/         # Model prediction & data logic
│   │   └── main.py           # Application entrypoint
│   ├── requirements.txt      # Python dependencies
│   └── .env.example          # Environment configuration template
│
├── frontend/                 # React Frontend
│   ├── src/
│   │   ├── assets/           # Premium aesthetic assets & logos
│   │   ├── components/       # Reusable UI cards, tables, charts
│   │   ├── pages/            # App pages (Dashboard, Landings)
│   │   ├── services/         # API clients (api.ts)
│   │   ├── App.tsx           # Layout and main routing
│   │   └── index.css         # Custom premium design system variables
│   ├── package.json          # Node configurations
│   └── .env.example          # Frontend configuration template
│
├── ml/                       # Machine Learning Engineering
│   ├── notebooks/            # EDA & experimental notebooks
│   ├── src/
│   │   ├── data_preprocessing.py # Preprocessing pipelines
│   │   ├── train_model.py    # Training execution scripts
│   │   ├── evaluate_model.py # Validation and metrics
│   │   └── predict.py        # Inference module for backend integration
│   ├── models/               # Saved model binaries (ignored by Git)
│   └── reports/              # Model metrics and validation reports
│
├── data/                     # Raw & Processed Datasets (ignored by Git)
│   ├── raw/
│   └── processed/
│
├── docs/                     # Product documentation
│   └── presentation-notes.md # Presentation/pitch notes for portfolio
│
├── README.md                 # Main Documentation
└── .gitignore                # Global Git Ignore rules
```

---

## 🚀 Getting Started

### 1. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```
The API will be available at `http://localhost:8000`. Test the health status at `http://localhost:8000/health`.

### 2. Frontend Setup
```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```
The React App will be available at `http://localhost:5173`.
