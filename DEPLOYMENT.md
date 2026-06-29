# GlowWise AI - Deployment Documentation 🚀

This document outlines the deployment configuration for GlowWise AI.

---

## 🎨 FastAPI Backend Deployment on Render

### Render Web Service Settings

* **Environment**: `Python`
* **Root Directory**: `.` (leave as repository root)
* **Build Command**: 
  ```bash
  pip install -r backend/requirements.txt
  ```
* **Start Command**: 
  ```bash
  PYTHONPATH=backend uvicorn app.main:app --host 0.0.0.0 --port $PORT
  ```

### Render Environment Variables

| Variable | Value | Description |
| :--- | :--- | :--- |
| **`PYTHON_VERSION`** | `3.11.9` | Force Render to build with the stable Python version |
| **`ALLOWED_ORIGINS`** | `http://localhost:5173,https://<your-vercel-domain>.vercel.app` | Comma-separated list of origins for CORS |
| **`FRONTEND_URL`** | `https://<your-vercel-domain>.vercel.app` | Optional: URL of your deployed frontend |
| **`MODEL_PATH`** | `ml/models/best_satisfaction_model.joblib` | Optional: Override model file path |

---

## 💅 React Frontend Deployment on Vercel

### Vercel Project Settings

* **Framework Preset**: `Vite`
* **Root Directory**: `frontend`
* **Build Command**: 
  ```bash
  npm run build
  ```
* **Output Directory**: `dist`

### Vercel Environment Variables

| Variable | Value | Description |
| :--- | :--- | :--- |
| **`VITE_API_URL`** | `https://<your-render-service>.onrender.com` | Base URL of your deployed FastAPI backend API |

---

## 🔍 Local Verification from Repo Root

To test the production backend configuration locally before pushing:

### 1. Start Backend Server
From the repository root folder, run:
```powershell
# On Windows PowerShell
$env:PYTHONPATH="backend"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
*(On Linux/macOS: `PYTHONPATH=backend uvicorn app.main:app --host 127.0.0.1 --port 8000`)*

### 2. Verify API Endpoints
* **Health Check**:
  ```bash
  curl http://localhost:8000/health
  ```
* **Model Loading Status**:
  ```bash
  curl http://localhost:8000/api/model/status
  ```
* **High Satisfaction Review Prediction**:
  ```bash
  curl -X POST http://localhost:8000/api/predict/satisfaction \
    -H "Content-Type: application/json" \
    -d "{\"review_title\": \"Holy grail\", \"review_text\": \"This moisturizer is amazing! It hydrates my skin perfectly and leaves a smooth glow.\"}"
  ```
* **Low Satisfaction Review Prediction**:
  ```bash
  curl -X POST http://localhost:8000/api/predict/satisfaction \
    -H "Content-Type: application/json" \
    -d "{\"review_title\": \"Horrible breakout\", \"review_text\": \"I was so disappointed. This product broke me out immediately and caused redness.\"}"
  ```
