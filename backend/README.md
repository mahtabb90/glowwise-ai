# GlowWise AI - FastAPI Backend API 🚀

This is the production-style FastAPI backend layer for the GlowWise AI project. It exposes ML inference endpoints and model insights to serve predictions to the React frontend.

---

## 📦 1. Installation

It is recommended to run the backend in a virtual environment. From the `backend/` directory:

1. **Activate local virtual environment**:
   - On Windows (PowerShell):
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - On Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃‍♂️ 2. Running the Backend Server

Launch the development server using `uvicorn`:
```bash
uvicorn app.main:app --reload
```
This runs the API server locally at `http://127.0.0.1:8000`.
- **Interactive Documentation (Swagger UI)**: Access `http://127.0.0.1:8000/docs` to test endpoints visually.
- **Alternative Redoc Documentation**: Access `http://127.0.0.1:8000/redoc`.

---

## ⚙️ 3. Required Local ML Model Artifacts

The API requires the serialized classification model:
- **Expected path**: `ml/models/best_satisfaction_model.joblib`

### ⚠️ Graceful Fallback if Missing:
If the model file is not found, the backend will **not crash on startup**. It allows the server to run so health checks remain active. However:
- `GET /api/model/status` will report `"model_loaded": false`.
- `POST /api/predict/satisfaction` will return an HTTP `503 Service Unavailable` status.

### 🔄 How to Generate the Model Artifact:
To generate or update the model binary, run the model comparison script from the project root:
```bash
python ml/src/model_comparison.py
```

*Note: Raw datasets and serialized model binaries are ignored by Git (via `.gitignore`) to prevent repository bloat. They are saved locally.*

---

## 🔌 4. Example API Specifications

### Post-Inference Classification:
- **Endpoint**: `POST /api/predict/satisfaction`
- **Request Body (JSON)**:
  ```json
  {
    "review_text": "I love this moisturizer! It makes my dry skin feel so hydrated, smooth, and lightweight.",
    "review_title": "Best moisturizer ever",
    "product_name": "Ultra Facial Cream",
    "brand_name": "Kiehl's"
  }
  ```
- **Response Body (JSON)**:
  ```json
  {
    "predicted_label": 1,
    "predicted_class": "high_satisfaction",
    "confidence": 0.9654,
    "probability_low_or_medium": 0.0346,
    "probability_high_satisfaction": 0.9654,
    "model_name": "logistic_regression_tuned",
    "input_preview": "Best moisturizer ever I love this moisturizer! It makes my dry skin feel so hydrated..."
  }
  ```

---

## 🧪 5. Running Manual Integration Tests

To verify that the API serves predictions and insights correctly, make sure uvicorn is running, and execute:
```bash
python test_api_manual.py
```
This script queries health, status, positive review classification, negative review classification, and dashboard analytics insights.
