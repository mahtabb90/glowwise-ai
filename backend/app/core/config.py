import os
from pathlib import Path

# Resolve directory paths relative to this config file
# This file is: backend/app/core/config.py
# Config folder: backend/app/core
# App folder: backend/app
# Backend folder: backend
# Project root folder: project root (containing backend/ and ml/)

CORE_DIR = Path(__file__).resolve().parent
APP_DIR = CORE_DIR.parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

# Model Configuration
DEFAULT_MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "best_satisfaction_model.joblib"
MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))

# Insight Reports Configuration
REPORTS_DIR = PROJECT_ROOT / "ml" / "reports"

EXPLAINABILITY_JSON_PATH = REPORTS_DIR / "model_explainability_results.json"
TOP_POS_TERMS_CSV_PATH = REPORTS_DIR / "top_positive_terms.csv"
TOP_NEG_TERMS_CSV_PATH = REPORTS_DIR / "top_negative_terms.csv"

CLUSTERING_JSON_PATH = REPORTS_DIR / "clustering_results.json"
CLUSTER_PROFILES_CSV_PATH = REPORTS_DIR / "cluster_profiles.csv"

# API Configuration
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
