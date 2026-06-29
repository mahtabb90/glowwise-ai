import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables first to ensure config properties evaluate correctly
load_dotenv()

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
# Keep localhost origins for local development
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# Add comma-separated ALLOWED_ORIGINS from environment variables
env_origins = os.getenv("ALLOWED_ORIGINS")
if env_origins:
    origins = [orig.strip() for orig in env_origins.split(",") if orig.strip()]
    ALLOWED_ORIGINS = origins

# Add any additional FRONTEND_URL environment variables
frontend_url = os.getenv("FRONTEND_URL")
if frontend_url:
    origins = [orig.strip() for orig in frontend_url.split(",") if orig.strip()]
    for orig in origins:
        if orig not in ALLOWED_ORIGINS:
            ALLOWED_ORIGINS.append(orig)
