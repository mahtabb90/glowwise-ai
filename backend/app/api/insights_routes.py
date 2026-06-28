import json
from fastapi import APIRouter
from app.core import config
from app.services import model_service

router = APIRouter(prefix="/api/insights", tags=["Insights"])

@router.get("/top-terms")
def get_top_terms():
    """
    Returns positive and negative review satisfaction drivers from explainability reports.
    Returns empty arrays and a warning message if the source file is missing.
    """
    json_path = config.EXPLAINABILITY_JSON_PATH
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "available": True,
                "top_positive_terms": data.get("top_positive_terms", []),
                "top_negative_terms": data.get("top_negative_terms", []),
                "message": "Top terms successfully retrieved."
            }
        except Exception as e:
            return {
                "available": False,
                "top_positive_terms": [],
                "top_negative_terms": [],
                "message": f"Error parsing explainability results: {e}"
            }
    else:
        return {
            "available": False,
            "top_positive_terms": [],
            "top_negative_terms": [],
            "message": "Explainability results JSON report file not found. Run model_explainability.py first."
        }

@router.get("/clusters")
def get_clusters():
    """
    Returns cluster profiles and customer segments discovered during unsupervised clustering.
    Returns empty data and a warning message if the source file is missing.
    """
    json_path = config.CLUSTERING_JSON_PATH
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "available": True,
                "selected_k": data.get("selected_k", 5),
                "cluster_profiles": data.get("cluster_profiles", []),
                "message": "Cluster profiles successfully retrieved."
            }
        except Exception as e:
            return {
                "available": False,
                "selected_k": 0,
                "cluster_profiles": [],
                "message": f"Error parsing clustering results: {e}"
            }
    else:
        return {
            "available": False,
            "selected_k": 0,
            "cluster_profiles": [],
            "message": "Clustering results JSON report file not found. Run clustering_insights.py first."
        }

@router.get("/summary")
def get_insights_summary():
    """
    Returns a compact summary of available ML models and insight files.
    """
    terms_available = config.EXPLAINABILITY_JSON_PATH.exists()
    clusters_available = config.CLUSTERING_JSON_PATH.exists()
    
    # Check if model is loaded or exists
    model_status = model_service.get_model_status()
    model_loaded = model_status["model_loaded"]
    model_name = model_status["model_name"]
    
    missing_files = []
    if not terms_available:
        missing_files.append("model_explainability_results.json")
    if not clusters_available:
        missing_files.append("clustering_results.json")
    if not model_loaded:
        missing_files.append("best_satisfaction_model.joblib")
        
    if missing_files:
        msg = f"Dashboard is partially configured. Missing artifacts: {', '.join(missing_files)}. Please execute ML pipeline scripts."
    else:
        msg = "Dashboard is fully configured and ready."
        
    return {
        "top_terms_available": terms_available,
        "cluster_profiles_available": clusters_available,
        "model_available": model_loaded,
        "model_name": model_name,
        "message": msg
    }
