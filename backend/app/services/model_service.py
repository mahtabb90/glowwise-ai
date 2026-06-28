import joblib
import numpy as np
from pathlib import Path
from app.core import config
from app.models.prediction import SatisfactionPredictionRequest, SatisfactionPredictionResponse

# Caching state for the model
_model = None

def load_model():
    """
    Lazily loads the model from config.MODEL_PATH.
    If the file does not exist, returns None instead of crashing.
    """
    global _model
    if _model is not None:
        return _model
        
    model_path = config.MODEL_PATH
    if not model_path.exists():
        print(f"[WARNING] Model artifact not found at: {model_path}")
        return None
        
    try:
        # Load the serialized scikit-learn pipeline
        _model = joblib.load(model_path)
        print(f"[INFO] Successfully loaded model from: {model_path}")
        return _model
    except Exception as e:
        print(f"[ERROR] Failed to load model from: {model_path}. Error: {e}")
        return None

def get_model_status() -> dict:
    """
    Returns the loaded status, resolved path, and message.
    """
    model_path = config.MODEL_PATH
    loaded = _model is not None or load_model() is not None
    
    if loaded:
        return {
            "model_loaded": True,
            "model_path": str(model_path.absolute().as_posix()),
            "model_name": "logistic_regression_tuned",
            "message": "Model is loaded and ready for inference."
        }
    else:
        return {
            "model_loaded": False,
            "model_path": str(model_path.absolute().as_posix()),
            "model_name": "unknown",
            "message": f"Best model artifact not found. Run ml/src/model_comparison.py first to generate ml/models/best_satisfaction_model.joblib."
        }

def predict_satisfaction(request: SatisfactionPredictionRequest) -> dict:
    """
    Performs inference on the request.
    Raises RuntimeError if the model is not loaded.
    """
    model = load_model()
    if model is None:
        raise RuntimeError("Model artifact is missing. Predictions cannot be processed.")
        
    # Combine title and text exactly matching training preprocessing: review_title + " " + review_text
    title = request.review_title or ""
    text = request.review_text or ""
    combined_text = f"{title} {text}".strip()
    
    # Run prediction (expects a list of strings)
    pred_label = int(model.predict([combined_text])[0])
    
    # Run predict_proba if supported
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([combined_text])[0]
        prob_low = float(probs[0])
        prob_high = float(probs[1])
        confidence = float(np.max(probs))
    else:
        # Fallback in case of custom model mock or lack of probability support
        prob_low = 0.0 if pred_label == 1 else 1.0
        prob_high = 1.0 if pred_label == 1 else 0.0
        confidence = 1.0
        
    predicted_class = "high_satisfaction" if pred_label == 1 else "low_or_medium_satisfaction"
    
    # Input preview snippet
    input_preview = combined_text[:100] + "..." if len(combined_text) > 100 else combined_text
    
    return {
        "predicted_label": pred_label,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probability_low_or_medium": prob_low,
        "probability_high_satisfaction": prob_high,
        "model_name": "logistic_regression_tuned",
        "input_preview": input_preview
    }
