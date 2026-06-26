"""
GlowWise AI - Model Inference Module
Loads serialized ML models from disk and exposes prediction functions
supporting single-record or batch inferences. Called directly by backend app/services.
"""

import os
import joblib

# Cache path for the sentiment model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/sentiment_model.joblib")
_model_cache = None

def get_model():
    """
    Loads and caches the model artifact.
    """
    global _model_cache
    if _model_cache is None:
        if os.path.exists(MODEL_PATH):
            _model_cache = joblib.load(MODEL_PATH)
        else:
            print(f"Warning: Model not found at {MODEL_PATH}. Using mock fallback.")
            _model_cache = {
                "model_name": "Mock Sentiment Classifier",
                "version": "0.0.0",
                "fallback": True
            }
    return _model_cache

def predict_sentiment(review_text: str) -> dict:
    """
    Predicts sentiment class and confidence for a single review string.
    """
    model = get_model()
    # Simple placeholder inference rule
    cleaned_text = review_text.lower().strip()
    
    # Very basic keywords rules to simulate ML model inference before training
    positive_words = ["love", "great", "excellent", "glow", "amazing", "best", "smooth", "perfect"]
    negative_words = ["waste", "bad", "breakout", "dry", "irritate", "broke", "expensive", "disappointed"]
    
    score = 0.5
    for word in positive_words:
        if word in cleaned_text:
            score += 0.15
    for word in negative_words:
        if word in cleaned_text:
            score -= 0.15
            
    score = max(0.0, min(1.0, score))
    
    if score > 0.6:
        sentiment = "positive"
        confidence = score
    elif score < 0.4:
        sentiment = "negative"
        confidence = 1.0 - score
    else:
        sentiment = "neutral"
        confidence = 0.5
        
    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 2),
        "rating_prediction": round(1 + score * 4, 1),
        "model_info": {
            "name": model.get("model_name", "Unknown"),
            "version": model.get("version", "0.0.0")
        }
    }

if __name__ == "__main__":
    # Test prediction
    test_review = "I absolutely love this cleanser! It leaves my skin glowing and super smooth."
    result = predict_sentiment(test_review)
    print(f"Review: {test_review}")
    print(f"Result: {result}")
