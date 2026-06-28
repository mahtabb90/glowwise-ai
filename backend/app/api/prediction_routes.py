from fastapi import APIRouter, HTTPException, status
from app.models.prediction import (
    SatisfactionPredictionRequest, 
    SatisfactionPredictionResponse, 
    ModelStatusResponse
)
from app.services import model_service

router = APIRouter(prefix="/api", tags=["Prediction"])

@router.get("/model/status", response_model=ModelStatusResponse)
def get_model_status():
    """
    Checks and returns the status of the model artifact.
    """
    status_data = model_service.get_model_status()
    return status_data

@router.post("/predict/satisfaction", response_model=SatisfactionPredictionResponse)
def predict_satisfaction(request: SatisfactionPredictionRequest):
    """
    Classifies customer review satisfaction as low/medium (0) or high (1).
    Returns 503 Service Unavailable if the model artifact is missing.
    """
    try:
        prediction_result = model_service.predict_satisfaction(request)
        return prediction_result
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
