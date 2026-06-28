from typing import Optional
from pydantic import BaseModel, Field, field_validator

class SatisfactionPredictionRequest(BaseModel):
    review_text: str = Field(..., description="The main text of the customer review.")
    review_title: Optional[str] = Field(default="", description="The optional title of the review.")
    product_name: Optional[str] = Field(default="", description="The optional name of the product.")
    brand_name: Optional[str] = Field(default="", description="The optional name of the product brand.")

    @field_validator("review_text")
    @classmethod
    def validate_and_trim_review_text(cls, v: str) -> str:
        if v is None:
            raise ValueError("review_text must not be None.")
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("review_text must not be empty or whitespace-only.")
        return trimmed

    @field_validator("review_title", "product_name", "brand_name")
    @classmethod
    def trim_optional_fields(cls, v: Optional[str]) -> str:
        if v is None:
            return ""
        return v.strip()

class SatisfactionPredictionResponse(BaseModel):
    predicted_label: int = Field(..., description="Inference label: 0 for low/med satisfaction, 1 for high satisfaction.")
    predicted_class: str = Field(..., description="Semantic label description: low_or_medium_satisfaction or high_satisfaction.")
    confidence: float = Field(..., description="Prediction confidence score as the maximum class probability.")
    probability_low_or_medium: float = Field(..., description="Model probability for low/medium satisfaction.")
    probability_high_satisfaction: float = Field(..., description="Model probability for high satisfaction.")
    model_name: str = Field(..., description="The name of the classification model used.")
    input_preview: str = Field(..., description="A short preview snippet of the compiled text.")

class ModelStatusResponse(BaseModel):
    model_loaded: bool = Field(..., description="Whether the model joblib artifact is currently loaded.")
    model_path: str = Field(..., description="The resolved system absolute path to the model file.")
    model_name: str = Field(..., description="Name of the model if loaded, or 'unknown'.")
    message: str = Field(..., description="Status message detailing instructions or success details.")
