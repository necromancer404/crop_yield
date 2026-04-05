from typing import Any

from pydantic import BaseModel, Field, field_validator


class CropPredictRequest(BaseModel):
    N: float = Field(..., ge=0, le=200, description="Nitrogen (kg/ha)")
    P: float = Field(..., ge=0, le=200, description="Phosphorus (kg/ha)")
    K: float = Field(..., ge=0, le=200, description="Potassium (kg/ha)")
    temperature: float = Field(..., ge=-10, le=55, description="Temperature (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Humidity (%)")
    ph: float = Field(..., ge=0, le=14, description="Soil pH")
    rainfall: float = Field(..., ge=0, le=2000, description="Rainfall (mm)")


class YieldPredictRequest(BaseModel):
    State: str = Field(..., min_length=1, max_length=120)
    District: str = Field(..., min_length=1, max_length=120)
    Crop: str = Field(..., min_length=1, max_length=120)
    Season: str = Field(..., min_length=1, max_length=80)
    Temperature: float = Field(..., ge=-10, le=55)
    Humidity: float = Field(..., ge=0, le=100)
    Soil_Moisture: float = Field(..., ge=0, le=100)
    Area: float = Field(..., gt=0, description="Area (hectares)")
    Crop_Year: int | None = Field(None, ge=1990, le=2035, description="Optional; defaults to recent year if omitted")

    @field_validator("State", "District", "Crop", "Season", mode="before")
    @classmethod
    def strip_strings(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip()
        return v


class FullRecommendRequest(CropPredictRequest, YieldPredictRequest):
    pass


class PredictCropResponse(BaseModel):
    recommended_crop: str
    model: str = "RandomForestClassifier"
    confidence_hint: str | None = None


class PredictYieldResponse(BaseModel):
    predicted_yield: float
    unit: str = "production per hectare (same units as training data)"
    models: list[str] = Field(default_factory=lambda: ["RandomForestRegressor", "XGBRegressor"])
    rf_contribution: float | None = None
    xgb_contribution: float | None = None


class RecommendResponse(BaseModel):
    recommended_crop: str
    mapped_crop_for_yield: str | None = Field(
        None, description="Crop name aligned to the yield training vocabulary"
    )
    predicted_yield: float
    predicted_yield_for_recommended: float | None = None
    suggestions: list[str]
    fertilizer_suggestions: list[str]
    yield_insights: list[str]


class ErrorResponse(BaseModel):
    detail: str
