from fastapi import APIRouter, HTTPException, Request

from models.schemas import (
    CropPredictRequest,
    FullRecommendRequest,
    PredictCropResponse,
    PredictYieldResponse,
    RecommendResponse,
    YieldPredictRequest,
)
from services.inference import ModelRegistry
from utils.suggestions import fertilizer_suggestions, yield_insights

router = APIRouter()


def _ml(request: Request) -> ModelRegistry:
    reg: ModelRegistry = request.app.state.ml
    return reg


def _ensure_ready(reg: ModelRegistry) -> None:
    if not reg.ready:
        msg = reg._last_error or "Models not loaded. Run: python -m training.train_models (from backend/)."
        raise HTTPException(status_code=503, detail=msg)


@router.post("/predict-crop", response_model=PredictCropResponse)
def predict_crop(req: CropPredictRequest, request: Request) -> PredictCropResponse:
    reg = _ml(request)
    _ensure_ready(reg)
    try:
        label, proba = reg.predict_crop(req)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    hint = None
    if proba is not None:
        top = float(proba.max())
        hint = f"Top class probability ~{top:.2f} (calibrated scores not applied)."
    return PredictCropResponse(recommended_crop=label, confidence_hint=hint)


@router.post("/predict-yield", response_model=PredictYieldResponse)
def predict_yield(req: YieldPredictRequest, request: Request) -> PredictYieldResponse:
    reg = _ml(request)
    _ensure_ready(reg)
    try:
        out = reg.predict_yield(req, crop_override=None)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictYieldResponse(
        predicted_yield=out["blend"],
        rf_contribution=out["rf"],
        xgb_contribution=out["xgb"],
    )


@router.post("/recommend", response_model=RecommendResponse)
def recommend(req: FullRecommendRequest, request: Request) -> RecommendResponse:
    reg = _ml(request)
    _ensure_ready(reg)
    try:
        crop_label, _ = reg.predict_crop(req)
        mapped = reg.map_recommended_to_yield_crop(crop_label)
        y_user = reg.predict_yield(req, crop_override=None)
        y_rec = reg.predict_yield(req, crop_override=mapped)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    fert = fertilizer_suggestions(req)
    insights = yield_insights(y_user["blend"], y_rec["blend"])
    summary = [
        f"Soil and climate classifier suggests: {crop_label}.",
        f"Yield model uses crop label: {mapped}.",
    ]

    return RecommendResponse(
        recommended_crop=crop_label,
        mapped_crop_for_yield=mapped,
        predicted_yield=float(y_user["blend"]),
        predicted_yield_for_recommended=float(y_rec["blend"]),
        suggestions=summary,
        fertilizer_suggestions=fert,
        yield_insights=insights,
    )
