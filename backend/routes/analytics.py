import pandas as pd
from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder

from preprocessing.cleaning import clean_crop_prediction_frame, clean_recommendation_frame
from utils.config import Settings

router = APIRouter()


def _points_records(df: pd.DataFrame) -> list[dict]:
    """Native Python scalars for JSON (avoids numpy serialization issues)."""
    return jsonable_encoder(df.to_dict(orient="records"))


@router.get("/analytics/rainfall-by-crop")
def rainfall_by_crop(top: int = 12) -> dict:
    settings = Settings.load()
    df = pd.read_csv(settings.crop_recommendation_csv)
    df = clean_recommendation_frame(df)
    g = df.groupby("label", as_index=False)["rainfall"].mean().sort_values("rainfall", ascending=False)
    g = g.head(max(3, min(top, 50))).rename(columns={"label": "crop"})
    return {"points": _points_records(g.astype({"crop": str}))}


@router.get("/analytics/yield-by-crop")
def yield_by_crop(top: int = 12) -> dict:
    settings = Settings.load()
    df = pd.read_csv(settings.crop_prediction_csv, low_memory=False)
    df = clean_crop_prediction_frame(df)
    g = df.groupby("Crop", as_index=False)["yield"].mean().sort_values("yield", ascending=False)
    g = g.head(max(3, min(top, 50))).rename(columns={"Crop": "crop", "yield": "avg_yield"})
    return {"points": _points_records(g.astype({"crop": str}))}


@router.get("/analytics/yield-vs-rainfall")
def yield_vs_rainfall(max_crops: int = 15) -> dict:
    """
    Join crops by normalized name between recommendation (rainfall) and production (yield) datasets.
    """
    settings = Settings.load()
    rec = clean_recommendation_frame(pd.read_csv(settings.crop_recommendation_csv))
    pred = clean_crop_prediction_frame(pd.read_csv(settings.crop_prediction_csv, low_memory=False))

    # Group by a real column name — groupby(Series) omits the key in recent pandas (breaks merge).
    rec = rec.assign(crop_key=rec["label"].astype(str).str.strip().str.lower())
    rain = rec.groupby("crop_key", as_index=False)["rainfall"].mean()

    pred = pred.assign(crop_key=pred["Crop"].astype(str).str.strip().str.lower())
    yld = pred.groupby("crop_key", as_index=False)["yield"].mean()

    merged = rain.merge(yld, on="crop_key", how="inner")
    merged = merged.sort_values("yield", ascending=False).head(max(5, min(max_crops, 40)))
    merged = merged.rename(columns={"crop_key": "crop", "yield": "avg_yield"})
    return {"points": _points_records(merged.astype({"crop": str}))}
