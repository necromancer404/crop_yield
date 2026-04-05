from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from feature_engineering.recommendation_features import RECOMMENDATION_FEATURE_COLUMNS
from models.schemas import CropPredictRequest, YieldPredictRequest
from utils.crop_matching import match_dataset_crop
from utils.paths import get_artifact_dir


class ModelRegistry:
    def __init__(self, artifact_dir: Path | None = None) -> None:
        self.artifact_dir = artifact_dir or get_artifact_dir()
        self._rec_pipe: Any | None = None
        self._yield_bundle: dict[str, Any] | None = None
        self._yield_vocab: dict[str, Any] | None = None
        self._last_error: str | None = None

    def refresh(self) -> bool:
        self._last_error = None
        rec_path = self.artifact_dir / "crop_recommendation_pipeline.pkl"
        y_path = self.artifact_dir / "yield_model_bundle.pkl"
        v_path = self.artifact_dir / "yield_crop_vocab.json"

        if not rec_path.is_file():
            self._last_error = f"Missing recommendation model: {rec_path}"
            return False
        if not y_path.is_file():
            self._last_error = f"Missing yield model: {y_path}"
            return False

        try:
            self._rec_pipe = joblib.load(rec_path)
            self._yield_bundle = joblib.load(y_path)
            if v_path.is_file():
                with open(v_path, encoding="utf-8") as f:
                    self._yield_vocab = json.load(f)
            else:
                self._yield_vocab = {"crops": [], "default_crop": "Rice"}
            return True
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            self._rec_pipe = None
            self._yield_bundle = None
            self._yield_vocab = None
            return False

    @property
    def ready(self) -> bool:
        return self._rec_pipe is not None and self._yield_bundle is not None

    def predict_crop(self, req: CropPredictRequest) -> tuple[str, np.ndarray]:
        if not self._rec_pipe:
            raise RuntimeError("Recommendation model not loaded")
        row = pd.DataFrame(
            [
                {
                    "N": req.N,
                    "P": req.P,
                    "K": req.K,
                    "temperature": req.temperature,
                    "humidity": req.humidity,
                    "ph": req.ph,
                    "rainfall": req.rainfall,
                }
            ]
        )[RECOMMENDATION_FEATURE_COLUMNS]
        label = str(self._rec_pipe.predict(row)[0])
        proba = None
        clf = self._rec_pipe.named_steps.get("clf")
        if clf is not None and hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(row)[0]
        return label, proba

    def _build_yield_frame(self, req: YieldPredictRequest, crop_override: str | None) -> pd.DataFrame:
        assert self._yield_bundle is not None
        cat_cols: list[str] = list(self._yield_bundle["cat_cols"])
        num_cols: list[str] = list(self._yield_bundle["num_cols"])
        crop = crop_override if crop_override is not None else req.Crop

        row: dict[str, Any] = {
            "State": req.State,
            "District": req.District,
            "Crop": crop,
            "Season": req.Season,
            "Temperature": req.Temperature,
            "Humidity": req.Humidity,
            "Soil_Moisture": req.Soil_Moisture,
            "Area": req.Area,
        }
        if "Crop_Year" in num_cols:
            row["Crop_Year"] = int(req.Crop_Year) if req.Crop_Year is not None else 2020

        df = pd.DataFrame([row])
        for c in cat_cols + num_cols:
            if c not in df.columns:
                raise ValueError(f"Missing feature column {c}")
        return df[cat_cols + num_cols]

    def predict_yield(self, req: YieldPredictRequest, crop_override: str | None = None) -> dict[str, float]:
        if not self._yield_bundle:
            raise RuntimeError("Yield model not loaded")
        prep = self._yield_bundle["preprocessor"]
        rf = self._yield_bundle["rf"]
        xgb = self._yield_bundle["xgb"]
        X = prep.transform(self._build_yield_frame(req, crop_override))
        pred_rf = max(0.0, float(rf.predict(X)[0]))
        pred_xgb = max(0.0, float(xgb.predict(X)[0]))
        w_rf = float(self._yield_bundle.get("ensemble_weight_rf", 0.5))
        w_xgb = float(self._yield_bundle.get("ensemble_weight_xgb", 0.5))
        s = w_rf + w_xgb
        if s <= 0:
            w_rf, w_xgb = 0.5, 0.5
        else:
            w_rf, w_xgb = w_rf / s, w_xgb / s
        blend = max(0.0, w_rf * pred_rf + w_xgb * pred_xgb)
        return {"rf": pred_rf, "xgb": pred_xgb, "blend": blend}

    def map_recommended_to_yield_crop(self, recommended_label: str) -> str:
        vocab = (self._yield_vocab or {}).get("crops") or []
        default_crop = str((self._yield_vocab or {}).get("default_crop") or "Rice")
        return match_dataset_crop(recommended_label, list(vocab), default_crop)
