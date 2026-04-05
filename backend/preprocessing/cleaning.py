import numpy as np
import pandas as pd


def coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def clean_recommendation_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Crop recommendation: N,P,K,temperature,humidity,ph,rainfall,label."""
    cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            raise ValueError(f"Missing column {c} in recommendation dataset")
    out = coerce_numeric_columns(out, cols)
    out = out.dropna(subset=cols + (["label"] if "label" in out.columns else []))
    out = out[out["label"].astype(str).str.len() > 0]
    return out.reset_index(drop=True)


def clean_crop_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects State_Name, District_Name, Season, Crop, Temperature, Humidity,
    Soil_Moisture, Area, Production (and optional Crop_Year).
    Adds yield = Production / Area.
    """
    out = df.copy()
    rename = {
        "State_Name": "State",
        "District_Name": "District",
        "Season": "Season",
        "Crop": "Crop",
        "Temperature": "Temperature",
        "Humidity": "Humidity",
        "Soil_Moisture": "Soil_Moisture",
        "Area": "Area",
        "Production": "Production",
    }
    for old, new in rename.items():
        if old in out.columns and new not in out.columns:
            out = out.rename(columns={old: new})

    for c in ["Season", "Crop", "State", "District"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()

    num = ["Temperature", "Humidity", "Soil_Moisture", "Area", "Production"]
    if "Crop_Year" in out.columns:
        num = ["Crop_Year"] + num
    out = coerce_numeric_columns(out, num)
    out = out.dropna(subset=["Area", "Production", "Crop", "Season", "State", "District"])
    out = out[out["Area"] > 0]
    out["yield"] = out["Production"] / out["Area"]
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["yield"])
    out = out[out["yield"] >= 0]
    # Reduce extreme outliers for stable regression (keeps most crops; drops absurd ratios)
    hi = out["yield"].quantile(0.999)
    if np.isfinite(hi) and hi > 0:
        out = out[out["yield"] <= hi]
    return out.reset_index(drop=True)
