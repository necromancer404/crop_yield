from dataclasses import dataclass
from pathlib import Path

from utils.paths import get_data_dir


@dataclass(frozen=True)
class Settings:
    crop_recommendation_csv: Path
    crop_prediction_csv: Path

    @staticmethod
    def load() -> "Settings":
        root = get_data_dir()
        rec = root / "Crop Recommendation dataset.csv"
        pred = root / "Crop Prediction dataset.csv"
        return Settings(crop_recommendation_csv=rec, crop_prediction_csv=pred)
