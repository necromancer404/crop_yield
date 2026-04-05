# Smart Crop Yield Prediction and Recommendation

End-to-end system: **FastAPI** backend with **RandomForest** classification, **RandomForest + XGBoost** regression, and a **React (Vite)** dashboard with **Recharts**.

## Layout

- `Crop Recommendation dataset.csv` and `Crop Prediction dataset.csv` live in the **project root** (same folder as `backend/` and `frontend/`).
- Trained artifacts are written to `backend/artifacts/*.pkl` after training.

## Prerequisites

- Python 3.10+
- Node.js 18+

## 1. Train models (required once)

From the `backend` folder:

```powershell
cd backend
python -m pip install -r requirements.txt
python -m training.train_models
```

To also **export the 80/20 train and test splits as CSV files** (same rows used for training vs metrics):

```powershell
python -m training.train_models --export-splits
```

Files are written under `backend/artifacts/splits/`:

- `crop_recommendation_train.csv`, `crop_recommendation_test.csv` — features + `label`
- `crop_yield_train.csv`, `crop_yield_test.csv` — model features + `yield`
- `split_manifest.json` — `test_size`, `random_state`, and file names

This will:

- Clean data, add `yield = Production / Area` for the prediction dataset, encode categoricals, scale numerics.
- Train **RandomForestClassifier** on N, P, K, temperature, humidity, pH, rainfall → crop label.
- Train **RandomForestRegressor** and **XGBRegressor** on State, District, Crop, Season, (+ Crop\_Year when present), Temperature, Humidity, Soil\_Moisture, Area → yield.
- Save `crop_recommendation_pipeline.pkl`, `yield_model_bundle.pkl`, and metadata JSON files under `backend/artifacts/`.

## 2. Run the API

```powershell
cd backend
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

- Docs: `http://127.0.0.1:8000/docs`
- Health: `GET http://127.0.0.1:8000/health`

### Windows: `WinError 10013` on port 8000

Some PCs block or reserve **port 8000**. Use another port (e.g. **8080**):

```powershell
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8080
```

Then point the frontend dev proxy at the same port:

1. Copy `frontend/.env.example` to `frontend/.env.development`
2. Set `VITE_API_PROXY_TARGET=http://127.0.0.1:8080`

Use `http://127.0.0.1:8080/docs` for Swagger when on 8080.

## 3. Run the frontend

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The dev server proxies API calls to port 8000.

For a production build served separately, set `VITE_API_URL` to your API origin (e.g. `http://127.0.0.1:8000`) and rebuild.

## Example API requests

**Classify crop (soil + weather)**

```http
POST http://127.0.0.1:8000/predict-crop
Content-Type: application/json

{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 22,
  "humidity": 80,
  "ph": 6.5,
  "rainfall": 200
}
```

**Predict yield**

```http
POST http://127.0.0.1:8000/predict-yield
Content-Type: application/json

{
  "State": "Karnataka",
  "District": "BANGALORE RURAL",
  "Crop": "Rice",
  "Season": "Kharif",
  "Temperature": 28,
  "Humidity": 65,
  "Soil_Moisture": 48,
  "Area": 2.5,
  "Crop_Year": 2014
}
```

**Full recommendation (crop + yield + suggestions)**

```http
POST http://127.0.0.1:8000/recommend
Content-Type: application/json

{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 22,
  "humidity": 80,
  "ph": 6.5,
  "rainfall": 200,
  "State": "Karnataka",
  "District": "BANGALORE RURAL",
  "Crop": "Rice",
  "Season": "Kharif",
  "Temperature": 28,
  "Humidity": 65,
  "Soil_Moisture": 48,
  "Area": 2.5,
  "Crop_Year": 2014
}
```

**Chart data**

- `GET http://127.0.0.1:8000/analytics/yield-vs-rainfall`
- `GET http://127.0.0.1:8000/analytics/yield-by-crop`
- `GET http://127.0.0.1:8000/analytics/rainfall-by-crop`

## Backend modules

| Area | Path |
|------|------|
| API | `backend/app.py`, `backend/routes/` |
| Schemas / validation | `backend/models/schemas.py` |
| Preprocessing | `backend/preprocessing/` |
| Feature lists | `backend/feature_engineering/` |
| Training | `backend/training/train_models.py` |
| Metrics | `backend/evaluation/` |
| Inference + model load | `backend/services/inference.py` |
| Suggestions | `backend/utils/suggestions.py`, `backend/utils/crop_matching.py` |

## Notes

- Yield values are in the same units as `Production / Area` in the government-style crop statistics file (mixed crops and scales). Use predictions comparatively rather than as absolute agronomic guarantees.
- If the API reports models missing, run the training step again from `backend/`.
