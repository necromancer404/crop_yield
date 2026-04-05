from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import api_router
from services.inference import ModelRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    reg = ModelRegistry()
    reg.refresh()
    app.state.ml = reg
    yield


app = FastAPI(
    title="Smart Crop Yield & Recommendation API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health")
def health():
    reg: ModelRegistry = app.state.ml
    return {"status": "ok", "models_loaded": reg.ready, "detail": reg._last_error}
