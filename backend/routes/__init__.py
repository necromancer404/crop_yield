from fastapi import APIRouter

from routes.analytics import router as analytics_router
from routes.predict import router as predict_router

api_router = APIRouter()
api_router.include_router(predict_router, tags=["prediction"])
api_router.include_router(analytics_router, tags=["analytics"])

__all__ = ["api_router"]
