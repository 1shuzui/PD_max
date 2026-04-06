from fastapi import APIRouter

from app import config as app_config
from app.api.v1.routes import auth, tl

api_router = APIRouter()
api_router.include_router(tl.router, tags=["TL比价模块"])
api_router.include_router(auth.router, tags=["用户认证"])
if app_config.AI_DETECTION_ENABLED:
    from app.api.v1.routes import ai_detection

    api_router.include_router(ai_detection.router, tags=["AI鉴伪模块"])
