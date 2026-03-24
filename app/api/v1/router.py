from fastapi import APIRouter

from app.api.v1.routes import tl

api_router = APIRouter()
api_router.include_router(tl.router, tags=["TL比价模块"])
