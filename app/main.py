from fastapi import FastAPI

from app.api.v1.router import api_router
from app.database import create_tables

app = FastAPI(title="TL比价系统", version="1.0.0")

app.include_router(api_router)


@app.on_event("startup")
def on_startup():
    create_tables()
