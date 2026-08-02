from fastapi import FastAPI

from config import Config
from app.api.routes.fixtures import router as fixtures_router
from app.api.routes.predictions import router as predictions_router

from app.api.routes.teams import router as teams_router
from app.api.routes.leagues import router as leagues_router
from app.api.routes.standings import router as standings_router
from app.dashboard.routes import router as dashboard_router
from pathlib import Path

from fastapi.staticfiles import StaticFiles


app = FastAPI(
    title=Config.APP_NAME,
    version=Config.APP_VERSION,
    description=(
        "REST API проекта Football AI Analyst."
    ),
)

BASE_DIR = Path(__file__).resolve().parent
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "dashboard" / "static")),
    name="static",
)

@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": Config.APP_NAME,
        "version": Config.APP_VERSION,
        "status": "running",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
    }


app.include_router(fixtures_router)
app.include_router(predictions_router)
app.include_router(teams_router)
app.include_router(leagues_router)
app.include_router(standings_router)
app.include_router(dashboard_router)