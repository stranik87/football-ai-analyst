from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.services.fixture_service import FixtureService


BASE_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)

router = APIRouter(
    prefix="/dashboard",
    tags=["Dashboard"],
)


@router.get(
    "",
    response_class=HTMLResponse,
)
def dashboard_home(
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Главная страница Dashboard.
    """

    fixture_service = FixtureService(db)

    fixtures = fixture_service.get_latest_matches(
        limit=10
    )

    serialized_fixtures = [
        fixture_service.serialize(fixture)
        for fixture in fixtures
    ]

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "title": "Football AI Analyst",
            "fixtures": serialized_fixtures,
        },
    )