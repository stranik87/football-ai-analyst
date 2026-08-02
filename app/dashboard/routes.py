from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.models.fixture import Fixture
from app.models.league import League
from app.models.standing import Standing
from app.models.team import Team
from app.services.fixture_service import FixtureService

from fastapi import APIRouter, Depends, HTTPException, Request
from app.services.prediction_service import PredictionService

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

    total_fixtures = (
        db.query(func.count(Fixture.id)).scalar()
        or 0
    )

    total_teams = (
        db.query(func.count(Team.id)).scalar()
        or 0
    )

    total_leagues = (
        db.query(func.count(League.id)).scalar()
        or 0
    )

    total_standings = (
        db.query(func.count(Standing.id)).scalar()
        or 0
    )

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "title": "Football AI Analyst",
            "fixtures": serialized_fixtures,
            "total_fixtures": total_fixtures,
            "total_teams": total_teams,
            "total_leagues": total_leagues,
            "total_standings": total_standings,
        },
    )

@router.get(
    "/predict/{fixture_id}",
    response_class=HTMLResponse,
)
def dashboard_prediction(
    fixture_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    HTML-страница прогноза матча.
    """

    try:
        prediction_service = PredictionService(db)
        prediction = prediction_service.predict(fixture_id)

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    probabilities = prediction["probabilities"]

    return templates.TemplateResponse(
        request=request,
        name="prediction.html",
        context={
            "title": "Прогноз матча",
            "prediction": prediction,
            "home_percent": (
                probabilities["home_win"] * 100
            ),
            "draw_percent": (
                probabilities["draw"] * 100
            ),
            "away_percent": (
                probabilities["away_win"] * 100
            ),
            "confidence_percent": (
                prediction["confidence"] * 100
            ),
        },
    )