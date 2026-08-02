from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
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
from app.services.prediction_service import PredictionService
from app.services.team_service import TeamService


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
        prediction = prediction_service.predict(
            fixture_id
        )

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


@router.get(
    "/fixtures",
    response_class=HTMLResponse,
)
def dashboard_fixtures(
    request: Request,
    team: str | None = None,
    db: Session = Depends(get_db),
):
    """
    HTML-страница списка матчей.
    """

    fixture_service = FixtureService(db)

    if team:
        fixtures = fixture_service.search_team_matches(
            team_name=team,
            limit=50,
        )
    else:
        fixtures = fixture_service.get_latest_matches(
            limit=50
        )

    serialized_fixtures = [
        fixture_service.serialize(fixture)
        for fixture in fixtures
    ]

    return templates.TemplateResponse(
        request=request,
        name="fixtures.html",
        context={
            "title": "Матчи",
            "fixtures": serialized_fixtures,
            "team_query": team or "",
        },
    )


@router.get(
    "/teams",
    response_class=HTMLResponse,
)
def dashboard_teams(
    request: Request,
    name: str | None = None,
    db: Session = Depends(get_db),
):
    """
    HTML-страница списка команд.
    """

    team_service = TeamService(db)

    if name:
        teams = team_service.search(
            name=name,
            limit=50,
        )
    else:
        teams = team_service.get_all(
            limit=50,
            offset=0,
        )

    serialized_teams = [
        team_service.serialize(team)
        for team in teams
    ]

    return templates.TemplateResponse(
        request=request,
        name="teams.html",
        context={
            "title": "Команды",
            "teams": serialized_teams,
            "name_query": name or "",
        },
    )


@router.get(
    "/team/{team_id}",
    response_class=HTMLResponse,
)
def dashboard_team(
    team_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    HTML-страница одной команды.
    """

    team_service = TeamService(db)
    fixture_service = FixtureService(db)

    team = team_service.get_by_id(team_id)

    if team is None:
        raise HTTPException(
            status_code=404,
            detail=f"Команда {team_id} не найдена.",
        )

    team_data = team_service.serialize(team)

    fixtures = fixture_service.search_team_matches(
        team_name=team.name,
        limit=10,
    )

    serialized_fixtures = [
        fixture_service.serialize(fixture)
        for fixture in fixtures
    ]

    return templates.TemplateResponse(
        request=request,
        name="team.html",
        context={
            "title": team.name,
            "team": team_data,
            "fixtures": serialized_fixtures,
        },
    )