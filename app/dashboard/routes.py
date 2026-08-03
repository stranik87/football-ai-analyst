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
from app.models.league_season import LeagueSeason
from app.models.standing import Standing
from app.models.team import Team
from app.services.fixture_service import FixtureService
from app.services.league_service import LeagueService
from app.services.prediction_service import PredictionService
from app.services.standing_service import StandingService
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

    completed_fixtures = (
        db.query(func.count(Fixture.id))
        .filter(
            Fixture.status_short.in_(
                ["FT", "AET", "PEN"]
            ),
            Fixture.home_goals.isnot(None),
            Fixture.away_goals.isnot(None),
        )
        .scalar()
        or 0
    )

    home_wins = (
        db.query(func.count(Fixture.id))
        .filter(
            Fixture.status_short.in_(
                ["FT", "AET", "PEN"]
            ),
            Fixture.home_goals
            > Fixture.away_goals,
        )
        .scalar()
        or 0
    )

    draws = (
        db.query(func.count(Fixture.id))
        .filter(
            Fixture.status_short.in_(
                ["FT", "AET", "PEN"]
            ),
            Fixture.home_goals
            == Fixture.away_goals,
        )
        .scalar()
        or 0
    )

    away_wins = (
        db.query(func.count(Fixture.id))
        .filter(
            Fixture.status_short.in_(
                ["FT", "AET", "PEN"]
            ),
            Fixture.home_goals
            < Fixture.away_goals,
        )
        .scalar()
        or 0
    )

    if completed_fixtures:
        home_win_percent = (
            home_wins / completed_fixtures * 100
        )
        draw_percent = (
            draws / completed_fixtures * 100
        )
        away_win_percent = (
            away_wins / completed_fixtures * 100
        )
    else:
        home_win_percent = 0.0
        draw_percent = 0.0
        away_win_percent = 0.0

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "title": "Football AI Analyst",
            "active_page": "dashboard",
            "fixtures": serialized_fixtures,
            "total_fixtures": total_fixtures,
            "total_teams": total_teams,
            "total_leagues": total_leagues,
            "total_standings": total_standings,
            "completed_fixtures": completed_fixtures,
            "home_wins": home_wins,
            "draws": draws,
            "away_wins": away_wins,
            "home_win_percent": home_win_percent,
            "draw_percent": draw_percent,
            "away_win_percent": away_win_percent,
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
            "active_page": "fixtures",
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
            "active_page": "fixtures",
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
            "active_page": "teams",
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
            "active_page": "teams",
            "team": team_data,
            "fixtures": serialized_fixtures,
        },
    )


@router.get(
    "/leagues",
    response_class=HTMLResponse,
)
def dashboard_leagues(
    request: Request,
    name: str | None = None,
    db: Session = Depends(get_db),
):
    """
    HTML-страница списка лиг.
    """

    league_service = LeagueService(db)

    if name:
        leagues = league_service.search(
            name=name,
            limit=50,
        )
    else:
        leagues = league_service.get_all(
            limit=50,
            offset=0,
        )

    serialized_leagues = [
        league_service.serialize(league)
        for league in leagues
    ]

    return templates.TemplateResponse(
        request=request,
        name="leagues.html",
        context={
            "title": "Лиги",
            "active_page": "leagues",
            "leagues": serialized_leagues,
            "name_query": name or "",
        },
    )


@router.get(
    "/league/{league_id}",
    response_class=HTMLResponse,
)
def dashboard_league(
    league_id: int,
    request: Request,
    season: int = 2024,
    db: Session = Depends(get_db),
):
    """
    HTML-страница одной лиги.
    """

    league_service = LeagueService(db)
    standing_service = StandingService(db)
    fixture_service = FixtureService(db)
    team_service = TeamService(db)

    league = league_service.get_by_id(league_id)

    if league is None:
        raise HTTPException(
            status_code=404,
            detail=f"Лига {league_id} не найдена.",
        )

    league_data = league_service.serialize(league)

    seasons = (
        db.query(LeagueSeason)
        .filter(
            LeagueSeason.league_id == league_id
        )
        .order_by(
            LeagueSeason.season.desc()
        )
        .all()
    )

    selected_season = next(
        (
            item
            for item in seasons
            if item.season == season
        ),
        None,
    )

    standings = (
        standing_service.get_by_league_and_season(
            league_id=league_id,
            season=season,
        )
    )

    serialized_standings = [
        standing_service.serialize(item)
        for item in standings
    ]

    teams = (
        db.query(Team)
        .filter(
            Team.league_id == league_id
        )
        .order_by(
            Team.name.asc()
        )
        .all()
    )

    serialized_teams = [
        team_service.serialize(team)
        for team in teams
    ]

    fixtures = []

    if selected_season is not None:
        fixtures = (
            db.query(Fixture)
            .filter(
                Fixture.league_season_id
                == selected_season.id
            )
            .order_by(
                Fixture.kickoff.desc()
            )
            .limit(10)
            .all()
        )

    serialized_fixtures = [
        fixture_service.serialize(fixture)
        for fixture in fixtures
    ]

    return templates.TemplateResponse(
        request=request,
        name="league.html",
        context={
            "title": league.name,
            "active_page": "leagues",
            "league": league_data,
            "seasons": seasons,
            "selected_season": season,
            "standings": serialized_standings,
            "teams": serialized_teams,
            "fixtures": serialized_fixtures,
        },
    )


@router.get(
    "/standings",
    response_class=HTMLResponse,
)
def dashboard_standings(
    request: Request,
    league_api_id: int = 39,
    season: int = 2024,
    db: Session = Depends(get_db),
):
    """
    HTML-страница турнирной таблицы.
    """

    standing_service = StandingService(db)
    league_service = LeagueService(db)

    standings = (
        standing_service.get_by_league_api_id_and_season(
            league_api_id=league_api_id,
            season=season,
        )
    )

    leagues = (
        db.query(League)
        .filter(
            League.api_id.in_(
                [39, 61, 78, 135, 140]
            )
        )
        .order_by(
            League.name.asc()
        )
        .all()
    )

    selected_league = (
        db.query(League)
        .filter(
            League.api_id == league_api_id
        )
        .first()
    )

    serialized_standings = [
        standing_service.serialize(item)
        for item in standings
    ]

    serialized_leagues = [
        league_service.serialize(league)
        for league in leagues
    ]

    return templates.TemplateResponse(
        request=request,
        name="standings.html",
        context={
            "title": "Турнирная таблица",
            "active_page": "standings",
            "standings": serialized_standings,
            "leagues": serialized_leagues,
            "selected_league": (
                league_service.serialize(
                    selected_league
                )
                if selected_league is not None
                else None
            ),
            "selected_league_api_id": league_api_id,
            "selected_season": season,
        },
    )
