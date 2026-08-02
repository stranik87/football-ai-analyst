from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.schemas.league import LeagueResponse
from app.services.league_service import LeagueService


router = APIRouter(
    prefix="/leagues",
    tags=["Leagues"],
)


@router.get(
    "",
    response_model=list[LeagueResponse],
)
def get_leagues(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> list[dict]:
    """
    Получить список лиг.
    """

    service = LeagueService(db)

    leagues = service.get_all(
        limit=limit,
        offset=offset,
    )

    return [
        service.serialize(league)
        for league in leagues
    ]


@router.get(
    "/search",
    response_model=list[LeagueResponse],
)
def search_leagues(
    name: str,
    limit: int = 20,
    db: Session = Depends(get_db),
) -> list[dict]:
    """
    Найти лиги по части названия.
    """

    service = LeagueService(db)

    leagues = service.search(
        name=name,
        limit=limit,
    )

    return [
        service.serialize(league)
        for league in leagues
    ]


@router.get(
    "/{league_id}",
    response_model=LeagueResponse,
)
def get_league(
    league_id: int,
    db: Session = Depends(get_db),
) -> dict:
    """
    Получить лигу по ID.
    """

    service = LeagueService(db)

    league = service.get_by_id(
        league_id
    )

    if league is None:
        raise HTTPException(
            status_code=404,
            detail=f"Лига {league_id} не найдена.",
        )

    return service.serialize(league)