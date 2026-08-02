from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.schemas.standing import StandingResponse
from app.services.standing_service import StandingService


router = APIRouter(
    prefix="/standings",
    tags=["Standings"],
)


@router.get(
    "/{league_api_id}/{season}",
    response_model=list[StandingResponse],
)
def get_standings(
    league_api_id: int,
    season: int,
    db: Session = Depends(get_db),
) -> list[dict]:
    """
    Получить турнирную таблицу по API ID лиги и сезону.
    """

    service = StandingService(db)

    standings = (
        service.get_by_league_api_id_and_season(
            league_api_id=league_api_id,
            season=season,
        )
    )

    if not standings:
        raise HTTPException(
            status_code=404,
            detail=(
                "Турнирная таблица не найдена: "
                f"league_api_id={league_api_id}, "
                f"season={season}."
            ),
        )

    return [
        service.serialize(standing)
        for standing in standings
    ]