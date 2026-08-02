from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.schemas.team import TeamResponse
from app.services.team_service import TeamService


router = APIRouter(
    prefix="/teams",
    tags=["Teams"],
)


@router.get(
    "",
    response_model=list[TeamResponse],
)
def get_teams(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> list[dict]:
    """
    Получить список команд.
    """

    service = TeamService(db)

    teams = service.get_all(
        limit=limit,
        offset=offset,
    )

    return [
        service.serialize(team)
        for team in teams
    ]


@router.get(
    "/search",
    response_model=list[TeamResponse],
)
def search_teams(
    name: str,
    limit: int = 20,
    db: Session = Depends(get_db),
) -> list[dict]:
    """
    Найти команды по части названия.
    """

    service = TeamService(db)

    teams = service.search(
        name=name,
        limit=limit,
    )

    return [
        service.serialize(team)
        for team in teams
    ]


@router.get(
    "/{team_id}",
    response_model=TeamResponse,
)
def get_team(
    team_id: int,
    db: Session = Depends(get_db),
) -> dict:
    """
    Получить команду по ID.
    """

    service = TeamService(db)

    team = service.get_by_id(
        team_id
    )

    if team is None:
        raise HTTPException(
            status_code=404,
            detail=f"Команда {team_id} не найдена.",
        )

    return service.serialize(team)