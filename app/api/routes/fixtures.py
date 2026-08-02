from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.schemas.fixture import FixtureResponse
from app.services.fixture_service import FixtureService


router = APIRouter(
    prefix="/fixtures",
    tags=["Fixtures"],
)


@router.get(
    "/latest",
    response_model=list[FixtureResponse],
)
def get_latest_fixtures(
    limit: int = 10,
    db: Session = Depends(get_db),
) -> list[dict]:
    """
    Получить последние завершённые матчи.
    """

    service = FixtureService(db)

    fixtures = service.get_latest_matches(
        limit=limit
    )

    return [
        service.serialize(fixture)
        for fixture in fixtures
    ]


@router.get(
    "/search",
    response_model=list[FixtureResponse],
)
def search_fixtures(
    team: str,
    limit: int = 10,
    db: Session = Depends(get_db),
) -> list[dict]:
    """
    Найти последние матчи команды по названию.
    """

    service = FixtureService(db)

    fixtures = service.search_team_matches(
        team_name=team,
        limit=limit,
    )

    return [
        service.serialize(fixture)
        for fixture in fixtures
    ]


@router.get(
    "/{fixture_id}",
    response_model=FixtureResponse,
)
def get_fixture(
    fixture_id: int,
    db: Session = Depends(get_db),
) -> dict:
    """
    Получить матч по ID.
    """

    service = FixtureService(db)

    fixture = service.get_by_id(
        fixture_id
    )

    if fixture is None:
        raise HTTPException(
            status_code=404,
            detail=f"Матч {fixture_id} не найден.",
        )

    return service.serialize(
        fixture
    )