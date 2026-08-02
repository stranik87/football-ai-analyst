from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from config import Config
from app.database.database import SessionLocal
from app.services.fixture_service import FixtureService
from app.services.prediction_service import PredictionService


app = FastAPI(
    title=Config.APP_NAME,
    version=Config.APP_VERSION,
    description=(
        "REST API проекта Football AI Analyst."
    ),
)


def get_db():
    """
    Создать сессию базы данных для запроса.
    """

    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


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


@app.get("/fixtures/latest")
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


@app.get("/fixtures/{fixture_id}")
def get_fixture(
    fixture_id: int,
    db: Session = Depends(get_db),
) -> dict:
    """
    Получить информацию о матче по ID.
    """

    service = FixtureService(db)
    fixture = service.get_by_id(fixture_id)

    if fixture is None:
        raise HTTPException(
            status_code=404,
            detail=f"Матч с ID {fixture_id} не найден.",
        )

    return service.serialize(fixture)

@app.get("/predict/{fixture_id}")
def predict_fixture(
    fixture_id: int,
    db: Session = Depends(get_db),
) -> dict:
    """
    Получить прогноз исхода матча.
    """

    try:
        service = PredictionService(db)

        return service.predict(fixture_id)

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Не удалось построить прогноз.",
        ) from error