from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.services.prediction_service import PredictionService


router = APIRouter(
    prefix="/predict",
    tags=["Predictions"],
)


@router.get("/{fixture_id}")
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