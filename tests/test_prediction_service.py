import pytest

from app.database.database import SessionLocal
from app.services.prediction_service import PredictionService


TEST_FIXTURE_ID = 1377


@pytest.fixture
def db_session():
    session = SessionLocal()

    try:
        yield session
    finally:
        session.rollback()
        session.close()


def test_prediction_service_returns_prediction(
    db_session,
):
    service = PredictionService(
        db_session
    )

    prediction = service.predict(
        TEST_FIXTURE_ID
    )

    assert prediction[
        "fixture_id"
    ] == TEST_FIXTURE_ID

    assert prediction[
        "predicted_result"
    ] in {
        "H",
        "D",
        "A",
    }

    assert 0.0 <= prediction[
        "confidence"
    ] <= 1.0

    probabilities = prediction[
        "probabilities"
    ]

    assert set(
        probabilities.keys()
    ) == {
        "home_win",
        "draw",
        "away_win",
    }

    assert all(
        0.0 <= probability <= 1.0
        for probability
        in probabilities.values()
    )

    assert sum(
        probabilities.values()
    ) == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_prediction_service_unknown_fixture(
    db_session,
):
    service = PredictionService(
        db_session
    )

    with pytest.raises(
        ValueError,
        match="не найден",
    ):
        service.predict(
            999999999
        )