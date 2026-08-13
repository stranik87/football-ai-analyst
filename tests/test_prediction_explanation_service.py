import pytest

from app.database.database import SessionLocal
from app.services.prediction_explanation_service import (
    PredictionExplanationService,
)


TEST_FIXTURE_ID = 1377


@pytest.fixture
def db_session():
    session = SessionLocal()

    try:
        yield session
    finally:
        session.rollback()
        session.close()


def test_prediction_explanation_returns_factors(
    db_session,
):
    service = PredictionExplanationService(
        db_session
    )

    explanation = service.explain(
        fixture_id=TEST_FIXTURE_ID,
        limit=8,
    )

    assert explanation[
        "fixture_id"
    ] == TEST_FIXTURE_ID

    assert explanation[
        "predicted_result"
    ] in {
        "H",
        "D",
        "A",
    }

    assert isinstance(
        explanation["summary"],
        str,
    )

    assert explanation["summary"]

    factors = explanation[
        "factors"
    ]

    assert isinstance(
        factors,
        list,
    )

    assert 0 < len(factors) <= 8

    for factor in factors:
        assert "feature" in factor
        assert "label" in factor
        assert "feature_value" in factor
        assert "shap_value" in factor
        assert "importance_percent" in factor
        assert factor[
            "direction"
        ] in {
            "support",
            "against",
        }


def test_prediction_explanation_unknown_fixture(
    db_session,
):
    service = PredictionExplanationService(
        db_session
    )

    with pytest.raises(
        ValueError,
        match="не найден",
    ):
        service.explain(
            fixture_id=999999999,
            limit=8,
        )