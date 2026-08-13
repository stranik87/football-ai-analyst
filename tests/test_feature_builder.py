import pytest

from app.database.database import SessionLocal
from app.ml.feature_builder import FeatureBuilder


TEST_FIXTURE_ID = 1377
HOME_TEAM_ID = 81
AWAY_TEAM_ID = 92


@pytest.fixture
def db_session():
    session = SessionLocal()

    try:
        yield session
    finally:
        session.rollback()
        session.close()


def test_feature_builder_returns_features(
    db_session,
):
    builder = FeatureBuilder(
        db_session
    )

    features = builder.build(
        home_team_id=HOME_TEAM_ID,
        away_team_id=AWAY_TEAM_ID,
        fixture_id=TEST_FIXTURE_ID,
    )

    assert isinstance(
        features,
        dict,
    )

    assert features

    expected_features = {
        "home_points_per_match",
        "away_points_per_match",
        "home_average_goals_for",
        "away_average_goals_for",
        "home_average_goals_against",
        "away_average_goals_against",
        "home_rest_days",
        "away_rest_days",
    }

    assert expected_features.issubset(
        features.keys()
    )

    assert all(
        isinstance(
            value,
            (int, float),
        )
        for value in features.values()
    )


def test_feature_builder_does_not_use_future_match(
    db_session,
):
    builder = FeatureBuilder(
        db_session
    )

    features = builder.build(
        home_team_id=HOME_TEAM_ID,
        away_team_id=AWAY_TEAM_ID,
        fixture_id=TEST_FIXTURE_ID,
    )

    assert features[
        "home_points_per_match"
    ] == 0.0

    assert features[
        "away_points_per_match"
    ] == 0.0