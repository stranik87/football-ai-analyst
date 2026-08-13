from fastapi.testclient import TestClient

from app.web_api import app


client = TestClient(app)

TEST_FIXTURE_ID = 1377


def test_prediction_api_returns_prediction():
    response = client.get(
        f"/predict/{TEST_FIXTURE_ID}"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["fixture_id"] == TEST_FIXTURE_ID

    assert data["predicted_result"] in {
        "H",
        "D",
        "A",
    }

    assert 0.0 <= data["confidence"] <= 1.0

    assert set(
        data["probabilities"].keys()
    ) == {
        "home_win",
        "draw",
        "away_win",
    }


def test_prediction_api_unknown_fixture():
    response = client.get(
        "/predict/999999999"
    )

    assert response.status_code == 404