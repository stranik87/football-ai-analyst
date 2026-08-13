from fastapi.testclient import TestClient

from app.web_api import app


client = TestClient(app)


def test_dashboard_home():
    response = client.get(
        "/dashboard"
    )

    assert response.status_code == 200

    assert "Football AI Analyst" in response.text


def test_dashboard_prediction():
    response = client.get(
        "/dashboard/predict/1377"
    )

    assert response.status_code == 200

    assert "Прогноз" in response.text


def test_dashboard_model_evaluation():
    response = client.get(
        "/dashboard/model-evaluation"
    )

    assert response.status_code == 200

    assert "Оценка модели" in response.text


def test_dashboard_upcoming():
    response = client.get(
        "/dashboard/upcoming"
    )

    assert response.status_code == 200

    assert "Ближайшие матчи" in response.text