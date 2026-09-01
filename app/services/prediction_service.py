from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from catboost import CatBoostClassifier

from app.ml.feature_builder import FeatureBuilder
from app.models.fixture import Fixture
from app.models.team import Team


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "data"
    / "models"
    / "match_result_catboost_optimized.cbm"
)

FEATURES_PATH = (
    BASE_DIR
    / "data"
    / "models"
    / "match_result_features_optimized.joblib"
)

RESULT_NAMES = {
    "H": "Победа хозяев",
    "D": "Ничья",
    "A": "Победа гостей",
}


class PredictionService:
    def __init__(self, session) -> None:
        self.session = session

        self.model = self._load_model()
        self.feature_columns = self._load_feature_columns()

        self.feature_builder = FeatureBuilder(
            session=session
        )

    def _load_model(self) -> CatBoostClassifier:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Модель не найдена: {MODEL_PATH}"
            )

        model = CatBoostClassifier()
        model.load_model(MODEL_PATH)

        return model

    def _load_feature_columns(self) -> list[str]:
        if not FEATURES_PATH.exists():
            raise FileNotFoundError(
                f"Файл признаков не найден: {FEATURES_PATH}"
            )

        feature_columns = joblib.load(
            FEATURES_PATH
        )

        if not isinstance(feature_columns, list):
            raise TypeError(
                "Файл признаков должен содержать список."
            )

        return feature_columns

    def _get_fixture(
        self,
        fixture_id: int,
    ) -> Fixture:
        fixture = (
            self.session.query(Fixture)
            .filter(Fixture.id == fixture_id)
            .first()
        )

        if fixture is None:
            raise ValueError(
                f"Матч с ID {fixture_id} не найден."
            )

        if fixture.home_team_id is None:
            raise ValueError(
                "У матча отсутствует команда хозяев."
            )

        if fixture.away_team_id is None:
            raise ValueError(
                "У матча отсутствует команда гостей."
            )

        return fixture

    def _get_team_name(
        self,
        team_id: int,
    ) -> str:
        team = (
            self.session.query(Team)
            .filter(Team.id == team_id)
            .first()
        )

        if team is None:
            return f"Team ID {team_id}"

        return team.name

    def _build_dataframe(
        self,
        fixture: Fixture,
    ) -> pd.DataFrame:
        features = self.feature_builder.build(
            home_team_id=fixture.home_team_id,
            away_team_id=fixture.away_team_id,
            fixture_id=fixture.id,
        )

        if not features:
            raise ValueError(
                "Не удалось построить признаки матча."
            )

        dataframe = pd.DataFrame(
            [features]
        )

        missing_columns = [
            column
            for column in self.feature_columns
            if column not in dataframe.columns
        ]

        for column in missing_columns:
            dataframe[column] = 0

        dataframe = dataframe[
            self.feature_columns
        ].copy()

        dataframe = dataframe.apply(
            pd.to_numeric,
            errors="coerce",
        )

        dataframe = dataframe.replace(
            [float("inf"), float("-inf")],
            pd.NA,
        )

        dataframe = dataframe.fillna(0)

        return dataframe

    def predict(
        self,
        fixture_id: int,
    ) -> dict[str, Any]:
        fixture = self._get_fixture(
            fixture_id
        )

        dataframe = self._build_dataframe(
            fixture
        )

        probabilities = self.model.predict_proba(
            dataframe
        )[0]

        model_classes = list(
            self.model.classes_
        )

        probability_map = {
            result_class: float(probability)
            for result_class, probability in zip(
                model_classes,
                probabilities,
            )
        }

        sorted_probabilities = sorted(
            probability_map.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        predicted_result = sorted_probabilities[0][0]
        confidence = sorted_probabilities[0][1]

        home_team_name = self._get_team_name(
            fixture.home_team_id
        )

        away_team_name = self._get_team_name(
            fixture.away_team_id
        )

        return {
            "fixture_id": fixture.id,
            "kickoff": fixture.kickoff,
            "home_team_id": fixture.home_team_id,
            "away_team_id": fixture.away_team_id,
            "home_team": home_team_name,
            "away_team": away_team_name,
            "predicted_result": predicted_result,
            "predicted_result_name": RESULT_NAMES.get(
                predicted_result,
                predicted_result,
            ),
            "confidence": confidence,
            "probabilities": {
                "home_win": probability_map.get(
                    "H",
                    0.0,
                ),
                "draw": probability_map.get(
                    "D",
                    0.0,
                ),
                "away_win": probability_map.get(
                    "A",
                    0.0,
                ),
            },
            "actual_score": {
                "home_goals": fixture.home_goals,
                "away_goals": fixture.away_goals,
            },
        }