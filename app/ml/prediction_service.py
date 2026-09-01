from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sqlalchemy.orm import Session

from app.ml.feature_builder import FeatureBuilder
from app.models.fixture import Fixture


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


class PredictionService:
    VALID_CLASSES = (
        "A",
        "D",
        "H",
    )

    def __init__(
        self,
        session: Session,
    ) -> None:
        self.session = session

        self.feature_builder = FeatureBuilder(
            session
        )

        self.model = CatBoostClassifier()

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Файл модели не найден: "
                f"{MODEL_PATH}"
            )

        self.model.load_model(
            MODEL_PATH
        )

        if not FEATURES_PATH.exists():
            raise FileNotFoundError(
                "Файл признаков не найден: "
                f"{FEATURES_PATH}"
            )

        self.feature_columns = joblib.load(
            FEATURES_PATH
        )

        if not isinstance(
            self.feature_columns,
            list,
        ):
            raise TypeError(
                "Файл признаков должен содержать список."
            )

    def predict_fixture(
        self,
        fixture_id: int,
    ) -> dict:
        fixture = (
            self.session.query(Fixture)
            .filter(
                Fixture.id == fixture_id
            )
            .first()
        )

        if not fixture:
            raise ValueError(
                "Матч не найден: "
                f"fixture_id={fixture_id}"
            )

        if fixture.home_team_id is None:
            raise ValueError(
                "У матча отсутствует домашняя команда: "
                f"fixture_id={fixture_id}"
            )

        if fixture.away_team_id is None:
            raise ValueError(
                "У матча отсутствует гостевая команда: "
                f"fixture_id={fixture_id}"
            )

        features = self.feature_builder.build(
            home_team_id=fixture.home_team_id,
            away_team_id=fixture.away_team_id,
            fixture_id=fixture.id,
        )

        prediction = self.predict(
            features=features
        )

        return {
            "fixture_id": fixture.id,
            "home_team_id": fixture.home_team_id,
            "away_team_id": fixture.away_team_id,
            **prediction,
        }

    def predict(
        self,
        features: dict,
    ) -> dict:
        dataframe = pd.DataFrame(
            [features]
        )

        dataframe = dataframe.reindex(
            columns=self.feature_columns,
            fill_value=0.0,
        )

        dataframe = dataframe.apply(
            pd.to_numeric,
            errors="coerce",
        )

        dataframe = dataframe.replace(
            [float("inf"), float("-inf")],
            pd.NA,
        )

        dataframe = dataframe.fillna(
            0.0
        )

        probabilities = self.model.predict_proba(
            dataframe
        )[0]

        raw_prediction = self.model.predict(
            dataframe
        ).reshape(-1)[0]

        predicted_class = str(
            raw_prediction
        )

        class_names = [
            str(class_name)
            for class_name in self.model.classes_
        ]

        probability_by_class = {
            class_name: float(probability)
            for class_name, probability in zip(
                class_names,
                probabilities,
            )
        }

        for class_name in self.VALID_CLASSES:
            probability_by_class.setdefault(
                class_name,
                0.0,
            )

        return {
            "prediction": predicted_class,
            "home_win": round(
                probability_by_class["H"],
                4,
            ),
            "draw": round(
                probability_by_class["D"],
                4,
            ),
            "away_win": round(
                probability_by_class["A"],
                4,
            ),
        }