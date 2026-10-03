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


STAT_MODEL_CONFIG = {
    "goals_total": "data/models/v12_goals_total_poisson.joblib",
    "goals_home": "data/models/v12_goals_home_poisson.joblib",
    "goals_away": "data/models/v12_goals_away_poisson.joblib",
    "goals_1h_total": "data/models/v12_goals_1h_total_poisson.joblib",
    "goals_1h_home": "data/models/v12_goals_1h_home_poisson.joblib",
    "goals_1h_away": "data/models/v12_goals_1h_away_poisson.joblib",
    "goals_2h_total": "data/models/v12_goals_2h_total_poisson.joblib",
    "goals_2h_home": "data/models/v12_goals_2h_home_poisson.joblib",
    "goals_2h_away": "data/models/v12_goals_2h_away_poisson.joblib",

    "corners_total": "data/models/v12_corners_total_catboost.joblib",
    "corners_home": "data/models/v12_corners_home_catboost.joblib",
    "corners_away": "data/models/v12_corners_away_catboost.joblib",

    "yellow_cards_total": "data/models/v12_yellow_cards_total_catboost.joblib",
    "yellow_cards_home": "data/models/v12_yellow_cards_home_catboost.joblib",
    "yellow_cards_away": "data/models/v12_yellow_cards_away_catboost.joblib",

    "shots_total": "data/models/v12_shots_total_catboost.joblib",
    "shots_home": "data/models/v12_shots_home_catboost.joblib",
    "shots_away": "data/models/v12_shots_away_catboost.joblib",

    "shots_on_target_total": "data/models/v12_shots_on_target_total_catboost.joblib",
    "shots_on_target_home": "data/models/v12_shots_on_target_home_catboost.joblib",
    "shots_on_target_away": "data/models/v12_shots_on_target_away_catboost.joblib",

    "offsides_total": "data/models/v12_offsides_total_catboost.joblib",
    "offsides_home": "data/models/v12_offsides_home_catboost.joblib",
    "offsides_away": "data/models/v12_offsides_away_catboost.joblib",

    "fouls_total": "data/models/v12_fouls_total_catboost.joblib",
    "fouls_home": "data/models/v12_fouls_home_catboost.joblib",
    "fouls_away": "data/models/v12_fouls_away_catboost.joblib",
}



class PredictionService:
    def __init__(self, session):
        self.session = session
        self.feature_builder = FeatureBuilder(session)

        self.model = self._load_model()
        self.feature_columns = self._load_feature_columns()

        self.stat_models = self._load_stat_models()

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

    def _load_stat_models(self) -> dict[str, dict[str, Any]]:
        models = {}

        for target, model_relative_path in STAT_MODEL_CONFIG.items():
            model_path = BASE_DIR / model_relative_path

            if not model_path.exists():
                raise FileNotFoundError(
                    f"READY-модель {target} не найдена: {model_path}"
                )

            bundle = joblib.load(model_path)

            if not isinstance(bundle, dict):
                raise TypeError(
                    f"Модель {target} должна быть сохранена как dict."
                )

            required_keys = {
                "model",
                "feature_columns",
                "target",
                "status",
            }

            missing_keys = required_keys - set(bundle.keys())

            if missing_keys:
                raise TypeError(
                    f"Модель {target} не содержит ключи: "
                    f"{sorted(missing_keys)}"
                )

            if bundle["status"] != "READY":
                raise ValueError(
                    f"Модель {target} имеет статус "
                    f"{bundle['status']!r}, ожидался READY."
                )

            feature_columns = bundle["feature_columns"]

            if not isinstance(feature_columns, list):
                raise TypeError(
                    f"feature_columns модели {target} должен быть списком."
                )

            config = {
                "model": bundle["model"],
                "feature_columns": feature_columns,
                "target": bundle["target"],
                "status": bundle["status"],
            }

            if "scaler" in bundle:
                config["scaler"] = bundle["scaler"]

            models[target] = config

        return models

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

    def _build_full_dataframe(
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

    def _build_dataframe(
        self,
        full_dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        dataframe = full_dataframe.copy()

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

        return dataframe

    def _build_stat_dataframe(
        self,
        full_dataframe: pd.DataFrame,
        feature_columns: list[str],
    ) -> pd.DataFrame:
        stat_dataframe = full_dataframe.copy()

        missing_columns = [
            column
            for column in feature_columns
            if column not in stat_dataframe.columns
        ]

        for column in missing_columns:
            stat_dataframe[column] = 0

        stat_dataframe = stat_dataframe[
            feature_columns
        ].copy()

        return stat_dataframe

    def _predict_statistics(
        self,
        full_dataframe: pd.DataFrame,
    ) -> dict[str, Any]:
        predictions = {}

        for target, config in self.stat_models.items():
            stat_dataframe = self._build_stat_dataframe(
                full_dataframe,
                config["feature_columns"],
            )

            # Poisson-модели v1.2 обучались на StandardScaler.
            # CatBoost-модели scaler не используют.
            model_input = stat_dataframe

            if config.get("scaler") is not None:
                model_input = config["scaler"].transform(
                    stat_dataframe
                )

            prediction = config["model"].predict(
                model_input
            )[0]

            predictions[target] = round(
                max(0.0, float(prediction)),
                3,
            )

        # Удобный объединённый блок для прогнозов голов.
        goal_targets = (
            "goals_total",
            "goals_home",
            "goals_away",
            "goals_1h_total",
            "goals_1h_home",
            "goals_1h_away",
            "goals_2h_total",
            "goals_2h_home",
            "goals_2h_away",
        )

        if all(target in predictions for target in goal_targets):
            expected_home_goals = predictions["goals_home"]
            expected_away_goals = predictions["goals_away"]

            predictions["goals"] = {
                "expected_home_goals": expected_home_goals,
                "expected_away_goals": expected_away_goals,
                "expected_total_goals": predictions[
                    "goals_total"
                ],
                "expected_1h_total_goals": predictions[
                    "goals_1h_total"
                ],
                "expected_1h_home_goals": predictions[
                    "goals_1h_home"
                ],
                "expected_1h_away_goals": predictions[
                    "goals_1h_away"
                ],
                "expected_2h_total_goals": predictions[
                    "goals_2h_total"
                ],
                "expected_2h_home_goals": predictions[
                    "goals_2h_home"
                ],
                "expected_2h_away_goals": predictions[
                    "goals_2h_away"
                ],
                "over_under": self._calculate_goal_probabilities(
                    expected_home_goals,
                    expected_away_goals,
                ),
            }

        return predictions

    @staticmethod
    def _calculate_goal_probabilities(
        expected_home_goals: float,
        expected_away_goals: float,
    ) -> dict[str, dict[str, float]]:
        import math

        expected_total = (
            expected_home_goals
            + expected_away_goals
        )

        probabilities = {}

        for line in (
            0.5,
            1.5,
            2.5,
            3.5,
            4.5,
        ):
            threshold = int(line)

            under_probability = 0.0

            for goals in range(threshold + 1):
                under_probability += (
                    math.exp(-expected_total)
                    * expected_total ** goals
                    / math.factorial(goals)
                )

            over_probability = 1.0 - under_probability

            probabilities[str(line)] = {
                "over": round(
                    max(0.0, min(1.0, over_probability)),
                    4,
                ),
                "under": round(
                    max(0.0, min(1.0, under_probability)),
                    4,
                ),
            }

        return probabilities

    def predict(
        self,
        fixture_id: int,
    ) -> dict[str, Any]:
        fixture = self._get_fixture(
            fixture_id
        )

        full_dataframe = self._build_full_dataframe(
            fixture
        )

        dataframe = self._build_dataframe(
            full_dataframe
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

        statistics_predictions = self._predict_statistics(
            full_dataframe
        )

        actual_score = None

        if (
            fixture.home_goals is not None
            and fixture.away_goals is not None
        ):
            actual_score = {
                "home_goals": fixture.home_goals,
                "away_goals": fixture.away_goals,
            }

        probabilities_compat = {
            "home_win": float(probability_map.get("H", 0.0)),
            "draw": float(probability_map.get("D", 0.0)),
            "away_win": float(probability_map.get("A", 0.0)),
        }

        return {
            "fixture_id": fixture.id,
            "home_team_id": fixture.home_team_id,
            "away_team_id": fixture.away_team_id,
            "home_team": self._get_team_name(
                fixture.home_team_id
            ),
            "away_team": self._get_team_name(
                fixture.away_team_id
            ),
            "kickoff": fixture.kickoff,
            "probabilities": probabilities_compat,
            "prediction": predicted_result,
            "prediction_name": RESULT_NAMES.get(
                predicted_result,
                predicted_result,
            ),
            "predicted_result": predicted_result,
            "predicted_result_name": RESULT_NAMES.get(
                predicted_result,
                predicted_result,
            ),
            "confidence": confidence,
            "actual_score": actual_score,
            "statistics_predictions": statistics_predictions,
        }
