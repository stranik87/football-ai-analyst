from typing import Any

import numpy as np
from catboost import Pool

from app.services.prediction_service import PredictionService


FEATURE_LABELS = {
    "points_per_match": "Форма и набранные очки",
    "average_goals_for": "Результативность атаки",
    "average_goals_against": "Надёжность защиты",
    "shots_on_goal": "Удары в створ",
    "ball_possession": "Владение мячом",
    "pass_accuracy": "Точность передач",
    "rest_days": "Количество дней отдыха",
    "home_points": "Домашние результаты хозяев",
    "away_points": "Гостевые результаты гостей",
    "home_goals": "Домашняя результативность",
    "away_goals": "Гостевая результативность",
    "possession_matches": "Матчи со статистикой владения",
    "average_possession": "Среднее владение мячом",
    "highest_possession": "Максимальное владение мячом",
    "possession_above_50_percentage": (
        "Матчи с владением выше 50%"
    ),
    "pass_matches": "Матчи со статистикой передач",
    "average_total_passes": "Среднее количество передач",
    "average_accurate_passes": "Точные передачи",
    "average_pass_accuracy": "Средняя точность передач",
    "above_85_accuracy_percentage": (
        "Матчи с точностью передач выше 85%"
    ),
    "shooting_matches": "Матчи со статистикой ударов",
    "average_total_shots": "Среднее количество ударов",
    "average_shots_on_goal": "Средние удары в створ",
    "shot_accuracy_percentage": "Точность ударов",
    "goal_conversion_percentage": "Реализация голевых моментов",
    "shots_per_goal": "Количество ударов на один гол",
    "goalkeeper_matches": "Матчи со статистикой вратаря",
    "average_saves": "Среднее количество сейвов",
    "average_goals_conceded_gk": "Голы, пропущенные вратарём",
    "save_percentage": "Процент отражённых ударов",
    "clean_sheet_percentage": "Процент сухих матчей",
}


class PredictionExplanationService:
    """
    Объяснение прогноза CatBoost через SHAP-значения.
    """

    RESULT_TO_INDEX = {
        "H": 0,
        "D": 1,
        "A": 2,
    }

    def __init__(self, session) -> None:
        self.session = session
        self.prediction_service = PredictionService(
            session=session
        )

    def explain(
        self,
        fixture_id: int,
        limit: int = 8,
    ) -> dict[str, Any]:
        """
        Получить главные факторы конкретного прогноза.
        """

        fixture = self.prediction_service._get_fixture(
            fixture_id
        )

        dataframe = (
            self.prediction_service._build_dataframe(
                fixture
            )
        )

        prediction = self.prediction_service.predict(
            fixture_id
        )

        pool = Pool(
            data=dataframe,
            feature_names=(
                self.prediction_service.feature_columns
            ),
        )

        shap_values = (
            self.prediction_service.model
            .get_feature_importance(
                pool,
                type="ShapValues",
            )
        )

        predicted_result = prediction[
            "predicted_result"
        ]

        class_index = self._get_class_index(
            predicted_result
        )

        feature_shap_values = self._extract_shap_values(
            shap_values=shap_values,
            class_index=class_index,
        )

        factors = self._build_factors(
            dataframe=dataframe,
            shap_values=feature_shap_values,
            predicted_result=predicted_result,
            limit=limit,
        )

        positive_factors = [
            factor
            for factor in factors
            if factor["direction"] == "support"
        ]

        negative_factors = [
            factor
            for factor in factors
            if factor["direction"] == "against"
        ]

        return {
            "fixture_id": fixture_id,
            "predicted_result": predicted_result,
            "predicted_result_name": prediction[
                "predicted_result_name"
            ],
            "summary": self._build_summary(
                prediction=prediction,
                positive_factors=positive_factors,
                negative_factors=negative_factors,
            ),
            "factors": factors,
            "supporting_factors": positive_factors,
            "opposing_factors": negative_factors,
        }

    def _get_class_index(
        self,
        predicted_result: str,
    ) -> int:
        """
        Найти индекс класса в модели.
        """

        classes = list(
            self.prediction_service.model.classes_
        )

        if predicted_result not in classes:
            raise ValueError(
                "Класс прогноза отсутствует в модели."
            )

        return classes.index(predicted_result)

    def _extract_shap_values(
        self,
        shap_values: Any,
        class_index: int,
    ) -> np.ndarray:
        """
        Извлечь SHAP конкретного класса.

        Последний элемент — базовое значение модели,
        поэтому он исключается.
        """

        values = np.asarray(
            shap_values,
            dtype=float,
        )

        if values.ndim == 3:
            return values[
                0,
                class_index,
                :-1,
            ]

        if values.ndim == 2:
            return values[
                0,
                :-1,
            ]

        raise ValueError(
            "Неизвестный формат SHAP-значений."
        )

    def _build_factors(
        self,
        dataframe,
        shap_values: np.ndarray,
        predicted_result: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        Построить список наиболее важных факторов.
        """

        feature_columns = (
            self.prediction_service.feature_columns
        )

        if len(feature_columns) != len(
            shap_values
        ):
            raise ValueError(
                "Количество признаков и SHAP-значений "
                "не совпадает."
            )

        raw_factors = []

        for index, feature_name in enumerate(
            feature_columns
        ):
            shap_value = float(
                shap_values[index]
            )

            feature_value = float(
                dataframe.iloc[0][feature_name]
            )

            raw_factors.append(
                {
                    "feature": feature_name,
                    "label": self._get_feature_label(
                        feature_name
                    ),
                    "team_side": self._get_team_side(
                        feature_name
                    ),
                    "feature_value": feature_value,
                    "shap_value": shap_value,
                    "absolute_importance": abs(
                        shap_value
                    ),
                    "direction": (
                        "support"
                        if shap_value > 0
                        else "against"
                    ),
                }
            )

        raw_factors.sort(
            key=lambda factor: factor[
                "absolute_importance"
            ],
            reverse=True,
        )

        selected = raw_factors[:limit]

        total_importance = sum(
            factor["absolute_importance"]
            for factor in selected
        )

        for factor in selected:
            if total_importance:
                importance_percent = (
                    factor["absolute_importance"]
                    / total_importance
                    * 100
                )
            else:
                importance_percent = 0.0

            factor["importance_percent"] = (
                importance_percent
            )

            factor["description"] = (
                self._build_description(
                    factor=factor,
                    predicted_result=(
                        predicted_result
                    ),
                )
            )

        return selected

    def _get_feature_label(
        self,
        feature_name: str,
    ) -> str:
        """
        Преобразовать техническое имя признака
        в понятное название.
        """

        normalized_name = feature_name

        if normalized_name.startswith(
            "home_"
        ):
            normalized_name = (
                normalized_name[5:]
            )

        elif normalized_name.startswith(
            "away_"
        ):
            normalized_name = (
                normalized_name[5:]
            )

        return FEATURE_LABELS.get(
            normalized_name,
            normalized_name.replace(
                "_",
                " ",
            ).capitalize(),
        )

    def _get_team_side(
        self,
        feature_name: str,
    ) -> str:
        if feature_name.startswith("home_"):
            return "home"

        if feature_name.startswith("away_"):
            return "away"

        return "neutral"

    def _build_description(
        self,
        factor: dict[str, Any],
        predicted_result: str,
    ) -> str:
        side_names = {
            "home": "хозяев",
            "away": "гостей",
            "neutral": "матча",
        }

        result_names = {
            "H": "победу хозяев",
            "D": "ничью",
            "A": "победу гостей",
        }

        side_name = side_names[
            factor["team_side"]
        ]

        result_name = result_names.get(
            predicted_result,
            "выбранный исход",
        )

        value = factor["feature_value"]

        if factor["direction"] == "support":
            return (
                f'Показатель «{factor["label"]}» '
                f"для {side_name} ({value:.2f}) "
                f"поддерживает {result_name}."
            )

        return (
            f'Показатель «{factor["label"]}» '
            f"для {side_name} ({value:.2f}) "
            f"снижает вероятность исхода "
            f"«{result_name}»."
        )

    def _build_summary(
        self,
        prediction: dict[str, Any],
        positive_factors: list[dict[str, Any]],
        negative_factors: list[dict[str, Any]],
    ) -> str:
        confidence_percent = (
            prediction["confidence"] * 100
        )

        if confidence_percent >= 65:
            confidence_text = "высокой"
        elif confidence_percent >= 50:
            confidence_text = "средней"
        else:
            confidence_text = "низкой"

        return (
            f'Модель прогнозирует исход '
            f'«{prediction["predicted_result_name"]}» '
            f"с {confidence_text} уверенностью "
            f"({confidence_percent:.1f}%). "
            f"Найдено {len(positive_factors)} "
            f"поддерживающих и "
            f"{len(negative_factors)} "
            f"сдерживающих факторов."
        )