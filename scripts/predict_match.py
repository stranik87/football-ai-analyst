import argparse
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from loguru import logger

from app.database.session import SessionLocal
from app.ml.feature_builder import FeatureBuilder
from app.models.fixture import Fixture
from app.models.team import Team


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "data"
    / "models"
    / "match_result_catboost.cbm"
)

FEATURES_PATH = (
    BASE_DIR
    / "data"
    / "models"
    / "match_result_features.joblib"
)

RESULT_NAMES = {
    "H": "Победа хозяев",
    "D": "Ничья",
    "A": "Победа гостей",
}


def load_model() -> tuple[
    CatBoostClassifier,
    list[str],
]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {MODEL_PATH}"
        )

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Файл признаков не найден: {FEATURES_PATH}"
        )

    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    feature_columns = joblib.load(
        FEATURES_PATH
    )

    return model, feature_columns


def get_team_name(
    session,
    team_id: int,
) -> str:
    team = (
        session.query(Team)
        .filter(Team.id == team_id)
        .first()
    )

    if team is None:
        return f"Team ID {team_id}"

    return team.name


def build_feature_dataframe(
    session,
    fixture: Fixture,
    feature_columns: list[str],
) -> pd.DataFrame:
    feature_builder = FeatureBuilder(
        session=session
    )

    features = feature_builder.build(
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
        for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        logger.warning(
            "Отсутствующие признаки будут заполнены нулями: {}",
            len(missing_columns),
        )

        for column in missing_columns:
            dataframe[column] = 0

    dataframe = dataframe[
        feature_columns
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


def predict_match(
    fixture_id: int,
) -> None:
    model, feature_columns = load_model()

    session = SessionLocal()

    try:
        fixture = (
            session.query(Fixture)
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

        home_team_name = get_team_name(
            session=session,
            team_id=fixture.home_team_id,
        )

        away_team_name = get_team_name(
            session=session,
            team_id=fixture.away_team_id,
        )

        logger.info(
            "Матч: {} — {}",
            home_team_name,
            away_team_name,
        )

        logger.info(
            "Fixture ID: {}",
            fixture.id,
        )

        logger.info(
            "Дата матча: {}",
            fixture.kickoff,
        )

        dataframe = build_feature_dataframe(
            session=session,
            fixture=fixture,
            feature_columns=feature_columns,
        )

        probabilities = model.predict_proba(
            dataframe
        )[0]

        model_classes = list(
            model.classes_
        )

        result_probabilities = {
            result_class: float(probability)
            for result_class, probability in zip(
                model_classes,
                probabilities,
            )
        }

        sorted_probabilities = sorted(
            result_probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        logger.info("Вероятности:")

        for result_class, probability in sorted_probabilities:
            result_name = RESULT_NAMES.get(
                result_class,
                result_class,
            )

            logger.info(
                "{}: {:.2f}%",
                result_name,
                probability * 100,
            )

        best_result_class = sorted_probabilities[0][0]
        best_probability = sorted_probabilities[0][1]

        logger.success(
            "Прогноз: {} — {:.2f}%",
            RESULT_NAMES.get(
                best_result_class,
                best_result_class,
            ),
            best_probability * 100,
        )

        if (
            fixture.home_goals is not None
            and fixture.away_goals is not None
        ):
            logger.info(
                "Фактический счёт: {}:{}",
                fixture.home_goals,
                fixture.away_goals,
            )

    finally:
        session.close()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Прогноз результата футбольного матча."
    )

    parser.add_argument(
        "--fixture-id",
        type=int,
        required=True,
        help="ID матча в базе данных.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    try:
        predict_match(
            fixture_id=arguments.fixture_id
        )
    except Exception as error:
        logger.exception(
            "Ошибка прогноза: {}",
            error,
        )
        raise