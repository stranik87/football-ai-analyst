import argparse
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from app.database.session import SessionLocal
from app.ml.feature_builder import FeatureBuilder
from app.models.fixture import Fixture
from app.models.team import Team


BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "datasets"
    / "matches_dataset.csv"
)

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

CLASS_ORDER = ["H", "D", "A"]

RESULT_NAMES = {
    "H": "Победа хозяев",
    "D": "Ничья",
    "A": "Победа гостей",
}

TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_resources() -> tuple[
    pd.DataFrame,
    CatBoostClassifier,
    list[str],
]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Датасет не найден: {DATASET_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {MODEL_PATH}"
        )

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Файл признаков не найден: {FEATURES_PATH}"
        )

    dataframe = pd.read_csv(DATASET_PATH)

    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    feature_columns = joblib.load(
        FEATURES_PATH
    )

    return dataframe, model, feature_columns


def get_test_fixture_ids(
    dataframe: pd.DataFrame,
) -> list[int]:
    required_columns = {
        "fixture_id",
        "result",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "В датасете отсутствуют колонки: "
            + ", ".join(sorted(missing_columns))
        )

    results = (
        dataframe["result"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    valid_mask = results.isin(CLASS_ORDER)

    valid_dataframe = (
        dataframe.loc[valid_mask]
        .reset_index(drop=True)
    )

    valid_results = (
        results.loc[valid_mask]
        .reset_index(drop=True)
    )

    train_indices, test_indices = train_test_split(
        valid_dataframe.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=valid_results,
    )

    del train_indices

    fixture_ids = (
        valid_dataframe.loc[
            test_indices,
            "fixture_id",
        ]
        .astype(int)
        .tolist()
    )

    return fixture_ids


def get_actual_result(
    fixture: Fixture,
) -> str:
    if fixture.home_goals is None:
        raise ValueError(
            "Отсутствуют голы хозяев."
        )

    if fixture.away_goals is None:
        raise ValueError(
            "Отсутствуют голы гостей."
        )

    if fixture.home_goals > fixture.away_goals:
        return "H"

    if fixture.home_goals < fixture.away_goals:
        return "A"

    return "D"


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


def build_features(
    feature_builder: FeatureBuilder,
    fixture: Fixture,
    feature_columns: list[str],
) -> pd.DataFrame:
    features = feature_builder.build(
        home_team_id=fixture.home_team_id,
        away_team_id=fixture.away_team_id,
        fixture_id=fixture.id,
    )

    dataframe = pd.DataFrame(
        [features]
    )

    missing_columns = [
        column
        for column in feature_columns
        if column not in dataframe.columns
    ]

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


def test_predictions(
    limit: int | None,
) -> None:
    dataframe, model, feature_columns = (
        load_resources()
    )

    fixture_ids = get_test_fixture_ids(
        dataframe
    )

    if limit is not None:
        fixture_ids = fixture_ids[:limit]

    logger.info(
        "Матчей для проверки: {}",
        len(fixture_ids),
    )

    session = SessionLocal()

    actual_results = []
    predicted_results = []

    correct_count = 0
    skipped_count = 0

    try:
        feature_builder = FeatureBuilder(
            session=session
        )

        for number, fixture_id in enumerate(
            fixture_ids,
            start=1,
        ):
            fixture = (
                session.query(Fixture)
                .filter(Fixture.id == fixture_id)
                .first()
            )

            if fixture is None:
                logger.warning(
                    "[{}/{}] Fixture ID {} не найден.",
                    number,
                    len(fixture_ids),
                    fixture_id,
                )

                skipped_count += 1
                continue

            if fixture.home_team_id is None:
                skipped_count += 1
                continue

            if fixture.away_team_id is None:
                skipped_count += 1
                continue

            try:
                actual_result = get_actual_result(
                    fixture
                )

                feature_dataframe = build_features(
                    feature_builder=feature_builder,
                    fixture=fixture,
                    feature_columns=feature_columns,
                )

                probabilities = model.predict_proba(
                    feature_dataframe
                )[0]

                model_classes = list(
                    model.classes_
                )

                best_index = int(
                    probabilities.argmax()
                )

                predicted_result = (
                    model_classes[best_index]
                )

                best_probability = float(
                    probabilities[best_index]
                )

                home_name = get_team_name(
                    session,
                    fixture.home_team_id,
                )

                away_name = get_team_name(
                    session,
                    fixture.away_team_id,
                )

                is_correct = (
                    predicted_result
                    == actual_result
                )

                if is_correct:
                    correct_count += 1

                status = (
                    "ВЕРНО"
                    if is_correct
                    else "ОШИБКА"
                )

                logger.info(
                    "[{}/{}] {} — {} | "
                    "прогноз: {} ({:.2f}%) | "
                    "факт: {} | {}",
                    number,
                    len(fixture_ids),
                    home_name,
                    away_name,
                    RESULT_NAMES[predicted_result],
                    best_probability * 100,
                    RESULT_NAMES[actual_result],
                    status,
                )

                actual_results.append(
                    actual_result
                )

                predicted_results.append(
                    predicted_result
                )

            except Exception as error:
                logger.warning(
                    "[{}/{}] Fixture ID {} пропущен: {}",
                    number,
                    len(fixture_ids),
                    fixture_id,
                    error,
                )

                skipped_count += 1

    finally:
        session.close()

    if not actual_results:
        raise ValueError(
            "Не удалось проверить ни одного матча."
        )

    accuracy = accuracy_score(
        actual_results,
        predicted_results,
    )

    logger.info(
        "Проверено матчей: {}",
        len(actual_results),
    )

    logger.info(
        "Правильных прогнозов: {}",
        correct_count,
    )

    logger.info(
        "Ошибочных прогнозов: {}",
        len(actual_results) - correct_count,
    )

    logger.info(
        "Пропущено матчей: {}",
        skipped_count,
    )

    logger.success(
        "Точность серии: {:.2f}%",
        accuracy * 100,
    )

    report = classification_report(
        actual_results,
        predicted_results,
        labels=CLASS_ORDER,
        target_names=[
            "Победа хозяев",
            "Ничья",
            "Победа гостей",
        ],
        zero_division=0,
    )

    logger.info(
        "Отчёт по классам:\n{}",
        report,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Проверка модели на серии матчей."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Количество матчей для проверки. "
            "Без параметра проверяются все тестовые матчи."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    try:
        test_predictions(
            limit=arguments.limit
        )
    except Exception as error:
        logger.exception(
            "Ошибка проверки прогнозов: {}",
            error,
        )
        raise