from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import train_test_split


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

TARGET_COLUMN = "result"

CLASS_ORDER = ["H", "D", "A"]

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

    feature_columns = joblib.load(
        FEATURES_PATH
    )

    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    return dataframe, model, feature_columns


def prepare_data(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    missing_columns = [
        column
        for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "В датасете отсутствуют признаки: "
            + ", ".join(missing_columns)
        )

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Нет целевой колонки: {TARGET_COLUMN}"
        )

    x = dataframe[feature_columns].copy()

    y = (
        dataframe[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    x = x.apply(
        pd.to_numeric,
        errors="coerce",
    )

    x = x.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    x = x.fillna(0)

    valid_mask = y.isin(CLASS_ORDER)

    x = x.loc[valid_mask].reset_index(
        drop=True
    )

    y = y.loc[valid_mask].reset_index(
        drop=True
    )

    if x.empty:
        raise ValueError(
            "После очистки данные отсутствуют."
        )

    return x, y


def evaluate_model() -> None:
    dataframe, model, feature_columns = (
        load_resources()
    )

    logger.info(
        "Загружена модель: {}",
        MODEL_PATH,
    )

    logger.info(
        "Количество признаков: {}",
        len(feature_columns),
    )

    x, y = prepare_data(
        dataframe,
        feature_columns,
    )

    _, x_test, _, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info(
        "Количество тестовых матчей: {}",
        len(x_test),
    )

    predictions = model.predict(
        x_test
    ).reshape(-1)

    probabilities = model.predict_proba(
        x_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    model_classes = list(
        model.classes_
    )

    loss = log_loss(
        y_test,
        probabilities,
        labels=model_classes,
    )

    logger.info(
        "Accuracy: {:.4f}",
        accuracy,
    )

    logger.info(
        "Log Loss: {:.4f}",
        loss,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=CLASS_ORDER,
        zero_division=0,
    )

    logger.info(
        "Classification report:\n{}",
        report,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=CLASS_ORDER,
    )

    matrix_dataframe = pd.DataFrame(
        matrix,
        index=[
            "actual_H",
            "actual_D",
            "actual_A",
        ],
        columns=[
            "predicted_H",
            "predicted_D",
            "predicted_A",
        ],
    )

    logger.info(
        "Confusion matrix:\n{}",
        matrix_dataframe.to_string(),
    )

    prediction_distribution = (
        pd.Series(predictions)
        .value_counts()
        .reindex(
            CLASS_ORDER,
            fill_value=0,
        )
    )

    logger.info(
        "Распределение прогнозов:\n{}",
        prediction_distribution.to_string(),
    )

    probability_dataframe = pd.DataFrame(
        probabilities,
        columns=model_classes,
    )

    logger.info(
        "Средние вероятности модели:\n{}",
        probability_dataframe.mean().to_string(),
    )

    logger.success(
        "Честная проверка качества завершена."
    )


if __name__ == "__main__":
    try:
        evaluate_model()
    except Exception as error:
        logger.exception(
            "Ошибка проверки модели: {}",
            error,
        )
        raise