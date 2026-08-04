from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from loguru import logger
from sklearn.metrics import accuracy_score, log_loss

from scripts.optimize_model import (
    CLASS_ORDER,
    DATASET_PATH,
    TARGET_COLUMN,
    load_dataset,
    prepare_dataframe,
    temporal_split,
)


BASE_DIR = Path(__file__).resolve().parents[1]

REPORT_DIR = BASE_DIR / "data" / "reports"
REPORT_PATH = REPORT_DIR / "catboost_model_comparison.csv"

RANDOM_SEED = 42


BASELINE_PARAMETERS = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3,
    "random_strength": 1.0,
    "bagging_temperature": 1.0,
}


OPTIMIZED_PARAMETERS = {
    "iterations": 300,
    "depth": 7,
    "learning_rate": 0.05,
    "l2_leaf_reg": 7,
    "random_strength": 1.0,
    "bagging_temperature": 1.0,
}


def create_model(parameters: dict) -> CatBoostClassifier:
    return CatBoostClassifier(
        **parameters,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def evaluate_model(
    name: str,
    parameters: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    logger.info(
        "Обучение модели: {}",
        name,
    )

    model = create_model(parameters)

    model.fit(
        x_train,
        y_train,
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

    loss = log_loss(
        y_test,
        probabilities,
        labels=list(model.classes_),
    )

    logger.info(
        "{} — Accuracy: {:.4f}",
        name,
        accuracy,
    )

    logger.info(
        "{} — Log Loss: {:.4f}",
        name,
        loss,
    )

    return {
        "model": name,
        **parameters,
        "test_accuracy": accuracy,
        "test_log_loss": loss,
    }


def compare_models() -> None:
    logger.info(
        "Загрузка датасета: {}",
        DATASET_PATH,
    )

    dataframe = load_dataset()

    (
        dataframe,
        feature_columns,
        date_column,
    ) = prepare_dataframe(dataframe)

    (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    ) = temporal_split(dataframe)

    combined_train_dataframe = pd.concat(
        [
            train_dataframe,
            validation_dataframe,
        ],
        ignore_index=True,
    )

    x_train = combined_train_dataframe[
        feature_columns
    ]

    y_train = combined_train_dataframe[
        TARGET_COLUMN
    ]

    x_test = test_dataframe[
        feature_columns
    ]

    y_test = test_dataframe[
        TARGET_COLUMN
    ]

    logger.info(
        "Обучение: {} матчей",
        len(x_train),
    )

    logger.info(
        "Финальный тест: {} матчей, период {} — {}",
        len(x_test),
        test_dataframe[date_column].min(),
        test_dataframe[date_column].max(),
    )

    baseline_result = evaluate_model(
        name="baseline",
        parameters=BASELINE_PARAMETERS,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )

    optimized_result = evaluate_model(
        name="optimized",
        parameters=OPTIMIZED_PARAMETERS,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )

    results = pd.DataFrame(
        [
            baseline_result,
            optimized_result,
        ]
    ).sort_values(
        by=[
            "test_log_loss",
            "test_accuracy",
        ],
        ascending=[
            True,
            False,
        ],
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        REPORT_PATH,
        index=False,
    )

    logger.info(
        "Сравнение моделей:\n{}",
        results.to_string(index=False),
    )

    baseline_accuracy = baseline_result[
        "test_accuracy"
    ]

    baseline_loss = baseline_result[
        "test_log_loss"
    ]

    optimized_accuracy = optimized_result[
        "test_accuracy"
    ]

    optimized_loss = optimized_result[
        "test_log_loss"
    ]

    if (
        optimized_loss < baseline_loss
        and optimized_accuracy >= baseline_accuracy
    ):
        logger.success(
            "Оптимизированная модель лучше baseline "
            "по Log Loss и не хуже по Accuracy."
        )
    else:
        logger.warning(
            "Оптимизированная модель не прошла критерий замены."
        )

    logger.success(
        "Отчёт сохранён: {}",
        REPORT_PATH,
    )


if __name__ == "__main__":
    try:
        compare_models()
    except Exception as error:
        logger.exception(
            "Ошибка сравнения моделей: {}",
            error,
        )
        raise