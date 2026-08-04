from itertools import product
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from loguru import logger
from sklearn.metrics import accuracy_score, log_loss


BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "datasets"
    / "matches_dataset.csv"
)

MODEL_DIR = BASE_DIR / "data" / "models"

OPTIMIZED_MODEL_PATH = (
    MODEL_DIR
    / "match_result_catboost_optimized.cbm"
)

OPTIMIZED_FEATURES_PATH = (
    MODEL_DIR
    / "match_result_features_optimized.joblib"
)

RESULTS_DIR = BASE_DIR / "data" / "reports"

RESULTS_PATH = (
    RESULTS_DIR
    / "catboost_optimization_results.csv"
)

TARGET_COLUMN = "result"

CLASS_ORDER = ["H", "D", "A"]

EXCLUDED_COLUMNS = [
    "fixture_id",
    "home_team_id",
    "away_team_id",
    "home_goals",
    "away_goals",
    TARGET_COLUMN,
]

DATE_COLUMN_CANDIDATES = [
    "kickoff",
    "date",
    "fixture_date",
    "match_date",
]

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

RANDOM_SEED = 42


PARAMETER_GRID = {
    "iterations": [300, 500, 800],
    "depth": [5, 6, 7],
    "learning_rate": [0.03, 0.05],
    "l2_leaf_reg": [3, 7],
    "random_strength": [0.5, 1.0],
    "bagging_temperature": [0.5, 1.0],
}


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Датасет не найден: {DATASET_PATH}\n"
            "Сначала запусти: python -m scripts.export_dataset"
        )

    dataframe = pd.read_csv(DATASET_PATH)

    if dataframe.empty:
        raise ValueError("Датасет пустой.")

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"В датасете отсутствует колонка: {TARGET_COLUMN}"
        )

    return dataframe


def detect_date_column(dataframe: pd.DataFrame) -> str:
    for column in DATE_COLUMN_CANDIDATES:
        if column in dataframe.columns:
            return column

    raise ValueError(
        "Не найдена колонка даты матча. "
        "Ожидалась одна из колонок: "
        + ", ".join(DATE_COLUMN_CANDIDATES)
    )


def prepare_dataframe(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], str]:
    date_column = detect_date_column(dataframe)

    prepared = dataframe.copy()

    prepared[date_column] = pd.to_datetime(
        prepared[date_column],
        errors="coerce",
    )

    invalid_dates = int(
        prepared[date_column].isna().sum()
    )

    if invalid_dates:
        logger.warning(
            "Удалено строк с некорректной датой: {}",
            invalid_dates,
        )

        prepared = prepared.loc[
            prepared[date_column].notna()
        ].copy()

    prepared[TARGET_COLUMN] = (
        prepared[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    valid_result_mask = prepared[
        TARGET_COLUMN
    ].isin(CLASS_ORDER)

    invalid_results = int(
        (~valid_result_mask).sum()
    )

    if invalid_results:
        logger.warning(
            "Удалено строк с неизвестным результатом: {}",
            invalid_results,
        )

        prepared = prepared.loc[
            valid_result_mask
        ].copy()

    prepared = prepared.sort_values(
        by=date_column,
        ascending=True,
    ).reset_index(drop=True)

    excluded_columns = set(
        EXCLUDED_COLUMNS + [date_column]
    )

    feature_columns = [
        column
        for column in prepared.columns
        if column not in excluded_columns
    ]

    if not feature_columns:
        raise ValueError(
            "Не найдены признаки для обучения."
        )

    prepared[feature_columns] = (
        prepared[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace(
            [float("inf"), float("-inf")],
            pd.NA,
        )
        .fillna(0)
    )

    if prepared.empty:
        raise ValueError(
            "После очистки датасет пуст."
        )

    return prepared, feature_columns, date_column


def temporal_split(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    total_rows = len(dataframe)

    train_end = int(
        total_rows * TRAIN_RATIO
    )

    validation_end = int(
        total_rows
        * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
    )

    if train_end <= 0:
        raise ValueError(
            "Недостаточно данных для обучения."
        )

    if validation_end <= train_end:
        raise ValueError(
            "Недостаточно данных для валидации."
        )

    if validation_end >= total_rows:
        raise ValueError(
            "Недостаточно данных для финального теста."
        )

    train_dataframe = dataframe.iloc[
        :train_end
    ].copy()

    validation_dataframe = dataframe.iloc[
        train_end:validation_end
    ].copy()

    test_dataframe = dataframe.iloc[
        validation_end:
    ].copy()

    return (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    )


def generate_parameter_combinations() -> list[dict]:
    parameter_names = list(
        PARAMETER_GRID.keys()
    )

    parameter_values = [
        PARAMETER_GRID[name]
        for name in parameter_names
    ]

    combinations = []

    for values in product(*parameter_values):
        combinations.append(
            dict(
                zip(
                    parameter_names,
                    values,
                )
            )
        )

    return combinations


def create_model(
    parameters: dict,
) -> CatBoostClassifier:
    return CatBoostClassifier(
        **parameters,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def evaluate_predictions(
    model: CatBoostClassifier,
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[float, float]:
    predictions = model.predict(
        x
    ).reshape(-1)

    probabilities = model.predict_proba(
        x
    )

    accuracy = accuracy_score(
        y,
        predictions,
    )

    loss = log_loss(
        y,
        probabilities,
        labels=list(model.classes_),
    )

    return accuracy, loss


def optimize_model() -> None:
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

    logger.info(
        "Всего подготовлено матчей: {}",
        len(dataframe),
    )

    logger.info(
        "Количество признаков: {}",
        len(feature_columns),
    )

    logger.info(
        "Колонка даты: {}",
        date_column,
    )

    (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    ) = temporal_split(dataframe)

    logger.info(
        "Обучение: {} матчей, период {} — {}",
        len(train_dataframe),
        train_dataframe[date_column].min(),
        train_dataframe[date_column].max(),
    )

    logger.info(
        "Валидация: {} матчей, период {} — {}",
        len(validation_dataframe),
        validation_dataframe[date_column].min(),
        validation_dataframe[date_column].max(),
    )

    logger.info(
        "Финальный тест: {} матчей, период {} — {}",
        len(test_dataframe),
        test_dataframe[date_column].min(),
        test_dataframe[date_column].max(),
    )

    x_train = train_dataframe[
        feature_columns
    ]

    y_train = train_dataframe[
        TARGET_COLUMN
    ]

    x_validation = validation_dataframe[
        feature_columns
    ]

    y_validation = validation_dataframe[
        TARGET_COLUMN
    ]

    x_test = test_dataframe[
        feature_columns
    ]

    y_test = test_dataframe[
        TARGET_COLUMN
    ]

    combinations = (
        generate_parameter_combinations()
    )

    logger.info(
        "Количество комбинаций параметров: {}",
        len(combinations),
    )

    results = []

    best_parameters = None
    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    best_iteration = None

    for index, parameters in enumerate(
        combinations,
        start=1,
    ):
        logger.info(
            "[{}/{}] Параметры: {}",
            index,
            len(combinations),
            parameters,
        )

        started_at = perf_counter()

        model = create_model(parameters)

        model.fit(
            x_train,
            y_train,
            eval_set=(
                x_validation,
                y_validation,
            ),
            early_stopping_rounds=50,
            use_best_model=True,
        )

        (
            validation_accuracy,
            validation_loss,
        ) = evaluate_predictions(
            model,
            x_validation,
            y_validation,
        )

        elapsed_seconds = (
            perf_counter() - started_at
        )

        current_best_iteration = (
            model.get_best_iteration()
        )

        result = {
            **parameters,
            "best_iteration": (
                current_best_iteration
            ),
            "validation_accuracy": (
                validation_accuracy
            ),
            "validation_log_loss": (
                validation_loss
            ),
            "elapsed_seconds": (
                round(elapsed_seconds, 2)
            ),
        }

        results.append(result)

        logger.info(
            "Accuracy: {:.4f}, Log Loss: {:.4f}, время: {:.2f} сек.",
            validation_accuracy,
            validation_loss,
            elapsed_seconds,
        )

        is_better_loss = (
            validation_loss
            < best_validation_loss
        )

        is_same_loss_better_accuracy = (
            abs(
                validation_loss
                - best_validation_loss
            )
            < 1e-9
            and validation_accuracy
            > best_validation_accuracy
        )

        if (
            is_better_loss
            or is_same_loss_better_accuracy
        ):
            best_parameters = parameters.copy()
            best_validation_loss = (
                validation_loss
            )
            best_validation_accuracy = (
                validation_accuracy
            )
            best_iteration = (
                current_best_iteration
            )

            logger.success(
                "Найдена новая лучшая конфигурация."
            )

    if best_parameters is None:
        raise RuntimeError(
            "Не удалось выбрать лучшую модель."
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dataframe = pd.DataFrame(
        results
    ).sort_values(
        by=[
            "validation_log_loss",
            "validation_accuracy",
        ],
        ascending=[
            True,
            False,
        ],
    )

    results_dataframe.to_csv(
        RESULTS_PATH,
        index=False,
    )

    logger.success(
        "Результаты оптимизации сохранены: {}",
        RESULTS_PATH,
    )

    logger.info(
        "Лучшие параметры: {}",
        best_parameters,
    )

    logger.info(
        "Лучшая validation Accuracy: {:.4f}",
        best_validation_accuracy,
    )

    logger.info(
        "Лучший validation Log Loss: {:.4f}",
        best_validation_loss,
    )

    final_iterations = best_parameters["iterations"]

    final_parameters = (
        best_parameters.copy()
    )

    final_parameters[
        "iterations"
    ] = final_iterations

    combined_train_dataframe = pd.concat(
        [
            train_dataframe,
            validation_dataframe,
        ],
        ignore_index=True,
    )

    x_final_train = combined_train_dataframe[
        feature_columns
    ]

    y_final_train = combined_train_dataframe[
        TARGET_COLUMN
    ]

    logger.info(
        "Обучение финальной оптимизированной модели "
        "на train + validation: {} матчей",
        len(combined_train_dataframe),
    )

    final_model = create_model(
        final_parameters
    )

    final_model.fit(
        x_final_train,
        y_final_train,
    )

    (
        test_accuracy,
        test_loss,
    ) = evaluate_predictions(
        final_model,
        x_test,
        y_test,
    )

    logger.info(
        "Финальная тестовая Accuracy: {:.4f}",
        test_accuracy,
    )

    logger.info(
        "Финальный тестовый Log Loss: {:.4f}",
        test_loss,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_model.save_model(
        OPTIMIZED_MODEL_PATH
    )

    joblib.dump(
        feature_columns,
        OPTIMIZED_FEATURES_PATH,
    )

    logger.success(
        "Оптимизированная модель сохранена: {}",
        OPTIMIZED_MODEL_PATH,
    )

    logger.success(
        "Признаки оптимизированной модели сохранены: {}",
        OPTIMIZED_FEATURES_PATH,
    )

    logger.success(
        "Оптимизация CatBoost завершена."
    )


if __name__ == "__main__":
    try:
        optimize_model()
    except Exception as error:
        logger.exception(
            "Ошибка оптимизации CatBoost: {}",
            error,
        )
        raise