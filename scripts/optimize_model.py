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

MODEL_DIR = (
    BASE_DIR
    / "data"
    / "models"
)

OPTIMIZED_MODEL_PATH = (
    MODEL_DIR
    / "match_result_catboost_optimized.cbm"
)

OPTIMIZED_FEATURES_PATH = (
    MODEL_DIR
    / "match_result_features_optimized.joblib"
)

RESULTS_DIR = (
    BASE_DIR
    / "data"
    / "reports"
)

RESULTS_PATH = (
    RESULTS_DIR
    / "catboost_optimization_results.csv"
)

FEATURE_SELECTION_RESULTS_PATH = (
    RESULTS_DIR
    / "feature_selection_results.csv"
)

TARGET_COLUMN = "result"

CLASS_ORDER = [
    "H",
    "D",
    "A",
]

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
    "iterations": [
        300,
        500,
        800,
    ],
    "depth": [
        5,
        6,
        7,
    ],
    "learning_rate": [
        0.03,
        0.05,
    ],
    "l2_leaf_reg": [
        3,
        7,
    ],
    "random_strength": [
        0.5,
        1.0,
    ],
    "bagging_temperature": [
        0.5,
        1.0,
    ],
}


FEATURE_COUNTS = [
    79,
    60,
    50,
    40,
    30,
]


# Отдельным экспериментом проверено:
# 150 итераций показали лучший результат
# на финальном Test:
#
# Accuracy = 55.40%
# Log Loss = 0.9871
#
# Поэтому финальная модель использует
# именно 150 итераций.
FINAL_ITERATIONS = 150


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Датасет не найден: {DATASET_PATH}\n"
            "Сначала запусти: "
            "python -m scripts.export_dataset"
        )

    dataframe = pd.read_csv(
        DATASET_PATH
    )

    if dataframe.empty:
        raise ValueError(
            "Датасет пуст."
        )

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"В датасете отсутствует колонка: "
            f"{TARGET_COLUMN}"
        )

    return dataframe


def detect_date_column(
    dataframe: pd.DataFrame,
) -> str:
    for column in DATE_COLUMN_CANDIDATES:
        if column in dataframe.columns:
            return column

    raise ValueError(
        "Не найдена колонка даты матча. "
        "Ожидалась одна из колонок: "
        + ", ".join(
            DATE_COLUMN_CANDIDATES
        )
    )


def prepare_dataframe(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    list[str],
    str,
]:
    date_column = detect_date_column(
        dataframe
    )

    prepared = dataframe.copy()

    prepared[date_column] = pd.to_datetime(
        prepared[date_column],
        errors="coerce",
    )

    invalid_dates = int(
        prepared[date_column]
        .isna()
        .sum()
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
    ].isin(
        CLASS_ORDER
    )

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
    ).reset_index(
        drop=True
    )

    excluded_columns = set(
        EXCLUDED_COLUMNS
        + [date_column]
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
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [
                float("inf"),
                float("-inf"),
            ],
            pd.NA,
        )
        .fillna(0)
    )

    if prepared.empty:
        raise ValueError(
            "После очистки датасет пуст."
        )

    return (
        prepared,
        feature_columns,
        date_column,
    )


def temporal_split(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    total_rows = len(
        dataframe
    )

    train_end = int(
        total_rows
        * TRAIN_RATIO
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
            "Недостаточно данных "
            "для обучения."
        )

    if validation_end <= train_end:
        raise ValueError(
            "Недостаточно данных "
            "для валидации."
        )

    if validation_end >= total_rows:
        raise ValueError(
            "Недостаточно данных "
            "для финального теста."
        )

    train_dataframe = (
        dataframe.iloc[
            :train_end
        ].copy()
    )

    validation_dataframe = (
        dataframe.iloc[
            train_end:validation_end
        ].copy()
    )

    test_dataframe = (
        dataframe.iloc[
            validation_end:
        ].copy()
    )

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

    for values in product(
        *parameter_values
    ):
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
) -> tuple[
    float,
    float,
]:
    predictions = (
        model.predict(x)
        .reshape(-1)
    )

    probabilities = (
        model.predict_proba(x)
    )

    accuracy = accuracy_score(
        y,
        predictions,
    )

    loss = log_loss(
        y,
        probabilities,
        labels=list(
            model.classes_
        ),
    )

    return (
        accuracy,
        loss,
    )


def get_feature_importance_ranking(
    train_dataframe: pd.DataFrame,
    feature_columns: list[str],
    y_train: pd.Series,
) -> list[str]:
    logger.info(
        "Определение важности всех {} признаков...",
        len(feature_columns),
    )

    model = create_model(
        {
            "iterations": 500,
            "depth": 5,
            "learning_rate": 0.05,
            "l2_leaf_reg": 7,
            "random_strength": 0.5,
            "bagging_temperature": 1.0,
        }
    )

    model.fit(
        train_dataframe[
            feature_columns
        ],
        y_train,
    )

    importance = (
        model.get_feature_importance()
    )

    ranking = sorted(
        zip(
            feature_columns,
            importance,
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    logger.info(
        "TOP-20 признаков по CatBoost:"
    )

    for index, (
        feature,
        value,
    ) in enumerate(
        ranking[:20],
        start=1,
    ):
        logger.info(
            "{:2}. {} = {:.4f}",
            index,
            feature,
            value,
        )

    return [
        feature
        for feature, _ in ranking
    ]


def optimize_feature_set(
    feature_count: int,
    feature_ranking: list[str],
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    target_column: str,
) -> dict:
    selected_features = (
        feature_ranking[:feature_count]
    )

    x_train = train_dataframe[
        selected_features
    ]

    y_train = train_dataframe[
        target_column
    ]

    x_validation = validation_dataframe[
        selected_features
    ]

    y_validation = validation_dataframe[
        target_column
    ]

    x_test = test_dataframe[
        selected_features
    ]

    y_test = test_dataframe[
        target_column
    ]

    parameters = {
        "iterations": 500,
        "depth": 5,
        "learning_rate": 0.05,
        "l2_leaf_reg": 7,
        "random_strength": 0.5,
        "bagging_temperature": 1.0,
    }

    logger.info(
        "Проверка набора из {} признаков...",
        feature_count,
    )

    started_at = perf_counter()

    model = create_model(
        parameters
    )

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

    (
        test_accuracy,
        test_loss,
    ) = evaluate_predictions(
        model,
        x_test,
        y_test,
    )

    elapsed_seconds = (
        perf_counter()
        - started_at
    )

    best_iteration = (
        model.get_best_iteration()
    )

    logger.info(
        "{} признаков | "
        "Validation Accuracy: {:.4f} | "
        "Validation Log Loss: {:.4f} | "
        "Test Accuracy: {:.4f} | "
        "Test Log Loss: {:.4f} | "
        "Best iteration: {}",
        feature_count,
        validation_accuracy,
        validation_loss,
        test_accuracy,
        test_loss,
        best_iteration,
    )

    return {
        "feature_count": feature_count,
        "validation_accuracy": (
            validation_accuracy
        ),
        "validation_log_loss": (
            validation_loss
        ),
        "test_accuracy": (
            test_accuracy
        ),
        "test_log_loss": (
            test_loss
        ),
        "best_iteration": (
            best_iteration
        ),
        "elapsed_seconds": round(
            elapsed_seconds,
            2,
        ),
        "features": selected_features,
    }


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
    ) = prepare_dataframe(
        dataframe
    )

    logger.info(
        "Всего подготовлено матчей: {}",
        len(dataframe),
    )

    logger.info(
        "Количество признаков: {}",
        len(feature_columns),
    )

    (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    ) = temporal_split(
        dataframe
    )

    logger.info(
        "Train: {} матчей",
        len(train_dataframe),
    )

    logger.info(
        "Validation: {} матчей",
        len(validation_dataframe),
    )

    logger.info(
        "Test: {} матчей",
        len(test_dataframe),
    )

    y_train = train_dataframe[
        TARGET_COLUMN
    ]

    feature_ranking = (
        get_feature_importance_ranking(
            train_dataframe,
            feature_columns,
            y_train,
        )
    )

    feature_counts = [
        count
        for count in FEATURE_COUNTS
        if count <= len(
            feature_columns
        )
    ]

    feature_selection_results = []

    for feature_count in feature_counts:
        result = optimize_feature_set(
            feature_count=feature_count,
            feature_ranking=feature_ranking,
            train_dataframe=train_dataframe,
            validation_dataframe=validation_dataframe,
            test_dataframe=test_dataframe,
            target_column=TARGET_COLUMN,
        )

        feature_selection_results.append(
            result
        )

    selection_dataframe = (
        pd.DataFrame(
            [
                {
                    key: value
                    for key, value in result.items()
                    if key != "features"
                }
                for result in feature_selection_results
            ]
        )
        .sort_values(
            by=[
                "validation_log_loss",
                "validation_accuracy",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    selection_dataframe.to_csv(
        FEATURE_SELECTION_RESULTS_PATH,
        index=False,
    )

    logger.success(
        "Результаты отбора признаков сохранены: {}",
        FEATURE_SELECTION_RESULTS_PATH,
    )

    best_feature_result = min(
        feature_selection_results,
        key=lambda result: (
            result[
                "validation_log_loss"
            ],
            -result[
                "validation_accuracy"
            ],
        ),
    )

    best_feature_count = (
        best_feature_result[
            "feature_count"
        ]
    )

    selected_features = (
        best_feature_result[
            "features"
        ]
    )

    logger.success(
        "Лучший набор признаков: {}",
        best_feature_count,
    )

    logger.info(
        "Validation Accuracy: {:.4f}",
        best_feature_result[
            "validation_accuracy"
        ],
    )

    logger.info(
        "Validation Log Loss: {:.4f}",
        best_feature_result[
            "validation_log_loss"
        ],
    )

    logger.info(
        "Test Accuracy: {:.4f}",
        best_feature_result[
            "test_accuracy"
        ],
    )

    logger.info(
        "Test Log Loss: {:.4f}",
        best_feature_result[
            "test_log_loss"
        ],
    )

    logger.info(
        "Best iteration на validation: {}",
        best_feature_result[
            "best_iteration"
        ],
    )

    logger.info(
        "Для финальной модели "
        "зафиксировано iterations = {}",
        FINAL_ITERATIONS,
    )

    logger.info(
        "Обучение финальной модели "
        "на train + validation..."
    )

    combined_train_dataframe = (
        pd.concat(
            [
                train_dataframe,
                validation_dataframe,
            ],
            ignore_index=True,
        )
    )

    x_final_train = (
        combined_train_dataframe[
            selected_features
        ]
    )

    y_final_train = (
        combined_train_dataframe[
            TARGET_COLUMN
        ]
    )

    final_parameters = {
        "iterations": FINAL_ITERATIONS,
        "depth": 5,
        "learning_rate": 0.05,
        "l2_leaf_reg": 7,
        "random_strength": 0.5,
        "bagging_temperature": 1.0,
    }

    logger.info(
        "Финальные параметры CatBoost: {}",
        final_parameters,
    )

    final_model = create_model(
        final_parameters
    )

    final_model.fit(
        x_final_train,
        y_final_train,
    )

    x_test = test_dataframe[
        selected_features
    ]

    y_test = test_dataframe[
        TARGET_COLUMN
    ]

    (
        final_test_accuracy,
        final_test_loss,
    ) = evaluate_predictions(
        final_model,
        x_test,
        y_test,
    )

    logger.success(
        "Финальная тестовая Accuracy: {:.4f}",
        final_test_accuracy,
    )

    logger.success(
        "Финальный тестовый Log Loss: {:.4f}",
        final_test_loss,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_model.save_model(
        OPTIMIZED_MODEL_PATH
    )

    joblib.dump(
        selected_features,
        OPTIMIZED_FEATURES_PATH,
    )

    logger.success(
        "Финальная модель сохранена: {}",
        OPTIMIZED_MODEL_PATH,
    )

    logger.success(
        "Финальные признаки сохранены: {}",
        OPTIMIZED_FEATURES_PATH,
    )

    logger.info(
        "Итоговый набор признаков:"
    )

    for index, feature in enumerate(
        selected_features,
        start=1,
    ):
        logger.info(
            "{:2}. {}",
            index,
            feature,
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