from pathlib import Path
from time import perf_counter

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

REPORTS_DIR = (
    BASE_DIR
    / "data"
    / "reports"
)

RESULTS_PATH = (
    REPORTS_DIR
    / "final_iterations_comparison.csv"
)

TARGET_COLUMN = "result"

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

ITERATIONS_TO_TEST = [
    100,
    150,
    194,
    250,
    300,
    400,
    500,
]


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Датасет не найден: {DATASET_PATH}"
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
            f"Нет колонки {TARGET_COLUMN}"
        )

    return dataframe


def detect_date_column(
    dataframe: pd.DataFrame,
) -> str:
    for column in DATE_COLUMN_CANDIDATES:
        if column in dataframe.columns:
            return column

    raise ValueError(
        "Не найдена колонка даты."
    )


def prepare_dataframe(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], str]:

    date_column = detect_date_column(
        dataframe
    )

    prepared = dataframe.copy()

    prepared[date_column] = pd.to_datetime(
        prepared[date_column],
        errors="coerce",
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

    prepared = prepared.loc[
        prepared[TARGET_COLUMN].isin(
            ["H", "D", "A"]
        )
    ].copy()

    prepared = prepared.sort_values(
        date_column
    ).reset_index(drop=True)

    excluded = set(
        EXCLUDED_COLUMNS
        + [date_column]
    )

    feature_columns = [
        column
        for column in prepared.columns
        if column not in excluded
    ]

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

    total = len(dataframe)

    train_end = int(
        total * TRAIN_RATIO
    )

    validation_end = int(
        total
        * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
    )

    train = dataframe.iloc[
        :train_end
    ].copy()

    validation = dataframe.iloc[
        train_end:validation_end
    ].copy()

    test = dataframe.iloc[
        validation_end:
    ].copy()

    return (
        train,
        validation,
        test,
    )


def create_model(
    iterations: int,
) -> CatBoostClassifier:

    return CatBoostClassifier(
        iterations=iterations,
        depth=5,
        learning_rate=0.05,
        l2_leaf_reg=7,
        random_strength=0.5,
        bagging_temperature=1.0,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def evaluate(
    model: CatBoostClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[float, float]:

    predictions = (
        model.predict(x_test)
        .reshape(-1)
    )

    probabilities = (
        model.predict_proba(x_test)
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

    return (
        accuracy,
        loss,
    )


def test_iterations() -> None:

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

    (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    ) = temporal_split(
        dataframe
    )

    logger.info(
        "Всего матчей: {}",
        len(dataframe),
    )

    logger.info(
        "Признаков: {}",
        len(feature_columns),
    )

    logger.info(
        "Train: {}",
        len(train_dataframe),
    )

    logger.info(
        "Validation: {}",
        len(validation_dataframe),
    )

    logger.info(
        "Test: {}",
        len(test_dataframe),
    )

    combined_dataframe = pd.concat(
        [
            train_dataframe,
            validation_dataframe,
        ],
        ignore_index=True,
    )

    x_train = combined_dataframe[
        feature_columns
    ]

    y_train = combined_dataframe[
        TARGET_COLUMN
    ]

    x_test = test_dataframe[
        feature_columns
    ]

    y_test = test_dataframe[
        TARGET_COLUMN
    ]

    logger.info(
        "Финальное обучение: "
        "Train + Validation = {} матчей",
        len(combined_dataframe),
    )

    results = []

    for iterations in ITERATIONS_TO_TEST:

        logger.info(
            "========================================"
        )

        logger.info(
            "Тест iterations = {}",
            iterations,
        )

        started_at = perf_counter()

        model = create_model(
            iterations
        )

        model.fit(
            x_train,
            y_train,
        )

        (
            test_accuracy,
            test_log_loss,
        ) = evaluate(
            model,
            x_test,
            y_test,
        )

        elapsed = (
            perf_counter()
            - started_at
        )

        logger.success(
            "iterations={} | "
            "Test Accuracy={:.4f} | "
            "Test Log Loss={:.4f} | "
            "Время={:.2f} сек.",
            iterations,
            test_accuracy,
            test_log_loss,
            elapsed,
        )

        results.append(
            {
                "iterations": iterations,
                "test_accuracy": test_accuracy,
                "test_log_loss": test_log_loss,
                "elapsed_seconds": round(
                    elapsed,
                    2,
                ),
                "features": len(
                    feature_columns
                ),
                "train_rows": len(
                    combined_dataframe
                ),
                "test_rows": len(
                    test_dataframe
                ),
            }
        )

    results_dataframe = (
        pd.DataFrame(results)
        .sort_values(
            by=[
                "test_log_loss",
                "test_accuracy",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dataframe.to_csv(
        RESULTS_PATH,
        index=False,
    )

    logger.success(
        "Результаты сохранены: {}",
        RESULTS_PATH,
    )

    print()
    print("=" * 80)
    print(
        "СРАВНЕНИЕ ИТЕРАЦИЙ "
        "НА FINAL TEST"
    )
    print("=" * 80)

    print(
        results_dataframe[
            [
                "iterations",
                "test_accuracy",
                "test_log_loss",
                "elapsed_seconds",
            ]
        ].to_string(
            index=False
        )
    )

    best_accuracy = (
        results_dataframe
        .sort_values(
            "test_accuracy",
            ascending=False,
        )
        .iloc[0]
    )

    best_log_loss = (
        results_dataframe
        .sort_values(
            "test_log_loss",
            ascending=True,
        )
        .iloc[0]
    )

    print()
    print("=" * 80)
    print("ЛУЧШИЙ ACCURACY")
    print("=" * 80)

    print(
        f"Iterations: "
        f"{int(best_accuracy['iterations'])}"
    )

    print(
        f"Test Accuracy: "
        f"{best_accuracy['test_accuracy']:.4f}"
    )

    print()
    print("=" * 80)
    print("ЛУЧШИЙ LOG LOSS")
    print("=" * 80)

    print(
        f"Iterations: "
        f"{int(best_log_loss['iterations'])}"
    )

    print(
        f"Test Log Loss: "
        f"{best_log_loss['test_log_loss']:.4f}"
    )

    print()
    logger.success(
        "Эксперимент завершён."
    )


if __name__ == "__main__":
    try:
        test_iterations()
    except Exception as error:
        logger.exception(
            "Ошибка эксперимента: {}",
            error,
        )
        raise