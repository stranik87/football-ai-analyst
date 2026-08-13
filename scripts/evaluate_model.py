import json
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

from app.database.session import SessionLocal
from app.models.fixture import Fixture
from app.models.league import League
from app.models.league_season import LeagueSeason
from scripts.optimize_model import (
    TARGET_COLUMN,
    load_dataset,
    prepare_dataframe,
    temporal_split,
)


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

REPORT_DIR = (
    BASE_DIR
    / "data"
    / "reports"
)

REPORT_PATH = (
    REPORT_DIR
    / "model_evaluation.json"
)

CLASS_ORDER = [
    "H",
    "D",
    "A",
]


def load_model_resources() -> tuple[
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

    feature_columns = joblib.load(
        FEATURES_PATH
    )

    model = CatBoostClassifier()
    model.load_model(
        MODEL_PATH
    )

    return (
        model,
        feature_columns,
    )


def prepare_model_features(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
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

    x = dataframe[
        feature_columns
    ].copy()

    x = x.apply(
        pd.to_numeric,
        errors="coerce",
    )

    x = x.replace(
        [
            float("inf"),
            float("-inf"),
        ],
        pd.NA,
    )

    x = x.fillna(0)

    return x


def get_fixture_leagues(
    fixture_ids: list[int],
) -> dict[int, str]:
    session = SessionLocal()

    try:
        rows = (
            session.query(
                Fixture.id,
                League.name,
            )
            .join(
                LeagueSeason,
                Fixture.league_season_id
                == LeagueSeason.id,
            )
            .join(
                League,
                LeagueSeason.league_id
                == League.id,
            )
            .filter(
                Fixture.id.in_(
                    fixture_ids
                )
            )
            .all()
        )

        return {
            fixture_id: league_name
            for fixture_id, league_name in rows
        }

    finally:
        session.close()


def calculate_classification_metrics(
    y_true: pd.Series,
    predictions,
) -> dict:
    report = classification_report(
        y_true,
        predictions,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )

    result = {}

    for class_name in CLASS_ORDER:
        class_report = report[
            class_name
        ]

        result[class_name] = {
            "precision": float(
                class_report["precision"]
            ),
            "recall": float(
                class_report["recall"]
            ),
            "f1_score": float(
                class_report["f1-score"]
            ),
            "support": int(
                class_report["support"]
            ),
        }

    return result


def calculate_confusion_matrix(
    y_true: pd.Series,
    predictions,
) -> dict:
    matrix = confusion_matrix(
        y_true,
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
        "Confusion Matrix:\n{}",
        matrix_dataframe.to_string(),
    )

    return {
        row_name: {
            column_name: int(
                matrix_dataframe.loc[
                    row_name,
                    column_name,
                ]
            )
            for column_name
            in matrix_dataframe.columns
        }
        for row_name
        in matrix_dataframe.index
    }


def calculate_baselines(
    train_dataframe: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    train_results = (
        train_dataframe[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    train_results = train_results[
        train_results.isin(
            CLASS_ORDER
        )
    ]

    most_frequent_class = (
        train_results
        .value_counts()
        .idxmax()
    )

    always_home_predictions = [
        "H"
        for _ in range(
            len(y_test)
        )
    ]

    most_frequent_predictions = [
        most_frequent_class
        for _ in range(
            len(y_test)
        )
    ]

    always_home_accuracy = (
        accuracy_score(
            y_test,
            always_home_predictions,
        )
    )

    most_frequent_accuracy = (
        accuracy_score(
            y_test,
            most_frequent_predictions,
        )
    )

    return {
        "always_home": {
            "class": "H",
            "accuracy": float(
                always_home_accuracy
            ),
        },
        "most_frequent_class": {
            "class": (
                most_frequent_class
            ),
            "accuracy": float(
                most_frequent_accuracy
            ),
        },
    }


def calculate_accuracy_by_league(
    test_dataframe: pd.DataFrame,
    y_test: pd.Series,
    predictions,
) -> dict:
    fixture_ids = (
        test_dataframe[
            "fixture_id"
        ]
        .astype(int)
        .tolist()
    )

    league_mapping = (
        get_fixture_leagues(
            fixture_ids
        )
    )

    evaluation_dataframe = (
        pd.DataFrame(
            {
                "fixture_id": fixture_ids,
                "actual": (
                    y_test
                    .reset_index(drop=True)
                ),
                "predicted": (
                    pd.Series(
                        predictions
                    )
                    .reset_index(drop=True)
                ),
            }
        )
    )

    evaluation_dataframe[
        "league"
    ] = (
        evaluation_dataframe[
            "fixture_id"
        ]
        .map(
            league_mapping
        )
        .fillna("Unknown")
    )

    results = {}

    for league_name, group in (
        evaluation_dataframe
        .groupby(
            "league"
        )
    ):
        accuracy = accuracy_score(
            group["actual"],
            group["predicted"],
        )

        results[
            str(league_name)
        ] = {
            "matches": int(
                len(group)
            ),
            "accuracy": float(
                accuracy
            ),
        }

    return results


def calculate_confidence_accuracy(
    y_test: pd.Series,
    predictions,
    probabilities,
) -> dict:
    probability_dataframe = (
        pd.DataFrame(
            probabilities
        )
    )

    confidence = (
        probability_dataframe
        .max(
            axis=1
        )
    )

    evaluation_dataframe = (
        pd.DataFrame(
            {
                "actual": (
                    y_test
                    .reset_index(drop=True)
                ),
                "predicted": (
                    pd.Series(
                        predictions
                    )
                    .reset_index(drop=True)
                ),
                "confidence": (
                    confidence
                    .reset_index(drop=True)
                ),
            }
        )
    )

    bins = [
        (
            "below_40",
            0.0,
            0.40,
        ),
        (
            "40_to_50",
            0.40,
            0.50,
        ),
        (
            "50_to_60",
            0.50,
            0.60,
        ),
        (
            "60_and_above",
            0.60,
            1.01,
        ),
    ]

    results = {}

    for (
        name,
        minimum,
        maximum,
    ) in bins:
        group = (
            evaluation_dataframe[
                (
                    evaluation_dataframe[
                        "confidence"
                    ]
                    >= minimum
                )
                & (
                    evaluation_dataframe[
                        "confidence"
                    ]
                    < maximum
                )
            ]
        )

        if group.empty:
            results[name] = {
                "matches": 0,
                "accuracy": None,
            }

            continue

        accuracy = accuracy_score(
            group["actual"],
            group["predicted"],
        )

        results[name] = {
            "matches": int(
                len(group)
            ),
            "accuracy": float(
                accuracy
            ),
        }

    return results


def evaluate_model() -> None:
    logger.info(
        "Загрузка датасета..."
    )

    dataframe = load_dataset()

    (
        prepared_dataframe,
        _,
        date_column,
    ) = prepare_dataframe(
        dataframe
    )

    (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    ) = temporal_split(
        prepared_dataframe
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

    (
        model,
        feature_columns,
    ) = load_model_resources()

    logger.info(
        "Модель: {}",
        MODEL_PATH,
    )

    logger.info(
        "Количество признаков модели: {}",
        len(feature_columns),
    )

    logger.info(
        "Train + validation: {} матчей",
        len(
            combined_train_dataframe
        ),
    )

    logger.info(
        "Финальный тест: {} матчей",
        len(
            test_dataframe
        ),
    )

    logger.info(
        "Период финального теста: {} — {}",
        test_dataframe[
            date_column
        ].min(),
        test_dataframe[
            date_column
        ].max(),
    )

    x_test = prepare_model_features(
        test_dataframe,
        feature_columns,
    )

    y_test = (
        test_dataframe[
            TARGET_COLUMN
        ]
        .astype(str)
        .str.strip()
        .str.upper()
        .reset_index(drop=True)
    )

    predictions = (
        model.predict(
            x_test
        )
        .reshape(-1)
    )

    probabilities = (
        model.predict_proba(
            x_test
        )
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

    classification_metrics = (
        calculate_classification_metrics(
            y_test,
            predictions,
        )
    )

    for (
        class_name,
        values,
    ) in classification_metrics.items():
        logger.info(
            "{} — Precision: {:.4f}, "
            "Recall: {:.4f}, "
            "F1: {:.4f}, "
            "Support: {}",
            class_name,
            values["precision"],
            values["recall"],
            values["f1_score"],
            values["support"],
        )

    confusion = (
        calculate_confusion_matrix(
            y_test,
            predictions,
        )
    )

    baselines = (
        calculate_baselines(
            combined_train_dataframe,
            y_test,
        )
    )

    logger.info(
        "Baseline always_home Accuracy: {:.4f}",
        baselines[
            "always_home"
        ][
            "accuracy"
        ],
    )

    logger.info(
        "Baseline most_frequent ({}) Accuracy: {:.4f}",
        baselines[
            "most_frequent_class"
        ][
            "class"
        ],
        baselines[
            "most_frequent_class"
        ][
            "accuracy"
        ],
    )

    accuracy_by_league = (
        calculate_accuracy_by_league(
            test_dataframe,
            y_test,
            predictions,
        )
    )

    logger.info(
        "Accuracy по лигам:"
    )

    for (
        league_name,
        values,
    ) in accuracy_by_league.items():
        logger.info(
            "{} — матчей: {}, Accuracy: {:.4f}",
            league_name,
            values["matches"],
            values["accuracy"],
        )

    confidence_accuracy = (
        calculate_confidence_accuracy(
            y_test,
            predictions,
            probabilities,
        )
    )

    logger.info(
        "Accuracy по уровням уверенности:"
    )

    for (
        group_name,
        values,
    ) in confidence_accuracy.items():
        if values[
            "accuracy"
        ] is None:
            logger.info(
                "{} — матчей: 0",
                group_name,
            )

            continue

        logger.info(
            "{} — матчей: {}, Accuracy: {:.4f}",
            group_name,
            values["matches"],
            values["accuracy"],
        )

    report = {
        "model": str(
            MODEL_PATH
        ),
        "features_count": int(
            len(
                feature_columns
            )
        ),
        "test": {
            "matches": int(
                len(
                    test_dataframe
                )
            ),
            "period_start": (
                test_dataframe[
                    date_column
                ]
                .min()
                .isoformat()
            ),
            "period_end": (
                test_dataframe[
                    date_column
                ]
                .max()
                .isoformat()
            ),
        },
        "metrics": {
            "accuracy": float(
                accuracy
            ),
            "log_loss": float(
                loss
            ),
        },
        "classification": (
            classification_metrics
        ),
        "confusion_matrix": (
            confusion
        ),
        "baselines": (
            baselines
        ),
        "accuracy_by_league": (
            accuracy_by_league
        ),
        "accuracy_by_confidence": (
            confidence_accuracy
        ),
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=4,
        )

    logger.success(
        "Отчёт сохранён: {}",
        REPORT_PATH,
    )

    logger.success(
        "Честная оценка модели завершена."
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