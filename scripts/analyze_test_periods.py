from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss


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
    / "match_result_catboost_optimized.cbm"
)

FEATURES_PATH = (
    BASE_DIR
    / "data"
    / "models"
    / "match_result_features_optimized.joblib"
)

TARGET_COLUMN = "result"

TRAIN_SIZE = 1297
VALIDATION_SIZE = 278


def main() -> None:
    print("=" * 90)
    print("АНАЛИЗ FINAL TEST ПО ВРЕМЕННЫМ ПЕРИОДАМ")
    print("=" * 90)

    dataframe = pd.read_csv(
        DATASET_PATH
    )

    dataframe["kickoff"] = pd.to_datetime(
        dataframe["kickoff"]
    )

    dataframe = dataframe.sort_values(
        "kickoff"
    ).reset_index(drop=True)

    features = joblib.load(
        FEATURES_PATH
    )

    test_start = (
        TRAIN_SIZE
        + VALIDATION_SIZE
    )

    test_dataframe = dataframe.iloc[
        test_start:
    ].copy()

    print(
        f"\nВсего матчей: {len(dataframe)}"
    )

    print(
        f"Train: 0:{TRAIN_SIZE}"
    )

    print(
        f"Validation: "
        f"{TRAIN_SIZE}:"
        f"{TRAIN_SIZE + VALIDATION_SIZE}"
    )

    print(
        f"Test: {test_start}:"
        f"{len(dataframe)}"
    )

    print(
        f"Признаков модели: "
        f"{len(features)}"
    )

    model = CatBoostClassifier()

    model.load_model(
        MODEL_PATH
    )

    x_test = test_dataframe[
        features
    ]

    y_test = test_dataframe[
        TARGET_COLUMN
    ]

    predictions = (
        model.predict(x_test)
        .reshape(-1)
    )

    probabilities = (
        model.predict_proba(x_test)
    )

    test_dataframe["prediction"] = (
        predictions
    )

    test_dataframe["correct"] = (
        test_dataframe[
            "prediction"
        ]
        == test_dataframe[
            TARGET_COLUMN
        ]
    )

    test_dataframe["month"] = (
        test_dataframe[
            "kickoff"
        ]
        .dt.to_period("M")
        .astype(str)
    )

    print("\n" + "=" * 90)
    print("ОБЩИЙ TEST")
    print("=" * 90)

    print(
        f"Accuracy: "
        f"{accuracy_score(y_test, predictions):.4f}"
    )

    print(
        f"Log Loss: "
        f"{log_loss(y_test, probabilities, labels=model.classes_):.4f}"
    )

    print("\n" + "=" * 90)
    print("ПО МЕСЯЦАМ")
    print("=" * 90)

    rows = []

    for month, group in (
        test_dataframe
        .groupby("month")
    ):
        indices = group.index

        group_predictions = (
            predictions[
                test_dataframe.index.get_indexer(
                    indices
                )
            ]
        )

        group_probabilities = (
            probabilities[
                test_dataframe.index.get_indexer(
                    indices
                )
            ]
        )

        group_actual = group[
            TARGET_COLUMN
        ]

        rows.append(
            {
                "month": month,
                "matches": len(group),
                "correct": int(
                    (
                        group_predictions
                        == group_actual.values
                    ).sum()
                ),
                "accuracy": accuracy_score(
                    group_actual,
                    group_predictions,
                ),
                "log_loss": log_loss(
                    group_actual,
                    group_probabilities,
                    labels=model.classes_,
                ),
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result["accuracy"] = (
            result["accuracy"]
            .map(
                lambda value:
                f"{value:.2%}"
            )
        )

        result["log_loss"] = (
            result["log_loss"]
            .map(
                lambda value:
                f"{value:.4f}"
            )
        )

        print(
            result.to_string(
                index=False
            )
        )

    print("\n" + "=" * 90)
    print("РЕЗУЛЬТАТЫ H / D / A")
    print("=" * 90)

    class_rows = []

    for result_class in [
        "H",
        "D",
        "A",
    ]:
        mask = (
            test_dataframe[
                TARGET_COLUMN
            ]
            == result_class
        )

        count = int(mask.sum())

        if count == 0:
            continue

        correct = int(
            (
                test_dataframe.loc[
                    mask,
                    "prediction",
                ]
                == result_class
            ).sum()
        )

        class_rows.append(
            {
                "class": result_class,
                "matches": count,
                "correct": correct,
                "accuracy": (
                    correct / count
                ),
            }
        )

    class_result = pd.DataFrame(
        class_rows
    )

    if not class_result.empty:
        class_result["accuracy"] = (
            class_result["accuracy"]
            .map(
                lambda value:
                f"{value:.2%}"
            )
        )

        print(
            class_result.to_string(
                index=False
            )
        )

    print("\n" + "=" * 90)
    print("РАЗБИВКА TEST НА 3 ЧАСТИ")
    print("=" * 90)

    test_size = len(
        test_dataframe
    )

    part_size = test_size // 3

    for number in range(3):
        start = (
            number * part_size
        )

        if number == 2:
            end = test_size
        else:
            end = (
                (number + 1)
                * part_size
            )

        part = test_dataframe.iloc[
            start:end
        ]

        part_predictions = (
            predictions[start:end]
        )

        part_probabilities = (
            probabilities[start:end]
        )

        part_actual = part[
            TARGET_COLUMN
        ]

        print(
            f"\nЧасть {number + 1}: "
            f"{part['kickoff'].min()} "
            f"→ "
            f"{part['kickoff'].max()}"
        )

        print(
            f"Матчей: {len(part)}"
        )

        print(
            f"Accuracy: "
            f"{accuracy_score(part_actual, part_predictions):.2%}"
        )

        print(
            f"Log Loss: "
            f"{log_loss(part_actual, part_probabilities, labels=model.classes_):.4f}"
        )


if __name__ == "__main__":
    main()
