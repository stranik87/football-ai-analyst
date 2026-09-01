from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix


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

THRESHOLDS = [
    1.00,
    0.95,
    0.90,
    0.85,
    0.80,
    0.75,
]


def load_test_data():
    dataframe = pd.read_csv(
        DATASET_PATH
    )

    dataframe["kickoff"] = pd.to_datetime(
        dataframe["kickoff"]
    )

    dataframe = (
        dataframe
        .sort_values("kickoff")
        .reset_index(drop=True)
    )

    features = joblib.load(
        FEATURES_PATH
    )

    test_start = (
        TRAIN_SIZE
        + VALIDATION_SIZE
    )

    test = dataframe.iloc[
        test_start:
    ].copy()

    return test, features


def apply_draw_threshold(
    probabilities,
    classes,
    threshold,
):
    class_index = {
        name: index
        for index, name
        in enumerate(classes)
    }

    p_home = probabilities[
        :, class_index["H"]
    ]

    p_draw = probabilities[
        :, class_index["D"]
    ]

    p_away = probabilities[
        :, class_index["A"]
    ]

    predictions = []

    for home, draw, away in zip(
        p_home,
        p_draw,
        p_away,
    ):
        best_non_draw = max(
            home,
            away,
        )

        if draw >= (
            best_non_draw
            * threshold
        ):
            predictions.append("D")
        elif home >= away:
            predictions.append("H")
        else:
            predictions.append("A")

    return predictions


def main():
    print("=" * 100)
    print("ЭКСПЕРИМЕНТ: ПОРОГ ДЛЯ НИЧЬЕЙ")
    print("=" * 100)

    test, features = load_test_data()

    model = CatBoostClassifier()

    model.load_model(
        MODEL_PATH
    )

    probabilities = model.predict_proba(
        test[features]
    )

    actual = test[
        TARGET_COLUMN
    ].values

    print(
        f"\nTest матчей: {len(test)}"
    )

    print(
        f"Признаков: {len(features)}"
    )

    results = []

    for threshold in THRESHOLDS:
        predictions = apply_draw_threshold(
            probabilities,
            model.classes_,
            threshold,
        )

        accuracy = accuracy_score(
            actual,
            predictions,
        )

        loss = log_loss(
            actual,
            probabilities,
            labels=model.classes_,
        )

        matrix = confusion_matrix(
            actual,
            predictions,
            labels=[
                "H",
                "D",
                "A",
            ],
        )

        correct_draws = matrix[
            1, 1
        ]

        predicted_draws = matrix[
            :, 1
        ].sum()

        actual_draws = matrix[
            1, :
        ].sum()

        results.append(
            {
                "threshold": threshold,
                "accuracy": accuracy,
                "log_loss": loss,
                "predicted_draws": int(
                    predicted_draws
                ),
                "actual_draws": int(
                    actual_draws
                ),
                "correct_draws": int(
                    correct_draws
                ),
            }
        )

        print()
        print("-" * 100)

        print(
            f"Порог D: "
            f"{threshold:.2f}"
        )

        print(
            f"Accuracy: "
            f"{accuracy:.4f}"
        )

        print(
            f"Log Loss: "
            f"{loss:.4f}"
        )

        print(
            f"Предсказано ничьих: "
            f"{predicted_draws}"
        )

        print(
            f"Правильных ничьих: "
            f"{correct_draws} "
            f"из {actual_draws}"
        )

        print()
        print(
            "Confusion Matrix:"
        )

        print(
            "        Pred H  Pred D  Pred A"
        )

        print(
            f"Fact H     {matrix[0,0]:3}     "
            f"{matrix[0,1]:3}     "
            f"{matrix[0,2]:3}"
        )

        print(
            f"Fact D     {matrix[1,0]:3}     "
            f"{matrix[1,1]:3}     "
            f"{matrix[1,2]:3}"
        )

        print(
            f"Fact A     {matrix[2,0]:3}     "
            f"{matrix[2,1]:3}     "
            f"{matrix[2,2]:3}"
        )

    result_dataframe = pd.DataFrame(
        results
    )

    print()
    print("=" * 100)
    print("СВОДНАЯ ТАБЛИЦА")
    print("=" * 100)

    display_dataframe = (
        result_dataframe.copy()
    )

    display_dataframe[
        "accuracy"
    ] = display_dataframe[
        "accuracy"
    ].map(
        lambda x: f"{x:.2%}"
    )

    display_dataframe[
        "log_loss"
    ] = display_dataframe[
        "log_loss"
    ].map(
        lambda x: f"{x:.4f}"
    )

    print(
        display_dataframe.to_string(
            index=False
        )
    )

    best_accuracy = max(
        results,
        key=lambda x: x[
            "accuracy"
        ],
    )

    best_draw = max(
        results,
        key=lambda x: x[
            "correct_draws"
        ],
    )

    print()
    print("=" * 100)
    print("ЛУЧШИЙ ACCURACY")
    print("=" * 100)

    print(
        f"Порог: "
        f"{best_accuracy['threshold']:.2f}"
    )

    print(
        f"Accuracy: "
        f"{best_accuracy['accuracy']:.2%}"
    )

    print()
    print("=" * 100)
    print("МАКСИМАЛЬНО ПРАВИЛЬНЫХ НИЧЬИХ")
    print("=" * 100)

    print(
        f"Порог: "
        f"{best_draw['threshold']:.2f}"
    )

    print(
        f"Правильных ничьих: "
        f"{best_draw['correct_draws']}"
    )


if __name__ == "__main__":
    main()
