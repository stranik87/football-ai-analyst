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

FEATURE_COUNTS = [
    30,
    40,
    50,
    60,
    79,
]

PARAMETERS = {
    "iterations": 150,
    "depth": 5,
    "learning_rate": 0.05,
    "l2_leaf_reg": 7,
    "random_strength": 0.5,
    "bagging_temperature": 1.0,
    "verbose": False,
    "random_seed": 42,
}


def load_dataset():
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

    return dataframe


def get_feature_columns(
    dataframe,
):
    excluded_columns = {
        "fixture_id",
        "api_id",
        "kickoff",
        "home_team",
        "away_team",
        "result",
        "home_goals",
        "away_goals",
    }

    return [
        column
        for column in dataframe.columns
        if column not in excluded_columns
    ]


def get_feature_ranking(
    train_dataframe,
    feature_columns,
):
    model = CatBoostClassifier(
        **PARAMETERS
    )

    x_train = train_dataframe[
        feature_columns
    ]

    y_train = train_dataframe[
        TARGET_COLUMN
    ]

    model.fit(
        x_train,
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

    return [
        feature
        for feature, _ in ranking
    ]


def evaluate_model(
    feature_count,
    selected_features,
    train_dataframe,
    validation_dataframe,
    test_dataframe,
):
    model = CatBoostClassifier(
        **PARAMETERS
    )

    x_train = train_dataframe[
        selected_features
    ]

    y_train = train_dataframe[
        TARGET_COLUMN
    ]

    x_validation = validation_dataframe[
        selected_features
    ]

    y_validation = validation_dataframe[
        TARGET_COLUMN
    ]

    x_test = test_dataframe[
        selected_features
    ]

    y_test = test_dataframe[
        TARGET_COLUMN
    ]

    model.fit(
        x_train,
        y_train,
        eval_set=(
            x_validation,
            y_validation,
        ),
        early_stopping_rounds=30,
        use_best_model=True,
    )

    validation_probabilities = (
        model.predict_proba(
            x_validation
        )
    )

    validation_predictions = (
        model.predict(
            x_validation
        )
        .reshape(-1)
    )

    test_probabilities = (
        model.predict_proba(
            x_test
        )
    )

    test_predictions = (
        model.predict(
            x_test
        )
        .reshape(-1)
    )

    validation_accuracy = (
        accuracy_score(
            y_validation,
            validation_predictions,
        )
    )

    validation_log_loss = (
        log_loss(
            y_validation,
            validation_probabilities,
            labels=model.classes_,
        )
    )

    test_accuracy = (
        accuracy_score(
            y_test,
            test_predictions,
        )
    )

    test_log_loss = (
        log_loss(
            y_test,
            test_probabilities,
            labels=model.classes_,
        )
    )

    matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=[
            "H",
            "D",
            "A",
        ],
    )

    draw_actual = int(
        matrix[1].sum()
    )

    draw_correct = int(
        matrix[1, 1]
    )

    home_actual = int(
        matrix[0].sum()
    )

    home_correct = int(
        matrix[0, 0]
    )

    away_actual = int(
        matrix[2].sum()
    )

    away_correct = int(
        matrix[2, 2]
    )

    best_iteration = (
        model.get_best_iteration()
    )

    return {
        "feature_count": feature_count,
        "validation_accuracy": (
            validation_accuracy
        ),
        "validation_log_loss": (
            validation_log_loss
        ),
        "test_accuracy": (
            test_accuracy
        ),
        "test_log_loss": (
            test_log_loss
        ),
        "draw_accuracy": (
            draw_correct / draw_actual
            if draw_actual
            else 0.0
        ),
        "draw_correct": (
            draw_correct
        ),
        "draw_total": (
            draw_actual
        ),
        "home_accuracy": (
            home_correct / home_actual
            if home_actual
            else 0.0
        ),
        "away_accuracy": (
            away_correct / away_actual
            if away_actual
            else 0.0
        ),
        "best_iteration": (
            best_iteration
        ),
    }


def main():
    print(
        "=" * 100
    )

    print(
        "СРАВНЕНИЕ НАБОРОВ ПРИЗНАКОВ"
    )

    print(
        "=" * 100
    )

    dataframe = load_dataset()

    feature_columns = (
        get_feature_columns(
            dataframe
        )
    )

    train_dataframe = dataframe.iloc[
        :TRAIN_SIZE
    ].copy()

    validation_dataframe = dataframe.iloc[
        TRAIN_SIZE:
        TRAIN_SIZE + VALIDATION_SIZE
    ].copy()

    test_dataframe = dataframe.iloc[
        TRAIN_SIZE + VALIDATION_SIZE:
    ].copy()

    print()
    print(
        f"Всего матчей: "
        f"{len(dataframe)}"
    )

    print(
        f"Всего признаков: "
        f"{len(feature_columns)}"
    )

    print(
        f"Train: "
        f"{len(train_dataframe)}"
    )

    print(
        f"Validation: "
        f"{len(validation_dataframe)}"
    )

    print(
        f"Test: "
        f"{len(test_dataframe)}"
    )

    print()
    print(
        "Получение рейтинга признаков..."
    )

    feature_ranking = (
        get_feature_ranking(
            train_dataframe,
            feature_columns,
        )
    )

    print(
        "Рейтинг признаков получен."
    )

    results = []

    for feature_count in FEATURE_COUNTS:
        if feature_count > len(
            feature_ranking
        ):
            continue

        selected_features = (
            feature_ranking[
                :feature_count
            ]
        )

        print()
        print(
            "-" * 100
        )

        print(
            f"Тест набора: "
            f"{feature_count} признаков"
        )

        result = evaluate_model(
            feature_count=feature_count,
            selected_features=selected_features,
            train_dataframe=train_dataframe,
            validation_dataframe=validation_dataframe,
            test_dataframe=test_dataframe,
        )

        results.append(
            result
        )

        print(
            f"Validation Accuracy: "
            f"{result['validation_accuracy']:.4f}"
        )

        print(
            f"Validation Log Loss: "
            f"{result['validation_log_loss']:.4f}"
        )

        print(
            f"Test Accuracy: "
            f"{result['test_accuracy']:.4f}"
        )

        print(
            f"Test Log Loss: "
            f"{result['test_log_loss']:.4f}"
        )

        print(
            f"Draw Accuracy: "
            f"{result['draw_accuracy']:.2%} "
            f"({result['draw_correct']}/"
            f"{result['draw_total']})"
        )

        print(
            f"Home Accuracy: "
            f"{result['home_accuracy']:.2%}"
        )

        print(
            f"Away Accuracy: "
            f"{result['away_accuracy']:.2%}"
        )

        print(
            f"Best iteration: "
            f"{result['best_iteration']}"
        )

    results_dataframe = pd.DataFrame(
        results
    )

    print()
    print(
        "=" * 100
    )

    print(
        "ИТОГОВОЕ СРАВНЕНИЕ"
    )

    print(
        "=" * 100
    )

    display_dataframe = (
        results_dataframe.copy()
    )

    display_dataframe[
        "validation_accuracy"
    ] = display_dataframe[
        "validation_accuracy"
    ].map(
        lambda value:
        f"{value:.2%}"
    )

    display_dataframe[
        "validation_log_loss"
    ] = display_dataframe[
        "validation_log_loss"
    ].map(
        lambda value:
        f"{value:.4f}"
    )

    display_dataframe[
        "test_accuracy"
    ] = display_dataframe[
        "test_accuracy"
    ].map(
        lambda value:
        f"{value:.2%}"
    )

    display_dataframe[
        "test_log_loss"
    ] = display_dataframe[
        "test_log_loss"
    ].map(
        lambda value:
        f"{value:.4f}"
    )

    display_dataframe[
        "draw_accuracy"
    ] = display_dataframe[
        "draw_accuracy"
    ].map(
        lambda value:
        f"{value:.2%}"
    )

    display_dataframe[
        "home_accuracy"
    ] = display_dataframe[
        "home_accuracy"
    ].map(
        lambda value:
        f"{value:.2%}"
    )

    display_dataframe[
        "away_accuracy"
    ] = display_dataframe[
        "away_accuracy"
    ].map(
        lambda value:
        f"{value:.2%}"
    )

    print(
        display_dataframe.to_string(
            index=False
        )
    )

    best_test = max(
        results,
        key=lambda item:
        item["test_accuracy"],
    )

    best_log_loss = min(
        results,
        key=lambda item:
        item["test_log_loss"],
    )

    best_draw = max(
        results,
        key=lambda item:
        item["draw_accuracy"],
    )

    print()
    print(
        "=" * 100
    )

    print(
        "ЛУЧШИЙ TEST ACCURACY"
    )

    print(
        "=" * 100
    )

    print(
        f"Признаков: "
        f"{best_test['feature_count']}"
    )

    print(
        f"Accuracy: "
        f"{best_test['test_accuracy']:.2%}"
    )

    print(
        f"Log Loss: "
        f"{best_test['test_log_loss']:.4f}"
    )

    print()
    print(
        "=" * 100
    )

    print(
        "ЛУЧШИЙ TEST LOG LOSS"
    )

    print(
        "=" * 100
    )

    print(
        f"Признаков: "
        f"{best_log_loss['feature_count']}"
    )

    print(
        f"Log Loss: "
        f"{best_log_loss['test_log_loss']:.4f}"
    )

    print(
        f"Accuracy: "
        f"{best_log_loss['test_accuracy']:.2%}"
    )

    print()
    print(
        "=" * 100
    )

    print(
        "ЛУЧШИЙ DRAW ACCURACY"
    )

    print(
        "=" * 100
    )

    print(
        f"Признаков: "
        f"{best_draw['feature_count']}"
    )

    print(
        f"Draw Accuracy: "
        f"{best_draw['draw_accuracy']:.2%}"
    )

    print(
        f"Ничьи: "
        f"{best_draw['draw_correct']}/"
        f"{best_draw['draw_total']}"
    )

    print()
    print(
        "=" * 100
    )

    print(
        "ТОП ПРИЗНАКОВ"
    )

    print(
        "=" * 100
    )

    for index, feature in enumerate(
        feature_ranking[:30],
        start=1,
    ):
        print(
            f"{index:2}. {feature}"
        )


if __name__ == "__main__":
    main()
