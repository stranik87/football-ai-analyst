from pathlib import Path

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

RESULTS_PATH = (
    BASE_DIR
    / "data"
    / "reports"
    / "class_weights_results.csv"
)

TARGET_COLUMN = "result"

TRAIN_SIZE = 1297
VALIDATION_SIZE = 278

FEATURE_COUNT = 60

DRAW_WEIGHTS = [
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
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

    return (
        dataframe
        .sort_values("kickoff")
        .reset_index(drop=True)
    )


def get_feature_columns(dataframe):
    excluded = {
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
        if column not in excluded
    ]


def get_feature_ranking(
    train_dataframe,
    feature_columns,
):
    ranking_model = CatBoostClassifier(
        **PARAMETERS
    )

    ranking_model.fit(
        train_dataframe[feature_columns],
        train_dataframe[TARGET_COLUMN],
    )

    importance = (
        ranking_model.get_feature_importance()
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
    train_dataframe,
    validation_dataframe,
    test_dataframe,
    selected_features,
    draw_weight,
):
    model_parameters = {
        **PARAMETERS,
        "class_weights": [
            1.0,
            draw_weight,
            1.0,
        ],
    }

    model = CatBoostClassifier(
        **model_parameters
    )

    model.fit(
        train_dataframe[selected_features],
        train_dataframe[TARGET_COLUMN],
        eval_set=(
            validation_dataframe[
                selected_features
            ],
            validation_dataframe[
                TARGET_COLUMN
            ],
        ),
        early_stopping_rounds=30,
        use_best_model=True,
    )

    validation_predictions = (
        model.predict(
            validation_dataframe[
                selected_features
            ]
        )
        .reshape(-1)
    )

    validation_probabilities = (
        model.predict_proba(
            validation_dataframe[
                selected_features
            ]
        )
    )

    test_predictions = (
        model.predict(
            test_dataframe[
                selected_features
            ]
        )
        .reshape(-1)
    )

    test_probabilities = (
        model.predict_proba(
            test_dataframe[
                selected_features
            ]
        )
    )

    y_validation = (
        validation_dataframe[
            TARGET_COLUMN
        ]
    )

    y_test = (
        test_dataframe[
            TARGET_COLUMN
        ]
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

    draw_total = int(
        matrix[1].sum()
    )

    draw_correct = int(
        matrix[1, 1]
    )

    home_total = int(
        matrix[0].sum()
    )

    home_correct = int(
        matrix[0, 0]
    )

    away_total = int(
        matrix[2].sum()
    )

    away_correct = int(
        matrix[2, 2]
    )

    predicted_draws = int(
        (test_predictions == "D").sum()
    )

    return {
        "draw_weight": draw_weight,
        "validation_accuracy": validation_accuracy,
        "validation_log_loss": validation_log_loss,
        "test_accuracy": test_accuracy,
        "test_log_loss": test_log_loss,
        "predicted_draws": predicted_draws,
        "actual_draws": draw_total,
        "correct_draws": draw_correct,
        "draw_accuracy": (
            draw_correct / draw_total
            if draw_total
            else 0.0
        ),
        "home_accuracy": (
            home_correct / home_total
            if home_total
            else 0.0
        ),
        "away_accuracy": (
            away_correct / away_total
            if away_total
            else 0.0
        ),
        "best_iteration": (
            model.get_best_iteration()
        ),
    }


def main():
    print("=" * 100)
    print("ЭКСПЕРИМЕНТ: CLASS WEIGHTS ДЛЯ НИЧЬЕЙ")
    print("=" * 100)

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
        f"Всего матчей: {len(dataframe)}"
    )

    print(
        f"Всего признаков: "
        f"{len(feature_columns)}"
    )

    print(
        f"Train: {len(train_dataframe)}"
    )

    print(
        f"Validation: "
        f"{len(validation_dataframe)}"
    )

    print(
        f"Test: {len(test_dataframe)}"
    )

    print()
    print(
        "Получение рейтинга признаков..."
    )

    ranking = get_feature_ranking(
        train_dataframe,
        feature_columns,
    )

    selected_features = ranking[
        :FEATURE_COUNT
    ]

    print(
        f"Выбрано признаков: "
        f"{len(selected_features)}"
    )

    print()
    print(
        "Первые 60 признаков:"
    )

    for index, feature in enumerate(
        selected_features,
        start=1,
    ):
        print(
            f"{index:2}. {feature}"
        )

    results = []

    for draw_weight in DRAW_WEIGHTS:
        print()
        print("-" * 100)

        print(
            f"Тест веса D = "
            f"{draw_weight:.1f}"
        )

        result = evaluate_model(
            train_dataframe,
            validation_dataframe,
            test_dataframe,
            selected_features,
            draw_weight,
        )

        results.append(
            result
        )

        print(
            f"Validation Accuracy: "
            f"{result['validation_accuracy']:.2%}"
        )

        print(
            f"Validation Log Loss: "
            f"{result['validation_log_loss']:.4f}"
        )

        print(
            f"Test Accuracy: "
            f"{result['test_accuracy']:.2%}"
        )

        print(
            f"Test Log Loss: "
            f"{result['test_log_loss']:.4f}"
        )

        print(
            f"Предсказано ничьих: "
            f"{result['predicted_draws']}"
        )

        print(
            f"Правильных ничьих: "
            f"{result['correct_draws']} "
            f"из {result['actual_draws']}"
        )

        print(
            f"Draw Accuracy: "
            f"{result['draw_accuracy']:.2%}"
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

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dataframe.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print()
    print("=" * 100)
    print("СВОДНАЯ ТАБЛИЦА")
    print("=" * 100)

    display = results_dataframe.copy()

    for column in [
        "validation_accuracy",
        "test_accuracy",
        "draw_accuracy",
        "home_accuracy",
        "away_accuracy",
    ]:
        display[column] = display[
            column
        ].map(
            lambda value:
            f"{value:.2%}"
        )

    display[
        "validation_log_loss"
    ] = display[
        "validation_log_loss"
    ].map(
        lambda value:
        f"{value:.4f}"
    )

    display[
        "test_log_loss"
    ] = display[
        "test_log_loss"
    ].map(
        lambda value:
        f"{value:.4f}"
    )

    print(
        display.to_string(
            index=False
        )
    )

    best_accuracy = max(
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
    print("=" * 100)
    print("ЛУЧШИЙ TEST ACCURACY")
    print("=" * 100)

    print(
        f"Вес D: "
        f"{best_accuracy['draw_weight']:.1f}"
    )

    print(
        f"Accuracy: "
        f"{best_accuracy['test_accuracy']:.2%}"
    )

    print(
        f"Log Loss: "
        f"{best_accuracy['test_log_loss']:.4f}"
    )

    print()
    print("=" * 100)
    print("ЛУЧШИЙ TEST LOG LOSS")
    print("=" * 100)

    print(
        f"Вес D: "
        f"{best_log_loss['draw_weight']:.1f}"
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
    print("=" * 100)
    print("ЛУЧШИЙ DRAW ACCURACY")
    print("=" * 100)

    print(
        f"Вес D: "
        f"{best_draw['draw_weight']:.1f}"
    )

    print(
        f"Draw Accuracy: "
        f"{best_draw['draw_accuracy']:.2%}"
    )

    print(
        f"Правильных ничьих: "
        f"{best_draw['correct_draws']}/"
        f"{best_draw['actual_draws']}"
    )

    print()
    print(
        f"Результаты сохранены: "
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()
