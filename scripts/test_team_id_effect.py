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

TARGET_COLUMN = "result"

TRAIN_SIZE = 1297
VALIDATION_SIZE = 278

FEATURE_COUNTS = [
    50,
    60,
]

BASE_PARAMETERS = {
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


def get_features(
    dataframe,
    include_team_ids,
):
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

    if not include_team_ids:
        excluded.update(
            {
                "home_team_id",
                "away_team_id",
            }
        )

    return [
        column
        for column in dataframe.columns
        if column not in excluded
    ]


def get_feature_ranking(
    train_dataframe,
    feature_columns,
):
    model = CatBoostClassifier(
        **BASE_PARAMETERS
    )

    model.fit(
        train_dataframe[feature_columns],
        train_dataframe[TARGET_COLUMN],
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


def evaluate(
    train_dataframe,
    validation_dataframe,
    test_dataframe,
    features,
):
    model = CatBoostClassifier(
        **BASE_PARAMETERS
    )

    model.fit(
        train_dataframe[features],
        train_dataframe[TARGET_COLUMN],
        eval_set=(
            validation_dataframe[features],
            validation_dataframe[TARGET_COLUMN],
        ),
        early_stopping_rounds=30,
        use_best_model=True,
    )

    validation_probability = (
        model.predict_proba(
            validation_dataframe[features]
        )
    )

    validation_prediction = (
        model.predict(
            validation_dataframe[features]
        )
        .reshape(-1)
    )

    test_probability = (
        model.predict_proba(
            test_dataframe[features]
        )
    )

    test_prediction = (
        model.predict(
            test_dataframe[features]
        )
        .reshape(-1)
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
            validation_prediction,
        )
    )

    validation_loss = (
        log_loss(
            y_validation,
            validation_probability,
            labels=model.classes_,
        )
    )

    test_accuracy = (
        accuracy_score(
            y_test,
            test_prediction,
        )
    )

    test_loss = (
        log_loss(
            y_test,
            test_probability,
            labels=model.classes_,
        )
    )

    matrix = confusion_matrix(
        y_test,
        test_prediction,
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

    return {
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
        "draw_accuracy": (
            draw_correct / draw_total
            if draw_total
            else 0.0
        ),
        "draw_correct": (
            draw_correct
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
    print("ТЕСТ ВЛИЯНИЯ TEAM ID")
    print("=" * 100)

    dataframe = load_dataset()

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
        f"Train: {len(train_dataframe)}"
    )

    print(
        f"Validation: "
        f"{len(validation_dataframe)}"
    )

    print(
        f"Test: {len(test_dataframe)}"
    )

    all_results = []

    for include_team_ids in [
        False,
        True,
    ]:
        mode_name = (
            "С TEAM ID"
            if include_team_ids
            else "БЕЗ TEAM ID"
        )

        print()
        print("=" * 100)
        print(mode_name)
        print("=" * 100)

        feature_columns = get_features(
            dataframe,
            include_team_ids,
        )

        print(
            f"Доступно признаков: "
            f"{len(feature_columns)}"
        )

        print(
            "Получение ranking..."
        )

        ranking = get_feature_ranking(
            train_dataframe,
            feature_columns,
        )

        print(
            "Ranking готов."
        )

        for feature_count in FEATURE_COUNTS:
            selected_features = ranking[
                :feature_count
            ]

            print()
            print(
                "-" * 100
            )

            print(
                f"{mode_name} | "
                f"{feature_count} признаков"
            )

            result = evaluate(
                train_dataframe,
                validation_dataframe,
                test_dataframe,
                selected_features,
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
                f"Draw Accuracy: "
                f"{result['draw_accuracy']:.2%} "
                f"({result['draw_correct']}/58)"
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

            all_results.append(
                {
                    "team_ids": (
                        "yes"
                        if include_team_ids
                        else "no"
                    ),
                    "feature_count": (
                        feature_count
                    ),
                    **result,
                }
            )

            print()
            print(
                "Используемые признаки:"
            )

            for index, feature in enumerate(
                selected_features,
                start=1,
            ):
                print(
                    f"{index:2}. {feature}"
                )

    print()
    print("=" * 100)
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print("=" * 100)

    result_dataframe = pd.DataFrame(
        all_results
    )

    display = result_dataframe[
        [
            "team_ids",
            "feature_count",
            "validation_accuracy",
            "validation_log_loss",
            "test_accuracy",
            "test_log_loss",
            "draw_accuracy",
            "home_accuracy",
            "away_accuracy",
            "best_iteration",
        ]
    ].copy()

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

    best = max(
        all_results,
        key=lambda item:
        item["test_accuracy"],
    )

    print()
    print("=" * 100)
    print("ЛУЧШИЙ ВАРИАНТ ПО TEST ACCURACY")
    print("=" * 100)

    print(
        f"Team ID: {best['team_ids']}"
    )

    print(
        f"Признаков: "
        f"{best['feature_count']}"
    )

    print(
        f"Test Accuracy: "
        f"{best['test_accuracy']:.2%}"
    )

    print(
        f"Test Log Loss: "
        f"{best['test_log_loss']:.4f}"
    )

    print(
        f"Draw Accuracy: "
        f"{best['draw_accuracy']:.2%}"
    )


if __name__ == "__main__":
    main()
