import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix

from scripts.optimize_model import (
    load_dataset,
    prepare_dataframe,
    temporal_split,
)


MODEL_PATH = (
    "data/models/match_result_catboost_optimized.cbm"
)


def main():
    df = load_dataset()

    df, features, date_column = prepare_dataframe(df)

    _, _, test = temporal_split(df)

    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    probabilities = model.predict_proba(
        test[features]
    )

    classes = list(model.classes_)

    predictions = [
        classes[index]
        for index in probabilities.argmax(axis=1)
    ]

    result = test[
        [
            "fixture_id",
            date_column,
            "result",
        ]
    ].copy()

    result["prediction"] = predictions

    result["confidence"] = probabilities.max(
        axis=1
    )

    result["correct"] = (
        result["result"]
        == result["prediction"]
    )

    print()
    print("=" * 80)
    print("ОШИБКИ МОДЕЛИ")
    print("=" * 80)

    print(
        f"Всего тестовых матчей: {len(result)}"
    )

    print(
        f"Правильных: {result['correct'].sum()}"
    )

    print(
        f"Ошибочных: {(~result['correct']).sum()}"
    )

    print()
    print("Матрица ошибок H / D / A:")
    print()

    matrix = confusion_matrix(
        result["result"],
        result["prediction"],
        labels=["H", "D", "A"],
    )

    print(
        pd.DataFrame(
            matrix,
            index=["Факт H", "Факт D", "Факт A"],
            columns=[
                "Прогноз H",
                "Прогноз D",
                "Прогноз A",
            ],
        )
    )

    print()
    print("=" * 80)
    print("ФАКТ → ПРОГНОЗ")
    print("=" * 80)

    combinations = (
        result
        .groupby(
            ["result", "prediction"]
        )
        .agg(
            matches=("prediction", "count"),
            avg_confidence=(
                "confidence",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        combinations.to_string(
            index=False,
            formatters={
                "avg_confidence":
                    "{:.2%}".format,
            },
        )
    )

    print()
    print("=" * 80)
    print("САМЫЕ УВЕРЕННЫЕ ОШИБКИ")
    print("=" * 80)

    wrong = result[
        ~result["correct"]
    ].sort_values(
        "confidence",
        ascending=False,
    )

    print(
        wrong.head(20).to_string(
            index=False,
            formatters={
                "confidence":
                    "{:.2%}".format,
            },
        )
    )

    print()
    print("=" * 80)
    print("САМЫЕ УВЕРЕННЫЕ ПРАВИЛЬНЫЕ ПРОГНОЗЫ")
    print("=" * 80)

    correct = result[
        result["correct"]
    ].sort_values(
        "confidence",
        ascending=False,
    )

    print(
        correct.head(20).to_string(
            index=False,
            formatters={
                "confidence":
                    "{:.2%}".format,
            },
        )
    )


if __name__ == "__main__":
    main()
