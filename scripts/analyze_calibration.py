import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

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

    prediction_indexes = probabilities.argmax(axis=1)

    predictions = [
        classes[index]
        for index in prediction_indexes
    ]

    confidence = probabilities.max(axis=1)

    result = test[
        [
            "fixture_id",
            date_column,
            "result",
        ]
    ].copy()

    result["prediction"] = predictions
    result["confidence"] = confidence

    result["correct"] = (
        result["prediction"]
        == result["result"]
    )

    bins = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        1.01,
    ]

    labels = [
        "50-55%",
        "55-60%",
        "60-65%",
        "65-70%",
        "70-75%",
        "75-80%",
        "80%+",
    ]

    result["confidence_group"] = pd.cut(
        result["confidence"],
        bins=bins,
        labels=labels,
        right=False,
    )

    grouped = (
        result
        .groupby(
            "confidence_group",
            observed=False,
        )
        .agg(
            predictions=("prediction", "count"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
            average_confidence=(
                "confidence",
                "mean",
            ),
        )
        .reset_index()
    )

    print()
    print("=" * 80)
    print("КАЧЕСТВО МОДЕЛИ ПО УВЕРЕННОСТИ")
    print("=" * 80)

    print(
        grouped.to_string(
            index=False,
            formatters={
                "accuracy": "{:.2%}".format,
                "average_confidence": "{:.2%}".format,
            },
        )
    )

    print()
    print("=" * 80)
    print("КАЧЕСТВО ПО КЛАССАМ")
    print("=" * 80)

    for result_class in ["H", "D", "A"]:
        mask = result["prediction"] == result_class

        count = int(mask.sum())

        if count == 0:
            print(
                f"{result_class}: прогнозов нет"
            )
            continue

        correct = int(
            result.loc[mask, "correct"].sum()
        )

        accuracy = correct / count

        avg_confidence = result.loc[
            mask,
            "confidence",
        ].mean()

        print(
            f"{result_class}: "
            f"прогнозов={count}, "
            f"правильных={correct}, "
            f"accuracy={accuracy:.2%}, "
            f"средняя уверенность={avg_confidence:.2%}"
        )

    print()
    print("=" * 80)
    print("ОБЩАЯ ACCURACY")
    print("=" * 80)

    print(
        f"{accuracy_score(result.result, predictions):.2%}"
    )


if __name__ == "__main__":
    main()
