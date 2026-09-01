from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier


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

    model = CatBoostClassifier()

    model.load_model(
        MODEL_PATH
    )

    probabilities = model.predict_proba(
        test[features]
    )

    class_index = {
        name: index
        for index, name
        in enumerate(model.classes_)
    }

    test["prob_H"] = probabilities[
        :, class_index["H"]
    ]

    test["prob_D"] = probabilities[
        :, class_index["D"]
    ]

    test["prob_A"] = probabilities[
        :, class_index["A"]
    ]

    test["prediction"] = (
        model.predict(
            test[features]
        )
        .reshape(-1)
    )

    print()
    print("=" * 100)
    print("АНАЛИЗ ВЕРОЯТНОСТИ НИЧЬЕЙ")
    print("=" * 100)

    print(
        f"Test матчей: {len(test)}"
    )

    print()

    print(
        "Средние вероятности по всем Test:"
    )

    print(
        f"H: {test['prob_H'].mean():.2%}"
    )

    print(
        f"D: {test['prob_D'].mean():.2%}"
    )

    print(
        f"A: {test['prob_A'].mean():.2%}"
    )

    draw = test[
        test[TARGET_COLUMN] == "D"
    ]

    home = test[
        test[TARGET_COLUMN] == "H"
    ]

    away = test[
        test[TARGET_COLUMN] == "A"
    ]

    print()
    print("=" * 100)
    print("СРЕДНИЕ ВЕРОЯТНОСТИ ПО ФАКТИЧЕСКОМУ РЕЗУЛЬТАТУ")
    print("=" * 100)

    rows = []

    for name, dataframe_group in [
        ("H", home),
        ("D", draw),
        ("A", away),
    ]:
        rows.append(
            {
                "fact": name,
                "matches": len(
                    dataframe_group
                ),
                "P(H)": dataframe_group[
                    "prob_H"
                ].mean(),
                "P(D)": dataframe_group[
                    "prob_D"
                ].mean(),
                "P(A)": dataframe_group[
                    "prob_A"
                ].mean(),
            }
        )

    result = pd.DataFrame(
        rows
    )

    print(
        result.to_string(
            index=False,
            formatters={
                "P(H)": "{:.2%}".format,
                "P(D)": "{:.2%}".format,
                "P(A)": "{:.2%}".format,
            },
        )
    )

    print()
    print("=" * 100)
    print("НИЧЬИ: РАСПРЕДЕЛЕНИЕ P(D)")
    print("=" * 100)

    print(
        f"Матчей с фактической ничьёй: "
        f"{len(draw)}"
    )

    print(
        f"Средняя P(D): "
        f"{draw['prob_D'].mean():.2%}"
    )

    print(
        f"Медиана P(D): "
        f"{draw['prob_D'].median():.2%}"
    )

    print(
        f"Минимум P(D): "
        f"{draw['prob_D'].min():.2%}"
    )

    print(
        f"Максимум P(D): "
        f"{draw['prob_D'].max():.2%}"
    )

    print()
    print("P(D) по диапазонам:")

    bins = [
        0.0,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.50,
        1.0,
    ]

    labels = [
        "0-10%",
        "10-15%",
        "15-20%",
        "20-25%",
        "25-30%",
        "30-35%",
        "35-40%",
        "40-50%",
        "50%+",
    ]

    draw["prob_group"] = pd.cut(
        draw["prob_D"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    distribution = (
        draw["prob_group"]
        .value_counts(
            sort=False
        )
    )

    for group, count in distribution.items():
        print(
            f"{str(group):>8}: {count:3}"
        )

    print()
    print("=" * 100)
    print("ФАКТИЧЕСКИЕ НИЧЬИ С НАИБОЛЬШЕЙ P(D)")
    print("=" * 100)

    columns = [
        "fixture_id",
        "kickoff",
        TARGET_COLUMN,
        "prediction",
        "prob_H",
        "prob_D",
        "prob_A",
    ]

    top_draws = (
        draw
        .sort_values(
            "prob_D",
            ascending=False,
        )
        .head(20)
    )

    print(
        top_draws[
            columns
        ].to_string(
            index=False,
            formatters={
                "prob_H": "{:.2%}".format,
                "prob_D": "{:.2%}".format,
                "prob_A": "{:.2%}".format,
            },
        )
    )

    print()
    print("=" * 100)
    print("ФАКТИЧЕСКИЕ НИЧЬИ С НАИМЕНЬШЕЙ P(D)")
    print("=" * 100)

    bottom_draws = (
        draw
        .sort_values(
            "prob_D",
            ascending=True,
        )
        .head(20)
    )

    print(
        bottom_draws[
            columns
        ].to_string(
            index=False,
            formatters={
                "prob_H": "{:.2%}".format,
                "prob_D": "{:.2%}".format,
                "prob_A": "{:.2%}".format,
            },
        )
    )


if __name__ == "__main__":
    main()
