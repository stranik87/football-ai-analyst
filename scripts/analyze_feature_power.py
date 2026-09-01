import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from scripts.optimize_model import (
    load_dataset,
    prepare_dataframe,
    temporal_split,
)


def main():
    df = load_dataset()

    df, features, date_column = prepare_dataframe(df)

    train, validation, test = temporal_split(df)

    print()
    print("=" * 100)
    print("СИЛА ПРИЗНАКОВ")
    print("=" * 100)

    print(f"Всего признаков: {len(features)}")
    print(f"Train: {len(train)}")
    print(f"Validation: {len(validation)}")
    print(f"Test: {len(test)}")

    x = train[features]
    y = train["result"]

    scores = mutual_info_classif(
        x,
        y,
        random_state=42,
    )

    result = pd.DataFrame(
        {
            "feature": features,
            "mutual_information": scores,
        }
    ).sort_values(
        "mutual_information",
        ascending=False,
    )

    print()
    print("=" * 100)
    print("ТОП-30 ПРИЗНАКОВ")
    print("=" * 100)

    print(
        result.head(30).to_string(
            index=False,
            formatters={
                "mutual_information": "{:.5f}".format,
            },
        )
    )

    print()
    print("=" * 100)
    print("САМЫЕ СЛАБЫЕ ПРИЗНАКИ")
    print("=" * 100)

    print(
        result.tail(15).to_string(
            index=False,
            formatters={
                "mutual_information": "{:.5f}".format,
            },
        )
    )

    print()
    print("=" * 100)
    print("ПРИЗНАКИ, СВЯЗАННЫЕ С НИЧЬЕЙ")
    print("=" * 100)

    draw_mask = y == "D"

    draw_means = x.loc[draw_mask].mean()
    all_means = x.mean()

    draw_result = pd.DataFrame(
        {
            "feature": features,
            "draw_mean": [
                draw_means[f]
                for f in features
            ],
            "all_mean": [
                all_means[f]
                for f in features
            ],
        }
    )

    draw_result["difference"] = (
        draw_result["draw_mean"]
        - draw_result["all_mean"]
    ).abs()

    draw_result = draw_result.sort_values(
        "difference",
        ascending=False,
    )

    print(
        draw_result.head(20).to_string(
            index=False,
            formatters={
                "draw_mean": "{:.3f}".format,
                "all_mean": "{:.3f}".format,
                "difference": "{:.3f}".format,
            },
        )
    )


if __name__ == "__main__":
    main()
