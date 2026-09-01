import pandas as pd

from scripts.optimize_model import (
    load_dataset,
    prepare_dataframe,
    temporal_split,
)


def main():
    df = load_dataset()

    df, features, date_column = prepare_dataframe(df)

    _, _, test = temporal_split(df)

    print()
    print("=" * 90)
    print("АНАЛИЗ ПРИЗНАКОВ ДЛЯ НИЧЬИХ")
    print("=" * 90)

    print(f"Тестовых матчей: {len(test)}")
    print()

    draw = test[test["result"] == "D"]
    home = test[test["result"] == "H"]
    away = test[test["result"] == "A"]

    rows = []

    for feature in features:
        draw_mean = draw[feature].mean()
        home_mean = home[feature].mean()
        away_mean = away[feature].mean()

        all_mean = test[feature].mean()

        draw_distance = abs(
            draw_mean - all_mean
        )

        rows.append(
            {
                "feature": feature,
                "draw_mean": draw_mean,
                "home_mean": home_mean,
                "away_mean": away_mean,
                "draw_distance": draw_distance,
            }
        )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "draw_distance",
        ascending=False,
    )

    print(
        result.head(30).to_string(
            index=False,
            formatters={
                "draw_mean": "{:.3f}".format,
                "home_mean": "{:.3f}".format,
                "away_mean": "{:.3f}".format,
                "draw_distance": "{:.3f}".format,
            },
        )
    )

    print()
    print("=" * 90)
    print("СРЕДНИЕ ЗНАЧЕНИЯ: DRAW vs HOME vs AWAY")
    print("=" * 90)

    comparison = pd.DataFrame(
        {
            "DRAW": draw[features].mean(),
            "HOME": home[features].mean(),
            "AWAY": away[features].mean(),
        }
    )

    comparison["DRAW_vs_HOME"] = (
        comparison["DRAW"]
        - comparison["HOME"]
    )

    comparison["DRAW_vs_AWAY"] = (
        comparison["DRAW"]
        - comparison["AWAY"]
    )

    comparison["DRAW_vs_ALL"] = (
        comparison["DRAW"]
        - test[features].mean()
    )

    print(
        comparison.sort_values(
            "DRAW_vs_ALL",
            key=lambda x: x.abs(),
            ascending=False,
        )
        .head(30)
        .to_string(
            formatters={
                "DRAW": "{:.3f}".format,
                "HOME": "{:.3f}".format,
                "AWAY": "{:.3f}".format,
                "DRAW_vs_HOME": "{:.3f}".format,
                "DRAW_vs_AWAY": "{:.3f}".format,
                "DRAW_vs_ALL": "{:.3f}".format,
            }
        )
    )


if __name__ == "__main__":
    main()
