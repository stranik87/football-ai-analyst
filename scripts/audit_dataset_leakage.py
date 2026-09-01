from pathlib import Path

import numpy as np
import pandas as pd


DATASET_PATH = Path("data/datasets/matches_dataset.csv")

TARGET = "result"

METADATA_COLUMNS = {
    "fixture_id",
    "fixture_api_id",
    "kickoff",
    "home_team_id",
    "away_team_id",
    "home_team_name",
    "away_team_name",
    "result",
}

# Поля, которые однозначно относятся к результату текущего матча.
POST_MATCH_EXACT = {
    "home_goals",
    "away_goals",
    "home_score",
    "away_score",
    "home_score_ht",
    "away_score_ht",
    "home_goals_for",
    "away_goals_for",
    "home_goals_against",
    "away_goals_against",
}

# Названия, которые требуют особого внимания.
SUSPICIOUS_KEYWORDS = [
    "score",
    "goals",
    "goal",
    "result",
    "fixture_result",
    "match_result",
    "winner",
    "winning",
    "losing",
    "half_time",
    "full_time",
]


def print_section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def check_basic_structure(df):
    print_section("1. СТРУКТУРА DATASET")

    print(f"Строк:     {len(df)}")
    print(f"Колонок:   {len(df.columns)}")

    print("\nПериод:")

    if "kickoff" in df.columns:
        kickoff = pd.to_datetime(
            df["kickoff"],
            errors="coerce",
        )

        print(f"  От: {kickoff.min()}")
        print(f"  До: {kickoff.max()}")

    print("\nTarget:")

    if TARGET in df.columns:
        print(
            df[TARGET]
            .value_counts()
            .sort_index()
            .to_string()
        )
    else:
        print("ОШИБКА: result отсутствует")


def check_duplicates(df):
    print_section("2. ДУБЛИКАТЫ")

    if "fixture_id" in df.columns:
        duplicates = df["fixture_id"].duplicated().sum()

        print(
            f"Дубликатов fixture_id: "
            f"{duplicates}"
        )

    if "fixture_api_id" in df.columns:
        duplicates = (
            df["fixture_api_id"]
            .duplicated()
            .sum()
        )

        print(
            f"Дубликатов fixture_api_id: "
            f"{duplicates}"
        )


def check_post_match_columns(df):
    print_section("3. POST-MATCH КОЛОНКИ")

    found = [
        column
        for column in sorted(POST_MATCH_EXACT)
        if column in df.columns
    ]

    if found:
        print(
            "Найдены потенциально опасные "
            "post-match поля:"
        )

        for column in found:
            print(f"  ❌ {column}")
    else:
        print(
            "Явных post-match колонок "
            "из списка не найдено."
        )


def check_suspicious_columns(df):
    print_section("4. ПОДОЗРИТЕЛЬНЫЕ НАЗВАНИЯ ПРИЗНАКОВ")

    suspicious = []

    for column in df.columns:
        column_lower = column.lower()

        if column in METADATA_COLUMNS:
            continue

        for keyword in SUSPICIOUS_KEYWORDS:
            if keyword in column_lower:
                suspicious.append(
                    (column, keyword)
                )
                break

    if suspicious:
        for column, keyword in suspicious:
            print(
                f"  ⚠ {column:<50} "
                f"keyword='{keyword}'"
            )
    else:
        print("Подозрительных названий нет.")


def check_missing_values(df):
    print_section("5. ПРОПУСКИ")

    missing = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing = missing[missing > 0]

    if missing.empty:
        print("Пропусков нет.")
        return

    print(
        missing.to_string()
    )


def check_constant_columns(df):
    print_section("6. КОНСТАНТНЫЕ ПРИЗНАКИ")

    constants = []

    for column in df.columns:
        if df[column].nunique(
            dropna=False
        ) <= 1:
            constants.append(column)

    if constants:
        for column in constants:
            print(
                f"  {column}"
            )
    else:
        print("Константных колонок нет.")


def check_target_correlations(df):
    print_section(
        "7. КОРРЕЛЯЦИЯ ЧИСЛОВЫХ ПРИЗНАКОВ С TARGET"
    )

    mapping = {
        "A": 0,
        "D": 1,
        "H": 2,
    }

    if TARGET not in df.columns:
        return

    target_numeric = df[TARGET].map(mapping)

    numeric_columns = [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(
            df[column]
        )
        and column not in METADATA_COLUMNS
    ]

    correlations = []

    for column in numeric_columns:
        series = df[column]

        if series.nunique() <= 1:
            continue

        corr = series.corr(
            target_numeric
        )

        if pd.notna(corr):
            correlations.append(
                (column, abs(corr), corr)
            )

    correlations.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    print(
        "\nТоп-30 по абсолютной корреляции:"
    )

    for column, absolute, corr in correlations[:30]:
        print(
            f"{column:<50} "
            f"corr={corr:+.5f}"
        )

    print(
        "\nВАЖНО: высокая корреляция сама по себе "
        "не означает leakage."
    )


def check_train_test_overlap(df):
    print_section(
        "8. ПРОВЕРКА TEMPORAL SPLIT"
    )

    if "kickoff" not in df.columns:
        print(
            "kickoff отсутствует."
        )
        return

    dates = pd.to_datetime(
        df["kickoff"],
        errors="coerce",
    )

    df = df.copy()
    df["_kickoff"] = dates

    df = (
        df.sort_values("_kickoff")
        .reset_index(drop=True)
    )

    n = len(df)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end]
    validation = df.iloc[
        train_end:val_end
    ]
    test = df.iloc[val_end:]

    print(
        f"Train:      {len(train)}"
    )

    print(
        f"Validation: {len(validation)}"
    )

    print(
        f"Test:       {len(test)}"
    )

    print("\nГраницы:")

    print(
        f"Train: "
        f"{train['_kickoff'].min()} "
        f"→ "
        f"{train['_kickoff'].max()}"
    )

    print(
        f"Validation: "
        f"{validation['_kickoff'].min()} "
        f"→ "
        f"{validation['_kickoff'].max()}"
    )

    print(
        f"Test: "
        f"{test['_kickoff'].min()} "
        f"→ "
        f"{test['_kickoff'].max()}"
    )

    if (
        train["_kickoff"].max()
        >= validation["_kickoff"].min()
    ):
        print(
            "\n❌ Train/Validation пересекаются!"
        )
    else:
        print(
            "\n✓ Train → Validation "
            "хронологически корректны."
        )

    if (
        validation["_kickoff"].max()
        >= test["_kickoff"].min()
    ):
        print(
            "❌ Validation/Test пересекаются!"
        )
    else:
        print(
            "✓ Validation → Test "
            "хронологически корректны."
        )


def check_same_kickoff(df):
    print_section(
        "9. ОДИНАКОВЫЕ ВРЕМЕНА НАЧАЛА"
    )

    if "kickoff" not in df.columns:
        return

    kickoff = pd.to_datetime(
        df["kickoff"],
        errors="coerce",
    )

    counts = (
        kickoff
        .value_counts()
    )

    duplicated_times = (
        counts[counts > 1]
        .sort_index()
    )

    print(
        f"Уникальных kickoff: "
        f"{kickoff.nunique()}"
    )

    print(
        f"Временных точек с несколькими "
        f"матчами: "
        f"{len(duplicated_times)}"
    )

    if not duplicated_times.empty:
        print(
            "\nПервые 10:"
        )

        print(
            duplicated_times.head(10)
            .to_string()
        )


def check_feature_names(df):
    print_section(
        "10. ФИНАЛЬНЫЙ ML FEATURE AUDIT"
    )

    excluded = (
        METADATA_COLUMNS
        | POST_MATCH_EXACT
    )

    features = [
        column
        for column in df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    print(
        f"Всего потенциальных ML-признаков: "
        f"{len(features)}"
    )

    leaked = (
        set(features)
        & POST_MATCH_EXACT
    )

    if leaked:
        print(
            "\n❌ КРИТИЧЕСКАЯ ОШИБКА!"
        )

        for column in sorted(leaked):
            print(
                f"  {column}"
            )

        raise RuntimeError(
            "Post-match признаки попали "
            "в ML features."
        )

    print(
        "\n✓ Явных post-match полей "
        "в ML features нет."
    )

    print("\nПервые признаки:")

    for feature in features:
        print(
            f"  {feature}"
        )


def check_extreme_values(df):
    print_section(
        "11. ЭКСТРЕМАЛЬНЫЕ ЗНАЧЕНИЯ"
    )

    numeric = df.select_dtypes(
        include=np.number
    )

    results = []

    for column in numeric.columns:
        values = numeric[column].replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if values.empty:
            continue

        results.append(
            {
                "feature": column,
                "min": values.min(),
                "max": values.max(),
                "mean": values.mean(),
            }
        )

    stats = pd.DataFrame(results)

    suspicious = stats[
        (
            stats["max"].abs()
            > 1000000
        )
        | (
            stats["min"].abs()
            > 1000000
        )
    ]

    if suspicious.empty:
        print(
            "Экстремальных значений "
            "> 1,000,000 не найдено."
        )
    else:
        print(
            suspicious.to_string(
                index=False
            )
        )


def main():
    print("=" * 100)
    print("DATASET LEAKAGE AUDIT")
    print("=" * 100)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset не найден: "
            f"{DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH
    )

    check_basic_structure(df)
    check_duplicates(df)
    check_post_match_columns(df)
    check_suspicious_columns(df)
    check_missing_values(df)
    check_constant_columns(df)
    check_target_correlations(df)
    check_train_test_overlap(df)
    check_same_kickoff(df)
    check_feature_names(df)
    check_extreme_values(df)

    print_section(
        "ИТОГ АУДИТА"
    )

    print(
        "✓ Dataset прочитан."
    )

    print(
        "✓ Хронологический порядок проверен."
    )

    print(
        "✓ Post-match признаки проверены."
    )

    print(
        "✓ ML feature list проверен."
    )

    print(
        "\nСледующий этап после этого аудита:"
    )

    print(
        "Poisson / expected goals model"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
