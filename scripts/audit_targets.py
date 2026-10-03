from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


DB_PATH = Path("football.db")
DATASET_PATH = Path("data/datasets/matches_dataset.csv")
REPORT_DIR = Path("data/reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

CSV_REPORT = REPORT_DIR / "targets_audit.csv"
MD_REPORT = REPORT_DIR / "targets_audit.md"


def get_engine():
    return create_engine(f"sqlite:///{DB_PATH}")


def classify_target(available, total, fill_rate):
    if available == 0:
        return "INSUFFICIENT_DATA"

    if fill_rate < 0.50:
        return "INSUFFICIENT_DATA"

    return "CANDIDATE"


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"База данных не найдена: {DB_PATH}"
        )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset не найден: {DATASET_PATH}"
        )

    dataset = pd.read_csv(DATASET_PATH)

    if "fixture_id" not in dataset.columns:
        raise RuntimeError(
            "В matches_dataset.csv отсутствует fixture_id."
        )

    fixture_ids = (
        pd.to_numeric(
            dataset["fixture_id"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
    )

    total_matches = len(fixture_ids)

    engine = get_engine()

    with engine.connect() as conn:
        fixtures = pd.read_sql(
            text("""
                SELECT
                    id,
                    kickoff,
                    home_goals,
                    away_goals
                FROM fixtures
                WHERE id IN :fixture_ids
            """).bindparams(
                __import__("sqlalchemy").bindparam(
                    "fixture_ids",
                    expanding=True,
                )
            ),
            conn,
            params={
                "fixture_ids": fixture_ids.tolist()
            },
        )

        stats = pd.read_sql(
            text("""
                SELECT
                    fixture_id,
                    team_id,
                    shots_on_goal,
                    total_shots,
                    fouls,
                    corner_kicks,
                    offsides,
                    yellow_cards
                FROM fixture_team_statistics
                WHERE fixture_id IN :fixture_ids
            """).bindparams(
                __import__("sqlalchemy").bindparam(
                    "fixture_ids",
                    expanding=True,
                )
            ),
            conn,
            params={
                "fixture_ids": fixture_ids.tolist()
            },
        )

    fixtures = fixtures.drop_duplicates(
        subset=["id"]
    )

    stats_pivot = stats.pivot_table(
        index="fixture_id",
        values=[
            "shots_on_goal",
            "total_shots",
            "fouls",
            "corner_kicks",
            "offsides",
            "yellow_cards",
        ],
        aggfunc="sum",
        min_count=2,
    )

    goals = fixtures[
        ["id", "home_goals", "away_goals"]
    ].copy()

    goals["goals"] = (
        pd.to_numeric(
            goals["home_goals"],
            errors="coerce",
        )
        +
        pd.to_numeric(
            goals["away_goals"],
            errors="coerce",
        )
    )

    goals["btts"] = (
        (
            pd.to_numeric(
                goals["home_goals"],
                errors="coerce",
            ) > 0
        )
        &
        (
            pd.to_numeric(
                goals["away_goals"],
                errors="coerce",
            ) > 0
        )
    ).astype("Int64")

    goals = goals.set_index("id")

    targets = {
        "goals": goals["goals"],
        "btts": goals["btts"],
        "corners": stats_pivot["corner_kicks"]
        if "corner_kicks" in stats_pivot.columns
        else pd.Series(dtype=float),
        "yellow_cards": stats_pivot["yellow_cards"]
        if "yellow_cards" in stats_pivot.columns
        else pd.Series(dtype=float),
        "shots": stats_pivot["total_shots"]
        if "total_shots" in stats_pivot.columns
        else pd.Series(dtype=float),
        "shots_on_target": stats_pivot["shots_on_goal"]
        if "shots_on_goal" in stats_pivot.columns
        else pd.Series(dtype=float),
        "offsides": stats_pivot["offsides"]
        if "offsides" in stats_pivot.columns
        else pd.Series(dtype=float),
        "fouls": stats_pivot["fouls"]
        if "fouls" in stats_pivot.columns
        else pd.Series(dtype=float),
    }

    rows = []

    for name, series in targets.items():
        series = pd.to_numeric(
            series,
            errors="coerce",
        ).dropna()

        available = len(series)
        missing = total_matches - available

        fill_rate = (
            available / total_matches
            if total_matches
            else 0
        )

        if available:
            minimum = series.min()
            maximum = series.max()
            mean = series.mean()
            median = series.median()
            std = series.std()
        else:
            minimum = None
            maximum = None
            mean = None
            median = None
            std = None

        status = classify_target(
            available,
            total_matches,
            fill_rate,
        )

        rows.append(
            {
                "target": name,
                "total_matches": total_matches,
                "available_matches": available,
                "missing_matches": missing,
                "fill_rate": fill_rate,
                "min": minimum,
                "max": maximum,
                "mean": mean,
                "median": median,
                "std": std,
                "status": status,
            }
        )

    rows.append(
        {
            "target": "throw_ins",
            "total_matches": total_matches,
            "available_matches": 0,
            "missing_matches": total_matches,
            "fill_rate": 0.0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "status": "INSUFFICIENT_DATA",
        }
    )

    report = pd.DataFrame(rows)

    report.to_csv(
        CSV_REPORT,
        index=False,
    )

    with open(
        MD_REPORT,
        "w",
        encoding="utf-8",
    ) as file:
        file.write("# Targets Audit\n\n")
        file.write(
            "Аудит target выполнен только на матчах, "
            "которые входят в текущий "
            "`matches_dataset.csv`.\n\n"
        )

        file.write(
            f"Количество матчей dataset: "
            f"**{total_matches}**.\n\n"
        )

        file.write(
            "| Target | Available | Missing | "
            "Fill rate | Min | Max | Mean | "
            "Median | Std | Status |\n"
        )

        file.write(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
        )

        for row in rows:
            def fmt(value):
                if pd.isna(value):
                    return "-"

                if isinstance(value, float):
                    return f"{value:.4f}"

                return str(value)

            file.write(
                f"| {row['target']} "
                f"| {row['available_matches']} "
                f"| {row['missing_matches']} "
                f"| {row['fill_rate']:.2%} "
                f"| {fmt(row['min'])} "
                f"| {fmt(row['max'])} "
                f"| {fmt(row['mean'])} "
                f"| {fmt(row['median'])} "
                f"| {fmt(row['std'])} "
                f"| {row['status']} |\n"
            )

        file.write("\n## Notes\n\n")
        file.write(
            "- CANDIDATE означает только наличие "
            "достаточного количества данных для "
            "дальнейшего исследования.\n"
        )
        file.write(
            "- CANDIDATE не означает READY.\n"
        )
        file.write(
            "- Финальный статус определяется после "
            "temporal baseline и model evaluation.\n"
        )
        file.write(
            "- Throw-ins: INSUFFICIENT_DATA, "
            "так как соответствующего поля нет "
            "в текущей БД.\n"
        )

    print("=" * 100)
    print("TARGETS AUDIT — DATASET ONLY")
    print("=" * 100)
    print()
    print(
        f"Матчей в matches_dataset.csv: "
        f"{total_matches}"
    )
    print()
    print(report.to_string(index=False))
    print()
    print(f"CSV: {CSV_REPORT}")
    print(f"MD:  {MD_REPORT}")
    print("=" * 100)


if __name__ == "__main__":
    main()
