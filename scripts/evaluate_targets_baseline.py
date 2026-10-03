from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


DB_PATH = Path("football.db")
DATASET_PATH = Path("data/datasets/matches_dataset.csv")
REPORT_DIR = Path("data/reports")

CSV_REPORT = REPORT_DIR / "targets_baseline.csv"
MD_REPORT = REPORT_DIR / "targets_baseline.md"


def temporal_split(df):
    df = df.copy()

    df["kickoff"] = pd.to_datetime(
        df["kickoff"],
        errors="coerce",
    )

    df = (
        df.dropna(subset=["kickoff"])
        .sort_values("kickoff")
        .reset_index(drop=True)
    )

    n = len(df)

    train_target = int(n * 0.70)
    validation_target = int(n * 0.85)

    train_end = train_target

    while (
        train_end < n
        and train_end > 0
        and df.loc[train_end - 1, "kickoff"]
        == df.loc[train_end, "kickoff"]
    ):
        train_end += 1

    if train_end >= n:
        raise RuntimeError(
            "Невозможно сформировать Validation."
        )

    validation_end = max(
        validation_target,
        train_end + 1,
    )

    while (
        validation_end < n
        and validation_end > 0
        and df.loc[validation_end - 1, "kickoff"]
        == df.loc[validation_end, "kickoff"]
    ):
        validation_end += 1

    if validation_end >= n:
        raise RuntimeError(
            "Невозможно сформировать Test."
        )

    train = df.iloc[:train_end].copy()
    validation = df.iloc[
        train_end:validation_end
    ].copy()
    test = df.iloc[validation_end:].copy()

    return train, validation, test


def regression_metrics(actual, prediction):
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    mae = np.mean(
        np.abs(actual - prediction)
    )

    rmse = np.sqrt(
        np.mean(
            (actual - prediction) ** 2
        )
    )

    return mae, rmse


def classification_metrics(actual, probability):
    actual = np.asarray(actual, dtype=int)
    probability = np.asarray(
        probability,
        dtype=float,
    )

    prediction = (
        probability >= 0.5
    ).astype(int)

    accuracy = np.mean(
        prediction == actual
    )

    tp = np.sum(
        (prediction == 1)
        & (actual == 1)
    )

    fp = np.sum(
        (prediction == 1)
        & (actual == 0)
    )

    fn = np.sum(
        (prediction == 0)
        & (actual == 1)
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    eps = 1e-15

    probability = np.clip(
        probability,
        eps,
        1 - eps,
    )

    log_loss = -np.mean(
        actual * np.log(probability)
        + (1 - actual)
        * np.log(1 - probability)
    )

    brier = np.mean(
        (probability - actual) ** 2
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        log_loss,
        brier,
    )


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"База данных не найдена: {DB_PATH}"
        )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset не найден: {DATASET_PATH}"
        )

    dataset = pd.read_csv(
        DATASET_PATH
    )

    if "fixture_id" not in dataset.columns:
        raise RuntimeError(
            "В dataset отсутствует fixture_id."
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

    engine = create_engine(
        f"sqlite:///{DB_PATH}"
    )

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
                "fixture_ids":
                    fixture_ids.tolist()
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
                "fixture_ids":
                    fixture_ids.tolist()
            },
        )

    fixtures = fixtures.drop_duplicates(
        subset=["id"]
    )

    fixtures["goals"] = (
        pd.to_numeric(
            fixtures["home_goals"],
            errors="coerce",
        )
        +
        pd.to_numeric(
            fixtures["away_goals"],
            errors="coerce",
        )
    )

    fixtures["btts"] = (
        (
            pd.to_numeric(
                fixtures["home_goals"],
                errors="coerce",
            ) > 0
        )
        &
        (
            pd.to_numeric(
                fixtures["away_goals"],
                errors="coerce",
            ) > 0
        )
    ).astype(int)

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
    ).reset_index()

    stats_pivot = stats_pivot.rename(
        columns={
            "fixture_id": "id"
        }
    )

    data = fixtures[
        [
            "id",
            "kickoff",
            "goals",
            "btts",
        ]
    ].merge(
        stats_pivot,
        on="id",
        how="left",
    )

    train, validation, test = temporal_split(
        data
    )

    print("=" * 100)
    print("TARGET BASELINE EVALUATION")
    print("=" * 100)
    print()
    print(
        f"Train:      {len(train)}"
    )
    print(
        f"Validation: {len(validation)}"
    )
    print(
        f"Test:       {len(test)}"
    )
    print()

    results = []

    regression_targets = [
        "goals",
        "corner_kicks",
        "yellow_cards",
        "total_shots",
        "shots_on_goal",
        "offsides",
        "fouls",
    ]

    for target in regression_targets:
        train_values = pd.to_numeric(
            train[target],
            errors="coerce",
        ).dropna()

        test_values = pd.to_numeric(
            test[target],
            errors="coerce",
        )

        mask = test_values.notna()

        if train_values.empty or not mask.any():
            continue

        baseline_value = train_values.mean()

        prediction = np.full(
            mask.sum(),
            baseline_value,
        )

        mae, rmse = regression_metrics(
            test_values[mask],
            prediction,
        )

        results.append(
            {
                "target": target,
                "type": "regression",
                "test_matches": int(mask.sum()),
                "baseline": baseline_value,
                "mae": mae,
                "rmse": rmse,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "log_loss": None,
                "brier": None,
            }
        )

    train_btts = pd.to_numeric(
        train["btts"],
        errors="coerce",
    ).dropna()

    test_btts = pd.to_numeric(
        test["btts"],
        errors="coerce",
    )

    mask = test_btts.notna()

    if not train_btts.empty and mask.any():
        baseline_probability = train_btts.mean()

        probability = np.full(
            mask.sum(),
            baseline_probability,
        )

        (
            accuracy,
            precision,
            recall,
            f1,
            log_loss,
            brier,
        ) = classification_metrics(
            test_btts[mask],
            probability,
        )

        results.append(
            {
                "target": "btts",
                "type": "classification",
                "test_matches": int(mask.sum()),
                "baseline": baseline_probability,
                "mae": None,
                "rmse": None,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "log_loss": log_loss,
                "brier": brier,
            }
        )

    report = pd.DataFrame(
        results
    )

    report.to_csv(
        CSV_REPORT,
        index=False,
    )

    with open(
        MD_REPORT,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "# Targets Baseline Evaluation\n\n"
        )

        file.write(
            "Baseline рассчитан только на "
            "temporal test-период.\n\n"
        )

        file.write(
            f"- Train: {len(train)}\n"
        )
        file.write(
            f"- Validation: {len(validation)}\n"
        )
        file.write(
            f"- Test: {len(test)}\n\n"
        )

        file.write(
            "| Target | Type | Test | Baseline | "
            "MAE | RMSE | Accuracy | Precision | "
            "Recall | F1 | LogLoss | Brier |\n"
        )

        file.write(
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )

        for _, row in report.iterrows():
            def fmt(value):
                if pd.isna(value):
                    return "-"

                return f"{value:.4f}"

            file.write(
                f"| {row['target']} "
                f"| {row['type']} "
                f"| {int(row['test_matches'])} "
                f"| {fmt(row['baseline'])} "
                f"| {fmt(row['mae'])} "
                f"| {fmt(row['rmse'])} "
                f"| {fmt(row['accuracy'])} "
                f"| {fmt(row['precision'])} "
                f"| {fmt(row['recall'])} "
                f"| {fmt(row['f1'])} "
                f"| {fmt(row['log_loss'])} "
                f"| {fmt(row['brier'])} |\n"
            )

    print(
        report.to_string(
            index=False
        )
    )

    print()
    print(
        f"CSV: {CSV_REPORT}"
    )
    print(
        f"MD:  {MD_REPORT}"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
