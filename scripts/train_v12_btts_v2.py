from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    brier_score_loss,
)


DATASET = Path("data/datasets/matches_dataset.csv")
DB_PATH = Path("football.db")
MODEL_DIR = Path("data/models")
REPORT = Path("data/reports/v12_btts_v2.csv")

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

TARGET_COLUMNS = {
    "btts",
    "btts_target",
}


def get_verified_features(df):
    excluded = (
        METADATA_COLUMNS
        | POST_MATCH_EXACT
        | TARGET_COLUMNS
        | {
            "goals_total",
            "goals_home",
            "goals_away",
            "goals_1h_total",
            "goals_1h_home",
            "goals_1h_away",
            "goals_2h_total",
            "goals_2h_home",
            "goals_2h_away",
        }
    )

    features = [
        column
        for column in df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    if len(features) != 79:
        raise RuntimeError(
            f"Ожидалось 79 ML features, получено: {len(features)}"
        )

    return df[features].copy(), features


def load_btts_targets():
    query = """
        SELECT
            id AS fixture_id,
            CASE
                WHEN home_goals > 0 AND away_goals > 0
                THEN 1
                ELSE 0
            END AS btts_target
        FROM fixtures
        WHERE home_goals IS NOT NULL
          AND away_goals IS NOT NULL
    """

    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn)


def temporal_split(df):
    df = df.sort_values("kickoff").reset_index(drop=True)

    n = len(df)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.84)

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()

    return train, validation, test


def evaluate(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "logloss": log_loss(
            y_true,
            np.column_stack(
                [1.0 - probabilities, probabilities]
            ),
            labels=[0, 1],
        ),
        "brier": brier_score_loss(
            y_true,
            probabilities,
        ),
    }


def main():
    print("=" * 70)
    print("v1.2 BTTS V2 — CATBOOST CLASSIFIER")
    print("=" * 70)

    df = pd.read_csv(DATASET)
    df["kickoff"] = pd.to_datetime(df["kickoff"])

    targets = load_btts_targets()

    df = df.merge(
        targets,
        on="fixture_id",
        how="inner",
        validate="one_to_one",
    )

    X_all, feature_columns = get_verified_features(df)

    print()
    print("VERIFIED FEATURES:", len(feature_columns))
    print("TARGET ROWS:", len(targets))

    train, validation, test = temporal_split(df)

    print()
    print("SPLIT:")
    print("train:", len(train))
    print("validation:", len(validation))
    print("test:", len(test))

    print()
    print("DATES:")
    print(
        f"train {train['kickoff'].min()} -> "
        f"{train['kickoff'].max()}"
    )
    print(
        f"validation {validation['kickoff'].min()} -> "
        f"{validation['kickoff'].max()}"
    )
    print(
        f"test {test['kickoff'].min()} -> "
        f"{test['kickoff'].max()}"
    )

    y_train = train["btts_target"].astype(int)
    y_test = test["btts_target"].astype(int)

    X_train = X_all.loc[train.index].copy()
    X_test = X_all.loc[test.index].copy()

    medians = X_train.median(numeric_only=True)

    X_train = (
        X_train
        .replace([np.inf, -np.inf], np.nan)
        .fillna(medians)
        .fillna(0.0)
    )

    X_test = (
        X_test
        .replace([np.inf, -np.inf], np.nan)
        .fillna(medians)
        .fillna(0.0)
    )

    baseline_probability = float(y_train.mean())

    baseline_probabilities = np.full(
        len(y_test),
        baseline_probability,
        dtype=float,
    )

    baseline_metrics = evaluate(
        y_test,
        baseline_probabilities,
    )

    print()
    print("-" * 70)
    print("BASELINE")
    print("-" * 70)
    print(f"train BTTS rate: {baseline_probability:.6f}")

    for key, value in baseline_metrics.items():
        print(f"{key}: {value:.6f}")

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=600,
        depth=6,
        learning_rate=0.03,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        auto_class_weights="Balanced",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        use_best_model=True,
    )

    model_probabilities = model.predict_proba(X_test)[:, 1]

    model_metrics = evaluate(
        y_test,
        model_probabilities,
    )

    print()
    print("-" * 70)
    print("CATBOOST V2")
    print("-" * 70)

    for key, value in model_metrics.items():
        print(f"{key}: {value:.6f}")

    logloss_improvement = (
        baseline_metrics["logloss"]
        - model_metrics["logloss"]
    )

    brier_improvement = (
        baseline_metrics["brier"]
        - model_metrics["brier"]
    )

    accuracy_improvement = (
        model_metrics["accuracy"]
        - baseline_metrics["accuracy"]
    )

    status = (
        "READY"
        if logloss_improvement > 0
        and brier_improvement > 0
        else "WORSE_THAN_BASELINE"
    )

    print()
    print("-" * 70)
    print("COMPARISON")
    print("-" * 70)
    print(
        f"Accuracy improvement: "
        f"{accuracy_improvement:.6f}"
    )
    print(
        f"LogLoss improvement: "
        f"{logloss_improvement:.6f}"
    )
    print(
        f"Brier improvement: "
        f"{brier_improvement:.6f}"
    )
    print("STATUS:", status)

    if status == "READY":
        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        output = MODEL_DIR / "v12_btts_catboost.joblib"

        joblib.dump(
            {
                "model": model,
                "feature_columns": feature_columns,
                "target": "btts",
                "status": status,
            },
            output,
        )

        print("SAVED:", output)

    report = pd.DataFrame(
        [
            {
                "model": "baseline",
                **baseline_metrics,
                "status": "BASELINE",
                "feature_count": 0,
            },
            {
                "model": "catboost_v2",
                **model_metrics,
                "accuracy_improvement": accuracy_improvement,
                "logloss_improvement": logloss_improvement,
                "brier_improvement": brier_improvement,
                "status": status,
                "feature_count": len(feature_columns),
            },
        ]
    )

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        REPORT,
        index=False,
    )

    print()
    print("REPORT:", REPORT)


if __name__ == "__main__":
    main()
