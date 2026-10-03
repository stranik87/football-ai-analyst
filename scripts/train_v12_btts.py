from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    brier_score_loss,
)

DATASET = Path("data/datasets/matches_dataset.csv")
OUTPUT_MODEL = Path("data/models/v12_btts_logistic.joblib")
OUTPUT_REPORT = Path("data/reports/v12_btts_model.csv")

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


def get_verified_features(df):
    excluded = METADATA_COLUMNS | POST_MATCH_EXACT | {"btts"}

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

    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    return X, features


def temporal_split(df):
    df = df.sort_values("kickoff").reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.84)

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    return train, validation, test


def main():
    print("=" * 70)
    print("v1.2 BTTS MODEL — VERIFIED 79 FEATURES")
    print("=" * 70)

    df = pd.read_csv(DATASET)
    df["kickoff"] = pd.to_datetime(df["kickoff"])

    required = {"fixture_id", "home_goals", "away_goals", "kickoff"}
    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(f"Отсутствуют обязательные колонки: {sorted(missing)}")

    df["btts"] = (
        (df["home_goals"] > 0) &
        (df["away_goals"] > 0)
    ).astype(int)

    X_all, feature_columns = get_verified_features(df)

    print()
    print(f"VERIFIED FEATURES: {len(feature_columns)}")

    train, validation, test = temporal_split(df)

    print()
    print("SPLIT:")
    print(f"train      {len(train)}")
    print(f"validation {len(validation)}")
    print(f"test       {len(test)}")

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

    train_ids = set(train["fixture_id"])
    validation_ids = set(validation["fixture_id"])
    test_ids = set(test["fixture_id"])

    if train_ids & validation_ids:
        raise RuntimeError("LEAKAGE: train/validation overlap")
    if train_ids & test_ids:
        raise RuntimeError("LEAKAGE: train/test overlap")
    if validation_ids & test_ids:
        raise RuntimeError("LEAKAGE: validation/test overlap")

    train_idx = train.index
    validation_idx = validation.index
    test_idx = test.index

    X_train = X_all.loc[train_idx].copy()
    X_validation = X_all.loc[validation_idx].copy()
    X_test = X_all.loc[test_idx].copy()

    y_train = train["btts"]
    y_validation = validation["btts"]
    y_test = test["btts"]

    train_mean = float(y_train.mean())

    baseline_probability = np.full(len(y_test), train_mean)

    baseline_pred = (baseline_probability >= 0.5).astype(int)

    print()
    print("-" * 70)
    print("BASELINE")
    print(f"train BTTS rate: {train_mean:.6f}")
    print(
        f"Accuracy:  {accuracy_score(y_test, baseline_pred):.6f}"
    )
    print(
        f"Precision: {precision_score(y_test, baseline_pred, zero_division=0):.6f}"
    )
    print(
        f"Recall:    {recall_score(y_test, baseline_pred, zero_division=0):.6f}"
    )
    print(
        f"F1:        {f1_score(y_test, baseline_pred, zero_division=0):.6f}"
    )
    print(
        f"LogLoss:   {log_loss(y_test, baseline_probability, labels=[0, 1]):.6f}"
    )
    print(
        f"Brier:     {brier_score_loss(y_test, baseline_probability):.6f}"
    )

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)

    validation_probability = model.predict_proba(X_validation)[:, 1]
    test_probability = model.predict_proba(X_test)[:, 1]

    test_pred = (test_probability >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, test_pred)
    precision = precision_score(y_test, test_pred, zero_division=0)
    recall = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)
    test_logloss = log_loss(y_test, test_probability, labels=[0, 1])
    test_brier = brier_score_loss(y_test, test_probability)

    baseline_accuracy = accuracy_score(y_test, baseline_pred)
    baseline_logloss = log_loss(
        y_test,
        baseline_probability,
        labels=[0, 1],
    )
    baseline_brier = brier_score_loss(
        y_test,
        baseline_probability,
    )

    status = "READY"

    if (
        test_logloss >= baseline_logloss
        and test_brier >= baseline_brier
        and accuracy <= baseline_accuracy
    ):
        status = "WORSE_THAN_BASELINE"

    print()
    print("-" * 70)
    print("MODEL")
    print(f"Accuracy:  {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall:    {recall:.6f}")
    print(f"F1:        {f1:.6f}")
    print(f"LogLoss:   {test_logloss:.6f}")
    print(f"Brier:     {test_brier:.6f}")

    print()
    print(f"STATUS: {status}")

    if status == "READY":
        OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "model": model,
                "feature_columns": feature_columns,
                "target": "btts",
                "status": status,
                "train_matches": len(train),
                "validation_matches": len(validation),
                "test_matches": len(test),
                "validation_probability_mean": float(
                    validation_probability.mean()
                ),
            },
            OUTPUT_MODEL,
        )

        print(f"SAVED: {OUTPUT_MODEL}")

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    report = pd.DataFrame(
        [
            {
                "target": "btts",
                "train_matches": len(train),
                "validation_matches": len(validation),
                "test_matches": len(test),
                "baseline_accuracy": baseline_accuracy,
                "baseline_logloss": baseline_logloss,
                "baseline_brier": baseline_brier,
                "model_accuracy": accuracy,
                "model_precision": precision,
                "model_recall": recall,
                "model_f1": f1,
                "model_logloss": test_logloss,
                "model_brier": test_brier,
                "status": status,
                "feature_count": len(feature_columns),
            }
        ]
    )

    report.to_csv(OUTPUT_REPORT, index=False)

    print(f"REPORT: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
