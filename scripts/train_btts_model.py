from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler


DATASET_PATH = Path("data/datasets/matches_dataset.csv")

TARGET_COLUMNS = {
    "home_goals",
    "away_goals",
    "result",
    "btts",
}

SERVICE_COLUMNS = {
    "fixture_id",
    "kickoff",
    "home_team_id",
    "away_team_id",
}


def temporal_split(df, train_ratio=0.70, validation_ratio=0.15):
    df = df.sort_values("kickoff").reset_index(drop=True)

    train_end = int(len(df) * train_ratio)
    validation_end = int(
        len(df) * (train_ratio + validation_ratio)
    )

    train_end = min(max(train_end, 1), len(df) - 2)
    validation_end = min(
        max(validation_end, train_end + 1),
        len(df) - 1,
    )

    while (
        train_end < len(df)
        and df.loc[train_end - 1, "kickoff"]
        == df.loc[train_end, "kickoff"]
    ):
        train_end += 1

    validation_end = max(validation_end, train_end + 1)

    while (
        validation_end < len(df)
        and df.loc[validation_end - 1, "kickoff"]
        == df.loc[validation_end, "kickoff"]
    ):
        validation_end += 1

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()

    if train["kickoff"].max() >= validation["kickoff"].min():
        raise ValueError("Train/Validation temporal overlap")

    if validation["kickoff"].max() >= test["kickoff"].min():
        raise ValueError("Validation/Test temporal overlap")

    return train, validation, test


def prepare_dataframe():
    df = pd.read_csv(DATASET_PATH)

    df["kickoff"] = pd.to_datetime(df["kickoff"])

    df["btts"] = (
        (df["home_goals"] > 0)
        & (df["away_goals"] > 0)
    ).astype(int)

    feature_columns = [
        column
        for column in df.columns
        if column not in TARGET_COLUMNS
        and column not in SERVICE_COLUMNS
    ]

    X = df[feature_columns].copy()
    y = df["btts"].copy()

    return df, X, y


def train_model(X_train, y_train, X_eval):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    model = LogisticRegression(
        max_iter=2000,
        C=1.0,
    )

    model.fit(X_train_scaled, y_train)

    probabilities = model.predict_proba(X_eval_scaled)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return model, scaler, predictions, probabilities


def evaluate(name, actual, predictions, probabilities):
    accuracy = accuracy_score(actual, predictions)
    precision = precision_score(
        actual,
        predictions,
        zero_division=0,
    )
    recall = recall_score(
        actual,
        predictions,
        zero_division=0,
    )
    f1 = f1_score(
        actual,
        predictions,
        zero_division=0,
    )
    logloss = log_loss(
        actual,
        probabilities,
    )
    brier = brier_score_loss(
        actual,
        probabilities,
    )

    print(name)
    print(
        f"Accuracy:  {accuracy:.4f}"
    )
    print(
        f"Precision: {precision:.4f}"
    )
    print(
        f"Recall:    {recall:.4f}"
    )
    print(
        f"F1:        {f1:.4f}"
    )
    print(
        f"LogLoss:   {logloss:.4f}"
    )
    print(
        f"Brier:     {brier:.4f}"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "logloss": logloss,
        "brier": brier,
    }


if __name__ == "__main__":
    df, X, y = prepare_dataframe()

    train, validation, test = temporal_split(df)

    train_idx = train.index
    validation_idx = validation.index
    test_idx = test.index

    X_train = X.loc[train_idx]
    X_validation = X.loc[validation_idx]
    X_test = X.loc[test_idx]

    y_train = y.loc[train_idx]
    y_validation = y.loc[validation_idx]
    y_test = y.loc[test_idx]

    print("BTTS MODEL")
    print("=" * 60)
    print()

    print(f"Матчей всего: {len(df)}")
    print(f"Признаков: {len(X.columns)}")
    print(f"Train: {len(train)}")
    print(f"Validation: {len(validation)}")
    print()

    print(
        f"Train BTTS rate: "
        f"{y_train.mean():.4f}"
    )

    print(
        f"Validation BTTS rate: "
        f"{y_validation.mean():.4f}"
    )

    print()
    print("Обучение Logistic Regression...")

    model, scaler, predictions, probabilities = train_model(
        X_train,
        y_train,
        X_validation,
    )

    print()
    print("VALIDATION RESULTS")
    print("=" * 60)

    evaluate(
        "BTTS Validation",
        y_validation,
        predictions,
        probabilities,
    )

    print()
    print("FINAL TEST")
    print("=" * 60)

    _, _, test_predictions, test_probabilities = train_model(
        X_train,
        y_train,
        X_test,
    )

    test_results = evaluate(
        "BTTS Test",
        y_test,
        test_predictions,
        test_probabilities,
    )

    print()
    print("TEST BASELINE COMPARISON")
    print("=" * 60)

    baseline_probability = y_train.mean()
    baseline_predictions = np.ones(
        len(y_test),
        dtype=int,
    )

    baseline_logloss = log_loss(
        y_test,
        np.full(len(y_test), baseline_probability),
    )

    baseline_brier = brier_score_loss(
        y_test,
        np.full(len(y_test), baseline_probability),
    )

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions,
    )

    baseline_precision = precision_score(
        y_test,
        baseline_predictions,
        zero_division=0,
    )

    baseline_recall = recall_score(
        y_test,
        baseline_predictions,
        zero_division=0,
    )

    baseline_f1 = f1_score(
        y_test,
        baseline_predictions,
        zero_division=0,
    )

    print(
        f"Baseline probability: {baseline_probability:.4f}"
    )
    print(
        f"Baseline Accuracy:    {baseline_accuracy:.4f}"
    )
    print(
        f"Baseline Precision:   {baseline_precision:.4f}"
    )
    print(
        f"Baseline Recall:      {baseline_recall:.4f}"
    )
    print(
        f"Baseline F1:          {baseline_f1:.4f}"
    )
    print(
        f"Baseline LogLoss:     {baseline_logloss:.4f}"
    )
    print(
        f"Baseline Brier:       {baseline_brier:.4f}"
    )

    print()
    print("BTTS TEST COMPLETE")
