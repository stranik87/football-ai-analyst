from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text


DATASET_PATH = Path("data/datasets/matches_dataset.csv")
MODEL_PATH = Path("data/models/shots_on_target_poisson.joblib")
FEATURES_PATH = Path("data/models/shots_on_target_features.joblib")

TARGET_COLUMNS = {
    "home_goals",
    "away_goals",
    "result",
}

SERVICE_COLUMNS = {
    "fixture_id",
    "kickoff",
    "home_team_id",
    "away_team_id",
}


def temporal_split(df):
    df = df.sort_values("kickoff").reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()

    while (
        len(train) > 0
        and len(validation) > 0
        and train["kickoff"].iloc[-1] == validation["kickoff"].iloc[0]
    ):
        validation = pd.concat(
            [train.tail(1), validation],
            ignore_index=True,
        )
        train = train.iloc[:-1].copy()

    while (
        len(validation) > 0
        and len(test) > 0
        and validation["kickoff"].iloc[-1] == test["kickoff"].iloc[0]
    ):
        test = pd.concat(
            [validation.tail(1), test],
            ignore_index=True,
        )
        validation = validation.iloc[:-1].copy()

    return train, validation, test


def evaluate(name, y_true, predictions):
    mae = mean_absolute_error(y_true, predictions)
    rmse = np.sqrt(mean_squared_error(y_true, predictions))

    print(name)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return mae, rmse


def main():
    print("SHOTS ON TARGET MODEL")
    print("=" * 60)

    df = pd.read_csv(
        DATASET_PATH,
        parse_dates=["kickoff"],
    )

    print(f"Матчей всего: {len(df)}")

    engine = create_engine("sqlite:///football.db")

    query = text(
        """
        SELECT
            fixture_id,
            SUM(shots_on_goal) AS shots_on_target
        FROM fixture_team_statistics
        GROUP BY fixture_id
        """
    )

    stats = pd.read_sql_query(
        query,
        engine,
    )

    stats["fixture_id"] = stats["fixture_id"].astype(int)

    fixture_ids = df["fixture_id"].astype(int)

    stats = stats[
        stats["fixture_id"].isin(fixture_ids)
    ].copy()

    df = df.merge(
        stats,
        on="fixture_id",
        how="left",
    )

    df = df.dropna(
        subset=["shots_on_target"]
    ).copy()

    df = df.reset_index(drop=True)

    df["shots_on_target"] = (
        df["shots_on_target"].astype(float)
    )

    print(
        f"Матчей с данными shots on target: {len(df)}"
    )

    feature_columns = [
        c
        for c in df.columns
        if c not in TARGET_COLUMNS
        and c not in SERVICE_COLUMNS
        and c != "shots_on_target"
    ]

    X = df[feature_columns].copy()
    y = df["shots_on_target"].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(
        X.median(numeric_only=True)
    )

    print(f"Признаков: {len(feature_columns)}")

    train, validation, test = temporal_split(df)

    X_train = X.loc[train.index]
    X_validation = X.loc[validation.index]
    X_test = X.loc[test.index]

    y_train = y.loc[train.index]
    y_validation = y.loc[validation.index]
    y_test = y.loc[test.index]

    print()
    print(f"Train: {len(train)}")
    print(f"Validation: {len(validation)}")
    print(f"Test: {len(test)}")

    print()
    print(
        f"Train period:      "
        f"{train['kickoff'].min()} -> "
        f"{train['kickoff'].max()}"
    )
    print(
        f"Validation period: "
        f"{validation['kickoff'].min()} -> "
        f"{validation['kickoff'].max()}"
    )
    print(
        f"Test period:        "
        f"{test['kickoff'].min()} -> "
        f"{test['kickoff'].max()}"
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(
        X_validation
    )

    model = PoissonRegressor(
        alpha=0.3,
        max_iter=2000,
    )

    print()
    print("Обучение Poisson Regression...")

    model.fit(
        X_train_scaled,
        y_train,
    )

    validation_predictions = model.predict(
        X_validation_scaled
    )

    validation_predictions = np.clip(
        validation_predictions,
        0.01,
        None,
    )

    print()
    print("VALIDATION RESULTS")
    print("=" * 60)

    evaluate(
        "Shots on Target Validation",
        y_validation,
        validation_predictions,
    )

    print()
    print("FINAL TEST")
    print("=" * 60)

    X_train_full = pd.concat(
        [X_train, X_validation],
        axis=0,
    )

    y_train_full = pd.concat(
        [y_train, y_validation],
        axis=0,
    )

    final_scaler = StandardScaler()

    X_train_full_scaled = final_scaler.fit_transform(
        X_train_full
    )

    X_test_scaled = final_scaler.transform(
        X_test
    )

    final_model = PoissonRegressor(
        alpha=0.3,
        max_iter=2000,
    )

    final_model.fit(
        X_train_full_scaled,
        y_train_full,
    )

    test_predictions = final_model.predict(
        X_test_scaled
    )

    test_predictions = np.clip(
        test_predictions,
        0.01,
        None,
    )

    test_mae, test_rmse = evaluate(
        "Shots on Target Test",
        y_test,
        test_predictions,
    )

    baseline = y_train_full.mean()

    baseline_predictions = np.full(
        len(y_test),
        baseline,
    )

    baseline_mae, baseline_rmse = evaluate(
        "Shots on Target Baseline",
        y_test,
        baseline_predictions,
    )

    print()
    print("BASELINE COMPARISON")
    print("=" * 60)
    print(f"Model MAE:     {test_mae:.4f}")
    print(f"Baseline MAE:  {baseline_mae:.4f}")
    print(f"Model RMSE:    {test_rmse:.4f}")
    print(f"Baseline RMSE: {baseline_rmse:.4f}")

    print()

    if (
        test_mae < baseline_mae
        and test_rmse < baseline_rmse
    ):
        print("STATUS: MODEL BEATS BASELINE")
    else:
        print("STATUS: MODEL DOES NOT BEAT BASELINE")

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": final_model,
            "scaler": final_scaler,
        },
        MODEL_PATH,
    )

    joblib.dump(
        feature_columns,
        FEATURES_PATH,
    )

    print()
    print(
        f"Модель сохранена: {MODEL_PATH}"
    )
    print(
        f"Признаки сохранены: {FEATURES_PATH}"
    )


if __name__ == "__main__":
    main()
