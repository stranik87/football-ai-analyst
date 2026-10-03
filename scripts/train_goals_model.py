from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.preprocessing import StandardScaler


DATASET_PATH = Path("data/datasets/matches_dataset.csv")


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


def temporal_split(df, train_ratio=0.70, validation_ratio=0.15):
    if df.empty:
        raise ValueError("Dataset is empty")

    if "kickoff" not in df.columns:
        raise ValueError("kickoff column is required")

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

    if validation_end >= len(df):
        raise ValueError("Unable to create valid temporal splits")

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()

    if train["kickoff"].max() >= validation["kickoff"].min():
        raise ValueError("Train/Validation temporal overlap detected")

    if validation["kickoff"].max() >= test["kickoff"].min():
        raise ValueError("Validation/Test temporal overlap detected")

    return train, validation, test


def prepare_goals_dataframe():
    df = pd.read_csv(DATASET_PATH)

    df["kickoff"] = pd.to_datetime(df["kickoff"])

    df["total_goals"] = (
        df["home_goals"] + df["away_goals"]
    )

    feature_columns = [
        column
        for column in df.columns
        if column not in TARGET_COLUMNS
        and column not in SERVICE_COLUMNS
        and column != "total_goals"
    ]

    X = df[feature_columns].copy()

    y_home = df["home_goals"].copy()
    y_away = df["away_goals"].copy()
    y_total = df["total_goals"].copy()

    return df, X, y_home, y_away, y_total


def evaluate_predictions(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    print(
        f"{name}: "
        f"MAE={mae:.4f}, "
        f"RMSE={rmse:.4f}"
    )

    return mae, rmse


def total_over_probability(lambda_total, line):
    return 1.0 - poisson.cdf(line, lambda_total)


def evaluate_totals(actual_total, lambda_total):
    lines = [0.5, 1.5, 2.5, 3.5, 4.5]

    print()
    print("TOTALS PROBABILITIES")
    print("=" * 60)

    results = []

    for line in lines:
        probabilities = np.asarray(
            [
                total_over_probability(
                    value,
                    line,
                )
                for value in lambda_total
            ]
        )

        actual_over = (
            np.asarray(actual_total) > line
        ).astype(int)

        actual_under = 1 - actual_over

        over_logloss = log_loss(
            actual_over,
            np.column_stack(
                [
                    1 - probabilities,
                    probabilities,
                ]
            ),
            labels=[0, 1],
        )

        over_brier = brier_score_loss(
            actual_over,
            probabilities,
        )

        print(
            f"Line {line:.1f}: "
            f"Over mean={probabilities.mean():.4f}, "
            f"Under mean={1 - probabilities.mean():.4f}, "
            f"LogLoss={over_logloss:.4f}, "
            f"Brier={over_brier:.4f}"
        )

        results.append(
            {
                "line": line,
                "over_probability_mean": probabilities.mean(),
                "under_probability_mean": 1 - probabilities.mean(),
                "log_loss": over_logloss,
                "brier": over_brier,
            }
        )

    return pd.DataFrame(results)


def train_poisson_model(X_train, y_train, X_eval):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    model = PoissonRegressor(
        alpha=0.3,
        max_iter=2000,
    )

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_eval_scaled)

    predictions = np.clip(
        predictions,
        0.01,
        10.0,
    )

    return model, scaler, predictions


if __name__ == "__main__":
    df, X, y_home, y_away, y_total = prepare_goals_dataframe()

    train, validation, test = temporal_split(df)

    train_idx = train.index
    validation_idx = validation.index
    test_idx = test.index

    X_train = X.loc[train_idx]
    X_validation = X.loc[validation_idx]
    X_test = X.loc[test_idx]

    y_home_train = y_home.loc[train_idx]
    y_away_train = y_away.loc[train_idx]

    y_home_validation = y_home.loc[validation_idx]
    y_away_validation = y_away.loc[validation_idx]

    y_home_test = y_home.loc[test_idx]
    y_away_test = y_away.loc[test_idx]

    print("GOALS POISSON MODEL")
    print("=" * 60)
    print()

    print(f"Train:      {len(train)}")
    print(f"Validation: {len(validation)}")
    print()

    print("Обучение модели home_goals...")

    home_model, home_scaler, home_predictions = train_poisson_model(
        X_train,
        y_home_train,
        X_validation,
    )

    print("Обучение модели away_goals...")

    away_model, away_scaler, away_predictions = train_poisson_model(
        X_train,
        y_away_train,
        X_validation,
    )

    print()
    print("VALIDATION RESULTS")
    print("=" * 60)

    home_mae, home_rmse = evaluate_predictions(
        "Home goals",
        y_home_validation,
        home_predictions,
    )

    away_mae, away_rmse = evaluate_predictions(
        "Away goals",
        y_away_validation,
        away_predictions,
    )

    actual_total = (
        y_home_validation.values
        + y_away_validation.values
    )

    predicted_total = (
        home_predictions
        + away_predictions
    )

    total_mae, total_rmse = evaluate_predictions(
        "Total goals",
        actual_total,
        predicted_total,
    )

    print()
    print("EXPECTED GOALS")
    print("=" * 60)
    print(
        f"Average λ home:  {home_predictions.mean():.4f}"
    )
    print(
        f"Average λ away:  {away_predictions.mean():.4f}"
    )
    print(
        f"Average λ total: {predicted_total.mean():.4f}"
    )

    print()
    print("FINAL TEST")
    print("=" * 60)

    _, _, home_test_predictions = train_poisson_model(
        X_train,
        y_home_train,
        X_test,
    )

    _, _, away_test_predictions = train_poisson_model(
        X_train,
        y_away_train,
        X_test,
    )

    print()
    home_test_mae, home_test_rmse = evaluate_predictions(
        "Home goals",
        y_home_test,
        home_test_predictions,
    )

    away_test_mae, away_test_rmse = evaluate_predictions(
        "Away goals",
        y_away_test,
        away_test_predictions,
    )

    actual_total_test = (
        y_home_test.values
        + y_away_test.values
    )

    predicted_total_test = (
        home_test_predictions
        + away_test_predictions
    )

    total_test_mae, total_test_rmse = evaluate_predictions(
        "Total goals",
        actual_total_test,
        predicted_total_test,
    )

    print()
    print("TEST BASELINE COMPARISON")
    print("=" * 60)

    baseline_total = y_total.loc[train_idx].mean()

    baseline_predictions = np.full(
        len(y_total.loc[test_idx]),
        baseline_total,
    )

    baseline_mae = mean_absolute_error(
        y_total.loc[test_idx],
        baseline_predictions,
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_total.loc[test_idx],
            baseline_predictions,
        )
    )

    print(
        f"Baseline total goals: "
        f"mean={baseline_total:.4f}, "
        f"MAE={baseline_mae:.4f}, "
        f"RMSE={baseline_rmse:.4f}"
    )

    print(
        f"Poisson total goals: "
        f"MAE={total_test_mae:.4f}, "
        f"RMSE={total_test_rmse:.4f}"
    )

    total_results = evaluate_totals(
        actual_total_test,
        predicted_total_test,
    )

    print()
    print("TOTALS CHECK")
    print("=" * 60)

    print(
        "Всего линий проверено:",
        len(total_results),
    )

    from joblib import dump

    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    dump(
        {
            "home_model": home_model,
            "home_scaler": home_scaler,
            "away_model": away_model,
            "away_scaler": away_scaler,
        },
        models_dir / "goals_poisson.joblib",
    )

    feature_columns = list(X.columns)
    dump(
        feature_columns,
        models_dir / "goals_features.joblib",
    )

    print()
    print("MODELS SAVED")
    print("=" * 60)
    print("goals_poisson.joblib")
    print("goals_features.joblib")

    print()
    print("FINAL TEST COMPLETE")
