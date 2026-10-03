from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATASET = Path("data/datasets/matches_dataset.csv")
DB_PATH = Path("football.db")
MODEL_DIR = Path("data/models")
REPORT = Path("data/reports/v12_yellow_cards_models.csv")

TARGETS = {
    "yellow_cards_total": "yellow_cards_total",
    "yellow_cards_home": "home_yellow_cards",
    "yellow_cards_away": "away_yellow_cards",
}

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
    excluded = (
        METADATA_COLUMNS
        | POST_MATCH_EXACT
        | {
            "yellow_cards_total",
            "home_yellow_cards",
            "away_yellow_cards",
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


def load_yellow_card_targets():
    query = """
        SELECT
            fixture_id,
            MAX(
                CASE
                    WHEN team_id = home_team_id
                    THEN yellow_cards
                END
            ) AS home_yellow_cards,
            MAX(
                CASE
                    WHEN team_id = away_team_id
                    THEN yellow_cards
                END
            ) AS away_yellow_cards
        FROM (
            SELECT
                f.id AS fixture_id,
                f.home_team_id,
                f.away_team_id,
                s.team_id,
                s.yellow_cards
            FROM fixtures f
            JOIN fixture_team_statistics s
                ON s.fixture_id = f.id
        )
        GROUP BY fixture_id
    """

    with sqlite3.connect(DB_PATH) as conn:
        stats = pd.read_sql_query(query, conn)

    stats["yellow_cards_total"] = (
        stats["home_yellow_cards"]
        + stats["away_yellow_cards"]
    )

    return stats[
        [
            "fixture_id",
            "yellow_cards_total",
            "home_yellow_cards",
            "away_yellow_cards",
        ]
    ]


def temporal_split(df):
    df = df.sort_values("kickoff").reset_index(drop=True)

    n = len(df)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.84)

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()

    return train, validation, test


def main():
    print("=" * 70)
    print("v1.2 YELLOW CARDS — CATBOOST REGRESSOR")
    print("=" * 70)

    df = pd.read_csv(DATASET)
    df["kickoff"] = pd.to_datetime(df["kickoff"])

    card_targets = load_yellow_card_targets()

    df = df.merge(
        card_targets,
        on="fixture_id",
        how="left",
        validate="one_to_one",
    )

    X_all, feature_columns = get_verified_features(df)

    print()
    print("VERIFIED FEATURES:", len(feature_columns))
    print("TARGET ROWS:", len(card_targets))

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

    results = []

    for target_name, target_column in TARGETS.items():
        print()
        print("-" * 70)
        print(target_name)

        target_train = train.dropna(subset=[target_column])
        target_test = test.dropna(subset=[target_column])

        X_train = X_all.loc[target_train.index].copy()
        X_test = X_all.loc[target_test.index].copy()

        y_train = target_train[target_column].astype(float)
        y_test = target_test[target_column].astype(float)

        # Заполняем пропуски только статистиками TRAIN.
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

        # Baseline = среднее значение TRAIN.
        baseline_mean = float(y_train.mean())

        baseline_prediction = np.full(
            len(y_test),
            baseline_mean,
            dtype=float,
        )

        baseline_mae = mean_absolute_error(
            y_test,
            baseline_prediction,
        )

        baseline_rmse = mean_squared_error(
            y_test,
            baseline_prediction,
        ) ** 0.5

        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="MAE",
            iterations=500,
            depth=6,
            learning_rate=0.03,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            use_best_model=True,
        )

        prediction = model.predict(X_test)

        model_mae = mean_absolute_error(
            y_test,
            prediction,
        )

        model_rmse = mean_squared_error(
            y_test,
            prediction,
        ) ** 0.5

        mae_improvement = baseline_mae - model_mae
        rmse_improvement = baseline_rmse - model_rmse

        status = (
            "READY"
            if mae_improvement > 0
            and rmse_improvement > 0
            else "WORSE_THAN_BASELINE"
        )

        print("train:", len(y_train))
        print("test: ", len(y_test))
        print(f"baseline MAE:  {baseline_mae:.6f}")
        print(f"model MAE:     {model_mae:.6f}")
        print(f"baseline RMSE: {baseline_rmse:.6f}")
        print(f"model RMSE:    {model_rmse:.6f}")
        print(f"MAE improvement:  {mae_improvement:.6f}")
        print(f"RMSE improvement: {rmse_improvement:.6f}")
        print("STATUS:", status)

        if status == "READY":
            MODEL_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            output = (
                MODEL_DIR
                / f"v12_{target_name}_catboost.joblib"
            )

            joblib.dump(
                {
                    "model": model,
                    "feature_columns": feature_columns,
                    "target": target_name,
                    "status": status,
                },
                output,
            )

            print("SAVED:", output)

        results.append(
            {
                "target": target_name,
                "train_matches": len(y_train),
                "test_matches": len(y_test),
                "baseline_mae": baseline_mae,
                "baseline_rmse": baseline_rmse,
                "model_mae": model_mae,
                "model_rmse": model_rmse,
                "mae_improvement": mae_improvement,
                "rmse_improvement": rmse_improvement,
                "status": status,
                "feature_count": len(feature_columns),
            }
        )

    report = pd.DataFrame(results)

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        REPORT,
        index=False,
    )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(report.to_string(index=False))
    print()
    print("REPORT:", REPORT)


if __name__ == "__main__":
    main()
