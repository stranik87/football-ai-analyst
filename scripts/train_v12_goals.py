from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "datasets" / "matches_dataset.csv"
DB = ROOT / "football.db"
REPORT_DIR = ROOT / "data" / "reports"
MODEL_DIR = ROOT / "data" / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


TARGET_COLUMNS = {
    "goals_total": "home_goals + away_goals",
    "goals_home": "home_goals",
    "goals_away": "away_goals",
    "goals_1h_total": "halftime_home + halftime_away",
    "goals_1h_home": "halftime_home",
    "goals_1h_away": "halftime_away",
    "goals_2h_total": "(home_goals - halftime_home) + (away_goals - halftime_away)",
    "goals_2h_home": "home_goals - halftime_home",
    "goals_2h_away": "away_goals - halftime_away",
}


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_halftime_data():
    engine = create_engine(f"sqlite:///{DB}")

    query = text(
        """
        SELECT
            id AS fixture_id,
            halftime_home,
            halftime_away
        FROM fixtures
        WHERE halftime_home IS NOT NULL
          AND halftime_away IS NOT NULL
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    return pd.DataFrame(
        rows,
        columns=[
            "fixture_id",
            "halftime_home",
            "halftime_away",
        ],
    )


def get_verified_features(df):
    """Точный ML feature list из audit_dataset_leakage.py."""

    metadata_columns = {
        "fixture_id",
        "fixture_api_id",
        "kickoff",
        "home_team_id",
        "away_team_id",
        "home_team_name",
        "away_team_name",
        "result",
    }

    post_match_exact = {
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

    target_columns = set(TARGET_COLUMNS.keys())

    excluded = (
        metadata_columns
        | post_match_exact
        | {"halftime_home", "halftime_away"}
        | target_columns
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

    X = df[features].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return X, features


def main():
    print("=" * 70)
    print("v1.2 GOALS MODELS — VERIFIED 79 FEATURES")
    print("=" * 70)

    df = pd.read_csv(DATASET)

    required = [
        "fixture_id",
        "kickoff",
        "home_goals",
        "away_goals",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Не хватает колонок: {missing}"
        )

    df["fixture_id"] = pd.to_numeric(
        df["fixture_id"],
        errors="coerce",
    )

    df["kickoff"] = pd.to_datetime(
        df["kickoff"]
    )

    halftime = load_halftime_data()

    halftime["fixture_id"] = pd.to_numeric(
        halftime["fixture_id"],
        errors="coerce",
    )

    print()
    print(
        "HALFTIME DATA FROM DB:",
        len(halftime),
    )

    df = df.merge(
        halftime,
        on="fixture_id",
        how="left",
    )

    df = df.dropna(
        subset=[
            "home_goals",
            "away_goals",
            "halftime_home",
            "halftime_away",
        ]
    ).copy()

    df = df.sort_values(
        "kickoff"
    ).reset_index(drop=True)

    # Targets.
    df["goals_total"] = (
        df["home_goals"]
        + df["away_goals"]
    )

    df["goals_home"] = df["home_goals"]
    df["goals_away"] = df["away_goals"]

    df["goals_1h_home"] = (
        df["halftime_home"]
    )

    df["goals_1h_away"] = (
        df["halftime_away"]
    )

    df["goals_1h_total"] = (
        df["goals_1h_home"]
        + df["goals_1h_away"]
    )

    df["goals_2h_home"] = (
        df["home_goals"]
        - df["halftime_home"]
    )

    df["goals_2h_away"] = (
        df["away_goals"]
        - df["halftime_away"]
    )

    df["goals_2h_total"] = (
        df["goals_2h_home"]
        + df["goals_2h_away"]
    )

    if (
        (df["goals_2h_home"] < 0).any()
        or (df["goals_2h_away"] < 0).any()
    ):
        raise RuntimeError(
            "Обнаружены отрицательные значения 2H."
        )

    # Temporal split.
    timestamps = sorted(
        df["kickoff"].dropna().unique()
    )

    n = len(timestamps)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_times = set(
        timestamps[:train_end]
    )

    validation_times = set(
        timestamps[train_end:val_end]
    )

    test_times = set(
        timestamps[val_end:]
    )

    df["split"] = np.select(
        [
            df["kickoff"].isin(train_times),
            df["kickoff"].isin(validation_times),
            df["kickoff"].isin(test_times),
        ],
        [
            "train",
            "validation",
            "test",
        ],
        default="unknown",
    )

    print()
    print("SPLIT:")
    print(
        df["split"]
        .value_counts()
        .sort_index()
    )

    print()
    print("DATES:")

    for split in [
        "train",
        "validation",
        "test",
    ]:
        part = df[
            df["split"] == split
        ]

        print(
            split,
            part["kickoff"].min(),
            "->",
            part["kickoff"].max(),
            "matches:",
            len(part),
        )

    X, feature_columns = get_verified_features(df)

    print()
    print(
        "VERIFIED FEATURES:",
        len(feature_columns),
    )

    print(
        "home_team_id excluded:",
        "home_team_id" not in feature_columns,
    )

    print(
        "away_team_id excluded:",
        "away_team_id" not in feature_columns,
    )

    train_mask = (
        df["split"] == "train"
    )

    test_mask = (
        df["split"] == "test"
    )

    # Медианы считаются ТОЛЬКО на train.
    train_medians = X.loc[
        train_mask
    ].median()

    X = X.fillna(
        train_medians
    )

    targets = list(
        TARGET_COLUMNS.keys()
    )

    results = []
    ready_count = 0

    for target in targets:

        print()
        print("-" * 70)
        print(target)

        train_valid = (
            train_mask
            & df[target].notna()
        )

        test_valid = (
            test_mask
            & df[target].notna()
        )

        X_train = X.loc[
            train_valid
        ]

        X_test = X.loc[
            test_valid
        ]

        y_train = df.loc[
            train_valid,
            target,
        ].astype(float)

        y_test = df.loc[
            test_valid,
            target,
        ].astype(float)

        baseline_mean = float(
            y_train.mean()
        )

        baseline_pred = np.full(
            len(y_test),
            baseline_mean,
        )

        baseline_mae = (
            mean_absolute_error(
                y_test,
                baseline_pred,
            )
        )

        baseline_rmse = rmse(
            y_test,
            baseline_pred,
        )

        scaler = StandardScaler()

        X_train_scaled = (
            scaler.fit_transform(
                X_train
            )
        )

        X_test_scaled = (
            scaler.transform(
                X_test
            )
        )

        model = PoissonRegressor(
            alpha=1.0,
            max_iter=1000,
        )

        model.fit(
            X_train_scaled,
            y_train,
        )

        predictions = model.predict(
            X_test_scaled
        )

        model_mae = (
            mean_absolute_error(
                y_test,
                predictions,
            )
        )

        model_rmse = rmse(
            y_test,
            predictions,
        )

        mae_improvement = (
            baseline_mae
            - model_mae
        )

        rmse_improvement = (
            baseline_rmse
            - model_rmse
        )

        if (
            model_mae < baseline_mae
            and model_rmse < baseline_rmse
        ):
            status = "READY"
        elif (
            model_mae >= baseline_mae
            and model_rmse >= baseline_rmse
        ):
            status = (
                "WORSE_THAN_BASELINE"
            )
        else:
            status = "NOT_READY"

        print(
            f"train: {len(y_train)}"
        )

        print(
            f"test:  {len(y_test)}"
        )

        print(
            f"baseline MAE:  "
            f"{baseline_mae:.6f}"
        )

        print(
            f"model MAE:     "
            f"{model_mae:.6f}"
        )

        print(
            f"baseline RMSE: "
            f"{baseline_rmse:.6f}"
        )

        print(
            f"model RMSE:    "
            f"{model_rmse:.6f}"
        )

        print(
            f"MAE improvement:  "
            f"{mae_improvement:.6f}"
        )

        print(
            f"RMSE improvement: "
            f"{rmse_improvement:.6f}"
        )

        print(
            f"STATUS: {status}"
        )

        results.append(
            {
                "target": target,
                "train_matches": len(y_train),
                "test_matches": len(y_test),
                "baseline_mean": baseline_mean,
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

        if status == "READY":
            model_bundle = {
                "model": model,
                "scaler": scaler,
                "feature_columns": feature_columns,
                "target": target,
                "status": status,
            }

            path = (
                MODEL_DIR
                / f"v12_{target}_poisson.joblib"
            )

            joblib.dump(
                model_bundle,
                path,
            )

            ready_count += 1

            print(
                "SAVED:",
                path,
            )

    results_df = pd.DataFrame(
        results
    )

    report_csv = (
        REPORT_DIR
        / "v12_goals_models.csv"
    )

    report_md = (
        REPORT_DIR
        / "v12_goals_models.md"
    )

    results_df.to_csv(
        report_csv,
        index=False,
    )

    with open(
        report_md,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "# v1.2 Goals Models\n\n"
        )
        f.write(
            results_df.to_string(
                index=False
            )
        )
        f.write("\n")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print(
        "READY models saved:",
        ready_count,
    )

    print(
        "REPORT:",
        report_csv,
    )

    print(
        "REPORT:",
        report_md,
    )


if __name__ == "__main__":
    main()
