from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler


DATASET_PATH = Path(
    "data/datasets/matches_dataset.csv"
)

OUTPUT_DIR = Path(
    "data/reports"
)

ALPHAS = [
    0.1,
    0.3,
    1.0,
]


def build_folds(n: int):
    return [
        {
            "fold": 1,
            "train_end": int(n * 0.50),
            "val_end": int(n * 0.55),
            "test_end": int(n * 0.65),
        },
        {
            "fold": 2,
            "train_end": int(n * 0.65),
            "val_end": int(n * 0.70),
            "test_end": int(n * 0.80),
        },
        {
            "fold": 3,
            "train_end": int(n * 0.80),
            "val_end": int(n * 0.85),
            "test_end": int(n * 0.95),
        },
    ]


def poisson_probabilities(
    home_lambda: float,
    away_lambda: float,
    max_goals: int = 10,
):
    home_probs = np.array(
        [
            math.exp(-home_lambda)
            * home_lambda**i
            / math.factorial(i)
            for i in range(max_goals + 1)
        ]
    )

    away_probs = np.array(
        [
            math.exp(-away_lambda)
            * away_lambda**i
            / math.factorial(i)
            for i in range(max_goals + 1)
        ]
    )

    matrix = np.outer(
        home_probs,
        away_probs,
    )

    p_home = np.tril(
        matrix,
        -1,
    ).sum()

    p_draw = np.trace(
        matrix
    )

    p_away = np.triu(
        matrix,
        1,
    ).sum()

    total = (
        p_home
        + p_draw
        + p_away
    )

    return (
        p_home / total,
        p_draw / total,
        p_away / total,
    )


def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    numeric = df.select_dtypes(
        include=["number"]
    ).copy()

    drop_columns = [
        "home_goals",
        "away_goals",
        "home_score",
        "away_score",
        "home_goals_for",
        "away_goals_for",
        "home_goals_against",
        "away_goals_against",
    ]

    for column in drop_columns:

        if column in numeric.columns:
            numeric = numeric.drop(
                columns=column
            )

    numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    numeric = numeric.fillna(0)

    # Защита от слишком больших значений.
    numeric = numeric.clip(
        lower=-1e6,
        upper=1e6,
    )

    return numeric


def fit_models(
    train: pd.DataFrame,
    alpha: float,
):
    X = build_features(
        train
    )

    y_home = train[
        "home_goals"
    ].astype(float)

    y_away = train[
        "away_goals"
    ].astype(float)

    # Масштабирование необходимо для PoissonRegressor,
    # потому что признаки имеют сильно разные масштабы.
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    X_scaled = np.nan_to_num(
        X_scaled,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    home_model = PoissonRegressor(
        alpha=alpha,
        max_iter=2000,
        tol=1e-6,
    )

    away_model = PoissonRegressor(
        alpha=alpha,
        max_iter=2000,
        tol=1e-6,
    )

    home_model.fit(
        X_scaled,
        y_home,
    )

    away_model.fit(
        X_scaled,
        y_away,
    )

    return (
        home_model,
        away_model,
        scaler,
        X.columns.tolist(),
    )


def predict(
    home_model,
    away_model,
    scaler,
    validation: pd.DataFrame,
    feature_columns: list[str],
):

    X = build_features(
        validation
    )

    for column in feature_columns:

        if column not in X.columns:
            X[column] = 0

    X = X[
        feature_columns
    ]

    X_scaled = scaler.transform(
        X
    )

    X_scaled = np.nan_to_num(
        X_scaled,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    home_lambda = (
        home_model.predict(
            X_scaled
        )
    )

    away_lambda = (
        away_model.predict(
            X_scaled
        )
    )

    home_lambda = np.clip(
        home_lambda,
        0.05,
        5.0,
    )

    away_lambda = np.clip(
        away_lambda,
        0.05,
        5.0,
    )

    rows = []

    labels = np.array(
        [
            "H",
            "D",
            "A",
        ]
    )

    for i in range(
        len(validation)
    ):

        p_home, p_draw, p_away = (
            poisson_probabilities(
                float(home_lambda[i]),
                float(away_lambda[i]),
            )
        )

        probabilities = np.array(
            [
                p_home,
                p_draw,
                p_away,
            ]
        )

        prediction = labels[
            np.argmax(
                probabilities
            )
        ]

        rows.append(
            {
                "fixture_id": validation.iloc[
                    i
                ]["fixture_id"],
                "kickoff": validation.iloc[
                    i
                ]["kickoff"],
                "home_team_id": validation.iloc[
                    i
                ]["home_team_id"],
                "away_team_id": validation.iloc[
                    i
                ]["away_team_id"],
                "actual": validation.iloc[
                    i
                ]["result"],
                "prediction": prediction,
                "home_lambda": home_lambda[i],
                "away_lambda": away_lambda[i],
                "p_home": p_home,
                "p_draw": p_draw,
                "p_away": p_away,
                "confidence": max(
                    p_home,
                    p_draw,
                    p_away,
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():

    print("=" * 90)
    print(
        "POISSON VALIDATION PREDICTIONS"
    )
    print("=" * 90)

    df = pd.read_csv(
        DATASET_PATH
    )

    df["kickoff"] = pd.to_datetime(
        df["kickoff"]
    )

    df = df.sort_values(
        "kickoff"
    ).reset_index(
        drop=True
    )

    folds = build_folds(
        len(df)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for alpha in ALPHAS:

        print()
        print(
            "#" * 90
        )
        print(
            f"ALPHA = {alpha}"
        )
        print(
            "#" * 90
        )

        for fold_info in folds:

            fold = fold_info[
                "fold"
            ]

            train_end = fold_info[
                "train_end"
            ]

            val_end = fold_info[
                "val_end"
            ]

            train = df.iloc[
                :train_end
            ].copy()

            validation = df.iloc[
                train_end:val_end
            ].copy()

            print()
            print(
                f"Fold {fold}: "
                f"train={len(train)}, "
                f"validation={len(validation)}"
            )

            (
                home_model,
                away_model,
                scaler,
                feature_columns,
            ) = fit_models(
                train,
                alpha,
            )

            print(
                f"Home iterations: "
                f"{home_model.n_iter_}"
            )

            print(
                f"Away iterations: "
                f"{away_model.n_iter_}"
            )

            predictions = predict(
                home_model,
                away_model,
                scaler,
                validation,
                feature_columns,
            )

            output_path = (
                OUTPUT_DIR
                / (
                    f"poisson_v3_{alpha}_"
                    f"fold_{fold}_"
                    f"validation.csv"
                )
            )

            predictions.to_csv(
                output_path,
                index=False,
            )

            print(
                f"Saved: {output_path}"
            )

    print()
    print("=" * 90)
    print(
        "POISSON VALIDATION EXPORT DONE"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
