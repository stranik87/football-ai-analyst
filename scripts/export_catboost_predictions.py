from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_selection import mutual_info_classif


DATASET_PATH = Path(
    "data/datasets/matches_dataset.csv"
)

REPORT_PATH = Path(
    "data/reports/catboost_walk_forward_predictions.csv"
)

RANDOM_STATE = 42

TOP_FEATURES = 60

TARGET = "result"

DRAW_WEIGHT = 1.0


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


POST_MATCH_COLUMNS = {
    "home_goals",
    "away_goals",
    "home_score",
    "away_score",
    "home_goals_for",
    "away_goals_for",
    "home_goals_against",
    "away_goals_against",
}


def build_folds(
    n: int,
) -> list[dict]:

    return [
        {
            "name": "Fold 1",
            "train_end": int(n * 0.50),
            "validation_end": int(n * 0.55),
            "test_end": int(n * 0.65),
        },
        {
            "name": "Fold 2",
            "train_end": int(n * 0.65),
            "validation_end": int(n * 0.70),
            "test_end": int(n * 0.80),
        },
        {
            "name": "Fold 3",
            "train_end": int(n * 0.80),
            "validation_end": int(n * 0.85),
            "test_end": int(n * 0.95),
        },
    ]


def prepare_features(
    df: pd.DataFrame,
) -> list[str]:

    excluded = (
        METADATA_COLUMNS
        | POST_MATCH_COLUMNS
    )

    return [
        column
        for column in df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]


def select_features(
    train: pd.DataFrame,
    features: list[str],
) -> list[str]:

    X = (
        train[features]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    mapping = {
        "A": 0,
        "D": 1,
        "H": 2,
    }

    y = train[TARGET].map(
        mapping
    )

    mi = mutual_info_classif(
        X,
        y,
        random_state=RANDOM_STATE,
    )

    ranking = (
        pd.DataFrame(
            {
                "feature": features,
                "mi": mi,
            }
        )
        .sort_values(
            "mi",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return ranking.head(
        TOP_FEATURES
    )["feature"].tolist()


def prepare_X(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:

    return (
        df[features]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )


def train_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> CatBoostClassifier:

    X_train = prepare_X(
        train,
        features,
    )

    X_validation = prepare_X(
        validation,
        features,
    )

    y_train = train[TARGET]

    y_validation = (
        validation[TARGET]
    )

    model = CatBoostClassifier(
        iterations=500,
        depth=5,
        learning_rate=0.05,
        l2_leaf_reg=7,
        random_strength=0.5,
        bagging_temperature=1.0,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        class_weights=[
            1.0,
            DRAW_WEIGHT,
            1.0,
        ],
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(
            X_validation,
            y_validation,
        ),
        early_stopping_rounds=50,
        verbose=False,
    )

    return model


def export_fold(
    df: pd.DataFrame,
    fold: dict,
) -> pd.DataFrame:

    train_end = fold[
        "train_end"
    ]

    validation_end = fold[
        "validation_end"
    ]

    test_end = fold[
        "test_end"
    ]

    train = df.iloc[
        :train_end
    ].copy()

    validation = df.iloc[
        train_end:validation_end
    ].copy()

    test = df.iloc[
        validation_end:test_end
    ].copy()

    print()
    print("=" * 90)
    print(
        f"{fold['name']}"
    )
    print("=" * 90)

    print(
        f"Train:      {len(train)}"
    )

    print(
        f"Validation: {len(validation)}"
    )

    print(
        f"Test:       {len(test)}"
    )

    all_features = prepare_features(
        train
    )

    selected_features = (
        select_features(
            train,
            all_features,
        )
    )

    print(
        f"Features:   "
        f"{len(selected_features)}"
    )

    print(
        "Top features:"
    )

    for feature in selected_features[
        :10
    ]:
        print(
            f"  - {feature}"
        )

    model = train_model(
        train,
        validation,
        selected_features,
    )

    print(
        f"Best iteration: "
        f"{model.get_best_iteration()}"
    )

    X_test = prepare_X(
        test,
        selected_features,
    )

    probabilities = (
        model.predict_proba(
            X_test
        )
    )

    predictions = (
        model.predict(
            X_test
        ).flatten()
    )

    classes = list(
        model.classes_
    )

    print(
        f"Model classes: "
        f"{classes}"
    )

    # На всякий случай получаем индексы
    # классов по их реальным названиям.
    class_index = {
        cls: index
        for index, cls in enumerate(
            classes
        )
    }

    rows: list[dict] = []

    for position, (
        (_, match),
        prediction,
    ) in enumerate(
        zip(
            test.iterrows(),
            predictions,
        )
    ):

        p_home = float(
            probabilities[
                position,
                class_index["H"],
            ]
        )

        p_draw = float(
            probabilities[
                position,
                class_index["D"],
            ]
        )

        p_away = float(
            probabilities[
                position,
                class_index["A"],
            ]
        )

        confidence = max(
            p_home,
            p_draw,
            p_away,
        )

        rows.append(
            {
                "fold": fold["name"],
                "draw_weight": DRAW_WEIGHT,

                "fixture_id": match[
                    "fixture_id"
                ],

                "kickoff": match[
                    "kickoff"
                ],

                "home_team_id": match[
                    "home_team_id"
                ],

                "away_team_id": match[
                    "away_team_id"
                ],

                "actual": str(
                    match[TARGET]
                ),

                "prediction": str(
                    prediction
                ),

                "p_home": p_home,
                "p_draw": p_draw,
                "p_away": p_away,

                "confidence": confidence,

                "best_iteration": (
                    model.get_best_iteration()
                ),

                "features_count": len(
                    selected_features
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    print()
    print(
        f"Exported TEST predictions: "
        f"{len(result)}"
    )

    print(
        f"Predicted draws: "
        f"{(result['prediction'] == 'D').sum()}"
    )

    print(
        f"Actual draws: "
        f"{(result['actual'] == 'D').sum()}"
    )

    return result


def main() -> None:

    print("=" * 90)
    print(
        "CATBOOST WALK-FORWARD "
        "PREDICTION EXPORT"
    )
    print("=" * 90)

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: "
            f"{DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH
    )

    required_columns = {
        "fixture_id",
        "kickoff",
        "home_team_id",
        "away_team_id",
        "result",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    df["kickoff"] = pd.to_datetime(
        df["kickoff"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "kickoff",
            "home_team_id",
            "away_team_id",
            "result",
        ]
    ).copy()

    df = (
        df.sort_values(
            "kickoff"
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        f"Dataset: {DATASET_PATH}"
    )

    print(
        f"Matches: {len(df)}"
    )

    print(
        f"Period: "
        f"{df['kickoff'].min()} -> "
        f"{df['kickoff'].max()}"
    )

    folds = build_folds(
        len(df)
    )

    all_predictions: list[
        pd.DataFrame
    ] = []

    for fold in folds:

        fold_predictions = (
            export_fold(
                df,
                fold,
            )
        )

        all_predictions.append(
            fold_predictions
        )

    result = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    # ========================================================
    # Проверки
    # ========================================================

    print()
    print("=" * 90)
    print(
        "EXPORT CHECKS"
    )
    print("=" * 90)

    print(
        f"Total predictions: "
        f"{len(result)}"
    )

    print(
        f"Unique fixture IDs: "
        f"{result['fixture_id'].nunique()}"
    )

    duplicates = (
        result["fixture_id"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate fixture IDs: "
        f"{duplicates}"
    )

    print()
    print(
        "Predictions by fold:"
    )

    print(
        result.groupby(
            "fold"
        ).size()
    )

    print()
    print(
        "Target distribution:"
    )

    print(
        result["actual"].value_counts()
    )

    print()
    print(
        "Prediction distribution:"
    )

    print(
        result["prediction"].value_counts()
    )

    # ========================================================
    # SAVE
    # ========================================================

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = result.sort_values(
        [
            "fold",
            "kickoff",
            "fixture_id",
        ]
    ).reset_index(
        drop=True
    )

    result.to_csv(
        REPORT_PATH,
        index=False,
    )

    print()
    print(
        f"Saved: {REPORT_PATH}"
    )

    print()
    print(
        "CATBOOST EXPORT DONE"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
