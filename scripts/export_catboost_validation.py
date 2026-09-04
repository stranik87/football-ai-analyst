from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_selection import mutual_info_classif


DATASET_PATH = Path(
    "data/datasets/matches_dataset.csv"
)

OUTPUT_PATH = Path(
    "data/reports/catboost_walk_forward_validation.csv"
)

RANDOM_STATE = 42
TOP_FEATURES = 60
TARGET = "result"

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


def build_folds(n: int):
    return [
        {
            "fold": "Fold 1",
            "train_end": int(n * 0.50),
            "val_end": int(n * 0.55),
            "test_end": int(n * 0.65),
        },
        {
            "fold": "Fold 2",
            "train_end": int(n * 0.65),
            "val_end": int(n * 0.70),
            "test_end": int(n * 0.80),
        },
        {
            "fold": "Fold 3",
            "train_end": int(n * 0.80),
            "val_end": int(n * 0.85),
            "test_end": int(n * 0.95),
        },
    ]


def prepare_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    X = df.copy()

    columns_to_drop = (
        METADATA_COLUMNS
        | POST_MATCH_COLUMNS
    )

    columns_to_drop = [
        c for c in columns_to_drop
        if c in X.columns
    ]

    X = X.drop(
        columns=columns_to_drop
    )

    # Только числовые признаки.
    X = X.select_dtypes(
        include=["number"]
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(0)

    return X


def select_features(
    train: pd.DataFrame,
    target: pd.Series,
) -> list[str]:

    X = prepare_features(train)

    y = target.map(
        {
            "A": 0,
            "D": 1,
            "H": 2,
        }
    ).astype(int)

    mi = mutual_info_classif(
        X,
        y,
        random_state=RANDOM_STATE,
    )

    scores = pd.DataFrame(
        {
            "feature": X.columns,
            "mi": mi,
        }
    ).sort_values(
        "mi",
        ascending=False,
    )

    return scores.head(
        min(TOP_FEATURES, len(scores))
    )["feature"].tolist()


def prepare_X(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:

    X = prepare_features(df)

    for feature in features:

        if feature not in X.columns:
            X[feature] = 0

    return X[features]


def train_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> CatBoostClassifier:

    X_train = prepare_X(
        train,
        features,
    )

    X_val = prepare_X(
        validation,
        features,
    )

    y_train = train[TARGET]
    y_val = validation[TARGET]

    model = CatBoostClassifier(
        iterations=500,
        depth=5,
        learning_rate=0.05,
        l2_leaf_reg=7,
        random_strength=0.5,
        bagging_temperature=1.0,
        loss_function="MultiClass",
        class_weights=[
            1.0,
            1.0,
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
            X_val,
            y_val,
        ),
        early_stopping_rounds=50,
        verbose=False,
    )

    return model


def main() -> None:

    print("=" * 90)
    print(
        "CATBOOST VALIDATION PREDICTIONS"
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

    all_predictions = []

    for fold_info in folds:

        fold = fold_info["fold"]

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
        print("-" * 90)
        print(
            f"{fold}: "
            f"train={len(train)}, "
            f"validation={len(validation)}"
        )

        features = select_features(
            train,
            train[TARGET],
        )

        print(
            f"Features selected: "
            f"{len(features)}"
        )

        model = train_model(
            train,
            validation,
            features,
        )

        X_val = prepare_X(
            validation,
            features,
        )

        probabilities = (
            model.predict_proba(X_val)
        )

        class_names = list(
            model.classes_
        )

        probability_map = {
            cls: probabilities[:, i]
            for i, cls in enumerate(
                class_names
            )
        }

        p_home = probability_map.get(
            "H",
            np.zeros(len(validation)),
        )

        p_draw = probability_map.get(
            "D",
            np.zeros(len(validation)),
        )

        p_away = probability_map.get(
            "A",
            np.zeros(len(validation)),
        )

        predictions = np.array(
            class_names
        )[
            np.argmax(
                probabilities,
                axis=1,
            )
        ]

        result = pd.DataFrame(
            {
                "fold": fold,
                "fixture_id": validation[
                    "fixture_id"
                ].values,
                "kickoff": validation[
                    "kickoff"
                ].values,
                "home_team_id": validation[
                    "home_team_id"
                ].values,
                "away_team_id": validation[
                    "away_team_id"
                ].values,
                "actual": validation[
                    TARGET
                ].values,
                "prediction": predictions,
                "p_home": p_home,
                "p_draw": p_draw,
                "p_away": p_away,
                "confidence": np.maximum.reduce(
                    [
                        p_home,
                        p_draw,
                        p_away,
                    ]
                ),
                "best_iteration": (
                    model.get_best_iteration()
                ),
                "features_count": len(
                    features
                ),
            }
        )

        all_predictions.append(
            result
        )

        print(
            f"Best iteration: "
            f"{model.get_best_iteration()}"
        )

    output = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 90)
    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print(
        f"Rows: {len(output)}"
    )
    print(
        f"Unique fixtures: "
        f"{output['fixture_id'].nunique()}"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
