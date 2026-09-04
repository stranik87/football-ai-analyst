from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


DATASET_PATH = Path(
    "data/datasets/matches_dataset.csv"
)

REPORT_PATH = Path(
    "data/reports/walk_forward_clean_results.csv"
)

SUMMARY_PATH = Path(
    "data/reports/walk_forward_clean_summary.csv"
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


# ============================================================
# CLEAN WALK-FORWARD
#
# Каждый TEST-период используется только один раз.
#
# Fold 1:
#   TRAIN      0-50%
#   VALIDATION 50-55%
#   TEST       55-65%
#
# Fold 2:
#   TRAIN      0-65%
#   VALIDATION 65-70%
#   TEST       70-80%
#
# Fold 3:
#   TRAIN      0-80%
#   VALIDATION 80-85%
#   TEST       85-95%
#
# Последние 5% оставляем нетронутыми
# для будущего финального holdout.
# ============================================================


def build_folds(n: int) -> list[dict]:

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

    y = train[TARGET].map(mapping)

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

    selected = ranking.head(
        TOP_FEATURES
    )["feature"].tolist()

    return selected


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


def evaluate(
    model: CatBoostClassifier,
    df: pd.DataFrame,
    features: list[str],
) -> dict:

    X = prepare_X(
        df,
        features,
    )

    y = df[TARGET]

    probabilities = (
        model.predict_proba(X)
    )

    predictions = (
        model.predict(X)
        .flatten()
    )

    accuracy = accuracy_score(
        y,
        predictions,
    )

    logloss = log_loss(
        y,
        probabilities,
        labels=list(
            model.classes_
        ),
    )

    draw_true = y == "D"
    draw_pred = predictions == "D"

    draw_precision = (
        precision_score(
            draw_true,
            draw_pred,
            zero_division=0,
        )
    )

    draw_recall = (
        recall_score(
            draw_true,
            draw_pred,
            zero_division=0,
        )
    )

    draw_f1 = (
        f1_score(
            draw_true,
            draw_pred,
            zero_division=0,
        )
    )

    home_true = y == "H"
    away_true = y == "A"

    home_accuracy = (
        (
            predictions[home_true]
            == "H"
        ).mean()
        if home_true.sum()
        else 0.0
    )

    away_accuracy = (
        (
            predictions[away_true]
            == "A"
        ).mean()
        if away_true.sum()
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "log_loss": logloss,
        "draw_precision": draw_precision,
        "draw_recall": draw_recall,
        "draw_f1": draw_f1,
        "predicted_draws": int(
            draw_pred.sum()
        ),
        "actual_draws": int(
            draw_true.sum()
        ),
        "correct_draws": int(
            (
                draw_true
                & draw_pred
            ).sum()
        ),
        "home_accuracy": home_accuracy,
        "away_accuracy": away_accuracy,
    }


def train_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    draw_weight: float,
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
            draw_weight,
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


def run_fold(
    df: pd.DataFrame,
    fold: dict,
    draw_weight: float,
) -> dict:

    train_end = fold["train_end"]

    validation_end = (
        fold["validation_end"]
    )

    test_end = fold["test_end"]

    train = df.iloc[
        :train_end
    ].copy()

    validation = df.iloc[
        train_end:validation_end
    ].copy()

    test = df.iloc[
        validation_end:test_end
    ].copy()

    all_features = prepare_features(
        df
    )

    leaked = (
        set(all_features)
        & POST_MATCH_COLUMNS
    )

    if leaked:
        raise RuntimeError(
            "ОБНАРУЖЕНА УТЕЧКА: "
            + ", ".join(
                sorted(leaked)
            )
        )

    selected_features = (
        select_features(
            train,
            all_features,
        )
    )

    model = train_model(
        train,
        validation,
        selected_features,
        draw_weight,
    )

    validation_metrics = evaluate(
        model,
        validation,
        selected_features,
    )

    test_metrics = evaluate(
        model,
        test,
        selected_features,
    )

    return {
        "fold": fold["name"],
        "draw_weight": draw_weight,

        "train_size": len(train),
        "validation_size": len(validation),
        "test_size": len(test),

        "train_start": train[
            "kickoff"
        ].min(),

        "train_end": train[
            "kickoff"
        ].max(),

        "validation_start": validation[
            "kickoff"
        ].min(),

        "validation_end": validation[
            "kickoff"
        ].max(),

        "test_start": test[
            "kickoff"
        ].min(),

        "test_end": test[
            "kickoff"
        ].max(),

        "features_count": len(
            selected_features
        ),

        "best_iteration": (
            model.get_best_iteration()
        ),

        "validation_accuracy":
            validation_metrics[
                "accuracy"
            ],

        "validation_log_loss":
            validation_metrics[
                "log_loss"
            ],

        "validation_draw_precision":
            validation_metrics[
                "draw_precision"
            ],

        "validation_draw_recall":
            validation_metrics[
                "draw_recall"
            ],

        "validation_draw_f1":
            validation_metrics[
                "draw_f1"
            ],

        "test_accuracy":
            test_metrics[
                "accuracy"
            ],

        "test_log_loss":
            test_metrics[
                "log_loss"
            ],

        "test_draw_precision":
            test_metrics[
                "draw_precision"
            ],

        "test_draw_recall":
            test_metrics[
                "draw_recall"
            ],

        "test_draw_f1":
            test_metrics[
                "draw_f1"
            ],

        "test_predicted_draws":
            test_metrics[
                "predicted_draws"
            ],

        "test_actual_draws":
            test_metrics[
                "actual_draws"
            ],

        "test_correct_draws":
            test_metrics[
                "correct_draws"
            ],

        "test_home_accuracy":
            test_metrics[
                "home_accuracy"
            ],

        "test_away_accuracy":
            test_metrics[
                "away_accuracy"
            ],
    }


def print_fold_result(
    result: dict,
) -> None:

    print(
        f"\n{result['fold']} "
        f"| D weight="
        f"{result['draw_weight']}"
    )

    print(
        f"Train:      "
        f"{result['train_size']}"
    )

    print(
        f"Validation: "
        f"{result['validation_size']}"
    )

    print(
        f"Test:       "
        f"{result['test_size']}"
    )

    print(
        f"Features:   "
        f"{result['features_count']}"
    )

    print(
        f"Best iter:  "
        f"{result['best_iteration']}"
    )

    print("\nValidation:")

    print(
        f"  Accuracy: "
        f"{result['validation_accuracy']:.2%}"
    )

    print(
        f"  LogLoss:  "
        f"{result['validation_log_loss']:.4f}"
    )

    print(
        f"  Draw Rec: "
        f"{result['validation_draw_recall']:.2%}"
    )

    print("\nTest:")

    print(
        f"  Accuracy: "
        f"{result['test_accuracy']:.2%}"
    )

    print(
        f"  LogLoss:  "
        f"{result['test_log_loss']:.4f}"
    )

    print(
        f"  Draw Precision: "
        f"{result['test_draw_precision']:.2%}"
    )

    print(
        f"  Draw Recall:    "
        f"{result['test_draw_recall']:.2%}"
    )

    print(
        f"  Draw F1:        "
        f"{result['test_draw_f1']:.2%}"
    )

    print(
        f"  Draws: "
        f"{result['test_correct_draws']}/"
        f"{result['test_actual_draws']}"
        f" "
        f"(predicted "
        f"{result['test_predicted_draws']})"
    )

    print(
        f"  H Accuracy: "
        f"{result['test_home_accuracy']:.2%}"
    )

    print(
        f"  A Accuracy: "
        f"{result['test_away_accuracy']:.2%}"
    )


def main() -> None:

    print("=" * 100)
    print(
        "CLEAN WALK-FORWARD "
        "CATBOOST EVALUATION"
    )
    print("=" * 100)

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset не найден: "
            f"{DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH
    )

    required_columns = {
        "kickoff",
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

    df = (
        df
        .dropna(
            subset=[
                "kickoff",
                TARGET,
            ]
        )
        .sort_values(
            "kickoff"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"\nDataset: "
        f"{DATASET_PATH}"
    )

    print(
        f"Matches: "
        f"{len(df)}"
    )

    print(
        f"Period: "
        f"{df['kickoff'].min()} -> "
        f"{df['kickoff'].max()}"
    )

    print("\nTarget:")

    print(
        df[TARGET].value_counts()
    )

    all_features = prepare_features(
        df
    )

    leaked = (
        set(all_features)
        & POST_MATCH_COLUMNS
    )

    if leaked:

        raise RuntimeError(
            "ОБНАРУЖЕНА УТЕЧКА: "
            + ", ".join(
                sorted(leaked)
            )
        )

    print(
        f"\nML features available: "
        f"{len(all_features)}"
    )

    folds = build_folds(
        len(df)
    )

    print("\n" + "=" * 100)
    print("FOLD STRUCTURE")
    print("=" * 100)

    for fold in folds:

        train_end = fold[
            "train_end"
        ]

        validation_end = fold[
            "validation_end"
        ]

        test_end = fold[
            "test_end"
        ]

        print(
            f"\n{fold['name']}:"
        )

        print(
            f"  Train: "
            f"0 -> {train_end}"
        )

        print(
            f"  Validation: "
            f"{train_end} -> "
            f"{validation_end}"
        )

        print(
            f"  Test: "
            f"{validation_end} -> "
            f"{test_end}"
        )

        train = df.iloc[
            :train_end
        ]

        validation = df.iloc[
            train_end:validation_end
        ]

        test = df.iloc[
            validation_end:test_end
        ]

        print(
            f"  Dates: "
            f"{train['kickoff'].min()} -> "
            f"{test['kickoff'].max()}"
        )

        print(
            f"  Sizes: "
            f"{len(train)} / "
            f"{len(validation)} / "
            f"{len(test)}"
        )

    print(
        "\nПоследние 5% dataset "
        "оставлены как финальный holdout."
    )

    all_results: list[dict] = []

    # Проверяем два наиболее интересных
    # варианта веса ничьей.
    for draw_weight in [1.0, 1.1]:

        print("\n" + "#" * 100)
        print(
            f"DRAW WEIGHT = "
            f"{draw_weight}"
        )
        print("#" * 100)

        for fold in folds:

            print(
                "\n" + "-" * 100
            )

            result = run_fold(
                df=df,
                fold=fold,
                draw_weight=draw_weight,
            )

            all_results.append(
                result
            )

            print_fold_result(
                result
            )

    results = pd.DataFrame(
        all_results
    )

    print("\n" + "=" * 100)
    print("DETAILED RESULTS")
    print("=" * 100)

    display_columns = [
        "draw_weight",
        "fold",
        "train_size",
        "validation_size",
        "test_size",
        "features_count",
        "best_iteration",
        "validation_accuracy",
        "validation_log_loss",
        "test_accuracy",
        "test_log_loss",
        "test_draw_precision",
        "test_draw_recall",
        "test_draw_f1",
        "test_predicted_draws",
        "test_correct_draws",
        "test_home_accuracy",
        "test_away_accuracy",
    ]

    print(
        results[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = (
        results
        .groupby("draw_weight")
        .agg(
            mean_validation_accuracy=(
                "validation_accuracy",
                "mean",
            ),
            mean_validation_log_loss=(
                "validation_log_loss",
                "mean",
            ),
            mean_test_accuracy=(
                "test_accuracy",
                "mean",
            ),
            mean_test_log_loss=(
                "test_log_loss",
                "mean",
            ),
            mean_test_draw_precision=(
                "test_draw_precision",
                "mean",
            ),
            mean_test_draw_recall=(
                "test_draw_recall",
                "mean",
            ),
            mean_test_draw_f1=(
                "test_draw_f1",
                "mean",
            ),
            mean_test_home_accuracy=(
                "test_home_accuracy",
                "mean",
            ),
            mean_test_away_accuracy=(
                "test_away_accuracy",
                "mean",
            ),
        )
        .reset_index()
    )

    print("\n" + "=" * 100)
    print("MEAN RESULTS")
    print("=" * 100)

    print(
        summary.to_string(
            index=False
        )
    )

    # ========================================================
    # STABILITY ПО FOLD
    # ========================================================

    print("\n" + "=" * 100)
    print("FOLD STABILITY")
    print("=" * 100)

    for draw_weight in [1.0, 1.1]:

        subset = results[
            results["draw_weight"]
            == draw_weight
        ]

        print(
            f"\nD weight={draw_weight}"
        )

        print(
            f"Test Accuracy "
            f"min/max: "
            f"{subset['test_accuracy'].min():.2%} / "
            f"{subset['test_accuracy'].max():.2%}"
        )

        print(
            f"Test LogLoss "
            f"min/max: "
            f"{subset['test_log_loss'].min():.4f} / "
            f"{subset['test_log_loss'].max():.4f}"
        )

        print(
            f"Test Draw Recall "
            f"min/max: "
            f"{subset['test_draw_recall'].min():.2%} / "
            f"{subset['test_draw_recall'].max():.2%}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        REPORT_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print("\n" + "=" * 100)
    print("FILES SAVED")
    print("=" * 100)

    print(
        f"Detailed: "
        f"{REPORT_PATH}"
    )

    print(
        f"Summary: "
        f"{SUMMARY_PATH}"
    )

    print("\n" + "=" * 100)
    print("CLEAN WALK-FORWARD DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
