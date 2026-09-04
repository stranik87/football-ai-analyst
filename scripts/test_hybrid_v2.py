from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss


CATBOOST_VALIDATION_PATH = Path(
    "data/reports/catboost_walk_forward_validation.csv"
)

CATBOOST_TEST_PATH = Path(
    "data/reports/catboost_walk_forward_predictions.csv"
)

POISSON_DIR = Path(
    "data/reports"
)

RESULTS_PATH = Path(
    "data/reports/hybrid_v2_results.csv"
)

SUMMARY_PATH = Path(
    "data/reports/hybrid_v2_summary.csv"
)

PREDICTIONS_PATH = Path(
    "data/reports/hybrid_v2_test_predictions.csv"
)

WEIGHTS = np.arange(
    0.0,
    1.01,
    0.05,
)


REQUIRED_COLUMNS = {
    "fixture_id",
    "actual",
    "p_home",
    "p_draw",
    "p_away",
}


def load_csv(path: Path) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    return pd.read_csv(path)


def validate_columns(
    df: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} missing columns: "
            f"{sorted(missing)}"
        )


def calculate_metrics(
    df: pd.DataFrame,
    weight: float,
) -> dict[str, float]:

    poisson_weight = 1.0 - weight

    p_home = (
        weight * df["cat_p_home"].to_numpy()
        + poisson_weight
        * df["poi_p_home"].to_numpy()
    )

    p_draw = (
        weight * df["cat_p_draw"].to_numpy()
        + poisson_weight
        * df["poi_p_draw"].to_numpy()
    )

    p_away = (
        weight * df["cat_p_away"].to_numpy()
        + poisson_weight
        * df["poi_p_away"].to_numpy()
    )

    total = (
        p_home
        + p_draw
        + p_away
    )

    p_home = p_home / total
    p_draw = p_draw / total
    p_away = p_away / total

    probabilities = np.column_stack(
        [
            p_away,
            p_draw,
            p_home,
        ]
    )

    prediction = np.where(
        (p_home >= p_draw)
        & (p_home >= p_away),
        "H",
        np.where(
            p_draw >= p_away,
            "D",
            "A",
        ),
    )

    actual = df["actual"].to_numpy()

    accuracy = accuracy_score(
        actual,
        prediction,
    )

    logloss = log_loss(
        actual,
        probabilities,
        labels=["A", "D", "H"],
    )

    actual_draw = actual == "D"
    predicted_draw = prediction == "D"

    draw_recall = (
        (
            prediction[actual_draw] == "D"
        ).mean()
        if actual_draw.any()
        else 0.0
    )

    draw_precision = (
        (
            actual[predicted_draw] == "D"
        ).mean()
        if predicted_draw.any()
        else 0.0
    )

    if draw_precision + draw_recall > 0:

        draw_f1 = (
            2
            * draw_precision
            * draw_recall
            / (
                draw_precision
                + draw_recall
            )
        )

    else:

        draw_f1 = 0.0

    return {
        "weight_catboost": float(weight),
        "weight_poisson": float(
            poisson_weight
        ),
        "matches": len(df),
        "accuracy": float(accuracy),
        "logloss": float(logloss),
        "draw_precision": float(
            draw_precision
        ),
        "draw_recall": float(
            draw_recall
        ),
        "draw_f1": float(
            draw_f1
        ),
        "predicted_draws": int(
            predicted_draw.sum()
        ),
        "actual_draws": int(
            actual_draw.sum()
        ),
    }


def prepare_merge(
    catboost: pd.DataFrame,
    poisson: pd.DataFrame,
) -> pd.DataFrame:

    cat = catboost[
        [
            "fixture_id",
            "actual",
            "p_home",
            "p_draw",
            "p_away",
        ]
    ].copy()

    cat = cat.rename(
        columns={
            "p_home": "cat_p_home",
            "p_draw": "cat_p_draw",
            "p_away": "cat_p_away",
        }
    )

    poi = poisson[
        [
            "fixture_id",
            "actual",
            "p_home",
            "p_draw",
            "p_away",
        ]
    ].copy()

    poi = poi.rename(
        columns={
            "p_home": "poi_p_home",
            "p_draw": "poi_p_draw",
            "p_away": "poi_p_away",
        }
    )

    cat["fixture_id"] = (
        cat["fixture_id"]
        .astype(str)
    )

    poi["fixture_id"] = (
        poi["fixture_id"]
        .astype(str)
    )

    if cat["fixture_id"].duplicated().any():
        raise ValueError(
            "Duplicate fixture_id in CatBoost."
        )

    if poi["fixture_id"].duplicated().any():
        raise ValueError(
            "Duplicate fixture_id in Poisson."
        )

    merged = cat.merge(
        poi,
        on="fixture_id",
        how="inner",
        suffixes=(
            "_cat",
            "_poi",
        ),
    )

    if len(merged) != len(cat):

        raise ValueError(
            "Merge потерял матчи: "
            f"CatBoost={len(cat)}, "
            f"merged={len(merged)}"
        )

    if (
        merged["actual_cat"]
        != merged["actual_poi"]
    ).any():

        raise ValueError(
            "Actual result mismatch "
            "between CatBoost and Poisson."
        )

    merged["actual"] = (
        merged["actual_cat"]
    )

    return merged


def poisson_path(
    alpha: float,
    fold: int,
    split: str,
) -> Path:

    return (
        POISSON_DIR
        / (
            f"poisson_v3_{alpha}_"
            f"fold_{fold}_"
            f"{split}.csv"
        )
    )


def load_poisson(
    alpha: float,
    fold: int,
    split: str,
) -> pd.DataFrame:

    path = poisson_path(
        alpha,
        fold,
        split,
    )

    df = load_csv(path)

    validate_columns(
        df,
        REQUIRED_COLUMNS,
        f"Poisson {path}",
    )

    return df


def evaluate_validation(
    catboost: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for alpha in [
        0.1,
        0.3,
        1.0,
    ]:

        for fold in [
            1,
            2,
            3,
        ]:

            fold_name = (
                f"Fold {fold}"
            )

            cat_fold = catboost[
                catboost["fold"]
                == fold_name
            ].copy()

            poisson = load_poisson(
                alpha,
                fold,
                "validation",
            )

            merged = prepare_merge(
                cat_fold,
                poisson,
            )

            print(
                f"Validation {fold_name}, "
                f"alpha={alpha}: "
                f"{len(merged)} matched"
            )

            for weight in WEIGHTS:

                metrics = calculate_metrics(
                    merged,
                    float(weight),
                )

                rows.append(
                    {
                        "alpha": alpha,
                        "fold": fold_name,
                        **metrics,
                    }
                )

    return pd.DataFrame(rows)


def evaluate_test(
    catboost: pd.DataFrame,
    alpha: float,
    weight: float,
):

    rows = []
    prediction_rows = []

    for fold in [
        1,
        2,
        3,
    ]:

        fold_name = (
            f"Fold {fold}"
        )

        cat_fold = catboost[
            catboost["fold"]
            == fold_name
        ].copy()

        poisson = load_poisson(
            alpha,
            fold,
            "test",
        )

        merged = prepare_merge(
            cat_fold,
            poisson,
        )

        metrics = calculate_metrics(
            merged,
            weight,
        )

        metrics["alpha"] = alpha
        metrics["fold"] = fold_name

        rows.append(metrics)

        w = weight
        pw = 1.0 - w

        p_home = (
            w * merged["cat_p_home"].to_numpy()
            + pw * merged["poi_p_home"].to_numpy()
        )

        p_draw = (
            w * merged["cat_p_draw"].to_numpy()
            + pw * merged["poi_p_draw"].to_numpy()
        )

        p_away = (
            w * merged["cat_p_away"].to_numpy()
            + pw * merged["poi_p_away"].to_numpy()
        )

        total = (
            p_home
            + p_draw
            + p_away
        )

        p_home /= total
        p_draw /= total
        p_away /= total

        prediction = np.where(
            (p_home >= p_draw)
            & (p_home >= p_away),
            "H",
            np.where(
                p_draw >= p_away,
                "D",
                "A",
            ),
        )

        detailed = pd.DataFrame(
            {
                "fold": fold_name,
                "fixture_id": merged[
                    "fixture_id"
                ],
                "actual": merged[
                    "actual"
                ],
                "prediction": prediction,
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
            }
        )

        prediction_rows.append(
            detailed
        )

    return (
        pd.DataFrame(rows),
        pd.concat(
            prediction_rows,
            ignore_index=True,
        ),
    )


def main():

    print("=" * 90)
    print(
        "HYBRID V2 — VALIDATION-BASED WEIGHT SELECTION"
    )
    print("=" * 90)

    # ========================================================
    # LOAD VALIDATION
    # ========================================================

    catboost_validation = load_csv(
        CATBOOST_VALIDATION_PATH
    )

    validate_columns(
        catboost_validation,
        REQUIRED_COLUMNS | {"fold"},
        "CatBoost validation",
    )

    print()
    print(
        f"CatBoost validation rows: "
        f"{len(catboost_validation)}"
    )

    # ========================================================
    # VALIDATION GRID
    # ========================================================

    print()
    print("=" * 90)
    print(
        "VALIDATION GRID"
    )
    print("=" * 90)

    validation_results = (
        evaluate_validation(
            catboost_validation
        )
    )

    validation_summary = (
        validation_results
        .groupby(
            [
                "alpha",
                "weight_catboost",
                "weight_poisson",
            ]
        )
        .agg(
            mean_accuracy=(
                "accuracy",
                "mean",
            ),
            mean_logloss=(
                "logloss",
                "mean",
            ),
            mean_draw_precision=(
                "draw_precision",
                "mean",
            ),
            mean_draw_recall=(
                "draw_recall",
                "mean",
            ),
            mean_draw_f1=(
                "draw_f1",
                "mean",
            ),
            mean_predicted_draws=(
                "predicted_draws",
                "mean",
            ),
        )
        .reset_index()
    )

    # Главный критерий — LogLoss.
    # Accuracy выводим дополнительно.
    best = (
        validation_summary
        .sort_values(
            [
                "mean_logloss",
                "mean_accuracy",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .iloc[0]
    )

    best_alpha = float(
        best["alpha"]
    )

    best_weight = float(
        best["weight_catboost"]
    )

    print()
    print("=" * 90)
    print(
        "BEST CONFIGURATION FROM VALIDATION"
    )
    print("=" * 90)

    print(
        f"Alpha:              {best_alpha}"
    )

    print(
        f"CatBoost weight:    "
        f"{best_weight:.2f}"
    )

    print(
        f"Poisson weight:     "
        f"{1.0 - best_weight:.2f}"
    )

    print(
        f"Validation accuracy:"
        f" {best['mean_accuracy']:.4f}"
    )

    print(
        f"Validation LogLoss: "
        f"{best['mean_logloss']:.4f}"
    )

    print()
    print(
        "Top 15 validation configurations:"
    )

    print(
        validation_summary
        .sort_values(
            "mean_logloss"
        )
        .head(15)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # LOCK CONFIGURATION
    # ========================================================

    print()
    print("=" * 90)
    print(
        "CONFIGURATION LOCKED"
    )
    print("=" * 90)

    print(
        "Теперь alpha и вес НЕ меняются "
        "при проверке TEST."
    )

    # ========================================================
    # TEST
    # ========================================================

    catboost_test = load_csv(
        CATBOOST_TEST_PATH
    )

    validate_columns(
        catboost_test,
        REQUIRED_COLUMNS | {"fold"},
        "CatBoost test",
    )

    print()
    print("=" * 90)
    print(
        "FINAL TEST"
    )
    print("=" * 90)

    test_results, test_predictions = (
        evaluate_test(
            catboost_test,
            best_alpha,
            best_weight,
        )
    )

    print()
    print(
        test_results.to_string(
            index=False
        )
    )

    test_summary = {
        "alpha": best_alpha,
        "weight_catboost": best_weight,
        "weight_poisson": 1.0 - best_weight,
        "mean_accuracy": test_results[
            "accuracy"
        ].mean(),
        "mean_logloss": test_results[
            "logloss"
        ].mean(),
        "mean_draw_precision": test_results[
            "draw_precision"
        ].mean(),
        "mean_draw_recall": test_results[
            "draw_recall"
        ].mean(),
        "mean_draw_f1": test_results[
            "draw_f1"
        ].mean(),
        "mean_predicted_draws": test_results[
            "predicted_draws"
        ].mean(),
    }

    print()
    print("-" * 90)

    print(
        f"TEST Accuracy:       "
        f"{test_summary['mean_accuracy']:.4f}"
    )

    print(
        f"TEST LogLoss:        "
        f"{test_summary['mean_logloss']:.4f}"
    )

    print(
        f"TEST Draw Precision: "
        f"{test_summary['mean_draw_precision']:.4f}"
    )

    print(
        f"TEST Draw Recall:    "
        f"{test_summary['mean_draw_recall']:.4f}"
    )

    print(
        f"TEST Draw F1:        "
        f"{test_summary['mean_draw_f1']:.4f}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_results.to_csv(
        RESULTS_PATH,
        index=False,
    )

    pd.DataFrame(
        [test_summary]
    ).to_csv(
        SUMMARY_PATH,
        index=False,
    )

    test_predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print()
    print(
        f"Results:     {RESULTS_PATH}"
    )

    print(
        f"Summary:     {SUMMARY_PATH}"
    )

    print(
        f"Predictions: {PREDICTIONS_PATH}"
    )

    print()
    print("=" * 90)
    print(
        "HYBRID V2 DONE"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
