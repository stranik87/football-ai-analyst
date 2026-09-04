from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss


CATBOOST_PATH = Path(
    "data/reports/catboost_walk_forward_predictions.csv"
)

POISSON_DIR = Path(
    "data/reports"
)

REPORT_PATH = Path(
    "data/reports/hybrid_v1_results.csv"
)

SUMMARY_PATH = Path(
    "data/reports/hybrid_v1_summary.csv"
)

WEIGHTS = np.arange(
    0.0,
    1.01,
    0.05,
)


def load_catboost() -> pd.DataFrame:

    if not CATBOOST_PATH.exists():
        raise FileNotFoundError(
            f"Not found: {CATBOOST_PATH}"
        )

    df = pd.read_csv(
        CATBOOST_PATH
    )

    required = {
        "fold",
        "fixture_id",
        "actual",
        "p_home",
        "p_draw",
        "p_away",
    }

    missing = (
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            "CatBoost file missing: "
            f"{sorted(missing)}"
        )

    return df


def load_poisson(
    alpha: float,
    fold: str,
) -> pd.DataFrame:

    filename = (
        f"poisson_v3_{alpha}_"
        f"{fold.lower().replace(' ', '_')}_"
        f"test.csv"
    )

    path = POISSON_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Not found: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "fixture_id",
        "actual",
        "p_home",
        "p_draw",
        "p_away",
    }

    missing = (
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Poisson file {path} "
            f"missing: {sorted(missing)}"
        )

    return df


def evaluate(
    merged: pd.DataFrame,
    cat_weight: float,
) -> dict[str, float]:

    poisson_weight = (
        1.0 - cat_weight
    )

    p_home = (
        cat_weight * merged["cat_p_home"]
        + poisson_weight * merged["poi_p_home"]
    )

    p_draw = (
        cat_weight * merged["cat_p_draw"]
        + poisson_weight * merged["poi_p_draw"]
    )

    p_away = (
        cat_weight * merged["cat_p_away"]
        + poisson_weight * merged["poi_p_away"]
    )

    # На случай очень небольших floating-point
    # отклонений нормализуем вероятности.
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

    predictions = np.where(
        (p_home >= p_draw)
        & (p_home >= p_away),
        "H",
        np.where(
            p_draw >= p_away,
            "D",
            "A",
        ),
    )

    actual = merged["actual"].to_numpy()

    accuracy = accuracy_score(
        actual,
        predictions,
    )

    logloss = log_loss(
        actual,
        probabilities,
        labels=["A", "D", "H"],
    )

    actual_draw = (
        actual == "D"
    )

    predicted_draw = (
        predictions == "D"
    )

    draw_recall = (
        (
            predictions[actual_draw]
            == "D"
        ).mean()
        if actual_draw.any()
        else 0.0
    )

    draw_precision = (
        (
            actual[predicted_draw]
            == "D"
        ).mean()
        if predicted_draw.any()
        else 0.0
    )

    if (
        draw_precision
        + draw_recall
        > 0
    ):
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
        "catboost_weight": cat_weight,
        "poisson_weight": poisson_weight,
        "matches": len(merged),
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


def merge_models(
    catboost: pd.DataFrame,
    poisson: pd.DataFrame,
) -> pd.DataFrame:

    cat = catboost[
        [
            "fixture_id",
            "fold",
            "actual",
            "p_home",
            "p_draw",
            "p_away",
        ]
    ].rename(
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
    ].rename(
        columns={
            "p_home": "poi_p_home",
            "p_draw": "poi_p_draw",
            "p_away": "poi_p_away",
        }
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
            "Количество матчей после "
            "merge отличается от CatBoost."
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


def main() -> None:

    print("=" * 90)
    print(
        "HYBRID V1"
    )
    print("=" * 90)

    catboost = load_catboost()

    print()
    print(
        f"CatBoost predictions: "
        f"{len(catboost)}"
    )

    print(
        f"Unique fixtures: "
        f"{catboost['fixture_id'].nunique()}"
    )

    all_results: list[dict] = []

    for alpha in [
        0.1,
        0.3,
        1.0,
    ]:

        print()
        print("#" * 90)
        print(
            f"POISSON ALPHA = {alpha}"
        )
        print("#" * 90)

        for fold in [
            "Fold 1",
            "Fold 2",
            "Fold 3",
        ]:

            poisson = load_poisson(
                alpha,
                fold,
            )

            cat_fold = catboost[
                catboost["fold"] == fold
            ].copy()

            merged = merge_models(
                cat_fold,
                poisson,
            )

            print()
            print(
                f"{fold}: "
                f"{len(merged)} matched"
            )

            for weight in WEIGHTS:

                metrics = evaluate(
                    merged,
                    float(weight),
                )

                all_results.append(
                    {
                        "alpha": alpha,
                        "fold": fold,
                        **metrics,
                    }
                )

    results = pd.DataFrame(
        all_results
    )

    # ========================================================
    # MEAN ACROSS THREE TEST FOLDS
    # ========================================================

    summary = (
        results
        .groupby(
            [
                "alpha",
                "catboost_weight",
                "poisson_weight",
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

    # ========================================================
    # PRINT BEST BY LOGLOSS
    # ========================================================

    print()
    print("=" * 90)
    print(
        "BEST HYBRID BY MEAN TEST LOGLOSS"
    )
    print("=" * 90)

    best_logloss = (
        summary
        .sort_values(
            "mean_logloss"
        )
        .head(15)
    )

    print(
        best_logloss.to_string(
            index=False
        )
    )

    # ========================================================
    # PRINT BEST BY ACCURACY
    # ========================================================

    print()
    print("=" * 90)
    print(
        "BEST HYBRID BY MEAN TEST ACCURACY"
    )
    print("=" * 90)

    best_accuracy = (
        summary
        .sort_values(
            "mean_accuracy",
            ascending=False,
        )
        .head(15)
    )

    print(
        best_accuracy.to_string(
            index=False
        )
    )

    # ========================================================
    # BEST ALPHA BY VALIDATION
    #
    # Важно:
    # alpha выбираем не по TEST.
    # Здесь просто показываем результаты
    # для последующего анализа.
    # ========================================================

    print()
    print("=" * 90)
    print(
        "ALPHA COMPARISON"
    )
    print("=" * 90)

    alpha_summary = (
        summary
        .groupby("alpha")
        .agg(
            best_logloss=(
                "mean_logloss",
                "min",
            ),
            best_accuracy=(
                "mean_accuracy",
                "max",
            ),
        )
        .reset_index()
    )

    print(
        alpha_summary.to_string(
            index=False
        )
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

    print()
    print(
        f"Detailed results: "
        f"{REPORT_PATH}"
    )

    print(
        f"Summary: "
        f"{SUMMARY_PATH}"
    )

    print()
    print("=" * 90)
    print(
        "HYBRID V1 DONE"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
