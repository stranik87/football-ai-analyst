from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


CATBOOST_PATH = Path(
    "data/reports/catboost_walk_forward_predictions.csv"
)

HYBRID_PATH = Path(
    "data/reports/hybrid_v2_test_predictions.csv"
)

OUTPUT_DIR = Path(
    "data/reports/error_analysis"
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:

    cat = pd.read_csv(
        CATBOOST_PATH
    )

    hybrid = pd.read_csv(
        HYBRID_PATH
    )

    required_cat = {
        "fold",
        "fixture_id",
        "actual",
        "prediction",
        "p_home",
        "p_draw",
        "p_away",
        "confidence",
    }

    required_hybrid = {
        "fold",
        "fixture_id",
        "actual",
        "prediction",
        "p_home",
        "p_draw",
        "p_away",
        "confidence",
    }

    missing_cat = (
        required_cat - set(cat.columns)
    )

    missing_hybrid = (
        required_hybrid - set(hybrid.columns)
    )

    if missing_cat:
        raise ValueError(
            f"CatBoost missing: "
            f"{sorted(missing_cat)}"
        )

    if missing_hybrid:
        raise ValueError(
            f"Hybrid missing: "
            f"{sorted(missing_hybrid)}"
        )

    cat["fixture_id"] = (
        cat["fixture_id"]
        .astype(str)
    )

    hybrid["fixture_id"] = (
        hybrid["fixture_id"]
        .astype(str)
    )

    return cat, hybrid


def confusion_table(
    actual: pd.Series,
    prediction: pd.Series,
) -> pd.DataFrame:

    labels = [
        "H",
        "D",
        "A",
    ]

    cm = confusion_matrix(
        actual,
        prediction,
        labels=labels,
    )

    return pd.DataFrame(
        cm,
        index=[
            "Actual H",
            "Actual D",
            "Actual A",
        ],
        columns=[
            "Pred H",
            "Pred D",
            "Pred A",
        ],
    )


def class_accuracy(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for cls in [
        "H",
        "D",
        "A",
    ]:

        actual = (
            df["actual"] == cls
        )

        total = int(
            actual.sum()
        )

        correct = int(
            (
                actual
                & (
                    df["prediction"]
                    == cls
                )
            ).sum()
        )

        recall = (
            correct / total
            if total
            else 0
        )

        rows.append(
            {
                "class": cls,
                "actual_matches": total,
                "correct_predictions": correct,
                "recall": recall,
            }
        )

    return pd.DataFrame(rows)


def fold_accuracy(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return (
        df.groupby("fold")
        .agg(
            matches=(
                "actual",
                "size",
            ),
            accuracy=(
                "correct",
                "mean",
            ),
        )
        .reset_index()
    )


def disagreement_analysis(
    cat: pd.DataFrame,
    hybrid: pd.DataFrame,
) -> pd.DataFrame:

    merged = cat[
        [
            "fold",
            "fixture_id",
            "actual",
            "prediction",
            "p_home",
            "p_draw",
            "p_away",
        ]
    ].rename(
        columns={
            "prediction": "cat_prediction",
            "p_home": "cat_p_home",
            "p_draw": "cat_p_draw",
            "p_away": "cat_p_away",
        }
    )

    hyb = hybrid[
        [
            "fixture_id",
            "prediction",
            "p_home",
            "p_draw",
            "p_away",
        ]
    ].rename(
        columns={
            "prediction": "hybrid_prediction",
            "p_home": "hybrid_p_home",
            "p_draw": "hybrid_p_draw",
            "p_away": "hybrid_p_away",
        }
    )

    merged = merged.merge(
        hyb,
        on="fixture_id",
        how="inner",
    )

    merged["cat_correct"] = (
        merged["cat_prediction"]
        == merged["actual"]
    )

    merged["hybrid_correct"] = (
        merged["hybrid_prediction"]
        == merged["actual"]
    )

    merged["changed"] = (
        merged["cat_prediction"]
        != merged["hybrid_prediction"]
    )

    return merged


def high_confidence_errors(
    df: pd.DataFrame,
) -> pd.DataFrame:

    errors = df[
        df["prediction"]
        != df["actual"]
    ].copy()

    return errors.sort_values(
        "confidence",
        ascending=False,
    )


def draw_analysis(
    df: pd.DataFrame,
) -> pd.DataFrame:

    draws = df[
        df["actual"] == "D"
    ].copy()

    draws["max_non_draw"] = np.maximum(
        draws["p_home"],
        draws["p_away"],
    )

    draws["draw_rank"] = (
        draws[
            [
                "p_home",
                "p_draw",
                "p_away",
            ]
        ]
        .rank(
            axis=1,
            ascending=False,
            method="min",
        )["p_draw"]
    )

    return draws.sort_values(
        "p_draw",
        ascending=False,
    )


def hybrid_changes(
    disagreement: pd.DataFrame,
) -> pd.DataFrame:

    changed = disagreement[
        disagreement["changed"]
    ].copy()

    changed["change_type"] = (
        changed["cat_prediction"]
        + " -> "
        + changed["hybrid_prediction"]
    )

    changed["hybrid_change_correct"] = (
        changed["hybrid_correct"]
        & ~changed["cat_correct"]
    )

    changed["hybrid_change_wrong"] = (
        ~changed["hybrid_correct"]
        & changed["cat_correct"]
    )

    return changed.sort_values(
        "hybrid_p_"
        + changed[
            "hybrid_prediction"
        ].iloc[0]
        if len(changed)
        else "fixture_id"
    )


def save_table(
    df: pd.DataFrame,
    name: str,
) -> None:

    path = (
        OUTPUT_DIR
        / name
    )

    df.to_csv(
        path,
        index=False,
    )

    print(
        f"Saved: {path}"
    )


def print_section(
    title: str,
) -> None:

    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def main():

    print("=" * 90)
    print(
        "ERROR ANALYSIS V1"
    )
    print("=" * 90)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cat, hybrid = load_data()

    print()
    print(
        f"CatBoost rows: "
        f"{len(cat)}"
    )

    print(
        f"Hybrid rows:   "
        f"{len(hybrid)}"
    )

    # ========================================================
    # CATBOOST CORRECT
    # ========================================================

    cat["correct"] = (
        cat["prediction"]
        == cat["actual"]
    )

    # ========================================================
    # 1. CONFUSION MATRIX
    # ========================================================

    print_section(
        "CATBOOST CONFUSION MATRIX"
    )

    cm_cat = confusion_table(
        cat["actual"],
        cat["prediction"],
    )

    print(
        cm_cat.to_string()
    )

    save_table(
        cm_cat.reset_index(),
        "catboost_confusion_matrix.csv",
    )

    # ========================================================
    # 2. CLASS ANALYSIS
    # ========================================================

    print_section(
        "CATBOOST CLASS ANALYSIS"
    )

    classes_cat = class_accuracy(
        cat
    )

    print(
        classes_cat.to_string(
            index=False
        )
    )

    save_table(
        classes_cat,
        "catboost_class_analysis.csv",
    )

    # ========================================================
    # 3. FOLD ANALYSIS
    # ========================================================

    print_section(
        "CATBOOST FOLD ANALYSIS"
    )

    folds = fold_accuracy(
        cat
    )

    print(
        folds.to_string(
            index=False
        )
    )

    save_table(
        folds,
        "catboost_fold_analysis.csv",
    )

    # ========================================================
    # 4. HIGH CONFIDENCE ERRORS
    # ========================================================

    print_section(
        "CATBOOST HIGH-CONFIDENCE ERRORS"
    )

    high_errors = (
        high_confidence_errors(cat)
        .head(30)
    )

    print(
        high_errors[
            [
                "fold",
                "fixture_id",
                "actual",
                "prediction",
                "p_home",
                "p_draw",
                "p_away",
                "confidence",
            ]
        ].to_string(
            index=False
        )
    )

    save_table(
        high_errors,
        "catboost_high_confidence_errors.csv",
    )

    # ========================================================
    # 5. DRAW ANALYSIS
    # ========================================================

    print_section(
        "CATBOOST DRAW ANALYSIS"
    )

    draws = draw_analysis(
        cat
    )

    print(
        draws[
            [
                "fold",
                "fixture_id",
                "actual",
                "prediction",
                "p_home",
                "p_draw",
                "p_away",
                "confidence",
                "draw_rank",
            ]
        ].head(30).to_string(
            index=False
        )
    )

    print()
    print(
        f"Actual draws: "
        f"{len(draws)}"
    )

    print(
        f"Mean P(D) on actual draws: "
        f"{draws['p_draw'].mean():.4f}"
    )

    print(
        f"Max P(D) on actual draws: "
        f"{draws['p_draw'].max():.4f}"
    )

    save_table(
        draws,
        "catboost_draw_analysis.csv",
    )

    # ========================================================
    # 6. CATBOOST vs HYBRID
    # ========================================================

    print_section(
        "CATBOOST vs HYBRID"
    )

    disagreement = disagreement_analysis(
        cat,
        hybrid,
    )

    print(
        f"Total matches: "
        f"{len(disagreement)}"
    )

    print(
        f"Changed prediction: "
        f"{disagreement['changed'].sum()}"
    )

    print(
        f"CatBoost correct: "
        f"{disagreement['cat_correct'].sum()}"
    )

    print(
        f"Hybrid correct: "
        f"{disagreement['hybrid_correct'].sum()}"
    )

    changed = disagreement[
        disagreement["changed"]
    ]

    if len(changed):

        print()
        print(
            "Hybrid changes:"
        )

        change_summary = (
            changed.groupby(
                [
                    "cat_prediction",
                    "hybrid_prediction",
                ]
            )
            .agg(
                matches=(
                    "fixture_id",
                    "count",
                ),
                hybrid_correct=(
                    "hybrid_correct",
                    "sum",
                ),
                cat_correct=(
                    "cat_correct",
                    "sum",
                ),
            )
            .reset_index()
        )

        print(
            change_summary.to_string(
                index=False
            )
        )

        save_table(
            change_summary,
            "hybrid_change_summary.csv",
        )

    save_table(
        disagreement,
        "catboost_hybrid_comparison.csv",
    )

    # ========================================================
    # 7. HYBRID CHANGES THAT MATTER
    # ========================================================

    print_section(
        "HYBRID CHANGES — CORRECT VS WRONG"
    )

    changed = disagreement[
        disagreement["changed"]
    ].copy()

    changed["result"] = np.where(
        changed["hybrid_correct"]
        & ~changed["cat_correct"],
        "HYBRID_FIXED",
        np.where(
            ~changed["hybrid_correct"]
            & changed["cat_correct"],
            "HYBRID_BROKE",
            "BOTH_WRONG",
        ),
    )

    print(
        changed[
            [
                "fold",
                "fixture_id",
                "actual",
                "cat_prediction",
                "hybrid_prediction",
                "cat_p_home",
                "cat_p_draw",
                "cat_p_away",
                "hybrid_p_home",
                "hybrid_p_draw",
                "hybrid_p_away",
                "result",
            ]
        ].to_string(
            index=False
        )
    )

    save_table(
        changed,
        "hybrid_changed_matches.csv",
    )

    # ========================================================
    # 8. HYBRID CONFUSION MATRIX
    # ========================================================

    print_section(
        "HYBRID CONFUSION MATRIX"
    )

    cm_hybrid = confusion_table(
        hybrid["actual"],
        hybrid["prediction"],
    )

    print(
        cm_hybrid.to_string()
    )

    save_table(
        cm_hybrid.reset_index(),
        "hybrid_confusion_matrix.csv",
    )

    # ========================================================
    # 9. SUMMARY
    # ========================================================

    print_section(
        "SUMMARY"
    )

    summary = pd.DataFrame(
        [
            {
                "model": "CatBoost",
                "accuracy": cat[
                    "correct"
                ].mean(),
                "matches": len(cat),
            },
            {
                "model": "Hybrid",
                "accuracy": (
                    hybrid["prediction"]
                    == hybrid["actual"]
                ).mean(),
                "matches": len(hybrid),
            },
        ]
    )

    print(
        summary.to_string(
            index=False
        )
    )

    save_table(
        summary,
        "error_analysis_summary.csv",
    )

    print()
    print("=" * 90)
    print(
        "ERROR ANALYSIS V1 DONE"
    )
    print("=" * 90)

    print()
    print(
        f"Reports directory: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
