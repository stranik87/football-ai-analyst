from pathlib import Path

import joblib
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


DATASET_PATH = Path("data/datasets/matches_dataset.csv")
REPORT_PATH = Path("data/reports/draw_features_results.csv")

RANDOM_STATE = 42
TOP_FEATURES = 60

TARGET = "result"

# ============================================================
# КОЛОНКИ, КОТОРЫЕ НЕЛЬЗЯ ИСПОЛЬЗОВАТЬ ДЛЯ ПРОГНОЗА
# ============================================================

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

# Критически важно:
# это данные, которые становятся известны только ПОСЛЕ матча.
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


def add_draw_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ========================================================
    # 1. СХОЖЕСТЬ СИЛЫ КОМАНД
    # ========================================================

    if (
        "home_points_per_match" in df.columns
        and "away_points_per_match" in df.columns
    ):
        df["draw_points_similarity"] = (
            1
            - (
                (
                    df["home_points_per_match"]
                    - df["away_points_per_match"]
                ).abs()
                / (
                    df["home_points_per_match"]
                    + df["away_points_per_match"]
                    + 1e-6
                )
            )
        ).clip(0, 1)

    if (
        "home_average_goals_for" in df.columns
        and "away_average_goals_for" in df.columns
    ):
        df["draw_attack_similarity"] = (
            1
            - (
                (
                    df["home_average_goals_for"]
                    - df["away_average_goals_for"]
                ).abs()
                / (
                    df["home_average_goals_for"]
                    + df["away_average_goals_for"]
                    + 1e-6
                )
            )
        ).clip(0, 1)

    if (
        "home_average_goals_against" in df.columns
        and "away_average_goals_against" in df.columns
    ):
        df["draw_defense_similarity"] = (
            1
            - (
                (
                    df["home_average_goals_against"]
                    - df["away_average_goals_against"]
                ).abs()
                / (
                    df["home_average_goals_against"]
                    + df["away_average_goals_against"]
                    + 1e-6
                )
            )
        ).clip(0, 1)

    # ========================================================
    # 2. ОБЩАЯ СХОЖЕСТЬ КОМАНД
    # ========================================================

    similarity_columns = []

    pairs = [
        ("home_points_per_match", "away_points_per_match"),
        ("home_average_goals_for", "away_average_goals_for"),
        ("home_average_goals_against", "away_average_goals_against"),
        ("home_average_total_shots", "away_average_total_shots"),
        ("home_average_shots_on_goal", "away_average_shots_on_goal"),
        ("home_average_possession", "away_average_possession"),
        ("home_average_pass_accuracy", "away_average_pass_accuracy"),
        ("home_save_percentage", "away_save_percentage"),
    ]

    for home_col, away_col in pairs:
        if home_col in df.columns and away_col in df.columns:
            name = f"similarity_{home_col.replace('home_', '')}"

            denominator = (
                df[home_col].abs()
                + df[away_col].abs()
                + 1e-6
            )

            df[name] = (
                1
                - (
                    (df[home_col] - df[away_col]).abs()
                    / denominator
                )
            ).clip(0, 1)

            similarity_columns.append(name)

    if similarity_columns:
        df["draw_overall_similarity"] = (
            df[similarity_columns].mean(axis=1)
        )

    # ========================================================
    # 3. БАЛАНС УДАРОВ
    # ========================================================

    if (
        "home_average_total_shots" in df.columns
        and "away_average_total_shots" in df.columns
    ):
        df["draw_shots_balance"] = (
            1
            - (
                (
                    df["home_average_total_shots"]
                    - df["away_average_total_shots"]
                ).abs()
                / (
                    df["home_average_total_shots"]
                    + df["away_average_total_shots"]
                    + 1e-6
                )
            )
        ).clip(0, 1)

    if (
        "home_average_shots_on_goal" in df.columns
        and "away_average_shots_on_goal" in df.columns
    ):
        df["draw_sot_balance"] = (
            1
            - (
                (
                    df["home_average_shots_on_goal"]
                    - df["away_average_shots_on_goal"]
                ).abs()
                / (
                    df["home_average_shots_on_goal"]
                    + df["away_average_shots_on_goal"]
                    + 1e-6
                )
            )
        ).clip(0, 1)

    # ========================================================
    # 4. ВЛАДЕНИЕ
    # ========================================================

    if (
        "home_average_possession" in df.columns
        and "away_average_possession" in df.columns
    ):
        df["draw_possession_balance"] = (
            1
            - (
                (
                    df["home_average_possession"]
                    - df["away_average_possession"]
                ).abs()
                / 100.0
            )
        ).clip(0, 1)

        average_possession = (
            df["home_average_possession"]
            + df["away_average_possession"]
        ) / 2

        df["draw_possession_centered"] = (
            1
            - (
                (average_possession - 50).abs()
                / 50
            )
        ).clip(0, 1)

    # ========================================================
    # 5. ПЕРЕДАЧИ
    # ========================================================

    if (
        "home_average_total_passes" in df.columns
        and "away_average_total_passes" in df.columns
    ):
        df["draw_passes_balance"] = (
            1
            - (
                (
                    df["home_average_total_passes"]
                    - df["away_average_total_passes"]
                ).abs()
                / (
                    df["home_average_total_passes"]
                    + df["away_average_total_passes"]
                    + 1e-6
                )
            )
        ).clip(0, 1)

    if (
        "home_average_accurate_passes" in df.columns
        and "away_average_accurate_passes" in df.columns
    ):
        df["draw_accurate_passes_balance"] = (
            1
            - (
                (
                    df["home_average_accurate_passes"]
                    - df["away_average_accurate_passes"]
                ).abs()
                / (
                    df["home_average_accurate_passes"]
                    + df["away_average_accurate_passes"]
                    + 1e-6
                )
            )
        ).clip(0, 1)

    # ========================================================
    # 6. БАЛАНС ГОЛОВ В ИСТОРИЧЕСКИХ МАТЧАХ
    #
    # Здесь используются НЕ home_goals / away_goals текущего
    # матча, а средние показатели прошлых матчей.
    # ========================================================

    if (
        "home_average_goals_for" in df.columns
        and "away_average_goals_for" in df.columns
    ):
        df["draw_goals_for_balance"] = (
            1
            - (
                (
                    df["home_average_goals_for"]
                    - df["away_average_goals_for"]
                ).abs()
                / (
                    df["home_average_goals_for"]
                    + df["away_average_goals_for"]
                    + 1e-6
                )
            )
        ).clip(0, 1)

    if (
        "home_average_goals_against" in df.columns
        and "away_average_goals_against" in df.columns
    ):
        df["draw_goals_against_balance"] = (
            1
            - (
                (
                    df["home_average_goals_against"]
                    - df["away_average_goals_against"]
                ).abs()
                / (
                    df["home_average_goals_against"]
                    + df["away_average_goals_against"]
                    + 1e-6
                )
            )
        ).clip(0, 1)

    # ========================================================
    # 7. НИЗКАЯ РАЗНИЦА СИЛЫ
    # ========================================================

    if "points_difference" in df.columns:
        df["draw_low_points_difference"] = (
            1 / (1 + df["points_difference"].abs())
        )

    if "goals_for_difference" in df.columns:
        df["draw_low_goals_difference"] = (
            1 / (1 + df["goals_for_difference"].abs())
        )

    if "total_shots_difference" in df.columns:
        df["draw_low_shots_difference"] = (
            1 / (1 + df["total_shots_difference"].abs())
        )

    if "possession_difference" in df.columns:
        df["draw_low_possession_difference"] = (
            1 / (1 + df["possession_difference"].abs())
        )

    # ========================================================
    # 8. КОМБИНИРОВАННЫЕ ПРИЗНАКИ
    # ========================================================

    if (
        "draw_points_similarity" in df.columns
        and "draw_attack_similarity" in df.columns
    ):
        df["draw_strength_attack_balance"] = (
            df["draw_points_similarity"]
            * df["draw_attack_similarity"]
        )

    if (
        "draw_overall_similarity" in df.columns
        and "draw_possession_balance" in df.columns
    ):
        df["draw_game_balance"] = (
            df["draw_overall_similarity"]
            * df["draw_possession_balance"]
        )

    draw_score_columns = [
        "draw_points_similarity",
        "draw_attack_similarity",
        "draw_defense_similarity",
        "draw_overall_similarity",
        "draw_shots_balance",
        "draw_sot_balance",
        "draw_possession_balance",
        "draw_passes_balance",
        "draw_accurate_passes_balance",
        "draw_goals_for_balance",
        "draw_goals_against_balance",
        "draw_low_points_difference",
        "draw_low_goals_difference",
        "draw_low_shots_difference",
        "draw_low_possession_difference",
    ]

    available = [
        column
        for column in draw_score_columns
        if column in df.columns
    ]

    if available:
        df["draw_balance_score"] = (
            df[available].mean(axis=1)
        )

    return df


def temporal_split(df):
    n = len(df)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    return train, validation, test


def prepare_features(df):
    excluded = (
        METADATA_COLUMNS
        | POST_MATCH_COLUMNS
    )

    features = [
        column
        for column in df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    return features


def select_features(train, features):
    X = (
        train[features]
        .replace([np.inf, -np.inf], np.nan)
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

    selected = (
        ranking
        .head(TOP_FEATURES)["feature"]
        .tolist()
    )

    return selected, ranking


def evaluate_model(model, X, y):
    probabilities = model.predict_proba(X)
    predictions = model.predict(X).flatten()

    accuracy = accuracy_score(
        y,
        predictions,
    )

    # CatBoost при строковых классах возвращает вероятности
    # в порядке model.classes_.
    class_order = list(model.classes_)

    logloss = log_loss(
        y,
        probabilities,
        labels=class_order,
    )

    draw_true = y == "D"
    draw_pred = predictions == "D"

    draw_precision = precision_score(
        draw_true,
        draw_pred,
        zero_division=0,
    )

    draw_recall = recall_score(
        draw_true,
        draw_pred,
        zero_division=0,
    )

    draw_f1 = f1_score(
        draw_true,
        draw_pred,
        zero_division=0,
    )

    home_mask = y == "H"
    away_mask = y == "A"

    home_accuracy = (
        (predictions[home_mask] == "H").mean()
        if home_mask.sum() > 0
        else 0
    )

    away_accuracy = (
        (predictions[away_mask] == "A").mean()
        if away_mask.sum() > 0
        else 0
    )

    return {
        "accuracy": accuracy,
        "log_loss": logloss,
        "draw_precision": draw_precision,
        "draw_recall": draw_recall,
        "draw_f1": draw_f1,
        "predicted_draws": int(draw_pred.sum()),
        "actual_draws": int(draw_true.sum()),
        "correct_draws": int(
            (draw_true & draw_pred).sum()
        ),
        "home_accuracy": home_accuracy,
        "away_accuracy": away_accuracy,
    }


def train_and_evaluate(
    train,
    validation,
    test,
    features,
):
    X_train = (
        train[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    X_val = (
        validation[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    X_test = (
        test[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    y_train = train[TARGET]
    y_val = validation[TARGET]
    y_test = test[TARGET]

    model = CatBoostClassifier(
        iterations=500,
        depth=5,
        learning_rate=0.05,
        l2_leaf_reg=7,
        random_strength=0.5,
        bagging_temperature=1.0,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )

    validation_metrics = evaluate_model(
        model,
        X_val,
        y_val,
    )

    test_metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    return (
        model,
        validation_metrics,
        test_metrics,
    )


def print_metrics(title, metrics):
    print(title)

    print(
        f"Accuracy:          "
        f"{metrics['accuracy']:.2%}"
    )

    print(
        f"Log Loss:          "
        f"{metrics['log_loss']:.4f}"
    )

    print(
        f"Draw Precision:    "
        f"{metrics['draw_precision']:.2%}"
    )

    print(
        f"Draw Recall:       "
        f"{metrics['draw_recall']:.2%}"
    )

    print(
        f"Draw F1:            "
        f"{metrics['draw_f1']:.2%}"
    )

    print(
        f"Draws:             "
        f"{metrics['correct_draws']}/"
        f"{metrics['actual_draws']} "
        f"(predicted "
        f"{metrics['predicted_draws']})"
    )

    print(
        f"Home Accuracy:     "
        f"{metrics['home_accuracy']:.2%}"
    )

    print(
        f"Away Accuracy:     "
        f"{metrics['away_accuracy']:.2%}"
    )


def main():
    print("=" * 100)
    print("ЭКСПЕРИМЕНТ: DRAW-SPECIFIC FEATURES")
    print("=" * 100)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset не найден: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print(
        f"\nИсходный dataset: "
        f"{len(df)} матчей"
    )

    print(
        f"Колонки в dataset: "
        f"{len(df.columns)}"
    )

    if TARGET not in df.columns:
        raise ValueError(
            f"В dataset отсутствует "
            f"колонка '{TARGET}'"
        )

    print("\nПроверка post-match колонок...")

    leaked_columns = [
        column
        for column in POST_MATCH_COLUMNS
        if column in df.columns
    ]

    if leaked_columns:
        print(
            "Исключаем из ML:"
        )

        for column in sorted(leaked_columns):
            print(
                f"  - {column}"
            )

    if "kickoff" in df.columns:
        df["kickoff"] = pd.to_datetime(
            df["kickoff"]
        )

        df = (
            df.sort_values("kickoff")
            .reset_index(drop=True)
        )

    df = (
        df.dropna(subset=[TARGET])
        .reset_index(drop=True)
    )

    print(
        "\nРаспределение target:"
    )

    print(
        df[TARGET]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nДобавляем draw-specific признаки..."
    )

    df = add_draw_features(df)

    all_features = prepare_features(df)

    print(
        f"Числовых признаков после добавления: "
        f"{len(all_features)}"
    )

    # Дополнительная защита:
    # post-match признаки не должны попасть в модель
    forbidden_in_features = (
        set(all_features)
        & POST_MATCH_COLUMNS
    )

    if forbidden_in_features:
        raise RuntimeError(
            "ОБНАРУЖЕНА УТЕЧКА TARGET: "
            + ", ".join(
                sorted(forbidden_in_features)
            )
        )

    train, validation, test = temporal_split(df)

    print(
        f"\nTrain:       {len(train)}"
    )

    print(
        f"Validation:  {len(validation)}"
    )

    print(
        f"Test:        {len(test)}"
    )

    selected_features, ranking = select_features(
        train,
        all_features,
    )

    print(
        f"\nВыбрано признаков: "
        f"{len(selected_features)}"
    )

    print("\nПервые 60 признаков:")

    for i, row in ranking.head(
        TOP_FEATURES
    ).iterrows():
        print(
            f"{i + 1:2d}. "
            f"{row['feature']:<50} "
            f"MI={row['mi']:.5f}"
        )

    draw_features = [
        feature
        for feature in selected_features
        if (
            feature.startswith("draw_")
            or feature.startswith("similarity_")
        )
    ]

    print("\n" + "-" * 100)
    print("DRAW-SPECIFIC ПРИЗНАКИ В TOP-60")
    print("-" * 100)

    if draw_features:
        for feature in draw_features:
            mi_value = ranking.loc[
                ranking["feature"] == feature,
                "mi",
            ].iloc[0]

            print(
                f"{feature:<50} "
                f"MI={mi_value:.5f}"
            )
    else:
        print(
            "Ни один draw-specific признак "
            "не попал в TOP-60."
        )

    print("\n" + "=" * 100)
    print("ОБУЧЕНИЕ МОДЕЛИ")
    print("=" * 100)

    (
        model,
        validation_metrics,
        test_metrics,
    ) = train_and_evaluate(
        train,
        validation,
        test,
        selected_features,
    )

    print("\n" + "-" * 100)
    print("VALIDATION")
    print("-" * 100)

    print_metrics(
        "",
        validation_metrics,
    )

    print("\n" + "-" * 100)
    print("TEST")
    print("-" * 100)

    print_metrics(
        "",
        test_metrics,
    )

    best_iteration = model.get_best_iteration()

    print(
        f"\nBest iteration: "
        f"{best_iteration}"
    )

    # ========================================================
    # СРАВНЕНИЕ
    # ========================================================

    baseline = {
        "test_accuracy": 0.5324,
        "test_log_loss": 0.9914,
        "test_draw_recall": 0.0690,
    }

    print("\n" + "=" * 100)
    print("СРАВНЕНИЕ С БАЗОВОЙ МОДЕЛЬЮ")
    print("=" * 100)

    accuracy_delta = (
        test_metrics["accuracy"]
        - baseline["test_accuracy"]
    )

    logloss_delta = (
        test_metrics["log_loss"]
        - baseline["test_log_loss"]
    )

    draw_recall_delta = (
        test_metrics["draw_recall"]
        - baseline["test_draw_recall"]
    )

    print(
        f"Test Accuracy: "
        f"{baseline['test_accuracy']:.2%} -> "
        f"{test_metrics['accuracy']:.2%} "
        f"({accuracy_delta:+.2%})"
    )

    print(
        f"Test Log Loss: "
        f"{baseline['test_log_loss']:.4f} -> "
        f"{test_metrics['log_loss']:.4f} "
        f"({logloss_delta:+.4f})"
    )

    print(
        f"Draw Recall: "
        f"{baseline['test_draw_recall']:.2%} -> "
        f"{test_metrics['draw_recall']:.2%} "
        f"({draw_recall_delta:+.2%})"
    )

    # ========================================================
    # СОХРАНЕНИЕ
    # ========================================================

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "model": "draw_specific_features",
        "features_count": len(selected_features),
        "best_iteration": best_iteration,
        "validation_accuracy":
            validation_metrics["accuracy"],
        "validation_log_loss":
            validation_metrics["log_loss"],
        "validation_draw_precision":
            validation_metrics["draw_precision"],
        "validation_draw_recall":
            validation_metrics["draw_recall"],
        "validation_draw_f1":
            validation_metrics["draw_f1"],
        "test_accuracy":
            test_metrics["accuracy"],
        "test_log_loss":
            test_metrics["log_loss"],
        "test_draw_precision":
            test_metrics["draw_precision"],
        "test_draw_recall":
            test_metrics["draw_recall"],
        "test_draw_f1":
            test_metrics["draw_f1"],
        "test_predicted_draws":
            test_metrics["predicted_draws"],
        "test_actual_draws":
            test_metrics["actual_draws"],
        "test_correct_draws":
            test_metrics["correct_draws"],
        "test_home_accuracy":
            test_metrics["home_accuracy"],
        "test_away_accuracy":
            test_metrics["away_accuracy"],
    }

    pd.DataFrame(
        [result]
    ).to_csv(
        REPORT_PATH,
        index=False,
    )

    ranking_path = Path(
        "data/reports/"
        "draw_features_feature_ranking.csv"
    )

    ranking.to_csv(
        ranking_path,
        index=False,
    )

    model_path = Path(
        "data/reports/"
        "draw_features_experimental.cbm"
    )

    model.save_model(
        model_path
    )

    selected_path = Path(
        "data/reports/"
        "draw_features_selected.joblib"
    )

    joblib.dump(
        selected_features,
        selected_path,
    )

    print(
        f"\nРезультаты сохранены: "
        f"{REPORT_PATH}"
    )

    print(
        f"Рейтинг признаков сохранён: "
        f"{ranking_path}"
    )

    print(
        f"Экспериментальная модель сохранена: "
        f"{model_path}"
    )

    print(
        f"Список признаков сохранён: "
        f"{selected_path}"
    )

    print("\n" + "=" * 100)
    print("ЭКСПЕРИМЕНТ ЗАВЕРШЁН")
    print("=" * 100)


if __name__ == "__main__":
    main()
