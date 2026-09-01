from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score


DATASET_PATH = Path("data/datasets/matches_dataset.csv")
REPORT_PATH = Path("data/reports/walk_forward_results.csv")

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


def prepare_features(df):
    excluded = METADATA_COLUMNS | POST_MATCH_COLUMNS

    return [
        column
        for column in df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(df[column])
    ]


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
        .sort_values("mi", ascending=False)
        .reset_index(drop=True)
    )

    return ranking.head(TOP_FEATURES)["feature"].tolist()


def evaluate(model, df, features):
    X = (
        df[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    y = df[TARGET]

    probabilities = model.predict_proba(X)
    predictions = model.predict(X).flatten()

    accuracy = accuracy_score(y, predictions)

    logloss = log_loss(
        y,
        probabilities,
        labels=list(model.classes_),
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

    home_true = y == "H"
    away_true = y == "A"

    home_accuracy = (
        (predictions[home_true] == "H").mean()
        if home_true.sum()
        else 0
    )

    away_accuracy = (
        (predictions[away_true] == "A").mean()
        if away_true.sum()
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


def train_model(train, validation, features, draw_weight=1.0):
    X_train = (
        train[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    X_validation = (
        validation[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    y_train = train[TARGET]
    y_validation = validation[TARGET]

    model = CatBoostClassifier(
        iterations=500,
        depth=5,
        learning_rate=0.05,
        l2_leaf_reg=7,
        random_strength=0.5,
        bagging_temperature=1.0,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        class_weights=[1.0, draw_weight, 1.0],
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_validation, y_validation),
        early_stopping_rounds=50,
        verbose=False,
    )

    return model


def run_fold(
    df,
    train_start,
    train_end,
    validation_end,
    test_end,
    fold_name,
    draw_weight,
):
    train = df.iloc[train_start:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:test_end].copy()

    features = prepare_features(df)

    selected_features = select_features(
        train,
        features,
    )

    model = train_model(
        train,
        validation,
        selected_features,
        draw_weight=draw_weight,
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
        "fold": fold_name,
        "train_size": len(train),
        "validation_size": len(validation),
        "test_size": len(test),
        "train_start": train["kickoff"].min(),
        "train_end": train["kickoff"].max(),
        "validation_start": validation["kickoff"].min(),
        "validation_end": validation["kickoff"].max(),
        "test_start": test["kickoff"].min(),
        "test_end": test["kickoff"].max(),
        "features_count": len(selected_features),
        "best_iteration": model.get_best_iteration(),

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


def main():
    print("=" * 100)
    print("WALK-FORWARD VALIDATION")
    print("=" * 100)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset не найден: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    df["kickoff"] = pd.to_datetime(
        df["kickoff"]
    )

    df = (
        df.sort_values("kickoff")
        .dropna(subset=[TARGET])
        .reset_index(drop=True)
    )

    print(
        f"\nВсего матчей: {len(df)}"
    )

    print(
        f"Период: "
        f"{df['kickoff'].min()} — "
        f"{df['kickoff'].max()}"
    )

    # Проверка отсутствия очевидной утечки.
    features = prepare_features(df)

    leaked = set(features) & POST_MATCH_COLUMNS

    if leaked:
        raise RuntimeError(
            "ОБНАРУЖЕНА УТЕЧКА: "
            + ", ".join(sorted(leaked))
        )

    print(
        f"Доступно ML-признаков: "
        f"{len(features)}"
    )

    # ========================================================
    # WALK-FORWARD СХЕМА
    #
    # Fold 1:
    # 50% train
    # 15% validation
    # 15% test
    #
    # Fold 2:
    # 65% train
    # 15% validation
    # 15% test
    #
    # Оставшиеся 5% не используются.
    # ========================================================

    n = len(df)

    folds = [
        (
            0,
            int(n * 0.50),
            int(n * 0.65),
            int(n * 0.80),
            "Fold 1",
        ),
        (
            0,
            int(n * 0.65),
            int(n * 0.80),
            int(n * 0.95),
            "Fold 2",
        ),
    ]

    print("\nСхема folds:")

    for (
        train_start,
        train_end,
        validation_end,
        test_end,
        name,
    ) in folds:
        print(
            f"{name}: "
            f"Train={train_end - train_start}, "
            f"Val={validation_end - train_end}, "
            f"Test={test_end - validation_end}"
        )

    # ========================================================
    # ДВА ВАРИАНТА:
    #
    # 1. D = 1.0
    # 2. D = 1.1
    #
    # Так мы одновременно проверим наиболее перспективный
    # class weight на нескольких временных периодах.
    # ========================================================

    all_results = []

    for draw_weight in [1.0, 1.1]:

        print("\n" + "=" * 100)
        print(
            f"DRAW WEIGHT = {draw_weight}"
        )
        print("=" * 100)

        for (
            train_start,
            train_end,
            validation_end,
            test_end,
            fold_name,
        ) in folds:

            print("\n" + "-" * 100)
            print(f"{fold_name}")
            print("-" * 100)

            result = run_fold(
                df=df,
                train_start=train_start,
                train_end=train_end,
                validation_end=validation_end,
                test_end=test_end,
                fold_name=fold_name,
                draw_weight=draw_weight,
            )

            result["draw_weight"] = draw_weight

            all_results.append(result)

            print(
                f"Train: {result['train_size']}"
            )

            print(
                f"Validation: "
                f"{result['validation_size']}"
            )

            print(
                f"Test: "
                f"{result['test_size']}"
            )

            print(
                f"Features: "
                f"{result['features_count']}"
            )

            print(
                f"Best iteration: "
                f"{result['best_iteration']}"
            )

            print("\nValidation:")

            print(
                f"  Accuracy: "
                f"{result['validation_accuracy']:.2%}"
            )

            print(
                f"  Log Loss: "
                f"{result['validation_log_loss']:.4f}"
            )

            print(
                f"  Draw Recall: "
                f"{result['validation_draw_recall']:.2%}"
            )

            print("\nTest:")

            print(
                f"  Accuracy: "
                f"{result['test_accuracy']:.2%}"
            )

            print(
                f"  Log Loss: "
                f"{result['test_log_loss']:.4f}"
            )

            print(
                f"  Draw Precision: "
                f"{result['test_draw_precision']:.2%}"
            )

            print(
                f"  Draw Recall: "
                f"{result['test_draw_recall']:.2%}"
            )

            print(
                f"  Draw F1: "
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
                f"  Home Accuracy: "
                f"{result['test_home_accuracy']:.2%}"
            )

            print(
                f"  Away Accuracy: "
                f"{result['test_away_accuracy']:.2%}"
            )

    results = pd.DataFrame(all_results)

    print("\n" + "=" * 100)
    print("СВОДНАЯ ТАБЛИЦА")
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
        results[display_columns].to_string(
            index=False
        )
    )

    # ========================================================
    # СРЕДНИЕ РЕЗУЛЬТАТЫ ПО ВЕСУ
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
    print("СРЕДНИЕ РЕЗУЛЬТАТЫ")
    print("=" * 100)

    print(
        summary.to_string(
            index=False
        )
    )

    # ========================================================
    # СОХРАНЕНИЕ
    # ========================================================

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        REPORT_PATH,
        index=False,
    )

    summary_path = Path(
        "data/reports/"
        "walk_forward_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        f"\nПодробные результаты: "
        f"{REPORT_PATH}"
    )

    print(
        f"Средние результаты: "
        f"{summary_path}"
    )

    print("\n" + "=" * 100)
    print("WALK-FORWARD ЗАВЕРШЁН")
    print("=" * 100)


if __name__ == "__main__":
    main()
