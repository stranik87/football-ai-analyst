from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import label_binarize


DATASET_PATH = Path("data/datasets/matches_dataset.csv")
MODEL_PATH = Path("data/models/match_result_catboost.cbm")
FEATURES_PATH = Path("data/models/match_result_features.joblib")
REPORT_DIR = Path("data/reports/calibration")

CLASS_ORDER = ["A", "D", "H"]


def load_data():
    return pd.read_csv(DATASET_PATH)


def load_model():
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    return model


def load_features():
    import joblib
    return joblib.load(FEATURES_PATH)


def prepare_features(df, features):
    X = df[features].copy()

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].fillna("")
        else:
            X[col] = X[col].fillna(0)

    return X


def split_temporal(df):
    n = len(df)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    return train, val, test


def temperature_scale(probs, temperature):
    probs = np.clip(probs, 1e-12, 1.0)

    logits = np.log(probs)
    scaled_logits = logits / temperature

    scaled_logits -= scaled_logits.max(axis=1, keepdims=True)

    exp_logits = np.exp(scaled_logits)

    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def predict_labels(probs):
    indexes = np.argmax(probs, axis=1)
    return np.array(CLASS_ORDER)[indexes]


def calculate_metrics(y_true, probs):
    predictions = predict_labels(probs)

    accuracy = accuracy_score(y_true, predictions)

    logloss = log_loss(
        y_true,
        probs,
        labels=CLASS_ORDER,
    )

    y_bin = label_binarize(
        y_true,
        classes=CLASS_ORDER,
    )

    brier = np.mean(
        np.sum((y_bin - probs) ** 2, axis=1)
    )

    draw_mask = y_true == "D"

    mean_draw_probability = (
        probs[draw_mask, 1].mean()
        if draw_mask.any()
        else np.nan
    )

    mean_max_probability = probs.max(axis=1).mean()

    predicted_draws = np.sum(predictions == "D")

    return {
        "accuracy": accuracy,
        "logloss": logloss,
        "brier": brier,
        "mean_p_draw_actual_draws": mean_draw_probability,
        "mean_max_probability": mean_max_probability,
        "predicted_draws": predicted_draws,
    }


def find_best_temperature(y_true, probs):
    # V1.3:
    # V1.2 hit the lower boundary 0.50.
    # Therefore we extend the search substantially downward.
    temperatures = np.arange(0.10, 3.01, 0.05)

    results = []

    for temperature in temperatures:
        calibrated = temperature_scale(
            probs,
            temperature,
        )

        metrics = calculate_metrics(
            y_true,
            calibrated,
        )

        results.append(
            {
                "temperature": round(float(temperature), 2),
                "logloss": metrics["logloss"],
                "accuracy": metrics["accuracy"],
                "brier": metrics["brier"],
            }
        )

    results_df = pd.DataFrame(results)

    best = results_df.loc[
        results_df["logloss"].idxmin()
    ]

    return float(best["temperature"]), results_df


def main():
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("CALIBRATION V1.3")
    print("=" * 80)

    df = load_data()

    print(f"Dataset: {df.shape[0]} matches")
    print(f"Columns: {df.shape[1]}")

    train, val, test = split_temporal(df)

    print()
    print("Temporal split:")
    print(f"Train: {len(train)}")
    print(f"Validation: {len(val)}")
    print(f"Test: {len(test)}")

    model = load_model()
    features = load_features()

    print()
    print("CatBoost classes:")
    print(model.classes_)

    print()
    print("Expected class order:")
    print(CLASS_ORDER)

    if list(model.classes_) != CLASS_ORDER:
        raise RuntimeError(
            f"Unexpected CatBoost class order: {model.classes_}. "
            f"Expected: {CLASS_ORDER}"
        )

    # ================================================================
    # VALIDATION
    # ================================================================

    X_val = prepare_features(
        val,
        features,
    )

    y_val = val["result"].astype(str).values

    val_probs = model.predict_proba(X_val)

    baseline_val = calculate_metrics(
        y_val,
        val_probs,
    )

    best_temperature, temperature_results = find_best_temperature(
        y_val,
        val_probs,
    )

    temperature_results.to_csv(
        REPORT_DIR / "temperature_search_v1_3.csv",
        index=False,
    )

    calibrated_val_probs = temperature_scale(
        val_probs,
        best_temperature,
    )

    calibrated_val = calculate_metrics(
        y_val,
        calibrated_val_probs,
    )

    print()
    print("-" * 80)
    print("VALIDATION")
    print("-" * 80)

    print(
        f"Baseline Accuracy: "
        f"{baseline_val['accuracy']:.4%}"
    )

    print(
        f"Baseline LogLoss: "
        f"{baseline_val['logloss']:.6f}"
    )

    print(
        f"Calibrated Accuracy: "
        f"{calibrated_val['accuracy']:.4%}"
    )

    print(
        f"Calibrated LogLoss: "
        f"{calibrated_val['logloss']:.6f}"
    )

    print(
        f"Baseline Brier: "
        f"{baseline_val['brier']:.6f}"
    )

    print(
        f"Calibrated Brier: "
        f"{calibrated_val['brier']:.6f}"
    )

    print(
        f"Best Temperature: "
        f"{best_temperature:.2f}"
    )

    # ================================================================
    # TEST
    # ================================================================

    X_test = prepare_features(
        test,
        features,
    )

    y_test = test["result"].astype(str).values

    test_probs = model.predict_proba(X_test)

    baseline_test = calculate_metrics(
        y_test,
        test_probs,
    )

    calibrated_test_probs = temperature_scale(
        test_probs,
        best_temperature,
    )

    calibrated_test = calculate_metrics(
        y_test,
        calibrated_test_probs,
    )

    print()
    print("-" * 80)
    print("TEST")
    print("-" * 80)

    print(
        f"Baseline Accuracy: "
        f"{baseline_test['accuracy']:.4%}"
    )

    print(
        f"Baseline LogLoss: "
        f"{baseline_test['logloss']:.6f}"
    )

    print(
        f"Calibrated Accuracy: "
        f"{calibrated_test['accuracy']:.4%}"
    )

    print(
        f"Calibrated LogLoss: "
        f"{calibrated_test['logloss']:.6f}"
    )

    print(
        f"Baseline Brier: "
        f"{baseline_test['brier']:.6f}"
    )

    print(
        f"Calibrated Brier: "
        f"{calibrated_test['brier']:.6f}"
    )

    print(
        f"Baseline mean P(draw) on actual draws: "
        f"{baseline_test['mean_p_draw_actual_draws']:.6f}"
    )

    print(
        f"Calibrated mean P(draw) on actual draws: "
        f"{calibrated_test['mean_p_draw_actual_draws']:.6f}"
    )

    print(
        f"Baseline predicted draws: "
        f"{baseline_test['predicted_draws']}"
    )

    print(
        f"Calibrated predicted draws: "
        f"{calibrated_test['predicted_draws']}"
    )

    # ================================================================
    # TOP TEMPERATURES
    # ================================================================

    print()
    print("-" * 80)
    print("TOP 10 TEMPERATURES BY VALIDATION LOGLOSS")
    print("-" * 80)

    top10 = temperature_results.sort_values(
        "logloss"
    ).head(10)

    print(
        top10.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ================================================================
    # SAVE SUMMARY
    # ================================================================

    summary = pd.DataFrame(
        [
            {
                "model": "CatBoost",
                "temperature": 1.0,
                **baseline_test,
            },
            {
                "model": "CatBoost_calibrated_v1_3",
                "temperature": best_temperature,
                **calibrated_test,
            },
        ]
    )

    summary.to_csv(
        REPORT_DIR / "calibration_v1_3_summary.csv",
        index=False,
    )

    # ================================================================
    # SAVE TEST PREDICTIONS
    # ================================================================

    predictions = test[
        [
            "fixture_id",
            "kickoff",
            "home_team_id",
            "away_team_id",
        ]
    ].copy()

    predictions["actual"] = y_test

    predictions["p_away_baseline"] = test_probs[:, 0]
    predictions["p_draw_baseline"] = test_probs[:, 1]
    predictions["p_home_baseline"] = test_probs[:, 2]

    predictions["prediction_baseline"] = predict_labels(
        test_probs
    )

    predictions["p_away_calibrated"] = calibrated_test_probs[:, 0]
    predictions["p_draw_calibrated"] = calibrated_test_probs[:, 1]
    predictions["p_home_calibrated"] = calibrated_test_probs[:, 2]

    predictions["prediction_calibrated"] = predict_labels(
        calibrated_test_probs
    )

    predictions["temperature"] = best_temperature

    predictions.to_csv(
        REPORT_DIR / "calibration_v1_3_test_predictions.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(
        REPORT_DIR / "temperature_search_v1_3.csv"
    )

    print(
        REPORT_DIR / "calibration_v1_3_summary.csv"
    )

    print(
        REPORT_DIR / "calibration_v1_3_test_predictions.csv"
    )

    print()
    print("Calibration V1.3 completed successfully.")


if __name__ == "__main__":
    main()
