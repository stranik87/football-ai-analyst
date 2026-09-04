from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss


DATASET_PATH = Path("data/datasets/matches_dataset.csv")
MODEL_PATH = Path("data/models/match_result_catboost.cbm")
FEATURES_PATH = Path("data/models/match_result_features.joblib")
REPORT_DIR = Path("data/reports/calibration_v2")

CLASS_ORDER = ["A", "D", "H"]
TEMPERATURE = 0.15


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

    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


def temperature_scale(probs, temperature):
    probs = np.clip(probs, 1e-12, 1.0)

    logits = np.log(probs)
    logits = logits / temperature

    logits -= logits.max(axis=1, keepdims=True)

    exp_logits = np.exp(logits)

    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def predict_labels(probs):
    indexes = np.argmax(probs, axis=1)
    return np.array(CLASS_ORDER)[indexes]


def calculate_ece(y_true, probs, n_bins=10):
    predictions = predict_labels(probs)
    confidence = probs.max(axis=1)
    correct = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)

    rows = []

    for i in range(n_bins):
        left = bins[i]
        right = bins[i + 1]

        if i == n_bins - 1:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)

        count = mask.sum()

        if count == 0:
            rows.append(
                {
                    "bin": i + 1,
                    "confidence_min": left,
                    "confidence_max": right,
                    "count": 0,
                    "mean_confidence": np.nan,
                    "accuracy": np.nan,
                    "gap": np.nan,
                }
            )
            continue

        mean_confidence = confidence[mask].mean()
        accuracy = correct[mask].mean()
        gap = abs(mean_confidence - accuracy)

        rows.append(
            {
                "bin": i + 1,
                "confidence_min": left,
                "confidence_max": right,
                "count": int(count),
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "gap": gap,
            }
        )

    calibration = pd.DataFrame(rows)

    total = len(y_true)

    ece = (
        calibration["count"]
        * calibration["gap"]
    ).sum() / total

    mce = calibration["gap"].max()

    return ece, mce, calibration


def class_calibration(y_true, probs):
    rows = []

    for index, class_name in enumerate(CLASS_ORDER):
        actual = (y_true == class_name).astype(int)
        predicted_probability = probs[:, index]

        bins = np.linspace(0.0, 1.0, 11)

        for i in range(10):
            left = bins[i]
            right = bins[i + 1]

            if i == 9:
                mask = (
                    (predicted_probability >= left)
                    & (predicted_probability <= right)
                )
            else:
                mask = (
                    (predicted_probability >= left)
                    & (predicted_probability < right)
                )

            count = mask.sum()

            if count == 0:
                continue

            mean_probability = predicted_probability[mask].mean()
            observed_frequency = actual[mask].mean()

            rows.append(
                {
                    "class": class_name,
                    "bin": i + 1,
                    "count": int(count),
                    "mean_probability": mean_probability,
                    "observed_frequency": observed_frequency,
                    "gap": abs(
                        mean_probability
                        - observed_frequency
                    ),
                }
            )

    return pd.DataFrame(rows)


def high_confidence_analysis(y_true, probs, model_name):
    predictions = predict_labels(probs)
    confidence = probs.max(axis=1)

    df = pd.DataFrame(
        {
            "actual": y_true,
            "prediction": predictions,
            "confidence": confidence,
            "p_away": probs[:, 0],
            "p_draw": probs[:, 1],
            "p_home": probs[:, 2],
        }
    )

    df["correct"] = (
        df["actual"] == df["prediction"]
    )

    thresholds = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ]

    rows = []

    for threshold in thresholds:
        subset = df[df["confidence"] >= threshold]

        if len(subset) == 0:
            rows.append(
                {
                    "model": model_name,
                    "threshold": threshold,
                    "count": 0,
                    "accuracy": np.nan,
                    "mean_confidence": np.nan,
                }
            )
            continue

        rows.append(
            {
                "model": model_name,
                "threshold": threshold,
                "count": len(subset),
                "accuracy": subset["correct"].mean(),
                "mean_confidence": subset["confidence"].mean(),
            }
        )

    return pd.DataFrame(rows)


def class_summary(y_true, probs, model_name):
    predictions = predict_labels(probs)

    rows = []

    for index, class_name in enumerate(CLASS_ORDER):
        actual_mask = y_true == class_name
        predicted_mask = predictions == class_name

        actual_count = actual_mask.sum()
        predicted_count = predicted_mask.sum()

        recall = (
            (
                predicted_mask
                & actual_mask
            ).sum()
            / actual_count
            if actual_count
            else np.nan
        )

        precision = (
            (
                predicted_mask
                & actual_mask
            ).sum()
            / predicted_count
            if predicted_count
            else np.nan
        )

        mean_probability_actual = (
            probs[actual_mask, index].mean()
            if actual_count
            else np.nan
        )

        rows.append(
            {
                "model": model_name,
                "class": class_name,
                "actual_count": int(actual_count),
                "predicted_count": int(predicted_count),
                "recall": recall,
                "precision": precision,
                "mean_probability_on_actual": mean_probability_actual,
            }
        )

    return pd.DataFrame(rows)


def evaluate_model(y_true, probs, model_name):
    predictions = predict_labels(probs)

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    logloss = log_loss(
        y_true,
        probs,
        labels=CLASS_ORDER,
    )

    ece, mce, calibration = calculate_ece(
        y_true,
        probs,
    )

    class_data = class_summary(
        y_true,
        probs,
        model_name,
    )

    confidence_data = high_confidence_analysis(
        y_true,
        probs,
        model_name,
    )

    calibration["model"] = model_name

    return {
        "model": model_name,
        "accuracy": accuracy,
        "logloss": logloss,
        "ece": ece,
        "mce": mce,
        "predicted_draws": int(
            np.sum(predictions == "D")
        ),
        "mean_confidence": probs.max(axis=1).mean(),
    }, calibration, class_data, confidence_data


def main():
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("CALIBRATION V2 — RELIABILITY ANALYSIS")
    print("=" * 80)

    df = load_data()

    _, _, test = split_temporal(df)

    model = load_model()
    features = load_features()

    print()
    print(f"Test matches: {len(test)}")
    print(f"Features: {len(features)}")
    print(f"Classes: {list(model.classes_)}")
    print(f"Temperature under test: {TEMPERATURE}")

    X_test = prepare_features(
        test,
        features,
    )

    y_test = test["result"].astype(str).values

    baseline_probs = model.predict_proba(X_test)

    calibrated_probs = temperature_scale(
        baseline_probs,
        TEMPERATURE,
    )

    # ---------------------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------------------

    baseline_summary, baseline_cal, baseline_class, baseline_conf = (
        evaluate_model(
            y_test,
            baseline_probs,
            "CatBoost_baseline",
        )
    )

    calibrated_summary, calibrated_cal, calibrated_class, calibrated_conf = (
        evaluate_model(
            y_test,
            calibrated_probs,
            "CatBoost_T0.15",
        )
    )

    summary = pd.DataFrame(
        [
            baseline_summary,
            calibrated_summary,
        ]
    )

    calibration = pd.concat(
        [
            baseline_cal,
            calibrated_cal,
        ],
        ignore_index=True,
    )

    classes = pd.concat(
        [
            baseline_class,
            calibrated_class,
        ],
        ignore_index=True,
    )

    confidence = pd.concat(
        [
            baseline_conf,
            calibrated_conf,
        ],
        ignore_index=True,
    )

    # ---------------------------------------------------------------
    # PRINT SUMMARY
    # ---------------------------------------------------------------

    print()
    print("-" * 80)
    print("SUMMARY")
    print("-" * 80)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("-" * 80)
    print("CLASS CALIBRATION")
    print("-" * 80)

    print(
        classes.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("-" * 80)
    print("CONFIDENCE ANALYSIS")
    print("-" * 80)

    print(
        confidence.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ---------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------

    summary.to_csv(
        REPORT_DIR / "summary.csv",
        index=False,
    )

    calibration.to_csv(
        REPORT_DIR / "reliability_bins.csv",
        index=False,
    )

    classes.to_csv(
        REPORT_DIR / "class_calibration.csv",
        index=False,
    )

    confidence.to_csv(
        REPORT_DIR / "confidence_analysis.csv",
        index=False,
    )

    predictions = test[
        [
            "fixture_id",
            "kickoff",
            "home_team_id",
            "away_team_id",
        ]
    ].copy()

    predictions["actual"] = y_test

    predictions["baseline_prediction"] = predict_labels(
        baseline_probs
    )

    predictions["baseline_p_away"] = baseline_probs[:, 0]
    predictions["baseline_p_draw"] = baseline_probs[:, 1]
    predictions["baseline_p_home"] = baseline_probs[:, 2]

    predictions["calibrated_prediction"] = predict_labels(
        calibrated_probs
    )

    predictions["calibrated_p_away"] = calibrated_probs[:, 0]
    predictions["calibrated_p_draw"] = calibrated_probs[:, 1]
    predictions["calibrated_p_home"] = calibrated_probs[:, 2]

    predictions["baseline_confidence"] = baseline_probs.max(axis=1)
    predictions["calibrated_confidence"] = calibrated_probs.max(axis=1)

    predictions["baseline_correct"] = (
        predictions["actual"]
        == predictions["baseline_prediction"]
    )

    predictions["calibrated_correct"] = (
        predictions["actual"]
        == predictions["calibrated_prediction"]
    )

    predictions.to_csv(
        REPORT_DIR / "test_predictions.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(REPORT_DIR / "summary.csv")
    print(REPORT_DIR / "reliability_bins.csv")
    print(REPORT_DIR / "class_calibration.csv")
    print(REPORT_DIR / "confidence_analysis.csv")
    print(REPORT_DIR / "test_predictions.csv")

    print()
    print("Calibration V2 completed successfully.")


if __name__ == "__main__":
    main()
