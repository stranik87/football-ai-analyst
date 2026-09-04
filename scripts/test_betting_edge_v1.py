from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/reports/calibration_v2/test_predictions.csv"
)

REPORT_DIR = Path(
    "data/reports/betting_edge"
)

# Минимальный edge, при котором считаем событие потенциально value.
MIN_EDGE = 0.05

# Минимальная вероятность модели.
MIN_PROBABILITY = 0.50

# Для демонстрации используем несколько коэффициентов.
# Это НЕ реальные исторические odds и НЕ backtest.
TEST_ODDS = {
    "home": [1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 2.00],
    "draw": [2.80, 3.00, 3.20, 3.40, 3.60, 4.00],
    "away": [1.50, 1.70, 1.90, 2.10, 2.30, 2.50, 3.00],
}


def load_predictions():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    return pd.read_csv(INPUT_PATH)


def calculate_implied_probability(odds):
    if odds <= 1.0:
        return np.nan

    return 1.0 / odds


def calculate_fair_odds(probability):
    if probability <= 0:
        return np.inf

    return 1.0 / probability


def calculate_edge(model_probability, odds):
    implied_probability = calculate_implied_probability(odds)

    if pd.isna(implied_probability):
        return np.nan

    return model_probability - implied_probability


def calculate_ev(model_probability, odds):
    """
    Expected value for a 1-unit stake.

    EV = p * (odds - 1) - (1-p)
       = p * odds - 1
    """

    if odds <= 1.0:
        return np.nan

    return model_probability * odds - 1.0


def recommendation(
    model_probability,
    odds,
    edge,
):
    if pd.isna(edge):
        return "INVALID"

    if model_probability < MIN_PROBABILITY:
        return "NO_BET"

    if edge >= MIN_EDGE:
        return "VALUE"

    return "NO_BET"


def analyse_market(
    df,
    probability_column,
    market,
):
    rows = []

    for odds in TEST_ODDS[market]:
        probabilities = df[probability_column].values

        implied_probability = calculate_implied_probability(
            odds
        )

        fair_odds = np.where(
            probabilities > 0,
            1.0 / probabilities,
            np.inf,
        )

        edge = probabilities - implied_probability

        ev = probabilities * odds - 1.0

        value_mask = (
            (probabilities >= MIN_PROBABILITY)
            & (edge >= MIN_EDGE)
        )

        rows.append(
            {
                "market": market,
                "odds": odds,
                "implied_probability": implied_probability,
                "matches": len(df),
                "value_matches": int(value_mask.sum()),
                "value_rate": float(value_mask.mean()),
                "mean_model_probability": float(
                    probabilities.mean()
                ),
                "mean_fair_odds": float(
                    np.mean(fair_odds)
                ),
                "mean_edge": float(
                    edge.mean()
                ),
                "mean_ev": float(
                    ev.mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def create_example_table():
    """
    Small deterministic mathematical examples.
    """

    examples = [
        {
            "market": "HOME",
            "model_probability": 0.72,
            "odds": 1.60,
        },
        {
            "market": "HOME",
            "model_probability": 0.65,
            "odds": 1.70,
        },
        {
            "market": "DRAW",
            "model_probability": 0.30,
            "odds": 3.60,
        },
        {
            "market": "AWAY",
            "model_probability": 0.55,
            "odds": 2.10,
        },
        {
            "market": "AWAY",
            "model_probability": 0.45,
            "odds": 2.50,
        },
    ]

    rows = []

    for item in examples:
        p = item["model_probability"]
        odds = item["odds"]

        implied = calculate_implied_probability(odds)
        fair = calculate_fair_odds(p)
        edge = calculate_edge(p, odds)
        ev = calculate_ev(p, odds)

        rows.append(
            {
                "market": item["market"],
                "model_probability": p,
                "odds": odds,
                "fair_odds": fair,
                "implied_probability": implied,
                "edge": edge,
                "edge_percentage": edge * 100,
                "ev": ev,
                "ev_percentage": ev * 100,
                "recommendation": recommendation(
                    p,
                    odds,
                    edge,
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("BETTING EDGE ENGINE V1")
    print("=" * 80)

    df = load_predictions()

    print()
    print(f"Input: {INPUT_PATH}")
    print(f"Matches: {len(df)}")

    required_columns = [
        "calibrated_p_away",
        "calibrated_p_draw",
        "calibrated_p_home",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # ---------------------------------------------------------------
    # BASIC PROBABILITY CHECK
    # ---------------------------------------------------------------

    probability_columns = [
        "calibrated_p_away",
        "calibrated_p_draw",
        "calibrated_p_home",
    ]

    probability_sum = df[
        probability_columns
    ].sum(axis=1)

    print()
    print("-" * 80)
    print("PROBABILITY CHECK")
    print("-" * 80)

    print(
        f"Min probability sum: "
        f"{probability_sum.min():.10f}"
    )

    print(
        f"Max probability sum: "
        f"{probability_sum.max():.10f}"
    )

    print(
        f"Mean probability sum: "
        f"{probability_sum.mean():.10f}"
    )

    if not np.allclose(
        probability_sum,
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Probability columns do not sum to 1."
        )

    # ---------------------------------------------------------------
    # EXAMPLE CALCULATIONS
    # ---------------------------------------------------------------

    examples = create_example_table()

    print()
    print("-" * 80)
    print("MATHEMATICAL EXAMPLES")
    print("-" * 80)

    print(
        examples.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    examples.to_csv(
        REPORT_DIR / "examples.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # MARKET ANALYSIS
    # ---------------------------------------------------------------

    markets = {
        "home": "calibrated_p_home",
        "draw": "calibrated_p_draw",
        "away": "calibrated_p_away",
    }

    all_results = []

    for market, probability_column in markets.items():
        result = analyse_market(
            df,
            probability_column,
            market,
        )

        all_results.append(result)

        print()
        print("-" * 80)
        print(f"{market.upper()} MARKET")
        print("-" * 80)

        print(
            result.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

    market_results = pd.concat(
        all_results,
        ignore_index=True,
    )

    market_results.to_csv(
        REPORT_DIR / "market_analysis.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # SAMPLE MATCHES
    # ---------------------------------------------------------------

    sample = df[
        [
            "fixture_id",
            "kickoff",
            "actual",
            "calibrated_prediction",
            "calibrated_p_away",
            "calibrated_p_draw",
            "calibrated_p_home",
            "calibrated_confidence",
        ]
    ].copy()

    sample["fair_odds_away"] = (
        1.0 / sample["calibrated_p_away"]
    )

    sample["fair_odds_draw"] = (
        1.0 / sample["calibrated_p_draw"]
    )

    sample["fair_odds_home"] = (
        1.0 / sample["calibrated_p_home"]
    )

    sample.to_csv(
        REPORT_DIR / "fair_odds_test.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "min_edge": MIN_EDGE,
                "min_probability": MIN_PROBABILITY,
                "matches": len(df),
                "mean_p_home": df[
                    "calibrated_p_home"
                ].mean(),
                "mean_p_draw": df[
                    "calibrated_p_draw"
                ].mean(),
                "mean_p_away": df[
                    "calibrated_p_away"
                ].mean(),
                "mean_confidence": df[
                    "calibrated_confidence"
                ].mean(),
            }
        ]
    )

    summary.to_csv(
        REPORT_DIR / "summary.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(REPORT_DIR / "examples.csv")
    print(REPORT_DIR / "market_analysis.csv")
    print(REPORT_DIR / "fair_odds_test.csv")
    print(REPORT_DIR / "summary.csv")

    print()
    print("IMPORTANT:")
    print(
        "The odds used here are synthetic test values. "
        "This is NOT a historical ROI backtest."
    )

    print()
    print("Betting Edge Engine V1 completed successfully.")


if __name__ == "__main__":
    main()
