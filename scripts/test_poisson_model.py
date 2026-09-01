from __future__ import annotations

from pathlib import Path
from math import exp, factorial

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss


DATASET_PATH = Path("data/datasets/matches_dataset.csv")

TARGET = "result"

MAX_GOALS = 10

LEAGUE_WEIGHT = 5.0

FOLDS = [
    {
        "name": "Fold 1",
        "train_end": 926,
        "val_end": 1204,
        "test_end": 1482,
    },
    {
        "name": "Fold 2",
        "train_end": 1204,
        "val_end": 1482,
        "test_end": 1760,
    },
]


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0

    return exp(-lam) * (lam ** k) / factorial(k)


def poisson_match_probabilities(
    home_lambda: float,
    away_lambda: float,
) -> tuple[float, float, float]:

    home_probs = np.array(
        [
            poisson_pmf(i, home_lambda)
            for i in range(MAX_GOALS + 1)
        ]
    )

    away_probs = np.array(
        [
            poisson_pmf(i, away_lambda)
            for i in range(MAX_GOALS + 1)
        ]
    )

    home_probs /= home_probs.sum()
    away_probs /= away_probs.sum()

    matrix = np.outer(
        home_probs,
        away_probs,
    )

    p_home = float(
        np.tril(matrix, -1).sum()
    )

    p_draw = float(
        np.trace(matrix)
    )

    p_away = float(
        np.triu(matrix, 1).sum()
    )

    total = p_home + p_draw + p_away

    return (
        p_home / total,
        p_draw / total,
        p_away / total,
    )


def shrink(
    value: float,
    matches: int,
    league_average: float,
) -> float:

    if matches <= 0:
        return league_average

    weight = matches / (
        matches + LEAGUE_WEIGHT
    )

    return (
        weight * value
        + (1.0 - weight) * league_average
    )


class TeamHistory:

    def __init__(self) -> None:
        self.matches: dict[int, list[dict]] = {}

    def add_match(
        self,
        team_id: int,
        goals_for: float,
        goals_against: float,
        is_home: bool,
    ) -> None:

        self.matches.setdefault(
            team_id,
            [],
        ).append(
            {
                "goals_for": float(goals_for),
                "goals_against": float(goals_against),
                "is_home": is_home,
            }
        )

    def get(
        self,
        team_id: int,
        home_only: bool | None = None,
        limit: int | None = None,
    ) -> list[dict]:

        history = self.matches.get(
            team_id,
            [],
        )

        if home_only is not None:
            history = [
                item
                for item in history
                if item["is_home"] == home_only
            ]

        if limit is not None:
            history = history[-limit:]

        return history

    def count(
        self,
        team_id: int,
    ) -> int:

        return len(
            self.matches.get(
                team_id,
                [],
            )
        )


def calculate_global_averages(
    history: TeamHistory,
) -> dict[str, float]:

    all_matches = []

    for matches in history.matches.values():
        all_matches.extend(matches)

    if not all_matches:
        return {
            "home_goals": 1.35,
            "away_goals": 1.10,
            "goals_for": 1.25,
            "goals_against": 1.25,
        }

    home_goals = [
        x["goals_for"]
        for x in all_matches
        if x["is_home"]
    ]

    away_goals = [
        x["goals_for"]
        for x in all_matches
        if not x["is_home"]
    ]

    goals_for = [
        x["goals_for"]
        for x in all_matches
    ]

    goals_against = [
        x["goals_against"]
        for x in all_matches
    ]

    return {
        "home_goals": float(
            np.mean(home_goals)
        ),
        "away_goals": float(
            np.mean(away_goals)
        ),
        "goals_for": float(
            np.mean(goals_for)
        ),
        "goals_against": float(
            np.mean(goals_against)
        ),
    }


def attack_strength(
    history: TeamHistory,
    team_id: int,
    is_home: bool,
    global_avg: dict[str, float],
) -> float:

    all_history = history.get(
        team_id
    )

    if not all_history:
        return 1.0

    recent5 = history.get(
        team_id,
        limit=5,
    )

    recent10 = history.get(
        team_id,
        limit=10,
    )

    venue_history = history.get(
        team_id,
        home_only=is_home,
    )

    recent5_goals = np.mean(
        [
            x["goals_for"]
            for x in recent5
        ]
    )

    recent10_goals = np.mean(
        [
            x["goals_for"]
            for x in recent10
        ]
    )

    if venue_history:
        venue_goals = np.mean(
            [
                x["goals_for"]
                for x in venue_history[-5:]
            ]
        )
    else:
        venue_goals = (
            global_avg["home_goals"]
            if is_home
            else global_avg["away_goals"]
        )

    overall_goals = np.mean(
        [
            x["goals_for"]
            for x in all_history
        ]
    )

    raw_goals = (
        recent5_goals * 0.45
        + recent10_goals * 0.25
        + venue_goals * 0.20
        + overall_goals * 0.10
    )

    baseline = (
        global_avg["home_goals"]
        if is_home
        else global_avg["away_goals"]
    )

    raw_goals = shrink(
        raw_goals,
        len(all_history),
        baseline,
    )

    return max(
        0.05,
        raw_goals / baseline,
    )


def defense_strength(
    history: TeamHistory,
    team_id: int,
    is_home: bool,
    global_avg: dict[str, float],
) -> float:

    all_history = history.get(
        team_id
    )

    if not all_history:
        return 1.0

    recent5 = history.get(
        team_id,
        limit=5,
    )

    recent10 = history.get(
        team_id,
        limit=10,
    )

    venue_history = history.get(
        team_id,
        home_only=is_home,
    )

    recent5_conceded = np.mean(
        [
            x["goals_against"]
            for x in recent5
        ]
    )

    recent10_conceded = np.mean(
        [
            x["goals_against"]
            for x in recent10
        ]
    )

    if venue_history:
        venue_conceded = np.mean(
            [
                x["goals_against"]
                for x in venue_history[-5:]
            ]
        )
    else:
        venue_conceded = (
            global_avg["goals_against"]
        )

    overall_conceded = np.mean(
        [
            x["goals_against"]
            for x in all_history
        ]
    )

    raw_conceded = (
        recent5_conceded * 0.45
        + recent10_conceded * 0.25
        + venue_conceded * 0.20
        + overall_conceded * 0.10
    )

    raw_conceded = shrink(
        raw_conceded,
        len(all_history),
        global_avg["goals_against"],
    )

    return max(
        0.05,
        raw_conceded
        / global_avg["goals_against"],
    )


def predict_match(
    history: TeamHistory,
    home_team_id: int,
    away_team_id: int,
    global_avg: dict[str, float],
) -> tuple[float, float, tuple[float, float, float]]:

    home_attack = attack_strength(
        history,
        home_team_id,
        True,
        global_avg,
    )

    away_attack = attack_strength(
        history,
        away_team_id,
        False,
        global_avg,
    )

    home_defense = defense_strength(
        history,
        home_team_id,
        True,
        global_avg,
    )

    away_defense = defense_strength(
        history,
        away_team_id,
        False,
        global_avg,
    )

    home_lambda = (
        global_avg["home_goals"]
        * home_attack
        * away_defense
    )

    away_lambda = (
        global_avg["away_goals"]
        * away_attack
        * home_defense
    )

    home_lambda = float(
        np.clip(
            home_lambda,
            0.05,
            4.5,
        )
    )

    away_lambda = float(
        np.clip(
            away_lambda,
            0.05,
            4.5,
        )
    )

    probabilities = poisson_match_probabilities(
        home_lambda,
        away_lambda,
    )

    return (
        home_lambda,
        away_lambda,
        probabilities,
    )


def add_match_to_history(
    history: TeamHistory,
    row: pd.Series,
) -> None:

    home_id = int(
        row["home_team_id"]
    )

    away_id = int(
        row["away_team_id"]
    )

    home_goals = float(
        row["home_goals"]
    )

    away_goals = float(
        row["away_goals"]
    )

    history.add_match(
        home_id,
        home_goals,
        away_goals,
        True,
    )

    history.add_match(
        away_id,
        away_goals,
        home_goals,
        False,
    )


def evaluate_period(
    rows: pd.DataFrame,
    history: TeamHistory,
    period_name: str,
) -> dict:

    y_true = []
    y_pred = []
    probabilities = []

    records = []

    for _, row in rows.iterrows():

        home_id = int(
            row["home_team_id"]
        )

        away_id = int(
            row["away_team_id"]
        )

        global_avg = calculate_global_averages(
            history
        )

        home_lambda, away_lambda, probs = (
            predict_match(
                history,
                home_id,
                away_id,
                global_avg,
            )
        )

        p_home, p_draw, p_away = probs

        probability_map = {
            "H": p_home,
            "D": p_draw,
            "A": p_away,
        }

        prediction = max(
            probability_map,
            key=probability_map.get,
        )

        actual = str(
            row[TARGET]
        )

        y_true.append(actual)
        y_pred.append(prediction)

        # sklearn labels must match probability columns.
        probabilities.append(
            [
                p_away,
                p_draw,
                p_home,
            ]
        )

        records.append(
            {
                "fixture_id": row["fixture_id"],
                "kickoff": row["kickoff"],
                "home_team_id": home_id,
                "away_team_id": away_id,
                "actual": actual,
                "prediction": prediction,
                "home_lambda": home_lambda,
                "away_lambda": away_lambda,
                "p_home": p_home,
                "p_draw": p_draw,
                "p_away": p_away,
            }
        )

        # КРИТИЧЕСКИ ВАЖНО:
        # результат текущего матча добавляется
        # только ПОСЛЕ prediction.
        add_match_to_history(
            history,
            row,
        )

    y_true_np = np.array(
        y_true
    )

    y_pred_np = np.array(
        y_pred
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    logloss = log_loss(
        y_true,
        probabilities,
        labels=["A", "D", "H"],
    )

    actual_draws = (
        y_true_np == "D"
    )

    predicted_draws = (
        y_pred_np == "D"
    )

    draw_recall = (
        (
            y_pred_np[actual_draws]
            == "D"
        ).mean()
        if actual_draws.any()
        else 0.0
    )

    draw_precision = (
        (
            y_true_np[predicted_draws]
            == "D"
        ).mean()
        if predicted_draws.any()
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

    print()
    print("-" * 80)
    print(period_name)
    print("-" * 80)

    print(
        f"Matches:             {len(rows)}"
    )

    print(
        f"Accuracy:            {accuracy:.2%}"
    )

    print(
        f"LogLoss:             {logloss:.4f}"
    )

    print(
        f"Predicted draws:     "
        f"{int(predicted_draws.sum())}"
    )

    print(
        f"Actual draws:        "
        f"{int(actual_draws.sum())}"
    )

    print(
        f"Draw precision:      "
        f"{draw_precision:.2%}"
    )

    print(
        f"Draw recall:         "
        f"{draw_recall:.2%}"
    )

    print(
        f"Draw F1:             "
        f"{draw_f1:.2%}"
    )

    for cls in ["H", "D", "A"]:

        mask = (
            y_true_np == cls
        )

        cls_accuracy = (
            (
                y_pred_np[mask]
                == cls
            ).mean()
            if mask.any()
            else 0.0
        )

        print(
            f"{cls} accuracy:        "
            f"{cls_accuracy:.2%}"
        )

    return {
        "period": period_name,
        "matches": len(rows),
        "accuracy": accuracy,
        "logloss": logloss,
        "draw_precision": draw_precision,
        "draw_recall": draw_recall,
        "draw_f1": draw_f1,
        "records": records,
    }


def main() -> None:

    print("=" * 80)
    print("POISSON / EXPECTED GOALS BASELINE")
    print("=" * 80)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH
    )

    required_columns = {
        "fixture_id",
        "kickoff",
        "home_team_id",
        "away_team_id",
        "home_goals",
        "away_goals",
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

    df = df.dropna(
        subset=[
            "kickoff",
            "home_team_id",
            "away_team_id",
            "home_goals",
            "away_goals",
            "result",
        ]
    ).copy()

    df = df.sort_values(
        "kickoff"
    ).reset_index(
        drop=True
    )

    print()
    print(
        f"Dataset:             "
        f"{DATASET_PATH}"
    )

    print(
        f"Matches:             "
        f"{len(df)}"
    )

    print(
        f"Period:              "
        f"{df['kickoff'].min()} → "
        f"{df['kickoff'].max()}"
    )

    print()
    print("Target:")
    print(
        df[TARGET].value_counts()
    )

    all_results = []

    for fold in FOLDS:

        train_end = fold["train_end"]
        val_end = fold["val_end"]
        test_end = fold["test_end"]

        train = df.iloc[
            :train_end
        ]

        validation = df.iloc[
            train_end:val_end
        ]

        test = df.iloc[
            val_end:test_end
        ]

        print()
        print("=" * 80)
        print(
            fold["name"]
        )
        print("=" * 80)

        print(
            f"Train:      {len(train)}"
        )

        print(
            f"Validation: {len(validation)}"
        )

        print(
            f"Test:       {len(test)}"
        )

        history = TeamHistory()

        # Только train-матчи формируют
        # начальную историю.
        for _, row in train.iterrows():
            add_match_to_history(
                history,
                row,
            )

        validation_result = evaluate_period(
            validation,
            history,
            f"{fold['name']} — VALIDATION",
        )

        all_results.append(
            {
                "fold": fold["name"],
                "period": "validation",
                "matches": validation_result["matches"],
                "accuracy": validation_result["accuracy"],
                "logloss": validation_result["logloss"],
                "draw_precision": validation_result[
                    "draw_precision"
                ],
                "draw_recall": validation_result[
                    "draw_recall"
                ],
                "draw_f1": validation_result[
                    "draw_f1"
                ],
            }
        )

        test_result = evaluate_period(
            test,
            history,
            f"{fold['name']} — TEST",
        )

        all_results.append(
            {
                "fold": fold["name"],
                "period": "test",
                "matches": test_result["matches"],
                "accuracy": test_result["accuracy"],
                "logloss": test_result["logloss"],
                "draw_precision": test_result[
                    "draw_precision"
                ],
                "draw_recall": test_result[
                    "draw_recall"
                ],
                "draw_f1": test_result[
                    "draw_f1"
                ],
            }
        )

        examples = pd.DataFrame(
            test_result["records"]
        )

        if not examples.empty:

            examples["confidence"] = (
                examples[
                    [
                        "p_home",
                        "p_draw",
                        "p_away",
                    ]
                ].max(axis=1)
            )

            print()
            print(
                "Самые уверенные TEST predictions:"
            )

            print(
                examples
                .sort_values(
                    "confidence",
                    ascending=False,
                )
                .head(10)
                [
                    [
                        "fixture_id",
                        "home_team_id",
                        "away_team_id",
                        "actual",
                        "prediction",
                        "home_lambda",
                        "away_lambda",
                        "p_home",
                        "p_draw",
                        "p_away",
                        "confidence",
                    ]
                ]
                .to_string(
                    index=False
                )
            )

    results = pd.DataFrame(
        all_results
    )

    print()
    print("=" * 80)
    print("WALK-FORWARD SUMMARY")
    print("=" * 80)

    print(
        results.to_string(
            index=False
        )
    )

    test_results = results[
        results["period"] == "test"
    ]

    if not test_results.empty:

        print()
        print("=" * 80)
        print("MEAN TEST RESULTS")
        print("=" * 80)

        print(
            f"Accuracy:       "
            f"{test_results['accuracy'].mean():.2%}"
        )

        print(
            f"LogLoss:        "
            f"{test_results['logloss'].mean():.4f}"
        )

        print(
            f"Draw precision: "
            f"{test_results['draw_precision'].mean():.2%}"
        )

        print(
            f"Draw recall:    "
            f"{test_results['draw_recall'].mean():.2%}"
        )

        print(
            f"Draw F1:        "
            f"{test_results['draw_f1'].mean():.2%}"
        )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
