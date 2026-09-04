from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss


DATASET_PATH = Path("data/datasets/matches_dataset.csv")

REPORT_PATH = Path(
    "data/reports/poisson_v3_results.csv"
)

SUMMARY_PATH = Path(
    "data/reports/poisson_v3_summary.csv"
)

TARGET = "result"

MAX_GOALS = 10

ALPHAS = [0.1, 0.3, 1.0]


# ============================================================
# CLEAN EXPANDING WALK-FORWARD
# ============================================================
#
# Fold 1:
#   train 0:926
#   validation 926:1019  = 93
#   test 1019:1204       = 185
#
# Fold 2:
#   train 0:1204
#   validation 1204:1297 = 93
#   test 1297:1482       = 185
#
# Fold 3:
#   train 0:1482
#   validation 1482:1575 = 93
#   test 1575:1760       = 185
#
# Последние 93 матча (5%) пока остаются финальным holdout.
#
# Важно:
# предыдущий test-период может войти в train
# следующего expanding fold. Это нормально для
# walk-forward схемы.
# ============================================================

FOLDS = [
    {
        "name": "Fold 1",
        "train_end": 926,
        "val_end": 1019,
        "test_end": 1204,
    },
    {
        "name": "Fold 2",
        "train_end": 1204,
        "val_end": 1297,
        "test_end": 1482,
    },
    {
        "name": "Fold 3",
        "train_end": 1482,
        "val_end": 1575,
        "test_end": 1760,
    },
]


@dataclass
class TeamStats:
    matches: int = 0

    goals_for: list[float] | None = None
    goals_against: list[float] | None = None

    home_goals_for: list[float] | None = None
    home_goals_against: list[float] | None = None

    away_goals_for: list[float] | None = None
    away_goals_against: list[float] | None = None

    def __post_init__(self) -> None:
        self.goals_for = []
        self.goals_against = []

        self.home_goals_for = []
        self.home_goals_against = []

        self.away_goals_for = []
        self.away_goals_against = []


class TeamHistory:

    def __init__(self) -> None:
        self.teams: dict[int, TeamStats] = {}

    def _get(
        self,
        team_id: int,
    ) -> TeamStats:

        if team_id not in self.teams:
            self.teams[team_id] = TeamStats()

        return self.teams[team_id]

    def add_match(
        self,
        team_id: int,
        goals_for: float,
        goals_against: float,
        is_home: bool,
    ) -> None:

        stats = self._get(team_id)

        stats.matches += 1

        stats.goals_for.append(
            float(goals_for)
        )

        stats.goals_against.append(
            float(goals_against)
        )

        if is_home:

            stats.home_goals_for.append(
                float(goals_for)
            )

            stats.home_goals_against.append(
                float(goals_against)
            )

        else:

            stats.away_goals_for.append(
                float(goals_for)
            )

            stats.away_goals_against.append(
                float(goals_against)
            )


def safe_mean(
    values: list[float],
    default: float,
) -> float:

    if not values:
        return default

    return float(
        np.mean(values)
    )


def recent_mean(
    values: list[float],
    window: int,
    default: float,
) -> float:

    if not values:
        return default

    return float(
        np.mean(values[-window:])
    )


def calculate_league_averages(
    history: TeamHistory,
) -> dict[str, float]:

    home_goals: list[float] = []
    away_goals: list[float] = []

    all_goals_for: list[float] = []
    all_goals_against: list[float] = []

    for stats in history.teams.values():

        home_goals.extend(
            stats.home_goals_for
        )

        away_goals.extend(
            stats.away_goals_for
        )

        all_goals_for.extend(
            stats.goals_for
        )

        all_goals_against.extend(
            stats.goals_against
        )

    return {
        "home_goals": safe_mean(
            home_goals,
            1.35,
        ),
        "away_goals": safe_mean(
            away_goals,
            1.10,
        ),
        "goals_for": safe_mean(
            all_goals_for,
            1.25,
        ),
        "goals_against": safe_mean(
            all_goals_against,
            1.25,
        ),
    }


def team_features(
    history: TeamHistory,
    team_id: int,
    is_home: bool,
    league: dict[str, float],
) -> dict[str, float]:

    stats = history.teams.get(
        team_id
    )

    if stats is None or stats.matches == 0:

        return {
            "matches": 0.0,

            "gf_5": league["goals_for"],
            "gf_10": league["goals_for"],
            "gf_20": league["goals_for"],

            "ga_5": league["goals_against"],
            "ga_10": league["goals_against"],
            "ga_20": league["goals_against"],

            "venue_gf_5": (
                league["home_goals"]
                if is_home
                else league["away_goals"]
            ),

            "venue_ga_5": (
                league["goals_against"]
            ),

            "overall_gf": league["goals_for"],
            "overall_ga": league["goals_against"],
        }

    if is_home:

        venue_gf = stats.home_goals_for
        venue_ga = stats.home_goals_against

    else:

        venue_gf = stats.away_goals_for
        venue_ga = stats.away_goals_against

    venue_gf_default = (
        league["home_goals"]
        if is_home
        else league["away_goals"]
    )

    return {
        "matches": float(
            stats.matches
        ),

        "gf_5": recent_mean(
            stats.goals_for,
            5,
            league["goals_for"],
        ),

        "gf_10": recent_mean(
            stats.goals_for,
            10,
            league["goals_for"],
        ),

        "gf_20": recent_mean(
            stats.goals_for,
            20,
            league["goals_for"],
        ),

        "ga_5": recent_mean(
            stats.goals_against,
            5,
            league["goals_against"],
        ),

        "ga_10": recent_mean(
            stats.goals_against,
            10,
            league["goals_against"],
        ),

        "ga_20": recent_mean(
            stats.goals_against,
            20,
            league["goals_against"],
        ),

        "venue_gf_5": recent_mean(
            venue_gf,
            5,
            venue_gf_default,
        ),

        "venue_ga_5": recent_mean(
            venue_ga,
            5,
            league["goals_against"],
        ),

        "overall_gf": safe_mean(
            stats.goals_for,
            league["goals_for"],
        ),

        "overall_ga": safe_mean(
            stats.goals_against,
            league["goals_against"],
        ),
    }


def build_match_features(
    history: TeamHistory,
    home_id: int,
    away_id: int,
) -> tuple[np.ndarray, dict[str, float]]:

    league = calculate_league_averages(
        history
    )

    home = team_features(
        history,
        home_id,
        True,
        league,
    )

    away = team_features(
        history,
        away_id,
        False,
        league,
    )

    features = {
        "home_matches": home["matches"],
        "away_matches": away["matches"],

        "home_gf_5": home["gf_5"],
        "away_gf_5": away["gf_5"],

        "home_gf_10": home["gf_10"],
        "away_gf_10": away["gf_10"],

        "home_gf_20": home["gf_20"],
        "away_gf_20": away["gf_20"],

        "home_ga_5": home["ga_5"],
        "away_ga_5": away["ga_5"],

        "home_ga_10": home["ga_10"],
        "away_ga_10": away["ga_10"],

        "home_ga_20": home["ga_20"],
        "away_ga_20": away["ga_20"],

        "home_venue_gf_5": (
            home["venue_gf_5"]
        ),

        "away_venue_gf_5": (
            away["venue_gf_5"]
        ),

        "home_venue_ga_5": (
            home["venue_ga_5"]
        ),

        "away_venue_ga_5": (
            away["venue_ga_5"]
        ),

        "home_overall_gf": (
            home["overall_gf"]
        ),

        "away_overall_gf": (
            away["overall_gf"]
        ),

        "home_overall_ga": (
            home["overall_ga"]
        ),

        "away_overall_ga": (
            away["overall_ga"]
        ),

        "gf_difference_5": (
            home["gf_5"]
            - away["gf_5"]
        ),

        "gf_difference_10": (
            home["gf_10"]
            - away["gf_10"]
        ),

        "gf_difference_20": (
            home["gf_20"]
            - away["gf_20"]
        ),

        "ga_difference_5": (
            home["ga_5"]
            - away["ga_5"]
        ),

        "ga_difference_10": (
            home["ga_10"]
            - away["ga_10"]
        ),

        "ga_difference_20": (
            home["ga_20"]
            - away["ga_20"]
        ),

        "venue_attack_difference": (
            home["venue_gf_5"]
            - away["venue_gf_5"]
        ),

        "venue_defense_difference": (
            home["venue_ga_5"]
            - away["venue_ga_5"]
        ),
    }

    return (
        np.array(
            list(features.values()),
            dtype=float,
        ),
        features,
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


def build_training_matrix(
    rows: pd.DataFrame,
    history: TeamHistory,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    X: list[np.ndarray] = []

    y_home: list[float] = []
    y_away: list[float] = []

    for _, row in rows.iterrows():

        home_id = int(
            row["home_team_id"]
        )

        away_id = int(
            row["away_team_id"]
        )

        features, _ = build_match_features(
            history,
            home_id,
            away_id,
        )

        X.append(features)

        y_home.append(
            float(row["home_goals"])
        )

        y_away.append(
            float(row["away_goals"])
        )

        # ВАЖНО:
        # текущий матч попадает в историю
        # только после формирования признаков.
        add_match_to_history(
            history,
            row,
        )

    return (
        np.vstack(X),
        np.array(y_home),
        np.array(y_away),
    )


def poisson_pmf(
    k: int,
    lam: float,
) -> float:

    return (
        exp(-lam)
        * (lam ** k)
        / factorial(k)
    )


def match_probabilities(
    home_lambda: float,
    away_lambda: float,
) -> tuple[float, float, float]:

    home_probs = np.array(
        [
            poisson_pmf(
                i,
                home_lambda,
            )
            for i in range(
                MAX_GOALS + 1
            )
        ]
    )

    away_probs = np.array(
        [
            poisson_pmf(
                i,
                away_lambda,
            )
            for i in range(
                MAX_GOALS + 1
            )
        ]
    )

    # Нормируем хвост после MAX_GOALS.
    home_probs /= home_probs.sum()
    away_probs /= away_probs.sum()

    matrix = np.outer(
        home_probs,
        away_probs,
    )

    # matrix[i, j]:
    # i = голы хозяев
    # j = голы гостей
    #
    # H: i > j
    # D: i == j
    # A: i < j

    p_home = float(
        np.tril(
            matrix,
            -1,
        ).sum()
    )

    p_draw = float(
        np.trace(matrix)
    )

    p_away = float(
        np.triu(
            matrix,
            1,
        ).sum()
    )

    total = (
        p_home
        + p_draw
        + p_away
    )

    return (
        p_home / total,
        p_draw / total,
        p_away / total,
    )


def train_models(
    X: np.ndarray,
    y_home: np.ndarray,
    y_away: np.ndarray,
    alpha: float,
) -> tuple[
    PoissonRegressor,
    PoissonRegressor,
]:

    home_model = PoissonRegressor(
        alpha=alpha,
        max_iter=1000,
    )

    away_model = PoissonRegressor(
        alpha=alpha,
        max_iter=1000,
    )

    home_model.fit(
        X,
        y_home,
    )

    away_model.fit(
        X,
        y_away,
    )

    return (
        home_model,
        away_model,
    )


def calculate_metrics(
    y_true: list[str],
    y_pred: list[str],
    probabilities: list[list[float]],
) -> dict[str, float]:

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

    return {
        "accuracy": float(
            accuracy
        ),
        "logloss": float(
            logloss
        ),
        "draw_precision": float(
            draw_precision
        ),
        "draw_recall": float(
            draw_recall
        ),
        "draw_f1": float(
            draw_f1
        ),
    }


def evaluate_period(
    rows: pd.DataFrame,
    history: TeamHistory,
    home_model: PoissonRegressor,
    away_model: PoissonRegressor,
    period_name: str,
) -> dict:

    y_true: list[str] = []
    y_pred: list[str] = []

    probabilities: list[
        list[float]
    ] = []

    records: list[dict] = []

    for _, row in rows.iterrows():

        home_id = int(
            row["home_team_id"]
        )

        away_id = int(
            row["away_team_id"]
        )

        X, feature_dict = (
            build_match_features(
                history,
                home_id,
                away_id,
            )
        )

        X = X.reshape(
            1,
            -1,
        )

        home_lambda = float(
            home_model.predict(X)[0]
        )

        away_lambda = float(
            away_model.predict(X)[0]
        )

        home_lambda = float(
            np.clip(
                home_lambda,
                0.10,
                4.00,
            )
        )

        away_lambda = float(
            np.clip(
                away_lambda,
                0.10,
                4.00,
            )
        )

        (
            p_home,
            p_draw,
            p_away,
        ) = match_probabilities(
            home_lambda,
            away_lambda,
        )

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

        y_true.append(
            actual
        )

        y_pred.append(
            prediction
        )

        # sklearn labels:
        # A, D, H
        probabilities.append(
            [
                p_away,
                p_draw,
                p_home,
            ]
        )

        records.append(
            {
                "fixture_id": row[
                    "fixture_id"
                ],
                "kickoff": row[
                    "kickoff"
                ],
                "home_team_id": home_id,
                "away_team_id": away_id,
                "actual": actual,
                "prediction": prediction,
                "home_lambda": home_lambda,
                "away_lambda": away_lambda,
                "p_home": p_home,
                "p_draw": p_draw,
                "p_away": p_away,
                **feature_dict,
            }
        )

        # Текущий результат добавляется
        # в историю только после прогноза.
        add_match_to_history(
            history,
            row,
        )

    metrics = calculate_metrics(
        y_true,
        y_pred,
        probabilities,
    )

    y_true_np = np.array(
        y_true
    )

    y_pred_np = np.array(
        y_pred
    )

    print()
    print("-" * 90)
    print(period_name)
    print("-" * 90)

    print(
        f"Matches:          {len(rows)}"
    )

    print(
        f"Accuracy:         "
        f"{metrics['accuracy']:.2%}"
    )

    print(
        f"LogLoss:          "
        f"{metrics['logloss']:.4f}"
    )

    print(
        f"Predicted draws:  "
        f"{int((y_pred_np == 'D').sum())}"
    )

    print(
        f"Actual draws:     "
        f"{int((y_true_np == 'D').sum())}"
    )

    print(
        f"Draw precision:   "
        f"{metrics['draw_precision']:.2%}"
    )

    print(
        f"Draw recall:      "
        f"{metrics['draw_recall']:.2%}"
    )

    print(
        f"Draw F1:          "
        f"{metrics['draw_f1']:.2%}"
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
            f"{cls} accuracy:     "
            f"{cls_accuracy:.2%}"
        )

    return {
        **metrics,
        "matches": len(rows),
        "records": records,
    }


def prepare_history(
    rows: pd.DataFrame,
) -> TeamHistory:

    history = TeamHistory()

    for _, row in rows.iterrows():

        add_match_to_history(
            history,
            row,
        )

    return history


def evaluate_alpha(
    df: pd.DataFrame,
    alpha: float,
) -> list[dict]:

    all_results: list[dict] = []

    for fold in FOLDS:

        train_end = fold[
            "train_end"
        ]

        val_end = fold[
            "val_end"
        ]

        test_end = fold[
            "test_end"
        ]

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
        print("=" * 90)

        print(
            f"{fold['name']} | "
            f"alpha={alpha}"
        )

        print("=" * 90)

        print(
            f"Train:      {len(train)}"
        )

        print(
            f"Validation: {len(validation)}"
        )

        print(
            f"Test:       {len(test)}"
        )

        # ====================================================
        # TRAIN
        # ====================================================

        train_history = TeamHistory()

        (
            X_train,
            y_home,
            y_away,
        ) = build_training_matrix(
            train,
            train_history,
        )

        print(
            f"Training matrix: "
            f"{X_train.shape}"
        )

        home_model, away_model = (
            train_models(
                X_train,
                y_home,
                y_away,
                alpha,
            )
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        validation_history = (
            prepare_history(train)
        )

        validation_result = (
            evaluate_period(
                validation,
                validation_history,
                home_model,
                away_model,
                f"{fold['name']} -> VALIDATION",
            )
        )

        all_results.append(
            {
                "alpha": alpha,
                "fold": fold["name"],
                "period": "validation",
                "matches": validation_result[
                    "matches"
                ],
                "accuracy": validation_result[
                    "accuracy"
                ],
                "logloss": validation_result[
                    "logloss"
                ],
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

        # ====================================================
        # TEST
        # ====================================================
        #
        # Для test используем всю историю:
        # train + validation.
        #
        # Но модель НЕ переобучаем на validation.
        # Это важно: test остаётся честным.
        # ====================================================

        test_history = prepare_history(
            pd.concat(
                [
                    train,
                    validation,
                ],
                ignore_index=True,
            )
        )

        test_result = evaluate_period(
            test,
            test_history,
            home_model,
            away_model,
            f"{fold['name']} -> TEST",
        )

        all_results.append(
            {
                "alpha": alpha,
                "fold": fold["name"],
                "period": "test",
                "matches": test_result[
                    "matches"
                ],
                "accuracy": test_result[
                    "accuracy"
                ],
                "logloss": test_result[
                    "logloss"
                ],
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

        # ====================================================
        # SAVE TEST PREDICTIONS
        # ====================================================

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
                .head(5)
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

            # Для Hybrid нам особенно важны
            # вероятности каждой модели.
            #
            # Сохраняем все TEST predictions.
            prediction_path = Path(
                "data/reports/"
                f"poisson_v3_"
                f"{alpha}_"
                f"{fold['name'].lower().replace(' ', '_')}_"
                f"test.csv"
            )

            examples.to_csv(
                prediction_path,
                index=False,
            )

    return all_results


def main() -> None:

    print("=" * 90)
    print(
        "POISSON V3 / CLEAN "
        "WALK-FORWARD"
    )
    print("=" * 90)

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: "
            f"{DATASET_PATH}"
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

    df = (
        df.sort_values(
            "kickoff"
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        f"Dataset:  {DATASET_PATH}"
    )

    print(
        f"Matches:  {len(df)}"
    )

    print(
        f"Period:   "
        f"{df['kickoff'].min()} -> "
        f"{df['kickoff'].max()}"
    )

    print()
    print("Target:")

    print(
        df[TARGET].value_counts()
    )

    # ========================================================
    # CHECK FOLDS
    # ========================================================

    for fold in FOLDS:

        if fold["test_end"] > len(df):

            raise ValueError(
                f"{fold['name']} выходит "
                f"за пределы dataset."
            )

        if not (
            fold["train_end"]
            < fold["val_end"]
            < fold["test_end"]
        ):

            raise ValueError(
                f"Некорректные границы "
                f"{fold['name']}."
            )

    # ========================================================
    # RUN ALL ALPHAS
    # ========================================================

    all_results: list[dict] = []

    for alpha in ALPHAS:

        print()
        print("#" * 90)

        print(
            f"ALPHA = {alpha}"
        )

        print("#" * 90)

        alpha_results = (
            evaluate_alpha(
                df,
                alpha,
            )
        )

        all_results.extend(
            alpha_results
        )

    results = pd.DataFrame(
        all_results
    )

    # ========================================================
    # FULL RESULTS
    # ========================================================

    print()
    print("=" * 90)
    print(
        "POISSON V3 RESULTS"
    )
    print("=" * 90)

    print(
        results.to_string(
            index=False
        )
    )

    # ========================================================
    # TEST SUMMARY
    # ========================================================

    test_results = results[
        results["period"] == "test"
    ]

    if not test_results.empty:

        summary = (
            test_results
            .groupby("alpha")
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
            )
            .reset_index()
        )

        print()
        print("=" * 90)
        print(
            "MEAN TEST RESULTS"
        )
        print("=" * 90)

        print(
            summary.to_string(
                index=False
            )
        )

        # ====================================================
        # VALIDATION-BASED ALPHA SELECTION
        # ====================================================

        validation_results = results[
            results["period"]
            == "validation"
        ]

        validation_summary = (
            validation_results
            .groupby("alpha")
            .agg(
                mean_validation_logloss=(
                    "logloss",
                    "mean",
                ),
                mean_validation_accuracy=(
                    "accuracy",
                    "mean",
                ),
            )
            .reset_index()
            .sort_values(
                "mean_validation_logloss"
            )
        )

        best_alpha = float(
            validation_summary.iloc[0][
                "alpha"
            ]
        )

        print()
        print(
            "VALIDATION SELECTION"
        )

        print(
            validation_summary.to_string(
                index=False
            )
        )

        print()
        print(
            f"Best alpha by validation "
            f"LogLoss: {best_alpha}"
        )

    else:

        summary = pd.DataFrame()

    # ========================================================
    # SAVE REPORTS
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
        "POISSON V3 DONE"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
