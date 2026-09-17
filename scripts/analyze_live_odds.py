from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from app.database.session import SessionLocal
from app.services.prediction_service import PredictionService


ODDS_PATH = Path("data/reports/odds/odds_snapshots.csv")
OUTPUT_PATH = Path("data/reports/odds/live_odds_analysis.csv")


def safe_float(value):
    try:
        value = float(value)
        if np.isnan(value):
            return None
        return value
    except (TypeError, ValueError):
        return None


def fair_odds(probability):
    if probability is None or probability <= 0:
        return None
    return 1.0 / probability


def calculate_ev(probability, odds):
    if probability is None or odds is None:
        return None

    if probability <= 0 or odds <= 1:
        return None

    return probability * odds - 1.0


def no_vig_probabilities(home_odds, draw_odds, away_odds):
    values = [
        home_odds,
        draw_odds,
        away_odds,
    ]

    if any(
        value is None or value <= 0
        for value in values
    ):
        return None, None, None

    inverse = np.array(
        [
            1.0 / home_odds,
            1.0 / draw_odds,
            1.0 / away_odds,
        ],
        dtype=float,
    )

    total = inverse.sum()

    if total <= 0:
        return None, None, None

    probabilities = inverse / total

    return (
        float(probabilities[0]),
        float(probabilities[1]),
        float(probabilities[2]),
    )


def load_odds():
    if not ODDS_PATH.exists():
        raise FileNotFoundError(
            f"Файл odds не найден: {ODDS_PATH}"
        )

    df = pd.read_csv(
        ODDS_PATH
    )

    required = {
        "snapshot_at",
        "fixture_id",
        "fixture_api_id",
        "kickoff",
        "bookmaker",
        "home_odds",
        "draw_odds",
        "away_odds",
        "home_team_name",
        "away_team_name",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "В odds отсутствуют колонки: "
            f"{sorted(missing)}"
        )

    df["fixture_id"] = pd.to_numeric(
        df["fixture_id"],
        errors="coerce",
    )

    df["snapshot_at"] = pd.to_datetime(
        df["snapshot_at"],
        errors="coerce",
        utc=True,
    )

    df["kickoff"] = pd.to_datetime(
        df["kickoff"],
        errors="coerce",
    )

    for column in [
        "home_odds",
        "draw_odds",
        "away_odds",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "fixture_id",
            "snapshot_at",
        ]
    ).copy()

    return df


def get_latest_snapshot_per_fixture(df):
    """
    Берём самый свежий snapshot ДЛЯ КАЖДОГО fixture_id.

    Это важно:
    глобальный последний snapshot может содержать
    только один или несколько матчей.
    """

    latest_times = (
        df.groupby("fixture_id")[
            "snapshot_at"
        ]
        .max()
        .reset_index()
        .rename(
            columns={
                "snapshot_at": "latest_snapshot"
            }
        )
    )

    latest = df.merge(
        latest_times,
        on="fixture_id",
        how="inner",
    )

    latest = latest[
        latest["snapshot_at"]
        == latest["latest_snapshot"]
    ].copy()

    return latest


def get_best_and_average_odds(fixture_rows):
    return {
        "avg_home": safe_float(
            fixture_rows["home_odds"].mean()
        ),
        "avg_draw": safe_float(
            fixture_rows["draw_odds"].mean()
        ),
        "avg_away": safe_float(
            fixture_rows["away_odds"].mean()
        ),
        "best_home": safe_float(
            fixture_rows["home_odds"].max()
        ),
        "best_draw": safe_float(
            fixture_rows["draw_odds"].max()
        ),
        "best_away": safe_float(
            fixture_rows["away_odds"].max()
        ),
    }


def main():
    logger.info("=" * 78)
    logger.info("LIVE ODDS ANALYZER")
    logger.info("=" * 78)

    odds = load_odds()

    logger.info(
        f"Всего odds записей: {len(odds)}"
    )

    latest = get_latest_snapshot_per_fixture(
        odds
    )

    logger.info(
        "Актуальных bookmaker records: "
        f"{len(latest)}"
    )

    fixture_ids = sorted(
        latest["fixture_id"]
        .astype(int)
        .unique()
    )

    logger.info(
        f"Матчей для анализа: {len(fixture_ids)}"
    )

    session = SessionLocal()

    try:
        prediction_service = PredictionService(
            session
        )

        results = []

        for index, fixture_id in enumerate(
            fixture_ids,
            start=1,
        ):
            logger.info(
                f"[{index}/{len(fixture_ids)}] "
                f"fixture={fixture_id}"
            )

            fixture_rows = latest[
                latest["fixture_id"]
                == fixture_id
            ].copy()

            if fixture_rows.empty:
                continue

            try:
                prediction = (
                    prediction_service.predict(
                        int(fixture_id)
                    )
                )

            except Exception as exc:
                logger.exception(
                    f"Ошибка прогноза "
                    f"fixture={fixture_id}: {exc}"
                )
                continue

            probabilities = prediction.get(
                "probabilities",
                {},
            )

            p_home = safe_float(
                probabilities.get("home_win")
            )

            p_draw = safe_float(
                probabilities.get("draw")
            )

            p_away = safe_float(
                probabilities.get("away_win")
            )

            if any(
                value is None
                for value in [
                    p_home,
                    p_draw,
                    p_away,
                ]
            ):
                logger.warning(
                    "Нет полных вероятностей "
                    f"fixture={fixture_id}"
                )
                continue

            home_team = str(
                prediction.get(
                    "home_team",
                    "",
                )
            )

            away_team = str(
                prediction.get(
                    "away_team",
                    "",
                )
            )

            if not home_team:
                home_team = str(
                    fixture_rows[
                        "home_team_name"
                    ]
                    .dropna()
                    .iloc[0]
                )

            if not away_team:
                away_team = str(
                    fixture_rows[
                        "away_team_name"
                    ]
                    .dropna()
                    .iloc[0]
                )

            odds_values = (
                get_best_and_average_odds(
                    fixture_rows
                )
            )

            avg_home = odds_values[
                "avg_home"
            ]

            avg_draw = odds_values[
                "avg_draw"
            ]

            avg_away = odds_values[
                "avg_away"
            ]

            best_home = odds_values[
                "best_home"
            ]

            best_draw = odds_values[
                "best_draw"
            ]

            best_away = odds_values[
                "best_away"
            ]

            (
                market_p_home,
                market_p_draw,
                market_p_away,
            ) = no_vig_probabilities(
                avg_home,
                avg_draw,
                avg_away,
            )

            fair_home = fair_odds(
                p_home
            )

            fair_draw = fair_odds(
                p_draw
            )

            fair_away = fair_odds(
                p_away
            )

            model_market_home = (
                p_home - market_p_home
                if market_p_home is not None
                else None
            )

            model_market_draw = (
                p_draw - market_p_draw
                if market_p_draw is not None
                else None
            )

            model_market_away = (
                p_away - market_p_away
                if market_p_away is not None
                else None
            )

            ev_home = calculate_ev(
                p_home,
                best_home,
            )

            ev_draw = calculate_ev(
                p_draw,
                best_draw,
            )

            ev_away = calculate_ev(
                p_away,
                best_away,
            )

            predicted_result = prediction.get(
                "predicted_result"
            )

            confidence = safe_float(
                prediction.get(
                    "confidence"
                )
            )

            fixture_api_id = None

            api_values = (
                fixture_rows[
                    "fixture_api_id"
                ]
                .dropna()
                .tolist()
            )

            if api_values:
                fixture_api_id = int(
                    float(api_values[0])
                )

            result = {
                "fixture_id": int(
                    fixture_id
                ),

                "fixture_api_id":
                    fixture_api_id,

                "kickoff":
                    fixture_rows[
                        "kickoff"
                    ].iloc[0],

                "home_team":
                    home_team,

                "away_team":
                    away_team,

                "prediction":
                    predicted_result,

                "confidence":
                    confidence,

                "p_home":
                    p_home,

                "p_draw":
                    p_draw,

                "p_away":
                    p_away,

                "fair_home_odds":
                    fair_home,

                "fair_draw_odds":
                    fair_draw,

                "fair_away_odds":
                    fair_away,

                "avg_home_odds":
                    avg_home,

                "avg_draw_odds":
                    avg_draw,

                "avg_away_odds":
                    avg_away,

                "best_home_odds":
                    best_home,

                "best_draw_odds":
                    best_draw,

                "best_away_odds":
                    best_away,

                "market_p_home":
                    market_p_home,

                "market_p_draw":
                    market_p_draw,

                "market_p_away":
                    market_p_away,

                "model_market_home":
                    model_market_home,

                "model_market_draw":
                    model_market_draw,

                "model_market_away":
                    model_market_away,

                "ev_home":
                    ev_home,

                "ev_draw":
                    ev_draw,

                "ev_away":
                    ev_away,

                "bookmakers":
                    int(
                        fixture_rows[
                            "bookmaker"
                        ]
                        .nunique()
                    ),

                "latest_snapshot":
                    str(
                        fixture_rows[
                            "snapshot_at"
                        ].max()
                    ),
            }

            results.append(result)

            logger.info(
                f"{home_team} - {away_team}"
            )

            logger.info(
                "Model: "
                f"H={p_home:.4f} "
                f"D={p_draw:.4f} "
                f"A={p_away:.4f}"
            )

            logger.info(
                "Fair: "
                f"H={fair_home:.3f} "
                f"D={fair_draw:.3f} "
                f"A={fair_away:.3f}"
            )

            logger.info(
                "Best odds: "
                f"H={best_home:.2f} "
                f"D={best_draw:.2f} "
                f"A={best_away:.2f}"
            )

            logger.info(
                "EV: "
                f"H={ev_home:+.4f} "
                f"D={ev_draw:+.4f} "
                f"A={ev_away:+.4f}"
            )

    finally:
        session.close()

    if not results:
        raise RuntimeError(
            "Не удалось построить ни одного прогноза."
        )

    result_df = pd.DataFrame(
        results
    )

    result_df = result_df.sort_values(
        by="kickoff"
    ).reset_index(
        drop=True
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info("")
    logger.info("=" * 78)
    logger.success("RESULT")
    logger.info("=" * 78)

    logger.info(
        f"Сохранено матчей: "
        f"{len(result_df)}"
    )

    logger.info(
        f"Файл: {OUTPUT_PATH}"
    )

    print("")
    print(
        "Матч | H | D | A | Прогноз | "
        "Fair H/D/A | Best H/D/A | EV H/D/A"
    )

    print("-" * 150)

    for _, row in result_df.iterrows():

        match = (
            f"{row['home_team']} - "
            f"{row['away_team']}"
        )

        print(
            f"{match:45} | "
            f"{row['p_home']:.1%} | "
            f"{row['p_draw']:.1%} | "
            f"{row['p_away']:.1%} | "
            f"{row['prediction']} | "
            f"{row['fair_home_odds']:.2f}/"
            f"{row['fair_draw_odds']:.2f}/"
            f"{row['fair_away_odds']:.2f} | "
            f"{row['best_home_odds']:.2f}/"
            f"{row['best_draw_odds']:.2f}/"
            f"{row['best_away_odds']:.2f} | "
            f"{row['ev_home']:+.1%}/"
            f"{row['ev_draw']:+.1%}/"
            f"{row['ev_away']:+.1%}"
        )


if __name__ == "__main__":
    main()
