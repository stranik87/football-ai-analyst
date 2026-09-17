from pathlib import Path

import pandas as pd
from loguru import logger

from app.database.session import SessionLocal
from app.models.fixture import Fixture


ODDS_PATH = Path(
    "data/reports/odds/odds_snapshots.csv"
)

ANALYSIS_PATH = Path(
    "data/reports/odds/live_odds_analysis.csv"
)

OUTPUT_PATH = Path(
    "data/reports/odds/odds_backtest.csv"
)


FINISHED_STATUSES = {
    "FT",
    "AET",
    "PEN",
}


def get_result(home_goals, away_goals):
    """
    Определяет результат 1X2 по финальному счёту.

    H = победа хозяев
    D = ничья
    A = победа гостей
    """

    if pd.isna(home_goals) or pd.isna(away_goals):
        return None

    home_goals = int(home_goals)
    away_goals = int(away_goals)

    if home_goals > away_goals:
        return "H"

    if home_goals < away_goals:
        return "A"

    return "D"


def calculate_roi(odds, actual_result, selected_result):
    """
    ROI одной условной ставки 1 единицей.

    Если выбранный результат совпал:
        profit = odds - 1

    Если нет:
        profit = -1

    None, если данных недостаточно.
    """

    if pd.isna(odds):
        return None

    odds = float(odds)

    if odds <= 0:
        return None

    if selected_result not in {"H", "D", "A"}:
        return None

    if actual_result not in {"H", "D", "A"}:
        return None

    if selected_result == actual_result:
        return odds - 1.0

    return -1.0


def main():
    logger.info("=" * 78)
    logger.info("ODDS SETTLEMENT / BACKTEST DATA BUILDER")
    logger.info("=" * 78)

    if not ODDS_PATH.exists():
        raise FileNotFoundError(
            f"Odds файл не найден: {ODDS_PATH}"
        )

    odds_df = pd.read_csv(
        ODDS_PATH
    )

    if odds_df.empty:
        logger.warning(
            "Odds CSV пустой."
        )
        return

    required_odds_columns = {
        "snapshot_at",
        "fixture_id",
        "fixture_api_id",
        "kickoff",
        "status_short",
        "home_team_id",
        "away_team_id",
        "bookmaker_id",
        "bookmaker",
        "home_odds",
        "draw_odds",
        "away_odds",
    }

    missing = (
        required_odds_columns
        - set(odds_df.columns)
    )

    if missing:
        raise ValueError(
            "В odds CSV отсутствуют колонки: "
            f"{sorted(missing)}"
        )

    logger.info(
        f"Odds записей: {len(odds_df)}"
    )

    # ---------------------------------------------------------
    # Загружаем существующий live analysis.
    # ---------------------------------------------------------

    analysis_df = None

    if ANALYSIS_PATH.exists():
        analysis_df = pd.read_csv(
            ANALYSIS_PATH
        )

        logger.info(
            f"Live analysis записей: "
            f"{len(analysis_df)}"
        )
    else:
        logger.warning(
            f"Live analysis не найден: "
            f"{ANALYSIS_PATH}"
        )

    # ---------------------------------------------------------
    # Уникальные fixture_id.
    # ---------------------------------------------------------

    fixture_ids = (
        odds_df["fixture_id"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    logger.info(
        f"Уникальных матчей: "
        f"{len(fixture_ids)}"
    )

    session = SessionLocal()

    try:
        fixtures = (
            session.query(Fixture)
            .filter(
                Fixture.id.in_(fixture_ids)
            )
            .all()
        )

        fixture_map = {
            int(fixture.id): fixture
            for fixture in fixtures
        }

        logger.info(
            f"Матчей найдено в DB: "
            f"{len(fixture_map)}"
        )

        missing_fixtures = (
            set(fixture_ids)
            - set(fixture_map.keys())
        )

        if missing_fixtures:
            logger.warning(
                "Матчи отсутствуют в DB: "
                f"{sorted(missing_fixtures)}"
            )

        records = []

        for _, row in odds_df.iterrows():
            fixture_id = int(
                row["fixture_id"]
            )

            fixture = fixture_map.get(
                fixture_id
            )

            if fixture is None:
                continue

            status = (
                str(fixture.status_short or "")
                .upper()
                .strip()
            )

            home_goals = fixture.home_goals
            away_goals = fixture.away_goals

            actual_result = get_result(
                home_goals,
                away_goals,
            )

            is_finished = (
                status in FINISHED_STATUSES
                and actual_result is not None
            )

            record = {
                "snapshot_at":
                    row["snapshot_at"],

                "fixture_id":
                    fixture_id,

                "fixture_api_id":
                    int(fixture.api_id),

                "kickoff":
                    row["kickoff"],

                "status_at_settlement":
                    status,

                "home_team_id":
                    int(row["home_team_id"]),

                "away_team_id":
                    int(row["away_team_id"]),

                "bookmaker_id":
                    int(row["bookmaker_id"]),

                "bookmaker":
                    row["bookmaker"],

                "home_odds":
                    row["home_odds"],

                "draw_odds":
                    row["draw_odds"],

                "away_odds":
                    row["away_odds"],

                "home_goals":
                    home_goals,

                "away_goals":
                    away_goals,

                "actual_result":
                    actual_result,

                "is_finished":
                    is_finished,
            }

            # -------------------------------------------------
            # Если есть live analysis — присоединяем модель.
            # -------------------------------------------------

            if (
                analysis_df is not None
                and not analysis_df.empty
            ):
                fixture_analysis = analysis_df[
                    analysis_df["fixture_id"].astype(int)
                    == fixture_id
                ]

                if not fixture_analysis.empty:
                    # Берём самый свежий analysis
                    # для этого fixture.
                    fixture_analysis = (
                        fixture_analysis
                        .sort_values(
                            "analysis_at"
                        )
                        .iloc[-1]
                    )

                    record.update(
                        {
                            "model_prediction":
                                fixture_analysis.get(
                                    "prediction"
                                ),

                            "model_p_home":
                                fixture_analysis.get(
                                    "p_home"
                                ),

                            "model_p_draw":
                                fixture_analysis.get(
                                    "p_draw"
                                ),

                            "model_p_away":
                                fixture_analysis.get(
                                    "p_away"
                                ),

                            "model_fair_home_odds":
                                fixture_analysis.get(
                                    "fair_home_odds"
                                ),

                            "model_fair_draw_odds":
                                fixture_analysis.get(
                                    "fair_draw_odds"
                                ),

                            "model_fair_away_odds":
                                fixture_analysis.get(
                                    "fair_away_odds"
                                ),

                            "analysis_at":
                                fixture_analysis.get(
                                    "analysis_at"
                                ),
                        }
                    )

            # -------------------------------------------------
            # Если матч завершён — считаем outcome каждой
            # из трёх возможных ставок.
            # -------------------------------------------------

            if is_finished:
                record[
                    "home_bet_roi"
                ] = calculate_roi(
                    row["home_odds"],
                    actual_result,
                    "H",
                )

                record[
                    "draw_bet_roi"
                ] = calculate_roi(
                    row["draw_odds"],
                    actual_result,
                    "D",
                )

                record[
                    "away_bet_roi"
                ] = calculate_roi(
                    row["away_odds"],
                    actual_result,
                    "A",
                )

                model_prediction = record.get(
                    "model_prediction"
                )

                if model_prediction in {
                    "H",
                    "D",
                    "A",
                }:
                    if model_prediction == "H":
                        selected_odds = row[
                            "home_odds"
                        ]
                    elif model_prediction == "D":
                        selected_odds = row[
                            "draw_odds"
                        ]
                    else:
                        selected_odds = row[
                            "away_odds"
                        ]

                    record[
                        "model_selected_roi"
                    ] = calculate_roi(
                        selected_odds,
                        actual_result,
                        model_prediction,
                    )

                    record[
                        "model_selected_odds"
                    ] = selected_odds

                    record[
                        "model_was_correct"
                    ] = (
                        model_prediction
                        == actual_result
                    )
                else:
                    record[
                        "model_selected_roi"
                    ] = None

                    record[
                        "model_selected_odds"
                    ] = None

                    record[
                        "model_was_correct"
                    ] = None

            else:
                record[
                    "home_bet_roi"
                ] = None

                record[
                    "draw_bet_roi"
                ] = None

                record[
                    "away_bet_roi"
                ] = None

                record[
                    "model_selected_roi"
                ] = None

                record[
                    "model_selected_odds"
                ] = None

                record[
                    "model_was_correct"
                ] = None

            records.append(record)

    finally:
        session.close()

    if not records:
        logger.warning(
            "Не удалось создать records."
        )
        return

    result_df = pd.DataFrame(
        records
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
    logger.info("RESULT")
    logger.info("=" * 78)

    logger.info(
        f"Всего odds записей: "
        f"{len(result_df)}"
    )

    logger.info(
        f"Завершённых записей: "
        f"{int(result_df['is_finished'].sum())}"
    )

    logger.info(
        f"Незавершённых записей: "
        f"{int((~result_df['is_finished']).sum())}"
    )

    logger.info(
        f"Файл: {OUTPUT_PATH}"
    )

    if result_df["is_finished"].any():
        finished = result_df[
            result_df["is_finished"]
        ].copy()

        logger.info("")
        logger.info(
            "Результаты завершённых матчей:"
        )

        summary = (
            finished[
                [
                    "fixture_id",
                    "home_goals",
                    "away_goals",
                    "actual_result",
                ]
            ]
            .drop_duplicates(
                subset=["fixture_id"]
            )
        )

        print(
            summary.to_string(
                index=False
            )
        )

        model_rows = finished[
            finished[
                "model_prediction"
            ].isin(
                ["H", "D", "A"]
            )
        ].copy()

        if not model_rows.empty:
            logger.info("")
            logger.info(
                "Модельные predictions:"
            )

            model_summary = (
                model_rows[
                    [
                        "fixture_id",
                        "model_prediction",
                        "actual_result",
                        "model_was_correct",
                    ]
                ]
                .drop_duplicates(
                    subset=["fixture_id"]
                )
            )

            print(
                model_summary.to_string(
                    index=False
                )
            )

    else:
        logger.info(
            "Пока нет завершённых матчей "
            "с сохранёнными odds."
        )


if __name__ == "__main__":
    main()
