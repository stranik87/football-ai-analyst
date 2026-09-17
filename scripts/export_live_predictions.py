from pathlib import Path

import pandas as pd
from loguru import logger

from app.database.database import SessionLocal
from app.models.fixture import Fixture
from app.services.prediction_service import PredictionService


BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "reports"
    / "live_predictions.csv"
)

LOOKAHEAD_DAYS = 7
LIMIT = 30


def main() -> None:
    logger.info("=" * 70)
    logger.info("LIVE PREDICTIONS")
    logger.info("=" * 70)

    session = SessionLocal()

    try:
        now = pd.Timestamp.now()
        end = now + pd.Timedelta(days=LOOKAHEAD_DAYS)

        fixtures = (
            session.query(Fixture)
            .filter(
                Fixture.kickoff >= now.to_pydatetime(),
                Fixture.kickoff <= end.to_pydatetime(),
                Fixture.home_team_id.isnot(None),
                Fixture.away_team_id.isnot(None),
            )
            .order_by(Fixture.kickoff.asc())
            .limit(LIMIT)
            .all()
        )

        logger.info(f"Найдено будущих матчей: {len(fixtures)}")

        if not fixtures:
            logger.warning("Будущих матчей не найдено.")
            return

        predictor = PredictionService(session)

        rows = []

        for index, fixture in enumerate(fixtures, 1):
            logger.info("")
            logger.info(
                f"[{index}/{len(fixtures)}] "
                f"fixture={fixture.id}"
            )

            try:
                prediction = predictor.predict(fixture.id)

                home = prediction["home_team"]
                away = prediction["away_team"]

                p_home = prediction["probabilities"]["home_win"]
                p_draw = prediction["probabilities"]["draw"]
                p_away = prediction["probabilities"]["away_win"]

                predicted_result = prediction["predicted_result"]

                rows.append(
                    {
                        "fixture_id": fixture.id,
                        "kickoff": fixture.kickoff,
                        "home_team_id": fixture.home_team_id,
                        "away_team_id": fixture.away_team_id,
                        "home_team": home,
                        "away_team": away,
                        "predicted_result": predicted_result,
                        "predicted_result_name": prediction[
                            "predicted_result_name"
                        ],
                        "confidence": p_home
                        if predicted_result == "H"
                        else p_draw
                        if predicted_result == "D"
                        else p_away,
                        "p_home": p_home,
                        "p_draw": p_draw,
                        "p_away": p_away,
                    }
                )

                logger.success(
                    f"{home} - {away} | "
                    f"H={p_home:.3f} "
                    f"D={p_draw:.3f} "
                    f"A={p_away:.3f} | "
                    f"Прогноз={predicted_result}"
                )

            except Exception as exc:
                logger.exception(
                    f"Ошибка prediction fixture={fixture.id}: {exc}"
                )

        if not rows:
            raise RuntimeError(
                "Не удалось получить ни одного прогноза."
            )

        df = pd.DataFrame(rows)

        df["kickoff"] = pd.to_datetime(
            df["kickoff"],
            errors="coerce",
        )

        df = df.sort_values("kickoff").reset_index(drop=True)

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        logger.info("")
        logger.info("=" * 70)
        logger.success("ПРОГНОЗЫ СОХРАНЕНЫ")
        logger.info("=" * 70)
        logger.info(f"Матчей: {len(df)}")
        logger.info(f"Файл: {OUTPUT_PATH}")

        print("")
        print(df[
            [
                "fixture_id",
                "kickoff",
                "home_team",
                "away_team",
                "p_home",
                "p_draw",
                "p_away",
                "predicted_result",
            ]
        ].to_string(index=False))

    finally:
        session.close()


if __name__ == "__main__":
    main()
