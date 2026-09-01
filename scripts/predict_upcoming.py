from datetime import datetime, timezone

from app.database.session import SessionLocal
from app.models.fixture import Fixture
from app.services.prediction_service import PredictionService


def main():
    session = SessionLocal()

    try:
        predictor = PredictionService(session)

        fixtures = (
            session.query(Fixture)
            .filter(
                Fixture.kickoff > datetime.now(timezone.utc),
                Fixture.home_goals.is_(None),
                Fixture.away_goals.is_(None),
            )
            .order_by(Fixture.kickoff)
            .limit(20)
            .all()
        )

        print()
        print(
            "Дата | Матч | H | D | A | Прогноз | TOP-2"
        )
        print("-" * 120)

        for fixture in fixtures:
            result = predictor.predict(fixture.id)

            probabilities = result["probabilities"]

            values = sorted(
                probabilities.values(),
                reverse=True,
            )

            gap = values[0] - values[1]

            print(
                f"{fixture.kickoff} | "
                f"{fixture.home_team.name} - "
                f"{fixture.away_team.name} | "
                f"{probabilities['home_win']:.1%} | "
                f"{probabilities['draw']:.1%} | "
                f"{probabilities['away_win']:.1%} | "
                f"{result['predicted_result']} | "
                f"{gap:.1%}"
            )

    finally:
        session.close()


if __name__ == "__main__":
    main()
