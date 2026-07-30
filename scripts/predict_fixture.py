import argparse

from app.database.session import SessionLocal
from app.ml.prediction_service import PredictionService
from app.models.fixture import Fixture
from app.models.team import Team


def get_confidence(result: dict) -> tuple[str, float]:
    probabilities = {
        "H": result["home_win"],
        "D": result["draw"],
        "A": result["away_win"],
    }

    ordered = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    gap = ordered[0][1] - ordered[1][1]

    if gap >= 0.20:
        confidence = "Очень высокая"
    elif gap >= 0.10:
        confidence = "Высокая"
    elif gap >= 0.05:
        confidence = "Средняя"
    else:
        confidence = "Низкая"

    return confidence, gap


def recommendation(confidence: str) -> str:
    if confidence == "Очень высокая":
        return "Можно рассматривать прогноз."

    if confidence == "Высокая":
        return "Прогноз выглядит хорошим."

    if confidence == "Средняя":
        return "Использовать вместе с дополнительным анализом."

    return "Лучше пропустить матч."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Прогноз футбольного матча"
    )

    parser.add_argument(
        "--fixture-id",
        type=int,
        required=True,
        help="ID матча",
    )

    args = parser.parse_args()

    session = SessionLocal()

    try:
        fixture = (
            session.query(Fixture)
            .filter(Fixture.id == args.fixture_id)
            .first()
        )

        if fixture is None:
            print("Матч не найден.")
            return

        home_team = (
            session.query(Team)
            .filter(Team.id == fixture.home_team_id)
            .first()
        )

        away_team = (
            session.query(Team)
            .filter(Team.id == fixture.away_team_id)
            .first()
        )

        predictor = PredictionService(session)

        result = predictor.predict_fixture(
            fixture.id
        )

        confidence, gap = get_confidence(result)

        print()
        print("=" * 60)
        print(
            f"Матч: {home_team.name} — {away_team.name}"
        )
        print(f"Fixture ID : {fixture.id}")
        print(f"Дата       : {fixture.kickoff}")
        print()

        print(f"Прогноз: {result['prediction']}")
        print()

        print(
            f"Победа хозяев : {result['home_win']:.2%}"
        )
        print(
            f"Ничья         : {result['draw']:.2%}"
        )
        print(
            f"Победа гостей : {result['away_win']:.2%}"
        )

        print()
        print(f"Уверенность : {confidence}")
        print(f"Разница TOP-2 : {gap:.2%}")
        print(
            f"Рекомендация : {recommendation(confidence)}"
        )

        print("=" * 60)

    finally:
        session.close()


if __name__ == "__main__":
    main()