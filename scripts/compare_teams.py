import argparse
import json
import sys

from loguru import logger
from sqlalchemy.orm import Session

from app.analytics.team_comparison_analyzer import (
    TeamComparisonAnalyzer,
)
from app.database.database import SessionLocal
from app.models.team import Team


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сравнение двух футбольных команд"
    )

    parser.add_argument(
        "home_team",
        type=str,
        help="Домашняя команда",
    )

    parser.add_argument(
        "away_team",
        type=str,
        help="Гостевая команда",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Количество последних матчей",
    )

    parser.add_argument(
        "--before-fixture-id",
        type=int,
        default=None,
        help="Использовать данные только до указанного матча",
    )

    return parser.parse_args()


def find_team(
    session: Session,
    value: str,
) -> Team | None:
    if value.isdigit():
        team = (
            session.query(Team)
            .filter(Team.id == int(value))
            .first()
        )

        if team:
            return team

    exact_team = (
        session.query(Team)
        .filter(Team.name.ilike(value))
        .first()
    )

    if exact_team:
        return exact_team

    return (
        session.query(Team)
        .filter(Team.name.ilike(f"%{value}%"))
        .order_by(Team.name.asc())
        .first()
    )


def main() -> int:
    args = parse_args()
    session = SessionLocal()

    try:
        home_team = find_team(
            session,
            args.home_team,
        )

        away_team = find_team(
            session,
            args.away_team,
        )

        if not home_team:
            logger.error(
                "Домашняя команда не найдена: {}",
                args.home_team,
            )
            return 1

        if not away_team:
            logger.error(
                "Гостевая команда не найдена: {}",
                args.away_team,
            )
            return 1

        analyzer = TeamComparisonAnalyzer(session)

        result = analyzer.analyze(
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            limit=args.limit,
            before_fixture_id=args.before_fixture_id,
        )

        print(
            json.dumps(
                result.to_dict(),
                ensure_ascii=False,
                indent=4,
            )
        )

        return 0

    except ValueError as error:
        logger.error(str(error))
        return 1

    except Exception:
        logger.exception(
            "Ошибка сравнения команд"
        )
        return 1

    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())