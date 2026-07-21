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
        "first_team",
        type=str,
        help="Название или внутренний ID первой команды",
    )

    parser.add_argument(
        "second_team",
        type=str,
        help="Название или внутренний ID второй команды",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Количество последних матчей",
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

    teams = (
        session.query(Team)
        .filter(
            Team.name.ilike(f"%{value}%")
        )
        .order_by(Team.name.asc())
        .limit(10)
        .all()
    )

    if not teams:
        return None

    if len(teams) > 1:
        logger.warning(
            "Найдено несколько команд: {}. "
            "Используется первая.",
            ", ".join(
                f"{team.id}: {team.name}"
                for team in teams
            ),
        )

    return teams[0]


def main() -> int:
    args = parse_args()

    session = SessionLocal()

    try:
        first_team = find_team(
            session=session,
            value=args.first_team,
        )

        if not first_team:
            logger.error(
                "Первая команда не найдена: {}",
                args.first_team,
            )
            return 1

        second_team = find_team(
            session=session,
            value=args.second_team,
        )

        if not second_team:
            logger.error(
                "Вторая команда не найдена: {}",
                args.second_team,
            )
            return 1

        analyzer = TeamComparisonAnalyzer(
            session=session
        )

        result = analyzer.analyze(
            first_team_id=first_team.id,
            second_team_id=second_team.id,
            limit=args.limit,
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