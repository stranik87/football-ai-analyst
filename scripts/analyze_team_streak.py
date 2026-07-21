import argparse
import json
import sys

from loguru import logger
from sqlalchemy.orm import Session

from app.analytics.team_streak_analyzer import (
    TeamStreakAnalyzer,
)
from app.database.database import SessionLocal
from app.models.team import Team


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Анализ текущих серий команды"
    )

    parser.add_argument(
        "team",
        type=str,
        help="Название команды или внутренний ID",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=30,
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

    matching_teams = (
        session.query(Team)
        .filter(
            Team.name.ilike(f"%{value}%")
        )
        .order_by(Team.name.asc())
        .limit(10)
        .all()
    )

    if not matching_teams:
        return None

    if len(matching_teams) > 1:
        logger.warning(
            "Найдено несколько команд: {}. "
            "Используется первая.",
            ", ".join(
                f"{team.id}: {team.name}"
                for team in matching_teams
            ),
        )

    return matching_teams[0]


def main() -> int:
    args = parse_args()

    session = SessionLocal()

    try:
        team = find_team(
            session=session,
            value=args.team,
        )

        if not team:
            logger.error(
                "Команда не найдена: {}",
                args.team,
            )
            return 1

        analyzer = TeamStreakAnalyzer(
            session=session
        )

        result = analyzer.analyze(
            team_id=team.id,
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
            "Ошибка анализа серий команды"
        )
        return 1

    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())