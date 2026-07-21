import argparse
import json
import sys

from loguru import logger
from sqlalchemy.orm import Session

from app.analytics.team_half_time_full_time_analyzer import (
    TeamHalfTimeFullTimeAnalyzer,
)
from app.database.database import SessionLocal
from app.models.team import Team


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Анализ результатов команды тайм/матч"
    )

    parser.add_argument(
        "team",
        type=str,
        help="Название команды или внутренний ID",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Количество последних матчей",
    )

    parser.add_argument(
        "--venue",
        type=str,
        choices=("all", "home", "away"),
        default="all",
        help="Все матчи, домашние или гостевые",
    )

    parser.add_argument(
        "--before-fixture-id",
        type=int,
        default=None,
        help=(
            "Учитывать только матчи до указанного "
            "внутреннего ID матча"
        ),
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

        analyzer = TeamHalfTimeFullTimeAnalyzer(
            session=session
        )

        result = analyzer.analyze(
            team_id=team.id,
            limit=args.limit,
            venue=args.venue,
            before_fixture_id=(
                args.before_fixture_id
            ),
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
            "Ошибка анализа тайм/матч"
        )
        return 1

    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())