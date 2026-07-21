import argparse
import json
import sys

from loguru import logger
from sqlalchemy.orm import Session

from app.analytics.standings_analyzer import (
    StandingsAnalyzer,
)
from app.database.database import SessionLocal
from app.models.team import Team


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Анализ турнирного положения команды"
    )

    parser.add_argument(
        "team",
        type=str,
        help="Название команды или внутренний ID",
    )

    parser.add_argument(
        "--league-season-id",
        type=int,
        default=None,
        help="Внутренний ID сезона лиги",
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

        analyzer = StandingsAnalyzer(
            session=session
        )

        result = analyzer.analyze(
            team_id=team.id,
            league_season_id=(
                args.league_season_id
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
            "Ошибка анализа турнирного положения"
        )
        return 1

    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())