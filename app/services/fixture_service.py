from datetime import datetime

from sqlalchemy import or_
from sqlalchemy.orm import joinedload

from app.models.fixture import Fixture
from app.models.team import Team


class FixtureService:
    """
    Сервис для поиска и получения футбольных матчей.
    """

    FINISHED_STATUSES = (
        "FT",
        "AET",
        "PEN",
    )

    def __init__(self, session) -> None:
        self.session = session

    def get_by_id(
        self,
        fixture_id: int,
    ) -> Fixture | None:
        """
        Получить матч по локальному ID.
        """

        return (
            self.session.query(Fixture)
            .options(
                joinedload(Fixture.home_team),
                joinedload(Fixture.away_team),
                joinedload(Fixture.league_season),
            )
            .filter(Fixture.id == fixture_id)
            .first()
        )

    def get_latest_matches(
        self,
        limit: int = 10,
    ) -> list[Fixture]:
        """
        Получить последние завершённые матчи.
        """

        safe_limit = max(1, min(limit, 50))

        return (
            self.session.query(Fixture)
            .options(
                joinedload(Fixture.home_team),
                joinedload(Fixture.away_team),
                joinedload(Fixture.league_season),
            )
            .filter(
                Fixture.status_short.in_(
                    self.FINISHED_STATUSES
                )
            )
            .order_by(Fixture.kickoff.desc())
            .limit(safe_limit)
            .all()
        )

    def get_upcoming_matches(
        self,
        limit: int = 10,
        from_datetime: datetime | None = None,
    ) -> list[Fixture]:
        """
        Получить ближайшие незавершённые матчи.
        """

        safe_limit = max(1, min(limit, 50))
        start_datetime = (
            from_datetime
            if from_datetime is not None
            else datetime.now()
        )

        return (
            self.session.query(Fixture)
            .options(
                joinedload(Fixture.home_team),
                joinedload(Fixture.away_team),
                joinedload(Fixture.league_season),
            )
            .filter(
                Fixture.kickoff >= start_datetime,
                ~Fixture.status_short.in_(
                    self.FINISHED_STATUSES
                ),
            )
            .order_by(Fixture.kickoff.asc())
            .limit(safe_limit)
            .all()
        )

    def search_team_matches(
        self,
        team_name: str,
        limit: int = 10,
    ) -> list[Fixture]:
        """
        Найти последние матчи команды по названию.
        """

        normalized_name = team_name.strip()

        if not normalized_name:
            return []

        safe_limit = max(1, min(limit, 50))

        team_ids = (
            self.session.query(Team.id)
            .filter(
                Team.name.ilike(
                    f"%{normalized_name}%"
                )
            )
            .subquery()
        )

        return (
            self.session.query(Fixture)
            .options(
                joinedload(Fixture.home_team),
                joinedload(Fixture.away_team),
                joinedload(Fixture.league_season),
            )
            .filter(
                or_(
                    Fixture.home_team_id.in_(
                        team_ids
                    ),
                    Fixture.away_team_id.in_(
                        team_ids
                    ),
                )
            )
            .order_by(Fixture.kickoff.desc())
            .limit(safe_limit)
            .all()
        )

    @staticmethod
    def serialize(
        fixture: Fixture,
    ) -> dict:
        """
        Преобразовать матч в словарь.
        """

        return {
            "fixture_id": fixture.id,
            "api_id": fixture.api_id,
            "kickoff": fixture.kickoff,
            "status_short": fixture.status_short,
            "status_long": fixture.status_long,
            "round": fixture.round,
            "home_team_id": fixture.home_team_id,
            "away_team_id": fixture.away_team_id,
            "home_team": (
                fixture.home_team.name
                if fixture.home_team is not None
                else None
            ),
            "away_team": (
                fixture.away_team.name
                if fixture.away_team is not None
                else None
            ),
            "home_goals": fixture.home_goals,
            "away_goals": fixture.away_goals,
        }