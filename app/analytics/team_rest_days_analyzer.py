from dataclasses import asdict, dataclass
from datetime import datetime

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.fixture import Fixture
from app.models.team import Team


@dataclass
class TeamRestDaysResult:
    team_id: int
    team_name: str
    requested_limit: int
    matches: int

    average_rest_days: float
    minimum_rest_days: int | None
    maximum_rest_days: int | None

    short_rest_matches: int
    normal_rest_matches: int
    long_rest_matches: int

    rest_days: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


class TeamRestDaysAnalyzer:
    FINISHED_STATUSES = (
        "FT",
        "AET",
        "PEN",
    )

    def __init__(self, session: Session):
        self.session = session

    def analyze(
        self,
        team_id: int,
        limit: int = 10,
    ) -> TeamRestDaysResult:
        if limit <= 0:
            raise ValueError(
                "limit должен быть больше нуля"
            )

        team = (
            self.session.query(Team)
            .filter(Team.id == team_id)
            .first()
        )

        if not team:
            raise ValueError(
                f"Команда не найдена: team_id={team_id}"
            )

        fixtures = (
            self.session.query(Fixture)
            .filter(
                Fixture.status_short.in_(
                    self.FINISHED_STATUSES
                ),
                Fixture.kickoff.isnot(None),
            )
            .filter(
                or_(
                    Fixture.home_team_id == team_id,
                    Fixture.away_team_id == team_id,
                )
            )
            .order_by(
                Fixture.kickoff.desc(),
                Fixture.id.desc(),
            )
            .limit(limit + 1)
            .all()
        )

        rest_days_items: list[dict] = []
        rest_values: list[int] = []

        for index in range(len(fixtures) - 1):
            current_fixture = fixtures[index]
            previous_fixture = fixtures[index + 1]

            rest_days = self._calculate_rest_days(
                previous_fixture.kickoff,
                current_fixture.kickoff,
            )

            if rest_days < 0:
                continue

            rest_values.append(rest_days)

            rest_days_items.append(
                {
                    "fixture_id": current_fixture.id,
                    "kickoff": (
                        current_fixture.kickoff.isoformat()
                    ),
                    "previous_fixture_id": (
                        previous_fixture.id
                    ),
                    "previous_kickoff": (
                        previous_fixture.kickoff.isoformat()
                    ),
                    "rest_days": rest_days,
                }
            )

        short_rest_matches = sum(
            1
            for value in rest_values
            if value <= 3
        )

        normal_rest_matches = sum(
            1
            for value in rest_values
            if 4 <= value <= 7
        )

        long_rest_matches = sum(
            1
            for value in rest_values
            if value >= 8
        )

        average_rest_days = (
            round(
                sum(rest_values) / len(rest_values),
                2,
            )
            if rest_values
            else 0.0
        )

        return TeamRestDaysResult(
            team_id=team.id,
            team_name=team.name,
            requested_limit=limit,
            matches=len(rest_values),
            average_rest_days=average_rest_days,
            minimum_rest_days=(
                min(rest_values)
                if rest_values
                else None
            ),
            maximum_rest_days=(
                max(rest_values)
                if rest_values
                else None
            ),
            short_rest_matches=short_rest_matches,
            normal_rest_matches=normal_rest_matches,
            long_rest_matches=long_rest_matches,
            rest_days=rest_days_items,
        )

    @staticmethod
    def _calculate_rest_days(
        previous_kickoff: datetime,
        current_kickoff: datetime,
    ) -> int:
        difference = (
            current_kickoff - previous_kickoff
        )

        return difference.days