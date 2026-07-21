from dataclasses import asdict, dataclass

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.fixture import Fixture
from app.models.fixture_team_statistics import (
    FixtureTeamStatistics,
)
from app.models.team import Team


@dataclass
class TeamSetPieceResult:
    team_id: int
    team_name: str
    venue: str
    requested_limit: int
    matches: int

    corner_kicks: int
    average_corner_kicks: float
    highest_corner_kicks: int
    lowest_corner_kicks: int

    matches_over_4_5_corners: int
    matches_over_5_5_corners: int
    matches_over_6_5_corners: int

    over_4_5_corners_percentage: float
    over_5_5_corners_percentage: float
    over_6_5_corners_percentage: float

    def to_dict(self) -> dict:
        return asdict(self)


class TeamSetPieceAnalyzer:
    FINISHED_STATUSES = (
        "FT",
        "AET",
        "PEN",
    )

    VALID_VENUES = (
        "all",
        "home",
        "away",
    )

    def __init__(self, session: Session):
        self.session = session

    def analyze(
        self,
        team_id: int,
        limit: int = 10,
        venue: str = "all",
    ) -> TeamSetPieceResult:
        if limit <= 0:
            raise ValueError(
                "limit должен быть больше нуля"
            )

        if venue not in self.VALID_VENUES:
            raise ValueError(
                "venue должен быть: all, home или away"
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

        fixture_query = (
            self.session.query(Fixture)
            .join(
                FixtureTeamStatistics,
                FixtureTeamStatistics.fixture_id
                == Fixture.id,
            )
            .filter(
                FixtureTeamStatistics.team_id == team_id,
                Fixture.status_short.in_(
                    self.FINISHED_STATUSES
                ),
                Fixture.home_goals.isnot(None),
                Fixture.away_goals.isnot(None),
            )
        )

        if venue == "home":
            fixture_query = fixture_query.filter(
                Fixture.home_team_id == team_id
            )

        elif venue == "away":
            fixture_query = fixture_query.filter(
                Fixture.away_team_id == team_id
            )

        else:
            fixture_query = fixture_query.filter(
                or_(
                    Fixture.home_team_id == team_id,
                    Fixture.away_team_id == team_id,
                )
            )

        fixtures = (
            fixture_query.order_by(
                Fixture.kickoff.desc(),
                Fixture.id.desc(),
            )
            .limit(limit)
            .all()
        )

        corner_values: list[int] = []

        for fixture in fixtures:
            statistic = (
                self.session.query(
                    FixtureTeamStatistics
                )
                .filter(
                    FixtureTeamStatistics.fixture_id
                    == fixture.id,
                    FixtureTeamStatistics.team_id
                    == team_id,
                )
                .first()
            )

            if not statistic:
                continue

            if statistic.corner_kicks is None:
                continue

            corner_values.append(
                statistic.corner_kicks
            )

        matches = len(corner_values)
        corner_kicks = sum(corner_values)

        matches_over_4_5_corners = sum(
            1
            for corners in corner_values
            if corners >= 5
        )

        matches_over_5_5_corners = sum(
            1
            for corners in corner_values
            if corners >= 6
        )

        matches_over_6_5_corners = sum(
            1
            for corners in corner_values
            if corners >= 7
        )

        return TeamSetPieceResult(
            team_id=team.id,
            team_name=team.name,
            venue=venue,
            requested_limit=limit,
            matches=matches,
            corner_kicks=corner_kicks,
            average_corner_kicks=self._average(
                corner_kicks,
                matches,
            ),
            highest_corner_kicks=(
                max(corner_values)
                if corner_values
                else 0
            ),
            lowest_corner_kicks=(
                min(corner_values)
                if corner_values
                else 0
            ),
            matches_over_4_5_corners=(
                matches_over_4_5_corners
            ),
            matches_over_5_5_corners=(
                matches_over_5_5_corners
            ),
            matches_over_6_5_corners=(
                matches_over_6_5_corners
            ),
            over_4_5_corners_percentage=(
                self._percentage(
                    matches_over_4_5_corners,
                    matches,
                )
            ),
            over_5_5_corners_percentage=(
                self._percentage(
                    matches_over_5_5_corners,
                    matches,
                )
            ),
            over_6_5_corners_percentage=(
                self._percentage(
                    matches_over_6_5_corners,
                    matches,
                )
            ),
        )

    @staticmethod
    def _average(
        value: int,
        matches: int,
    ) -> float:
        if matches == 0:
            return 0.0

        return round(
            value / matches,
            2,
        )

    @staticmethod
    def _percentage(
        value: int,
        total: int,
    ) -> float:
        if total == 0:
            return 0.0

        return round(
            value / total * 100,
            2,
        )