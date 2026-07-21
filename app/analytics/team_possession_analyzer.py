from dataclasses import asdict, dataclass

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.fixture import Fixture
from app.models.fixture_team_statistics import (
    FixtureTeamStatistics,
)
from app.models.team import Team


@dataclass
class TeamPossessionResult:
    team_id: int
    team_name: str
    venue: str
    requested_limit: int
    matches: int

    total_possession: float
    average_possession: float
    highest_possession: float
    lowest_possession: float

    matches_above_50: int
    matches_equal_50: int
    matches_below_50: int

    possession_above_50_percentage: float
    possession_below_50_percentage: float

    def to_dict(self) -> dict:
        return asdict(self)


class TeamPossessionAnalyzer:
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
    ) -> TeamPossessionResult:
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

        possession_values: list[float] = []

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

            possession = self._normalize_possession(
                statistic.ball_possession
            )

            if possession is None:
                continue

            possession_values.append(possession)

        matches = len(possession_values)

        total_possession = sum(possession_values)

        matches_above_50 = sum(
            1
            for possession in possession_values
            if possession > 50
        )

        matches_equal_50 = sum(
            1
            for possession in possession_values
            if possession == 50
        )

        matches_below_50 = sum(
            1
            for possession in possession_values
            if possession < 50
        )

        return TeamPossessionResult(
            team_id=team.id,
            team_name=team.name,
            venue=venue,
            requested_limit=limit,
            matches=matches,
            total_possession=round(
                total_possession,
                2,
            ),
            average_possession=self._average(
                total_possession,
                matches,
            ),
            highest_possession=(
                round(max(possession_values), 2)
                if possession_values
                else 0.0
            ),
            lowest_possession=(
                round(min(possession_values), 2)
                if possession_values
                else 0.0
            ),
            matches_above_50=matches_above_50,
            matches_equal_50=matches_equal_50,
            matches_below_50=matches_below_50,
            possession_above_50_percentage=(
                self._percentage(
                    matches_above_50,
                    matches,
                )
            ),
            possession_below_50_percentage=(
                self._percentage(
                    matches_below_50,
                    matches,
                )
            ),
        )

    @staticmethod
    def _normalize_possession(
        value: object,
    ) -> float | None:
        if value is None:
            return None

        if isinstance(value, str):
            cleaned_value = value.replace(
                "%",
                "",
            ).strip()

            if not cleaned_value:
                return None

            try:
                return float(cleaned_value)

            except ValueError:
                return None

        try:
            return float(value)

        except (TypeError, ValueError):
            return None

    @staticmethod
    def _average(
        value: float,
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