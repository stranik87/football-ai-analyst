from dataclasses import asdict, dataclass

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.fixture import Fixture
from app.models.fixture_team_statistics import (
    FixtureTeamStatistics,
)
from app.models.team import Team


@dataclass
class TeamPassResult:
    team_id: int
    team_name: str
    venue: str
    requested_limit: int
    matches: int

    total_passes: int
    accurate_passes: int

    average_total_passes: float
    average_accurate_passes: float
    average_pass_accuracy: float

    highest_total_passes: int
    lowest_total_passes: int

    matches_above_80_accuracy: int
    matches_above_85_accuracy: int
    matches_above_90_accuracy: int

    above_80_accuracy_percentage: float
    above_85_accuracy_percentage: float
    above_90_accuracy_percentage: float

    def to_dict(self) -> dict:
        return asdict(self)


class TeamPassAnalyzer:
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
    ) -> TeamPassResult:
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

        total_pass_values: list[int] = []
        accurate_pass_values: list[int] = []
        accuracy_values: list[float] = []

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

            if statistic.total_passes is None:
                continue

            if statistic.passes_accurate is None:
                continue

            total_passes = int(
                statistic.total_passes
            )

            accurate_passes = int(
                statistic.passes_accurate
            )

            if total_passes <= 0:
                continue

            pass_accuracy = round(
                accurate_passes
                / total_passes
                * 100,
                2,
            )

            total_pass_values.append(
                total_passes
            )

            accurate_pass_values.append(
                accurate_passes
            )

            accuracy_values.append(
                pass_accuracy
            )

        matches = len(total_pass_values)

        total_passes = sum(
            total_pass_values
        )

        accurate_passes = sum(
            accurate_pass_values
        )

        matches_above_80_accuracy = sum(
            1
            for value in accuracy_values
            if value >= 80
        )

        matches_above_85_accuracy = sum(
            1
            for value in accuracy_values
            if value >= 85
        )

        matches_above_90_accuracy = sum(
            1
            for value in accuracy_values
            if value >= 90
        )

        return TeamPassResult(
            team_id=team.id,
            team_name=team.name,
            venue=venue,
            requested_limit=limit,
            matches=matches,
            total_passes=total_passes,
            accurate_passes=accurate_passes,
            average_total_passes=self._average(
                total_passes,
                matches,
            ),
            average_accurate_passes=self._average(
                accurate_passes,
                matches,
            ),
            average_pass_accuracy=self._average_float(
                sum(accuracy_values),
                matches,
            ),
            highest_total_passes=(
                max(total_pass_values)
                if total_pass_values
                else 0
            ),
            lowest_total_passes=(
                min(total_pass_values)
                if total_pass_values
                else 0
            ),
            matches_above_80_accuracy=(
                matches_above_80_accuracy
            ),
            matches_above_85_accuracy=(
                matches_above_85_accuracy
            ),
            matches_above_90_accuracy=(
                matches_above_90_accuracy
            ),
            above_80_accuracy_percentage=(
                self._percentage(
                    matches_above_80_accuracy,
                    matches,
                )
            ),
            above_85_accuracy_percentage=(
                self._percentage(
                    matches_above_85_accuracy,
                    matches,
                )
            ),
            above_90_accuracy_percentage=(
                self._percentage(
                    matches_above_90_accuracy,
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
    def _average_float(
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