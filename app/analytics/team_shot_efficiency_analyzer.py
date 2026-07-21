from dataclasses import asdict, dataclass

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.fixture import Fixture
from app.models.fixture_team_statistics import (
    FixtureTeamStatistics,
)
from app.models.team import Team


@dataclass
class TeamShotEfficiencyResult:
    team_id: int
    team_name: str
    venue: str
    requested_limit: int
    matches: int

    goals: int
    total_shots: int
    shots_on_goal: int
    shots_off_goal: int
    blocked_shots: int

    average_goals: float
    average_total_shots: float
    average_shots_on_goal: float

    shot_accuracy_percentage: float
    goal_conversion_percentage: float
    shots_on_goal_conversion_percentage: float

    shots_per_goal: float
    shots_on_goal_per_goal: float

    def to_dict(self) -> dict:
        return asdict(self)


class TeamShotEfficiencyAnalyzer:
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
    ) -> TeamShotEfficiencyResult:
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

        goals = 0
        total_shots = 0
        shots_on_goal = 0
        shots_off_goal = 0
        blocked_shots = 0
        processed_matches = 0

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

            if fixture.home_team_id == team_id:
                match_goals = fixture.home_goals
            else:
                match_goals = fixture.away_goals

            goals += match_goals or 0
            total_shots += statistic.total_shots or 0
            shots_on_goal += statistic.shots_on_goal or 0
            shots_off_goal += statistic.shots_off_goal or 0
            blocked_shots += statistic.blocked_shots or 0

            processed_matches += 1

        return TeamShotEfficiencyResult(
            team_id=team.id,
            team_name=team.name,
            venue=venue,
            requested_limit=limit,
            matches=processed_matches,
            goals=goals,
            total_shots=total_shots,
            shots_on_goal=shots_on_goal,
            shots_off_goal=shots_off_goal,
            blocked_shots=blocked_shots,
            average_goals=self._average(
                goals,
                processed_matches,
            ),
            average_total_shots=self._average(
                total_shots,
                processed_matches,
            ),
            average_shots_on_goal=self._average(
                shots_on_goal,
                processed_matches,
            ),
            shot_accuracy_percentage=self._percentage(
                shots_on_goal,
                total_shots,
            ),
            goal_conversion_percentage=self._percentage(
                goals,
                total_shots,
            ),
            shots_on_goal_conversion_percentage=self._percentage(
                goals,
                shots_on_goal,
            ),
            shots_per_goal=self._ratio(
                total_shots,
                goals,
            ),
            shots_on_goal_per_goal=self._ratio(
                shots_on_goal,
                goals,
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

    @staticmethod
    def _ratio(
        value: int,
        divisor: int,
    ) -> float:
        if divisor == 0:
            return 0.0

        return round(
            value / divisor,
            2,
        )