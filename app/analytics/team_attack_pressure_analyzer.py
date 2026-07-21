from dataclasses import asdict, dataclass

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.fixture import Fixture
from app.models.fixture_team_statistics import (
    FixtureTeamStatistics,
)
from app.models.team import Team


@dataclass
class TeamAttackPressureResult:
    team_id: int
    team_name: str
    venue: str
    requested_limit: int
    matches: int

    total_shots: int
    shots_on_goal: int
    corner_kicks: int

    average_total_shots: float
    average_shots_on_goal: float
    average_corner_kicks: float
    average_possession: float

    shot_accuracy_percentage: float
    attacking_pressure_score: float

    high_pressure_matches: int
    medium_pressure_matches: int
    low_pressure_matches: int

    high_pressure_percentage: float
    medium_pressure_percentage: float
    low_pressure_percentage: float

    def to_dict(self) -> dict:
        return asdict(self)


class TeamAttackPressureAnalyzer:
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
    ) -> TeamAttackPressureResult:
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

        total_shots = 0
        shots_on_goal = 0
        corner_kicks = 0
        total_possession = 0.0

        pressure_scores: list[float] = []

        high_pressure_matches = 0
        medium_pressure_matches = 0
        low_pressure_matches = 0

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

            if statistic.total_shots is None:
                continue

            if statistic.shots_on_goal is None:
                continue

            if statistic.corner_kicks is None:
                continue

            possession = self._normalize_possession(
                statistic.ball_possession
            )

            if possession is None:
                continue

            match_total_shots = (
                statistic.total_shots
            )

            match_shots_on_goal = (
                statistic.shots_on_goal
            )

            match_corner_kicks = (
                statistic.corner_kicks
            )

            match_pressure_score = round(
                (
                    match_total_shots * 2
                    + match_shots_on_goal * 3
                    + match_corner_kicks * 2
                    + possession * 0.5
                ),
                2,
            )

            total_shots += match_total_shots
            shots_on_goal += match_shots_on_goal
            corner_kicks += match_corner_kicks
            total_possession += possession

            pressure_scores.append(
                match_pressure_score
            )

            if match_pressure_score >= 70:
                high_pressure_matches += 1

            elif match_pressure_score >= 45:
                medium_pressure_matches += 1

            else:
                low_pressure_matches += 1

        matches = len(pressure_scores)

        return TeamAttackPressureResult(
            team_id=team.id,
            team_name=team.name,
            venue=venue,
            requested_limit=limit,
            matches=matches,
            total_shots=total_shots,
            shots_on_goal=shots_on_goal,
            corner_kicks=corner_kicks,
            average_total_shots=self._average(
                total_shots,
                matches,
            ),
            average_shots_on_goal=self._average(
                shots_on_goal,
                matches,
            ),
            average_corner_kicks=self._average(
                corner_kicks,
                matches,
            ),
            average_possession=self._average_float(
                total_possession,
                matches,
            ),
            shot_accuracy_percentage=self._percentage(
                shots_on_goal,
                total_shots,
            ),
            attacking_pressure_score=(
                self._average_float(
                    sum(pressure_scores),
                    matches,
                )
            ),
            high_pressure_matches=(
                high_pressure_matches
            ),
            medium_pressure_matches=(
                medium_pressure_matches
            ),
            low_pressure_matches=(
                low_pressure_matches
            ),
            high_pressure_percentage=(
                self._percentage(
                    high_pressure_matches,
                    matches,
                )
            ),
            medium_pressure_percentage=(
                self._percentage(
                    medium_pressure_matches,
                    matches,
                )
            ),
            low_pressure_percentage=(
                self._percentage(
                    low_pressure_matches,
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