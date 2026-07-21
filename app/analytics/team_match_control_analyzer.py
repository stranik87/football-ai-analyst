from dataclasses import asdict, dataclass

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.fixture import Fixture
from app.models.fixture_team_statistics import (
    FixtureTeamStatistics,
)
from app.models.team import Team


@dataclass
class TeamMatchControlResult:
    team_id: int
    team_name: str
    venue: str
    requested_limit: int
    matches: int

    average_possession: float
    average_total_passes: float
    average_pass_accuracy: float
    average_total_shots: float
    average_shots_on_goal: float
    average_corner_kicks: float

    match_control_score: float

    dominant_matches: int
    balanced_matches: int
    passive_matches: int

    dominant_percentage: float
    balanced_percentage: float
    passive_percentage: float

    highest_control_score: float
    lowest_control_score: float

    def to_dict(self) -> dict:
        return asdict(self)


class TeamMatchControlAnalyzer:
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
    ) -> TeamMatchControlResult:
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
        total_pass_values: list[int] = []
        pass_accuracy_values: list[float] = []
        total_shot_values: list[int] = []
        shots_on_goal_values: list[int] = []
        corner_values: list[int] = []
        control_scores: list[float] = []

        dominant_matches = 0
        balanced_matches = 0
        passive_matches = 0

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

            if statistic.total_passes is None:
                continue

            if statistic.passes_accurate is None:
                continue

            if statistic.total_shots is None:
                continue

            if statistic.shots_on_goal is None:
                continue

            if statistic.corner_kicks is None:
                continue

            total_passes = int(
                statistic.total_passes
            )

            accurate_passes = int(
                statistic.passes_accurate
            )

            total_shots = int(
                statistic.total_shots
            )

            shots_on_goal = int(
                statistic.shots_on_goal
            )

            corner_kicks = int(
                statistic.corner_kicks
            )

            if total_passes <= 0:
                continue

            pass_accuracy = round(
                accurate_passes
                / total_passes
                * 100,
                2,
            )

            control_score = self._calculate_score(
                possession=possession,
                total_passes=total_passes,
                pass_accuracy=pass_accuracy,
                total_shots=total_shots,
                shots_on_goal=shots_on_goal,
                corner_kicks=corner_kicks,
            )

            possession_values.append(
                possession
            )

            total_pass_values.append(
                total_passes
            )

            pass_accuracy_values.append(
                pass_accuracy
            )

            total_shot_values.append(
                total_shots
            )

            shots_on_goal_values.append(
                shots_on_goal
            )

            corner_values.append(
                corner_kicks
            )

            control_scores.append(
                control_score
            )

            if control_score >= 70:
                dominant_matches += 1

            elif control_score >= 45:
                balanced_matches += 1

            else:
                passive_matches += 1

        matches = len(control_scores)

        return TeamMatchControlResult(
            team_id=team.id,
            team_name=team.name,
            venue=venue,
            requested_limit=limit,
            matches=matches,
            average_possession=self._average(
                possession_values
            ),
            average_total_passes=self._average(
                total_pass_values
            ),
            average_pass_accuracy=self._average(
                pass_accuracy_values
            ),
            average_total_shots=self._average(
                total_shot_values
            ),
            average_shots_on_goal=self._average(
                shots_on_goal_values
            ),
            average_corner_kicks=self._average(
                corner_values
            ),
            match_control_score=self._average(
                control_scores
            ),
            dominant_matches=dominant_matches,
            balanced_matches=balanced_matches,
            passive_matches=passive_matches,
            dominant_percentage=self._percentage(
                dominant_matches,
                matches,
            ),
            balanced_percentage=self._percentage(
                balanced_matches,
                matches,
            ),
            passive_percentage=self._percentage(
                passive_matches,
                matches,
            ),
            highest_control_score=(
                max(control_scores)
                if control_scores
                else 0.0
            ),
            lowest_control_score=(
                min(control_scores)
                if control_scores
                else 0.0
            ),
        )

    @staticmethod
    def _calculate_score(
        possession: float,
        total_passes: int,
        pass_accuracy: float,
        total_shots: int,
        shots_on_goal: int,
        corner_kicks: int,
    ) -> float:
        possession_score = min(
            possession,
            70.0,
        ) / 70.0 * 30.0

        pass_score = min(
            total_passes,
            700,
        ) / 700 * 20.0

        accuracy_score = min(
            pass_accuracy,
            100.0,
        ) / 100.0 * 20.0

        shot_score = min(
            total_shots,
            25,
        ) / 25 * 10.0

        shots_on_goal_score = min(
            shots_on_goal,
            12,
        ) / 12 * 15.0

        corner_score = min(
            corner_kicks,
            10,
        ) / 10 * 5.0

        score = (
            possession_score
            + pass_score
            + accuracy_score
            + shot_score
            + shots_on_goal_score
            + corner_score
        )

        return round(
            min(score, 100.0),
            2,
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
        values: list[int] | list[float],
    ) -> float:
        if not values:
            return 0.0

        return round(
            sum(values) / len(values),
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