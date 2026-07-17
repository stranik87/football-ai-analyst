from dataclasses import asdict, dataclass

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.fixture import Fixture
from app.models.fixture_team_statistics import FixtureTeamStatistics
from app.models.team import Team


@dataclass
class TeamStatisticsResult:
    team_id: int
    team_name: str
    venue: str
    requested_limit: int
    matches: int

    home_matches: int
    away_matches: int

    average_shots_on_goal: float
    average_shots_off_goal: float
    average_total_shots: float
    average_blocked_shots: float
    average_shots_inside_box: float
    average_shots_outside_box: float

    average_fouls: float
    average_corner_kicks: float
    average_offsides: float

    average_ball_possession: float

    average_yellow_cards: float
    average_red_cards: float

    average_goalkeeper_saves: float

    average_total_passes: float
    average_passes_accurate: float
    average_passes_percentage: float

    average_expected_goals: float
    average_goals_prevented: float

    shot_accuracy_percentage: float

    def to_dict(self) -> dict:
        return asdict(self)


class TeamStatisticsAnalyzer:
    """
    Анализ средней матчевой статистики команды.
    """

    FINISHED_STATUSES = (
        "FT",
        "AET",
        "PEN",
    )

    ALLOWED_VENUES = (
        "all",
        "home",
        "away",
    )

    STAT_FIELDS = (
        "shots_on_goal",
        "shots_off_goal",
        "total_shots",
        "blocked_shots",
        "shots_inside_box",
        "shots_outside_box",
        "fouls",
        "corner_kicks",
        "offsides",
        "ball_possession",
        "yellow_cards",
        "red_cards",
        "goalkeeper_saves",
        "total_passes",
        "passes_accurate",
        "passes_percentage",
        "expected_goals",
        "goals_prevented",
    )

    def __init__(self, session: Session):
        self.session = session

    def analyze(
        self,
        team_id: int,
        limit: int = 10,
        venue: str = "all",
        before_fixture_id: int | None = None,
    ) -> TeamStatisticsResult:
        if limit <= 0:
            raise ValueError(
                "limit должен быть больше нуля"
            )

        venue = venue.strip().lower()

        if venue not in self.ALLOWED_VENUES:
            raise ValueError(
                "venue должен иметь значение: "
                "all, home или away"
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

        query = (
            self.session.query(
                FixtureTeamStatistics,
                Fixture,
            )
            .join(
                Fixture,
                Fixture.id
                == FixtureTeamStatistics.fixture_id,
            )
            .filter(
                FixtureTeamStatistics.team_id == team_id,
                Fixture.status_short.in_(
                    self.FINISHED_STATUSES
                ),
            )
        )

        if venue == "home":
            query = query.filter(
                Fixture.home_team_id == team_id
            )

        elif venue == "away":
            query = query.filter(
                Fixture.away_team_id == team_id
            )

        else:
            query = query.filter(
                or_(
                    Fixture.home_team_id == team_id,
                    Fixture.away_team_id == team_id,
                )
            )

        if before_fixture_id is not None:
            target_fixture = (
                self.session.query(Fixture)
                .filter(
                    Fixture.id == before_fixture_id
                )
                .first()
            )

            if not target_fixture:
                raise ValueError(
                    "Матч не найден: "
                    f"fixture_id={before_fixture_id}"
                )

            query = query.filter(
                Fixture.kickoff
                < target_fixture.kickoff
            )

        rows = (
            query
            .order_by(Fixture.kickoff.desc())
            .limit(limit)
            .all()
        )

        totals = {
            field: 0.0
            for field in self.STAT_FIELDS
        }

        counts = {
            field: 0
            for field in self.STAT_FIELDS
        }

        home_matches = 0
        away_matches = 0

        total_shots_for_accuracy = 0.0
        shots_on_goal_for_accuracy = 0.0

        for statistics, fixture in rows:
            if fixture.home_team_id == team_id:
                home_matches += 1
            else:
                away_matches += 1

            for field in self.STAT_FIELDS:
                value = getattr(
                    statistics,
                    field,
                    None,
                )

                if value is None:
                    continue

                totals[field] += float(value)
                counts[field] += 1

            if statistics.total_shots is not None:
                total_shots_for_accuracy += float(
                    statistics.total_shots
                )

            if statistics.shots_on_goal is not None:
                shots_on_goal_for_accuracy += float(
                    statistics.shots_on_goal
                )

        def average(field: str) -> float:
            count = counts[field]

            if count == 0:
                return 0.0

            return round(
                totals[field] / count,
                2,
            )

        if total_shots_for_accuracy > 0:
            shot_accuracy_percentage = round(
                (
                    shots_on_goal_for_accuracy
                    / total_shots_for_accuracy
                )
                * 100,
                2,
            )
        else:
            shot_accuracy_percentage = 0.0

        return TeamStatisticsResult(
            team_id=team.id,
            team_name=team.name,
            venue=venue,
            requested_limit=limit,
            matches=len(rows),
            home_matches=home_matches,
            away_matches=away_matches,
            average_shots_on_goal=average(
                "shots_on_goal"
            ),
            average_shots_off_goal=average(
                "shots_off_goal"
            ),
            average_total_shots=average(
                "total_shots"
            ),
            average_blocked_shots=average(
                "blocked_shots"
            ),
            average_shots_inside_box=average(
                "shots_inside_box"
            ),
            average_shots_outside_box=average(
                "shots_outside_box"
            ),
            average_fouls=average(
                "fouls"
            ),
            average_corner_kicks=average(
                "corner_kicks"
            ),
            average_offsides=average(
                "offsides"
            ),
            average_ball_possession=average(
                "ball_possession"
            ),
            average_yellow_cards=average(
                "yellow_cards"
            ),
            average_red_cards=average(
                "red_cards"
            ),
            average_goalkeeper_saves=average(
                "goalkeeper_saves"
            ),
            average_total_passes=average(
                "total_passes"
            ),
            average_passes_accurate=average(
                "passes_accurate"
            ),
            average_passes_percentage=average(
                "passes_percentage"
            ),
            average_expected_goals=average(
                "expected_goals"
            ),
            average_goals_prevented=average(
                "goals_prevented"
            ),
            shot_accuracy_percentage=(
                shot_accuracy_percentage
            ),
        )