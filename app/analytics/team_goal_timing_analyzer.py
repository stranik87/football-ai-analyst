from dataclasses import asdict, dataclass

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.fixture import Fixture
from app.models.team import Team


@dataclass
class TeamGoalTimingResult:
    team_id: int
    team_name: str
    venue: str
    requested_limit: int
    matches: int

    first_half_goals_for: int
    first_half_goals_against: int

    second_half_goals_for: int
    second_half_goals_against: int

    average_first_half_goals_for: float
    average_first_half_goals_against: float

    average_second_half_goals_for: float
    average_second_half_goals_against: float

    first_half_scored_matches: int
    second_half_scored_matches: int

    first_half_clean_sheets: int
    second_half_clean_sheets: int

    def to_dict(self) -> dict:
        return asdict(self)


class TeamGoalTimingAnalyzer:
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
    ) -> TeamGoalTimingResult:
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

        query = (
            self.session.query(Fixture)
            .filter(
                Fixture.status_short.in_(
                    self.FINISHED_STATUSES
                ),
                Fixture.halftime_home.isnot(None),
                Fixture.halftime_away.isnot(None),
                Fixture.fulltime_home.isnot(None),
                Fixture.fulltime_away.isnot(None),
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

        fixtures = (
            query.order_by(
                Fixture.kickoff.desc(),
                Fixture.id.desc(),
            )
            .limit(limit)
            .all()
        )

        first_half_goals_for = 0
        first_half_goals_against = 0

        second_half_goals_for = 0
        second_half_goals_against = 0

        first_half_scored_matches = 0
        second_half_scored_matches = 0

        first_half_clean_sheets = 0
        second_half_clean_sheets = 0

        for fixture in fixtures:
            is_home = (
                fixture.home_team_id == team_id
            )

            if is_home:
                first_half_for = fixture.halftime_home
                first_half_against = fixture.halftime_away

                fulltime_for = fixture.fulltime_home
                fulltime_against = fixture.fulltime_away
            else:
                first_half_for = fixture.halftime_away
                first_half_against = fixture.halftime_home

                fulltime_for = fixture.fulltime_away
                fulltime_against = fixture.fulltime_home

            second_half_for = (
                fulltime_for - first_half_for
            )

            second_half_against = (
                fulltime_against - first_half_against
            )

            first_half_goals_for += first_half_for
            first_half_goals_against += first_half_against

            second_half_goals_for += second_half_for
            second_half_goals_against += (
                second_half_against
            )

            if first_half_for > 0:
                first_half_scored_matches += 1

            if second_half_for > 0:
                second_half_scored_matches += 1

            if first_half_against == 0:
                first_half_clean_sheets += 1

            if second_half_against == 0:
                second_half_clean_sheets += 1

        matches = len(fixtures)

        return TeamGoalTimingResult(
            team_id=team.id,
            team_name=team.name,
            venue=venue,
            requested_limit=limit,
            matches=matches,
            first_half_goals_for=(
                first_half_goals_for
            ),
            first_half_goals_against=(
                first_half_goals_against
            ),
            second_half_goals_for=(
                second_half_goals_for
            ),
            second_half_goals_against=(
                second_half_goals_against
            ),
            average_first_half_goals_for=(
                self._average(
                    first_half_goals_for,
                    matches,
                )
            ),
            average_first_half_goals_against=(
                self._average(
                    first_half_goals_against,
                    matches,
                )
            ),
            average_second_half_goals_for=(
                self._average(
                    second_half_goals_for,
                    matches,
                )
            ),
            average_second_half_goals_against=(
                self._average(
                    second_half_goals_against,
                    matches,
                )
            ),
            first_half_scored_matches=(
                first_half_scored_matches
            ),
            second_half_scored_matches=(
                second_half_scored_matches
            ),
            first_half_clean_sheets=(
                first_half_clean_sheets
            ),
            second_half_clean_sheets=(
                second_half_clean_sheets
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