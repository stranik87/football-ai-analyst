from dataclasses import asdict, dataclass

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.fixture import Fixture
from app.models.fixture_team_statistics import (
    FixtureTeamStatistics,
)
from app.models.team import Team


@dataclass
class TeamGoalkeeperResult:
    team_id: int
    team_name: str
    venue: str
    requested_limit: int
    matches: int

    goalkeeper_saves: int
    goals_conceded: int
    shots_on_goal_faced: int

    average_saves: float
    average_goals_conceded: float
    average_shots_on_goal_faced: float

    save_percentage: float

    clean_sheets: int
    matches_with_goal_conceded: int
    matches_with_three_or_more_saves: int
    matches_with_five_or_more_saves: int

    clean_sheet_percentage: float
    three_or_more_saves_percentage: float
    five_or_more_saves_percentage: float

    highest_saves: int
    lowest_saves: int

    def to_dict(self) -> dict:
        return asdict(self)


class TeamGoalkeeperAnalyzer:
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
    ) -> TeamGoalkeeperResult:
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

        save_values: list[int] = []

        goalkeeper_saves = 0
        goals_conceded = 0
        shots_on_goal_faced = 0

        clean_sheets = 0
        matches_with_goal_conceded = 0
        matches_with_three_or_more_saves = 0
        matches_with_five_or_more_saves = 0

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

            if statistic.goalkeeper_saves is None:
                continue

            match_saves = statistic.goalkeeper_saves

            if fixture.home_team_id == team_id:
                match_goals_conceded = (
                    fixture.away_goals or 0
                )
            else:
                match_goals_conceded = (
                    fixture.home_goals or 0
                )

            match_shots_on_goal_faced = (
                match_saves + match_goals_conceded
            )

            save_values.append(match_saves)

            goalkeeper_saves += match_saves
            goals_conceded += match_goals_conceded
            shots_on_goal_faced += (
                match_shots_on_goal_faced
            )

            if match_goals_conceded == 0:
                clean_sheets += 1
            else:
                matches_with_goal_conceded += 1

            if match_saves >= 3:
                matches_with_three_or_more_saves += 1

            if match_saves >= 5:
                matches_with_five_or_more_saves += 1

        matches = len(save_values)

        return TeamGoalkeeperResult(
            team_id=team.id,
            team_name=team.name,
            venue=venue,
            requested_limit=limit,
            matches=matches,
            goalkeeper_saves=goalkeeper_saves,
            goals_conceded=goals_conceded,
            shots_on_goal_faced=shots_on_goal_faced,
            average_saves=self._average(
                goalkeeper_saves,
                matches,
            ),
            average_goals_conceded=self._average(
                goals_conceded,
                matches,
            ),
            average_shots_on_goal_faced=self._average(
                shots_on_goal_faced,
                matches,
            ),
            save_percentage=self._percentage(
                goalkeeper_saves,
                shots_on_goal_faced,
            ),
            clean_sheets=clean_sheets,
            matches_with_goal_conceded=(
                matches_with_goal_conceded
            ),
            matches_with_three_or_more_saves=(
                matches_with_three_or_more_saves
            ),
            matches_with_five_or_more_saves=(
                matches_with_five_or_more_saves
            ),
            clean_sheet_percentage=self._percentage(
                clean_sheets,
                matches,
            ),
            three_or_more_saves_percentage=(
                self._percentage(
                    matches_with_three_or_more_saves,
                    matches,
                )
            ),
            five_or_more_saves_percentage=(
                self._percentage(
                    matches_with_five_or_more_saves,
                    matches,
                )
            ),
            highest_saves=(
                max(save_values)
                if save_values
                else 0
            ),
            lowest_saves=(
                min(save_values)
                if save_values
                else 0
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