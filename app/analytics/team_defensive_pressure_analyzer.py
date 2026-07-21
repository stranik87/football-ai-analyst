from dataclasses import asdict, dataclass

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.fixture import Fixture
from app.models.fixture_team_statistics import (
    FixtureTeamStatistics,
)
from app.models.team import Team


@dataclass
class TeamDefensivePressureResult:
    team_id: int
    team_name: str
    venue: str
    requested_limit: int
    matches: int

    opponent_total_shots: int
    opponent_shots_on_goal: int

    goalkeeper_saves: int
    fouls: int
    clean_sheets: int

    average_opponent_total_shots: float
    average_opponent_shots_on_goal: float
    average_goalkeeper_saves: float
    average_fouls: float

    save_percentage: float
    clean_sheet_percentage: float
    defensive_pressure_score: float

    strong_defensive_matches: int
    medium_defensive_matches: int
    weak_defensive_matches: int

    strong_defensive_percentage: float
    medium_defensive_percentage: float
    weak_defensive_percentage: float

    def to_dict(self) -> dict:
        return asdict(self)


class TeamDefensivePressureAnalyzer:
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
    ) -> TeamDefensivePressureResult:
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

        opponent_total_shots = 0
        opponent_shots_on_goal = 0
        goalkeeper_saves = 0
        fouls = 0
        clean_sheets = 0

        defensive_scores: list[float] = []

        strong_defensive_matches = 0
        medium_defensive_matches = 0
        weak_defensive_matches = 0

        for fixture in fixtures:
            team_statistic = (
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

            opponent_statistic = (
                self.session.query(
                    FixtureTeamStatistics
                )
                .filter(
                    FixtureTeamStatistics.fixture_id
                    == fixture.id,
                    FixtureTeamStatistics.team_id
                    != team_id,
                )
                .first()
            )

            if not team_statistic:
                continue

            if not opponent_statistic:
                continue

            if opponent_statistic.total_shots is None:
                continue

            if opponent_statistic.shots_on_goal is None:
                continue

            if team_statistic.goalkeeper_saves is None:
                continue

            if team_statistic.fouls is None:
                continue

            match_opponent_total_shots = int(
                opponent_statistic.total_shots
            )

            match_opponent_shots_on_goal = int(
                opponent_statistic.shots_on_goal
            )

            match_goalkeeper_saves = int(
                team_statistic.goalkeeper_saves
            )

            match_fouls = int(
                team_statistic.fouls
            )

            goals_conceded = self._get_goals_conceded(
                fixture=fixture,
                team_id=team_id,
            )

            is_clean_sheet = goals_conceded == 0

            match_defensive_score = self._calculate_score(
                opponent_total_shots=(
                    match_opponent_total_shots
                ),
                opponent_shots_on_goal=(
                    match_opponent_shots_on_goal
                ),
                goalkeeper_saves=(
                    match_goalkeeper_saves
                ),
                goals_conceded=goals_conceded,
            )

            opponent_total_shots += (
                match_opponent_total_shots
            )

            opponent_shots_on_goal += (
                match_opponent_shots_on_goal
            )

            goalkeeper_saves += (
                match_goalkeeper_saves
            )

            fouls += match_fouls

            if is_clean_sheet:
                clean_sheets += 1

            defensive_scores.append(
                match_defensive_score
            )

            if match_defensive_score >= 70:
                strong_defensive_matches += 1

            elif match_defensive_score >= 45:
                medium_defensive_matches += 1

            else:
                weak_defensive_matches += 1

        matches = len(defensive_scores)

        return TeamDefensivePressureResult(
            team_id=team.id,
            team_name=team.name,
            venue=venue,
            requested_limit=limit,
            matches=matches,
            opponent_total_shots=(
                opponent_total_shots
            ),
            opponent_shots_on_goal=(
                opponent_shots_on_goal
            ),
            goalkeeper_saves=goalkeeper_saves,
            fouls=fouls,
            clean_sheets=clean_sheets,
            average_opponent_total_shots=(
                self._average(
                    opponent_total_shots,
                    matches,
                )
            ),
            average_opponent_shots_on_goal=(
                self._average(
                    opponent_shots_on_goal,
                    matches,
                )
            ),
            average_goalkeeper_saves=(
                self._average(
                    goalkeeper_saves,
                    matches,
                )
            ),
            average_fouls=self._average(
                fouls,
                matches,
            ),
            save_percentage=self._percentage(
                goalkeeper_saves,
                opponent_shots_on_goal,
            ),
            clean_sheet_percentage=(
                self._percentage(
                    clean_sheets,
                    matches,
                )
            ),
            defensive_pressure_score=(
                self._average_float(
                    sum(defensive_scores),
                    matches,
                )
            ),
            strong_defensive_matches=(
                strong_defensive_matches
            ),
            medium_defensive_matches=(
                medium_defensive_matches
            ),
            weak_defensive_matches=(
                weak_defensive_matches
            ),
            strong_defensive_percentage=(
                self._percentage(
                    strong_defensive_matches,
                    matches,
                )
            ),
            medium_defensive_percentage=(
                self._percentage(
                    medium_defensive_matches,
                    matches,
                )
            ),
            weak_defensive_percentage=(
                self._percentage(
                    weak_defensive_matches,
                    matches,
                )
            ),
        )

    @staticmethod
    def _get_goals_conceded(
        fixture: Fixture,
        team_id: int,
    ) -> int:
        if fixture.home_team_id == team_id:
            return int(
                fixture.away_goals or 0
            )

        return int(
            fixture.home_goals or 0
        )

    @staticmethod
    def _calculate_score(
        opponent_total_shots: int,
        opponent_shots_on_goal: int,
        goalkeeper_saves: int,
        goals_conceded: int,
    ) -> float:
        score = 100.0

        score -= opponent_total_shots * 1.5
        score -= opponent_shots_on_goal * 2.0
        score -= goals_conceded * 12.0
        score += goalkeeper_saves * 3.0

        if goals_conceded == 0:
            score += 10.0

        score = max(
            0.0,
            min(
                score,
                100.0,
            ),
        )

        return round(
            score,
            2,
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