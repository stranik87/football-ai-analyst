from dataclasses import asdict, dataclass

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.fixture import Fixture
from app.models.team import Team


@dataclass
class VenueStatistics:
    matches: int

    wins: int
    draws: int
    losses: int
    points: int

    goals_for: int
    goals_against: int
    goal_difference: int

    average_goals_for: float
    average_goals_against: float
    points_per_match: float

    win_percentage: float
    draw_percentage: float
    loss_percentage: float

    clean_sheets: int
    failed_to_score: int


@dataclass
class TeamVenueSplitResult:
    team_id: int
    team_name: str
    requested_limit: int

    home: VenueStatistics
    away: VenueStatistics

    def to_dict(self) -> dict:
        return asdict(self)


class TeamVenueSplitAnalyzer:
    FINISHED_STATUSES = (
        "FT",
        "AET",
        "PEN",
    )

    def __init__(self, session: Session):
        self.session = session

    def analyze(
        self,
        team_id: int,
        limit: int = 20,
    ) -> TeamVenueSplitResult:
        if limit <= 0:
            raise ValueError(
                "limit должен быть больше нуля"
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

        fixtures = (
            self.session.query(Fixture)
            .filter(
                Fixture.status_short.in_(
                    self.FINISHED_STATUSES
                ),
                Fixture.home_goals.isnot(None),
                Fixture.away_goals.isnot(None),
            )
            .filter(
                or_(
                    Fixture.home_team_id == team_id,
                    Fixture.away_team_id == team_id,
                )
            )
            .order_by(
                Fixture.kickoff.desc(),
                Fixture.id.desc(),
            )
            .limit(limit)
            .all()
        )

        home_matches = []
        away_matches = []

        for fixture in fixtures:
            if fixture.home_team_id == team_id:
                home_matches.append(
                    {
                        "goals_for": fixture.home_goals,
                        "goals_against": fixture.away_goals,
                    }
                )
            else:
                away_matches.append(
                    {
                        "goals_for": fixture.away_goals,
                        "goals_against": fixture.home_goals,
                    }
                )

        return TeamVenueSplitResult(
            team_id=team.id,
            team_name=team.name,
            requested_limit=limit,
            home=self._calculate_statistics(
                home_matches
            ),
            away=self._calculate_statistics(
                away_matches
            ),
        )

    @staticmethod
    def _calculate_statistics(
        matches_data: list[dict],
    ) -> VenueStatistics:
        matches = len(matches_data)

        wins = 0
        draws = 0
        losses = 0

        goals_for = 0
        goals_against = 0

        clean_sheets = 0
        failed_to_score = 0

        for match in matches_data:
            match_goals_for = match["goals_for"]
            match_goals_against = match["goals_against"]

            goals_for += match_goals_for
            goals_against += match_goals_against

            if match_goals_for > match_goals_against:
                wins += 1
            elif match_goals_for == match_goals_against:
                draws += 1
            else:
                losses += 1

            if match_goals_against == 0:
                clean_sheets += 1

            if match_goals_for == 0:
                failed_to_score += 1

        points = wins * 3 + draws

        return VenueStatistics(
            matches=matches,
            wins=wins,
            draws=draws,
            losses=losses,
            points=points,
            goals_for=goals_for,
            goals_against=goals_against,
            goal_difference=(
                goals_for - goals_against
            ),
            average_goals_for=(
                TeamVenueSplitAnalyzer._average(
                    goals_for,
                    matches,
                )
            ),
            average_goals_against=(
                TeamVenueSplitAnalyzer._average(
                    goals_against,
                    matches,
                )
            ),
            points_per_match=(
                TeamVenueSplitAnalyzer._average(
                    points,
                    matches,
                )
            ),
            win_percentage=(
                TeamVenueSplitAnalyzer._percentage(
                    wins,
                    matches,
                )
            ),
            draw_percentage=(
                TeamVenueSplitAnalyzer._percentage(
                    draws,
                    matches,
                )
            ),
            loss_percentage=(
                TeamVenueSplitAnalyzer._percentage(
                    losses,
                    matches,
                )
            ),
            clean_sheets=clean_sheets,
            failed_to_score=failed_to_score,
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
        matches: int,
    ) -> float:
        if matches == 0:
            return 0.0

        return round(
            value / matches * 100,
            2,
        )