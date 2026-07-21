from dataclasses import asdict, dataclass

from sqlalchemy.orm import Session

from app.analytics.standings_analyzer import (
    StandingsAnalyzer,
)
from app.analytics.team_form_analyzer import (
    TeamFormAnalyzer,
)
from app.analytics.team_statistics_analyzer import (
    TeamStatisticsAnalyzer,
)
from app.analytics.team_streak_analyzer import (
    TeamStreakAnalyzer,
)
from app.models.team import Team


@dataclass
class TeamComparisonResult:
    first_team_id: int
    first_team_name: str

    second_team_id: int
    second_team_name: str

    limit: int

    first_team: dict
    second_team: dict

    def to_dict(self) -> dict:
        return asdict(self)


class TeamComparisonAnalyzer:
    def __init__(self, session: Session):
        self.session = session

        self.form_analyzer = TeamFormAnalyzer(
            session=session
        )

        self.statistics_analyzer = (
            TeamStatisticsAnalyzer(
                session=session
            )
        )

        self.streak_analyzer = TeamStreakAnalyzer(
            session=session
        )

        self.standings_analyzer = (
            StandingsAnalyzer(
                session=session
            )
        )

    def analyze(
        self,
        first_team_id: int,
        second_team_id: int,
        limit: int = 10,
    ) -> TeamComparisonResult:
        if first_team_id == second_team_id:
            raise ValueError(
                "Нельзя сравнивать команду саму с собой"
            )

        if limit <= 0:
            raise ValueError(
                "limit должен быть больше нуля"
            )

        first_team = self._get_team(
            team_id=first_team_id
        )

        second_team = self._get_team(
            team_id=second_team_id
        )

        first_team_data = self._analyze_team(
            team_id=first_team.id,
            limit=limit,
        )

        second_team_data = self._analyze_team(
            team_id=second_team.id,
            limit=limit,
        )

        return TeamComparisonResult(
            first_team_id=first_team.id,
            first_team_name=first_team.name,
            second_team_id=second_team.id,
            second_team_name=second_team.name,
            limit=limit,
            first_team=first_team_data,
            second_team=second_team_data,
        )

    def _get_team(
        self,
        team_id: int,
    ) -> Team:
        team = (
            self.session.query(Team)
            .filter(Team.id == team_id)
            .first()
        )

        if not team:
            raise ValueError(
                f"Команда не найдена: team_id={team_id}"
            )

        return team

    def _analyze_team(
        self,
        team_id: int,
        limit: int,
    ) -> dict:
        form = self.form_analyzer.analyze(
            team_id=team_id,
            limit=limit,
        )

        statistics = (
            self.statistics_analyzer.analyze(
                team_id=team_id,
                limit=limit,
            )
        )

        streaks = self.streak_analyzer.analyze(
            team_id=team_id,
            limit=limit,
        )

        standing = None

        try:
            standing_result = (
                self.standings_analyzer.analyze(
                    team_id=team_id
                )
            )

            standing = standing_result.to_dict()

        except ValueError:
            standing = None

        return {
            "form": form.to_dict(),
            "statistics": statistics.to_dict(),
            "streaks": streaks.to_dict(),
            "standing": standing,
        }