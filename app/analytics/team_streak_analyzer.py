from dataclasses import asdict, dataclass

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.fixture import Fixture
from app.models.team import Team


@dataclass
class TeamStreakResult:
    team_id: int
    team_name: str
    matches: int

    current_win_streak: int
    current_draw_streak: int
    current_loss_streak: int

    current_unbeaten_streak: int
    current_winless_streak: int

    current_scoring_streak: int
    current_clean_sheet_streak: int

    form: str

    def to_dict(self) -> dict:
        return asdict(self)


class TeamStreakAnalyzer:
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
        limit: int = 30,
    ) -> TeamStreakResult:
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

        matches_data: list[dict] = []

        for fixture in fixtures:
            is_home = (
                fixture.home_team_id == team_id
            )

            if is_home:
                goals_for = fixture.home_goals
                goals_against = fixture.away_goals
            else:
                goals_for = fixture.away_goals
                goals_against = fixture.home_goals

            if goals_for > goals_against:
                result = "W"
            elif goals_for == goals_against:
                result = "D"
            else:
                result = "L"

            matches_data.append(
                {
                    "result": result,
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                }
            )

        win_streak = self._count_streak(
            matches_data,
            lambda match: match["result"] == "W",
        )

        draw_streak = self._count_streak(
            matches_data,
            lambda match: match["result"] == "D",
        )

        loss_streak = self._count_streak(
            matches_data,
            lambda match: match["result"] == "L",
        )

        unbeaten_streak = self._count_streak(
            matches_data,
            lambda match: match["result"] != "L",
        )

        winless_streak = self._count_streak(
            matches_data,
            lambda match: match["result"] != "W",
        )

        scoring_streak = self._count_streak(
            matches_data,
            lambda match: match["goals_for"] > 0,
        )

        clean_sheet_streak = self._count_streak(
            matches_data,
            lambda match: match["goals_against"] == 0,
        )

        form = "".join(
            reversed(
                [
                    match["result"]
                    for match in matches_data
                ]
            )
        )

        return TeamStreakResult(
            team_id=team.id,
            team_name=team.name,
            matches=len(matches_data),
            current_win_streak=win_streak,
            current_draw_streak=draw_streak,
            current_loss_streak=loss_streak,
            current_unbeaten_streak=unbeaten_streak,
            current_winless_streak=winless_streak,
            current_scoring_streak=scoring_streak,
            current_clean_sheet_streak=(
                clean_sheet_streak
            ),
            form=form,
        )

    @staticmethod
    def _count_streak(
        matches_data: list[dict],
        condition,
    ) -> int:
        streak = 0

        for match in matches_data:
            if not condition(match):
                break

            streak += 1

        return streak