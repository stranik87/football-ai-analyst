from dataclasses import asdict, dataclass

from sqlalchemy.orm import Session

from app.models.standing import Standing
from app.models.team import Team


@dataclass
class TeamStandingResult:
    team_id: int
    team_name: str
    league_season_id: int

    rank: int | None
    points: int | None
    goals_difference: int | None
    form: str | None

    played: int | None
    wins: int | None
    draws: int | None
    losses: int | None

    goals_for: int | None
    goals_against: int | None

    group: str | None
    description: str | None

    def to_dict(self) -> dict:
        return asdict(self)


class StandingsAnalyzer:
    def __init__(self, session: Session):
        self.session = session

    def analyze(
        self,
        team_id: int,
        league_season_id: int | None = None,
    ) -> TeamStandingResult:
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
            self.session.query(Standing)
            .filter(Standing.team_id == team_id)
        )

        if league_season_id is not None:
            query = query.filter(
                Standing.league_season_id
                == league_season_id
            )

        standing = (
            query.order_by(
                Standing.league_season_id.desc()
            )
            .first()
        )

        if not standing:
            raise ValueError(
                f"Турнирная таблица команды "
                f"{team.name} не найдена"
            )

        return TeamStandingResult(
            team_id=team.id,
            team_name=team.name,
            league_season_id=(
                standing.league_season_id
            ),
            rank=self._get_value(
                standing,
                "rank",
            ),
            points=self._get_value(
                standing,
                "points",
            ),
            goals_difference=self._get_value(
                standing,
                "goals_difference",
                "goals_diff",
            ),
            form=self._get_value(
                standing,
                "form",
            ),
            played=self._get_value(
                standing,
                "played",
                "all_played",
            ),
            wins=self._get_value(
                standing,
                "wins",
                "all_win",
            ),
            draws=self._get_value(
                standing,
                "draws",
                "all_draw",
            ),
            losses=self._get_value(
                standing,
                "losses",
                "all_lose",
            ),
            goals_for=self._get_value(
                standing,
                "goals_for",
                "all_goals_for",
            ),
            goals_against=self._get_value(
                standing,
                "goals_against",
                "all_goals_against",
            ),
            group=self._get_value(
                standing,
                "group",
                "group_name",
            ),
            description=self._get_value(
                standing,
                "description",
            ),
        )

    @staticmethod
    def _get_value(
        standing: Standing,
        *field_names: str,
    ):
        for field_name in field_names:
            if hasattr(standing, field_name):
                return getattr(
                    standing,
                    field_name,
                )

        return None