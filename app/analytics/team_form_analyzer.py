from dataclasses import asdict, dataclass

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.fixture import Fixture
from app.models.team import Team


@dataclass
class TeamFormResult:
    """
    Результат анализа формы команды.
    """

    team_id: int
    team_name: str
    venue: str
    requested_limit: int
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

    home_matches: int
    away_matches: int

    form: str

    def to_dict(self) -> dict:
        return asdict(self)


class TeamFormAnalyzer:
    """
    Анализ формы команды по завершённым матчам.
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

    def __init__(self, session: Session):
        self.session = session

    def analyze(
        self,
        team_id: int,
        limit: int = 5,
        venue: str = "all",
        before_fixture_id: int | None = None,
    ) -> TeamFormResult:
        """
        Рассчитать форму команды по последним матчам.

        venue:
        - all — все матчи;
        - home — только домашние;
        - away — только выездные.

        before_fixture_id позволяет брать только матчи,
        сыгранные до определённого матча.
        """
        if limit <= 0:
            raise ValueError(
                "limit должен быть больше нуля"
            )

        venue = venue.lower().strip()

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
            self.session.query(Fixture)
            .filter(
                Fixture.status_short.in_(
                    self.FINISHED_STATUSES
                )
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
                    "Матч для ограничения истории "
                    f"не найден: fixture_id={before_fixture_id}"
                )

            query = query.filter(
                Fixture.kickoff < target_fixture.kickoff
            )

        fixtures = (
            query
            .order_by(Fixture.kickoff.desc())
            .limit(limit)
            .all()
        )

        wins = 0
        draws = 0
        losses = 0
        points = 0

        goals_for = 0
        goals_against = 0

        home_matches = 0
        away_matches = 0

        form_items: list[str] = []

        for fixture in fixtures:
            is_home = (
                fixture.home_team_id == team_id
            )

            if is_home:
                home_matches += 1
                team_goals = fixture.home_goals
                opponent_goals = fixture.away_goals
            else:
                away_matches += 1
                team_goals = fixture.away_goals
                opponent_goals = fixture.home_goals

            if (
                team_goals is None
                or opponent_goals is None
            ):
                continue

            goals_for += team_goals
            goals_against += opponent_goals

            if team_goals > opponent_goals:
                wins += 1
                points += 3
                form_items.append("W")

            elif team_goals == opponent_goals:
                draws += 1
                points += 1
                form_items.append("D")

            else:
                losses += 1
                form_items.append("L")

        matches = wins + draws + losses

        average_goals_for = (
            round(goals_for / matches, 2)
            if matches
            else 0.0
        )

        average_goals_against = (
            round(goals_against / matches, 2)
            if matches
            else 0.0
        )

        points_per_match = (
            round(points / matches, 2)
            if matches
            else 0.0
        )

        return TeamFormResult(
            team_id=team.id,
            team_name=team.name,
            venue=venue,
            requested_limit=limit,
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
            average_goals_for=average_goals_for,
            average_goals_against=(
                average_goals_against
            ),
            points_per_match=points_per_match,
            home_matches=home_matches,
            away_matches=away_matches,
            form="".join(reversed(form_items)),
        )