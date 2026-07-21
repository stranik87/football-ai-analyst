from dataclasses import asdict, dataclass

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from app.models.fixture import Fixture
from app.models.team import Team


@dataclass
class HeadToHeadResult:
    home_team_id: int
    home_team_name: str

    away_team_id: int
    away_team_name: str

    matches: int

    home_team_wins: int
    draws: int
    away_team_wins: int

    home_team_goals: int
    away_team_goals: int

    both_teams_scored: int
    over_1_5: int
    over_2_5: int
    over_3_5: int

    average_total_goals: float

    form: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


class HeadToHeadAnalyzer:
    FINISHED_STATUSES = (
        "FT",
        "AET",
        "PEN",
    )

    def __init__(self, session: Session):
        self.session = session

    def analyze(
        self,
        home_team_id: int,
        away_team_id: int,
        limit: int = 10,
    ) -> HeadToHeadResult:
        if home_team_id == away_team_id:
            raise ValueError(
                "Нельзя сравнивать команду саму с собой"
            )

        if limit <= 0:
            raise ValueError(
                "limit должен быть больше нуля"
            )

        home_team = (
            self.session.query(Team)
            .filter(Team.id == home_team_id)
            .first()
        )

        if not home_team:
            raise ValueError(
                f"Команда не найдена: "
                f"team_id={home_team_id}"
            )

        away_team = (
            self.session.query(Team)
            .filter(Team.id == away_team_id)
            .first()
        )

        if not away_team:
            raise ValueError(
                f"Команда не найдена: "
                f"team_id={away_team_id}"
            )

        fixtures = (
            self.session.query(Fixture)
            .filter(
                Fixture.status_short.in_(
                    self.FINISHED_STATUSES
                ),
                Fixture.home_goals.isnot(None),
                Fixture.away_goals.isnot(None),
                or_(
                    and_(
                        Fixture.home_team_id
                        == home_team_id,
                        Fixture.away_team_id
                        == away_team_id,
                    ),
                    and_(
                        Fixture.home_team_id
                        == away_team_id,
                        Fixture.away_team_id
                        == home_team_id,
                    ),
                ),
            )
            .order_by(
                Fixture.kickoff.desc(),
                Fixture.id.desc(),
            )
            .limit(limit)
            .all()
        )

        home_team_wins = 0
        draws = 0
        away_team_wins = 0

        home_team_goals = 0
        away_team_goals = 0

        both_teams_scored = 0
        over_1_5 = 0
        over_2_5 = 0
        over_3_5 = 0

        form = []

        for fixture in fixtures:
            first_team_is_home = (
                fixture.home_team_id
                == home_team_id
            )

            if first_team_is_home:
                first_team_goals = (
                    fixture.home_goals
                )
                second_team_goals = (
                    fixture.away_goals
                )
            else:
                first_team_goals = (
                    fixture.away_goals
                )
                second_team_goals = (
                    fixture.home_goals
                )

            home_team_goals += first_team_goals
            away_team_goals += second_team_goals

            if first_team_goals > second_team_goals:
                home_team_wins += 1
                result = "W"
            elif first_team_goals == second_team_goals:
                draws += 1
                result = "D"
            else:
                away_team_wins += 1
                result = "L"

            total_goals = (
                first_team_goals
                + second_team_goals
            )

            if (
                first_team_goals > 0
                and second_team_goals > 0
            ):
                both_teams_scored += 1

            if total_goals > 1.5:
                over_1_5 += 1

            if total_goals > 2.5:
                over_2_5 += 1

            if total_goals > 3.5:
                over_3_5 += 1

            form.append(
                {
                    "fixture_id": fixture.id,
                    "kickoff": (
                        fixture.kickoff.isoformat()
                        if fixture.kickoff
                        else None
                    ),
                    "home_team_id": (
                        fixture.home_team_id
                    ),
                    "away_team_id": (
                        fixture.away_team_id
                    ),
                    "home_goals": (
                        fixture.home_goals
                    ),
                    "away_goals": (
                        fixture.away_goals
                    ),
                    "result_for_first_team": result,
                }
            )

        matches = len(fixtures)

        average_total_goals = (
            round(
                (
                    home_team_goals
                    + away_team_goals
                )
                / matches,
                2,
            )
            if matches
            else 0.0
        )

        return HeadToHeadResult(
            home_team_id=home_team.id,
            home_team_name=home_team.name,
            away_team_id=away_team.id,
            away_team_name=away_team.name,
            matches=matches,
            home_team_wins=home_team_wins,
            draws=draws,
            away_team_wins=away_team_wins,
            home_team_goals=home_team_goals,
            away_team_goals=away_team_goals,
            both_teams_scored=both_teams_scored,
            over_1_5=over_1_5,
            over_2_5=over_2_5,
            over_3_5=over_3_5,
            average_total_goals=(
                average_total_goals
            ),
            form=form,
        )