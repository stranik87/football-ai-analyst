from dataclasses import asdict, dataclass

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.fixture import Fixture
from app.models.team import Team


@dataclass
class TeamHalfTimeFullTimeResult:
    team_id: int
    team_name: str
    venue: str
    requested_limit: int
    matches: int

    ht_win_ft_win: int
    ht_win_ft_draw: int
    ht_win_ft_loss: int

    ht_draw_ft_win: int
    ht_draw_ft_draw: int
    ht_draw_ft_loss: int

    ht_loss_ft_win: int
    ht_loss_ft_draw: int
    ht_loss_ft_loss: int

    ht_win_ft_win_percentage: float
    ht_win_ft_draw_percentage: float
    ht_win_ft_loss_percentage: float

    ht_draw_ft_win_percentage: float
    ht_draw_ft_draw_percentage: float
    ht_draw_ft_loss_percentage: float

    ht_loss_ft_win_percentage: float
    ht_loss_ft_draw_percentage: float
    ht_loss_ft_loss_percentage: float

    leading_at_half_time: int
    drawing_at_half_time: int
    losing_at_half_time: int

    leading_at_half_time_percentage: float
    drawing_at_half_time_percentage: float
    losing_at_half_time_percentage: float

    winning_at_full_time: int
    drawing_at_full_time: int
    losing_at_full_time: int

    winning_at_full_time_percentage: float
    drawing_at_full_time_percentage: float
    losing_at_full_time_percentage: float

    held_half_time_lead: int
    failed_to_hold_half_time_lead: int
    half_time_lead_conversion_percentage: float

    won_after_half_time_draw: int
    drew_after_half_time_draw: int
    lost_after_half_time_draw: int

    won_after_half_time_draw_percentage: float
    drew_after_half_time_draw_percentage: float
    lost_after_half_time_draw_percentage: float

    comeback_wins: int
    comeback_draws: int
    comeback_losses: int

    comeback_win_percentage: float
    avoided_defeat_after_trailing_percentage: float

    result_changed_after_half_time: int
    result_unchanged_after_half_time: int

    result_changed_percentage: float
    result_unchanged_percentage: float

    def to_dict(self) -> dict:
        return asdict(self)


class TeamHalfTimeFullTimeAnalyzer:
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
        before_fixture_id: int | None = None,
    ) -> TeamHalfTimeFullTimeResult:
        if limit <= 0:
            raise ValueError(
                "limit должен быть больше нуля"
            )

        venue = venue.strip().lower()

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
                Fixture.home_goals.isnot(None),
                Fixture.away_goals.isnot(None),
                Fixture.halftime_home.isnot(None),
                Fixture.halftime_away.isnot(None),
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

        fixtures = (
            query.order_by(
                Fixture.kickoff.desc(),
                Fixture.id.desc(),
            )
            .limit(limit)
            .all()
        )

        combinations = {
            ("win", "win"): 0,
            ("win", "draw"): 0,
            ("win", "loss"): 0,
            ("draw", "win"): 0,
            ("draw", "draw"): 0,
            ("draw", "loss"): 0,
            ("loss", "win"): 0,
            ("loss", "draw"): 0,
            ("loss", "loss"): 0,
        }

        leading_at_half_time = 0
        drawing_at_half_time = 0
        losing_at_half_time = 0

        winning_at_full_time = 0
        drawing_at_full_time = 0
        losing_at_full_time = 0

        result_changed_after_half_time = 0
        result_unchanged_after_half_time = 0

        for fixture in fixtures:
            if fixture.home_team_id == team_id:
                half_time_goals_for = int(
                    fixture.halftime_home
                )
                half_time_goals_against = int(
                    fixture.halftime_away
                )
                full_time_goals_for = int(
                    fixture.home_goals
                )
                full_time_goals_against = int(
                    fixture.away_goals
                )

            else:
                half_time_goals_for = int(
                    fixture.halftime_away
                )
                half_time_goals_against = int(
                    fixture.halftime_home
                )
                full_time_goals_for = int(
                    fixture.away_goals
                )
                full_time_goals_against = int(
                    fixture.home_goals
                )

            half_time_result = self._get_result(
                goals_for=half_time_goals_for,
                goals_against=half_time_goals_against,
            )

            full_time_result = self._get_result(
                goals_for=full_time_goals_for,
                goals_against=full_time_goals_against,
            )

            combinations[
                (half_time_result, full_time_result)
            ] += 1

            if half_time_result == "win":
                leading_at_half_time += 1

            elif half_time_result == "draw":
                drawing_at_half_time += 1

            else:
                losing_at_half_time += 1

            if full_time_result == "win":
                winning_at_full_time += 1

            elif full_time_result == "draw":
                drawing_at_full_time += 1

            else:
                losing_at_full_time += 1

            if half_time_result == full_time_result:
                result_unchanged_after_half_time += 1

            else:
                result_changed_after_half_time += 1

        matches = len(fixtures)

        ht_win_ft_win = combinations[
            ("win", "win")
        ]
        ht_win_ft_draw = combinations[
            ("win", "draw")
        ]
        ht_win_ft_loss = combinations[
            ("win", "loss")
        ]

        ht_draw_ft_win = combinations[
            ("draw", "win")
        ]
        ht_draw_ft_draw = combinations[
            ("draw", "draw")
        ]
        ht_draw_ft_loss = combinations[
            ("draw", "loss")
        ]

        ht_loss_ft_win = combinations[
            ("loss", "win")
        ]
        ht_loss_ft_draw = combinations[
            ("loss", "draw")
        ]
        ht_loss_ft_loss = combinations[
            ("loss", "loss")
        ]

        held_half_time_lead = ht_win_ft_win
        failed_to_hold_half_time_lead = (
            ht_win_ft_draw + ht_win_ft_loss
        )

        comeback_wins = ht_loss_ft_win
        comeback_draws = ht_loss_ft_draw
        comeback_losses = ht_loss_ft_loss

        avoided_defeat_after_trailing = (
            comeback_wins + comeback_draws
        )

        return TeamHalfTimeFullTimeResult(
            team_id=team.id,
            team_name=team.name,
            venue=venue,
            requested_limit=limit,
            matches=matches,
            ht_win_ft_win=ht_win_ft_win,
            ht_win_ft_draw=ht_win_ft_draw,
            ht_win_ft_loss=ht_win_ft_loss,
            ht_draw_ft_win=ht_draw_ft_win,
            ht_draw_ft_draw=ht_draw_ft_draw,
            ht_draw_ft_loss=ht_draw_ft_loss,
            ht_loss_ft_win=ht_loss_ft_win,
            ht_loss_ft_draw=ht_loss_ft_draw,
            ht_loss_ft_loss=ht_loss_ft_loss,
            ht_win_ft_win_percentage=(
                self._percentage(
                    ht_win_ft_win,
                    matches,
                )
            ),
            ht_win_ft_draw_percentage=(
                self._percentage(
                    ht_win_ft_draw,
                    matches,
                )
            ),
            ht_win_ft_loss_percentage=(
                self._percentage(
                    ht_win_ft_loss,
                    matches,
                )
            ),
            ht_draw_ft_win_percentage=(
                self._percentage(
                    ht_draw_ft_win,
                    matches,
                )
            ),
            ht_draw_ft_draw_percentage=(
                self._percentage(
                    ht_draw_ft_draw,
                    matches,
                )
            ),
            ht_draw_ft_loss_percentage=(
                self._percentage(
                    ht_draw_ft_loss,
                    matches,
                )
            ),
            ht_loss_ft_win_percentage=(
                self._percentage(
                    ht_loss_ft_win,
                    matches,
                )
            ),
            ht_loss_ft_draw_percentage=(
                self._percentage(
                    ht_loss_ft_draw,
                    matches,
                )
            ),
            ht_loss_ft_loss_percentage=(
                self._percentage(
                    ht_loss_ft_loss,
                    matches,
                )
            ),
            leading_at_half_time=(
                leading_at_half_time
            ),
            drawing_at_half_time=(
                drawing_at_half_time
            ),
            losing_at_half_time=(
                losing_at_half_time
            ),
            leading_at_half_time_percentage=(
                self._percentage(
                    leading_at_half_time,
                    matches,
                )
            ),
            drawing_at_half_time_percentage=(
                self._percentage(
                    drawing_at_half_time,
                    matches,
                )
            ),
            losing_at_half_time_percentage=(
                self._percentage(
                    losing_at_half_time,
                    matches,
                )
            ),
            winning_at_full_time=(
                winning_at_full_time
            ),
            drawing_at_full_time=(
                drawing_at_full_time
            ),
            losing_at_full_time=(
                losing_at_full_time
            ),
            winning_at_full_time_percentage=(
                self._percentage(
                    winning_at_full_time,
                    matches,
                )
            ),
            drawing_at_full_time_percentage=(
                self._percentage(
                    drawing_at_full_time,
                    matches,
                )
            ),
            losing_at_full_time_percentage=(
                self._percentage(
                    losing_at_full_time,
                    matches,
                )
            ),
            held_half_time_lead=(
                held_half_time_lead
            ),
            failed_to_hold_half_time_lead=(
                failed_to_hold_half_time_lead
            ),
            half_time_lead_conversion_percentage=(
                self._percentage(
                    held_half_time_lead,
                    leading_at_half_time,
                )
            ),
            won_after_half_time_draw=(
                ht_draw_ft_win
            ),
            drew_after_half_time_draw=(
                ht_draw_ft_draw
            ),
            lost_after_half_time_draw=(
                ht_draw_ft_loss
            ),
            won_after_half_time_draw_percentage=(
                self._percentage(
                    ht_draw_ft_win,
                    drawing_at_half_time,
                )
            ),
            drew_after_half_time_draw_percentage=(
                self._percentage(
                    ht_draw_ft_draw,
                    drawing_at_half_time,
                )
            ),
            lost_after_half_time_draw_percentage=(
                self._percentage(
                    ht_draw_ft_loss,
                    drawing_at_half_time,
                )
            ),
            comeback_wins=comeback_wins,
            comeback_draws=comeback_draws,
            comeback_losses=comeback_losses,
            comeback_win_percentage=(
                self._percentage(
                    comeback_wins,
                    losing_at_half_time,
                )
            ),
            avoided_defeat_after_trailing_percentage=(
                self._percentage(
                    avoided_defeat_after_trailing,
                    losing_at_half_time,
                )
            ),
            result_changed_after_half_time=(
                result_changed_after_half_time
            ),
            result_unchanged_after_half_time=(
                result_unchanged_after_half_time
            ),
            result_changed_percentage=(
                self._percentage(
                    result_changed_after_half_time,
                    matches,
                )
            ),
            result_unchanged_percentage=(
                self._percentage(
                    result_unchanged_after_half_time,
                    matches,
                )
            ),
        )

    @staticmethod
    def _get_result(
        goals_for: int,
        goals_against: int,
    ) -> str:
        if goals_for > goals_against:
            return "win"

        if goals_for == goals_against:
            return "draw"

        return "loss"

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