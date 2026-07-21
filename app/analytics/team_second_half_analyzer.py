from dataclasses import asdict, dataclass

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.fixture import Fixture
from app.models.team import Team


@dataclass
class TeamSecondHalfResult:
    team_id: int
    team_name: str
    venue: str
    requested_limit: int
    matches: int

    second_half_wins: int
    second_half_draws: int
    second_half_losses: int

    second_half_goals_for: int
    second_half_goals_against: int
    second_half_goal_difference: int

    average_second_half_goals_for: float
    average_second_half_goals_against: float

    second_half_win_percentage: float
    second_half_draw_percentage: float
    second_half_loss_percentage: float

    scored_second_half_matches: int
    conceded_second_half_matches: int
    second_half_clean_sheets: int
    second_half_goalless_matches: int

    scored_second_half_percentage: float
    conceded_second_half_percentage: float
    second_half_clean_sheet_percentage: float
    second_half_goalless_percentage: float

    second_half_over_0_5_matches: int
    second_half_over_1_5_matches: int
    second_half_over_2_5_matches: int

    second_half_over_0_5_percentage: float
    second_half_over_1_5_percentage: float
    second_half_over_2_5_percentage: float

    second_half_both_teams_scored: int
    second_half_both_teams_scored_percentage: float

    def to_dict(self) -> dict:
        return asdict(self)


class TeamSecondHalfAnalyzer:
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
    ) -> TeamSecondHalfResult:
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

        second_half_wins = 0
        second_half_draws = 0
        second_half_losses = 0

        second_half_goals_for = 0
        second_half_goals_against = 0

        scored_second_half_matches = 0
        conceded_second_half_matches = 0
        second_half_clean_sheets = 0
        second_half_goalless_matches = 0

        second_half_over_0_5_matches = 0
        second_half_over_1_5_matches = 0
        second_half_over_2_5_matches = 0

        second_half_both_teams_scored = 0

        for fixture in fixtures:
            home_second_half_goals = (
                int(fixture.home_goals)
                - int(fixture.halftime_home)
            )

            away_second_half_goals = (
                int(fixture.away_goals)
                - int(fixture.halftime_away)
            )

            if fixture.home_team_id == team_id:
                goals_for = home_second_half_goals
                goals_against = away_second_half_goals

            else:
                goals_for = away_second_half_goals
                goals_against = home_second_half_goals

            if goals_for < 0 or goals_against < 0:
                continue

            second_half_goals_for += goals_for
            second_half_goals_against += goals_against

            if goals_for > goals_against:
                second_half_wins += 1

            elif goals_for == goals_against:
                second_half_draws += 1

            else:
                second_half_losses += 1

            if goals_for > 0:
                scored_second_half_matches += 1

            if goals_against > 0:
                conceded_second_half_matches += 1

            if goals_against == 0:
                second_half_clean_sheets += 1

            total_second_half_goals = (
                goals_for + goals_against
            )

            if total_second_half_goals == 0:
                second_half_goalless_matches += 1

            if total_second_half_goals >= 1:
                second_half_over_0_5_matches += 1

            if total_second_half_goals >= 2:
                second_half_over_1_5_matches += 1

            if total_second_half_goals >= 3:
                second_half_over_2_5_matches += 1

            if (
                goals_for > 0
                and goals_against > 0
            ):
                second_half_both_teams_scored += 1

        matches = (
            second_half_wins
            + second_half_draws
            + second_half_losses
        )

        return TeamSecondHalfResult(
            team_id=team.id,
            team_name=team.name,
            venue=venue,
            requested_limit=limit,
            matches=matches,
            second_half_wins=second_half_wins,
            second_half_draws=second_half_draws,
            second_half_losses=second_half_losses,
            second_half_goals_for=(
                second_half_goals_for
            ),
            second_half_goals_against=(
                second_half_goals_against
            ),
            second_half_goal_difference=(
                second_half_goals_for
                - second_half_goals_against
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
            second_half_win_percentage=(
                self._percentage(
                    second_half_wins,
                    matches,
                )
            ),
            second_half_draw_percentage=(
                self._percentage(
                    second_half_draws,
                    matches,
                )
            ),
            second_half_loss_percentage=(
                self._percentage(
                    second_half_losses,
                    matches,
                )
            ),
            scored_second_half_matches=(
                scored_second_half_matches
            ),
            conceded_second_half_matches=(
                conceded_second_half_matches
            ),
            second_half_clean_sheets=(
                second_half_clean_sheets
            ),
            second_half_goalless_matches=(
                second_half_goalless_matches
            ),
            scored_second_half_percentage=(
                self._percentage(
                    scored_second_half_matches,
                    matches,
                )
            ),
            conceded_second_half_percentage=(
                self._percentage(
                    conceded_second_half_matches,
                    matches,
                )
            ),
            second_half_clean_sheet_percentage=(
                self._percentage(
                    second_half_clean_sheets,
                    matches,
                )
            ),
            second_half_goalless_percentage=(
                self._percentage(
                    second_half_goalless_matches,
                    matches,
                )
            ),
            second_half_over_0_5_matches=(
                second_half_over_0_5_matches
            ),
            second_half_over_1_5_matches=(
                second_half_over_1_5_matches
            ),
            second_half_over_2_5_matches=(
                second_half_over_2_5_matches
            ),
            second_half_over_0_5_percentage=(
                self._percentage(
                    second_half_over_0_5_matches,
                    matches,
                )
            ),
            second_half_over_1_5_percentage=(
                self._percentage(
                    second_half_over_1_5_matches,
                    matches,
                )
            ),
            second_half_over_2_5_percentage=(
                self._percentage(
                    second_half_over_2_5_matches,
                    matches,
                )
            ),
            second_half_both_teams_scored=(
                second_half_both_teams_scored
            ),
            second_half_both_teams_scored_percentage=(
                self._percentage(
                    second_half_both_teams_scored,
                    matches,
                )
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