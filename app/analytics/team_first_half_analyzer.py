from dataclasses import asdict, dataclass

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.fixture import Fixture
from app.models.team import Team


@dataclass
class TeamFirstHalfResult:
    team_id: int
    team_name: str
    venue: str
    requested_limit: int
    matches: int

    first_half_wins: int
    first_half_draws: int
    first_half_losses: int

    first_half_goals_for: int
    first_half_goals_against: int
    first_half_goal_difference: int

    average_first_half_goals_for: float
    average_first_half_goals_against: float

    first_half_win_percentage: float
    first_half_draw_percentage: float
    first_half_loss_percentage: float

    scored_first_half_matches: int
    conceded_first_half_matches: int
    first_half_clean_sheets: int
    first_half_goalless_matches: int

    scored_first_half_percentage: float
    conceded_first_half_percentage: float
    first_half_clean_sheet_percentage: float
    first_half_goalless_percentage: float

    first_half_over_0_5_matches: int
    first_half_over_1_5_matches: int
    first_half_over_2_5_matches: int

    first_half_over_0_5_percentage: float
    first_half_over_1_5_percentage: float
    first_half_over_2_5_percentage: float

    first_half_both_teams_scored: int
    first_half_both_teams_scored_percentage: float

    def to_dict(self) -> dict:
        return asdict(self)


class TeamFirstHalfAnalyzer:
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
    ) -> TeamFirstHalfResult:
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

        first_half_wins = 0
        first_half_draws = 0
        first_half_losses = 0

        first_half_goals_for = 0
        first_half_goals_against = 0

        scored_first_half_matches = 0
        conceded_first_half_matches = 0
        first_half_clean_sheets = 0
        first_half_goalless_matches = 0

        first_half_over_0_5_matches = 0
        first_half_over_1_5_matches = 0
        first_half_over_2_5_matches = 0

        first_half_both_teams_scored = 0

        for fixture in fixtures:
            if fixture.home_team_id == team_id:
                goals_for = int(
                    fixture.halftime_home
                )

                goals_against = int(
                    fixture.halftime_away
                )

            else:
                goals_for = int(
                    fixture.halftime_away
                )

                goals_against = int(
                    fixture.halftime_home
                )

            first_half_goals_for += goals_for
            first_half_goals_against += goals_against

            if goals_for > goals_against:
                first_half_wins += 1

            elif goals_for == goals_against:
                first_half_draws += 1

            else:
                first_half_losses += 1

            if goals_for > 0:
                scored_first_half_matches += 1

            if goals_against > 0:
                conceded_first_half_matches += 1

            if goals_against == 0:
                first_half_clean_sheets += 1

            total_first_half_goals = (
                goals_for + goals_against
            )

            if total_first_half_goals == 0:
                first_half_goalless_matches += 1

            if total_first_half_goals >= 1:
                first_half_over_0_5_matches += 1

            if total_first_half_goals >= 2:
                first_half_over_1_5_matches += 1

            if total_first_half_goals >= 3:
                first_half_over_2_5_matches += 1

            if (
                goals_for > 0
                and goals_against > 0
            ):
                first_half_both_teams_scored += 1

        matches = len(fixtures)

        return TeamFirstHalfResult(
            team_id=team.id,
            team_name=team.name,
            venue=venue,
            requested_limit=limit,
            matches=matches,
            first_half_wins=first_half_wins,
            first_half_draws=first_half_draws,
            first_half_losses=first_half_losses,
            first_half_goals_for=(
                first_half_goals_for
            ),
            first_half_goals_against=(
                first_half_goals_against
            ),
            first_half_goal_difference=(
                first_half_goals_for
                - first_half_goals_against
            ),
            average_first_half_goals_for=(
                self._average(
                    first_half_goals_for,
                    matches,
                )
            ),
            average_first_half_goals_against=(
                self._average(
                    first_half_goals_against,
                    matches,
                )
            ),
            first_half_win_percentage=(
                self._percentage(
                    first_half_wins,
                    matches,
                )
            ),
            first_half_draw_percentage=(
                self._percentage(
                    first_half_draws,
                    matches,
                )
            ),
            first_half_loss_percentage=(
                self._percentage(
                    first_half_losses,
                    matches,
                )
            ),
            scored_first_half_matches=(
                scored_first_half_matches
            ),
            conceded_first_half_matches=(
                conceded_first_half_matches
            ),
            first_half_clean_sheets=(
                first_half_clean_sheets
            ),
            first_half_goalless_matches=(
                first_half_goalless_matches
            ),
            scored_first_half_percentage=(
                self._percentage(
                    scored_first_half_matches,
                    matches,
                )
            ),
            conceded_first_half_percentage=(
                self._percentage(
                    conceded_first_half_matches,
                    matches,
                )
            ),
            first_half_clean_sheet_percentage=(
                self._percentage(
                    first_half_clean_sheets,
                    matches,
                )
            ),
            first_half_goalless_percentage=(
                self._percentage(
                    first_half_goalless_matches,
                    matches,
                )
            ),
            first_half_over_0_5_matches=(
                first_half_over_0_5_matches
            ),
            first_half_over_1_5_matches=(
                first_half_over_1_5_matches
            ),
            first_half_over_2_5_matches=(
                first_half_over_2_5_matches
            ),
            first_half_over_0_5_percentage=(
                self._percentage(
                    first_half_over_0_5_matches,
                    matches,
                )
            ),
            first_half_over_1_5_percentage=(
                self._percentage(
                    first_half_over_1_5_matches,
                    matches,
                )
            ),
            first_half_over_2_5_percentage=(
                self._percentage(
                    first_half_over_2_5_matches,
                    matches,
                )
            ),
            first_half_both_teams_scored=(
                first_half_both_teams_scored
            ),
            first_half_both_teams_scored_percentage=(
                self._percentage(
                    first_half_both_teams_scored,
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