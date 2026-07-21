from dataclasses import asdict, dataclass

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.fixture import Fixture
from app.models.team import Team


@dataclass
class TeamFullTimeMarketResult:
    team_id: int
    team_name: str
    venue: str
    requested_limit: int
    matches: int

    wins: int
    draws: int
    losses: int

    win_percentage: float
    draw_percentage: float
    loss_percentage: float

    goals_for: int
    goals_against: int
    goal_difference: int

    average_goals_for: float
    average_goals_against: float
    average_total_goals: float

    scored_matches: int
    conceded_matches: int
    clean_sheets: int
    failed_to_score_matches: int

    scored_percentage: float
    conceded_percentage: float
    clean_sheet_percentage: float
    failed_to_score_percentage: float

    over_0_5_matches: int
    over_1_5_matches: int
    over_2_5_matches: int
    over_3_5_matches: int

    over_0_5_percentage: float
    over_1_5_percentage: float
    over_2_5_percentage: float
    over_3_5_percentage: float

    under_1_5_matches: int
    under_2_5_matches: int
    under_3_5_matches: int

    under_1_5_percentage: float
    under_2_5_percentage: float
    under_3_5_percentage: float

    both_teams_scored_matches: int
    both_teams_scored_percentage: float

    both_teams_not_scored_matches: int
    both_teams_not_scored_percentage: float

    win_or_draw_matches: int
    win_or_draw_percentage: float

    win_or_loss_matches: int
    win_or_loss_percentage: float

    draw_or_loss_matches: int
    draw_or_loss_percentage: float

    team_over_0_5_matches: int
    team_over_1_5_matches: int
    team_over_2_5_matches: int

    team_over_0_5_percentage: float
    team_over_1_5_percentage: float
    team_over_2_5_percentage: float

    opponent_over_0_5_matches: int
    opponent_over_1_5_matches: int
    opponent_over_2_5_matches: int

    opponent_over_0_5_percentage: float
    opponent_over_1_5_percentage: float
    opponent_over_2_5_percentage: float

    def to_dict(self) -> dict:
        return asdict(self)


class TeamFullTimeMarketAnalyzer:
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
    ) -> TeamFullTimeMarketResult:
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

        wins = 0
        draws = 0
        losses = 0

        goals_for = 0
        goals_against = 0

        scored_matches = 0
        conceded_matches = 0
        clean_sheets = 0
        failed_to_score_matches = 0

        over_0_5_matches = 0
        over_1_5_matches = 0
        over_2_5_matches = 0
        over_3_5_matches = 0

        under_1_5_matches = 0
        under_2_5_matches = 0
        under_3_5_matches = 0

        both_teams_scored_matches = 0
        both_teams_not_scored_matches = 0

        team_over_0_5_matches = 0
        team_over_1_5_matches = 0
        team_over_2_5_matches = 0

        opponent_over_0_5_matches = 0
        opponent_over_1_5_matches = 0
        opponent_over_2_5_matches = 0

        for fixture in fixtures:
            if fixture.home_team_id == team_id:
                match_goals_for = int(
                    fixture.home_goals
                )

                match_goals_against = int(
                    fixture.away_goals
                )

            else:
                match_goals_for = int(
                    fixture.away_goals
                )

                match_goals_against = int(
                    fixture.home_goals
                )

            goals_for += match_goals_for
            goals_against += match_goals_against

            if match_goals_for > match_goals_against:
                wins += 1

            elif match_goals_for == match_goals_against:
                draws += 1

            else:
                losses += 1

            if match_goals_for > 0:
                scored_matches += 1

            else:
                failed_to_score_matches += 1

            if match_goals_against > 0:
                conceded_matches += 1

            else:
                clean_sheets += 1

            total_goals = (
                match_goals_for
                + match_goals_against
            )

            if total_goals >= 1:
                over_0_5_matches += 1

            if total_goals >= 2:
                over_1_5_matches += 1

            if total_goals >= 3:
                over_2_5_matches += 1

            if total_goals >= 4:
                over_3_5_matches += 1

            if total_goals <= 1:
                under_1_5_matches += 1

            if total_goals <= 2:
                under_2_5_matches += 1

            if total_goals <= 3:
                under_3_5_matches += 1

            if (
                match_goals_for > 0
                and match_goals_against > 0
            ):
                both_teams_scored_matches += 1

            else:
                both_teams_not_scored_matches += 1

            if match_goals_for >= 1:
                team_over_0_5_matches += 1

            if match_goals_for >= 2:
                team_over_1_5_matches += 1

            if match_goals_for >= 3:
                team_over_2_5_matches += 1

            if match_goals_against >= 1:
                opponent_over_0_5_matches += 1

            if match_goals_against >= 2:
                opponent_over_1_5_matches += 1

            if match_goals_against >= 3:
                opponent_over_2_5_matches += 1

        matches = len(fixtures)

        win_or_draw_matches = wins + draws
        win_or_loss_matches = wins + losses
        draw_or_loss_matches = draws + losses

        return TeamFullTimeMarketResult(
            team_id=team.id,
            team_name=team.name,
            venue=venue,
            requested_limit=limit,
            matches=matches,
            wins=wins,
            draws=draws,
            losses=losses,
            win_percentage=self._percentage(
                wins,
                matches,
            ),
            draw_percentage=self._percentage(
                draws,
                matches,
            ),
            loss_percentage=self._percentage(
                losses,
                matches,
            ),
            goals_for=goals_for,
            goals_against=goals_against,
            goal_difference=(
                goals_for - goals_against
            ),
            average_goals_for=self._average(
                goals_for,
                matches,
            ),
            average_goals_against=self._average(
                goals_against,
                matches,
            ),
            average_total_goals=self._average(
                goals_for + goals_against,
                matches,
            ),
            scored_matches=scored_matches,
            conceded_matches=conceded_matches,
            clean_sheets=clean_sheets,
            failed_to_score_matches=(
                failed_to_score_matches
            ),
            scored_percentage=self._percentage(
                scored_matches,
                matches,
            ),
            conceded_percentage=self._percentage(
                conceded_matches,
                matches,
            ),
            clean_sheet_percentage=self._percentage(
                clean_sheets,
                matches,
            ),
            failed_to_score_percentage=(
                self._percentage(
                    failed_to_score_matches,
                    matches,
                )
            ),
            over_0_5_matches=over_0_5_matches,
            over_1_5_matches=over_1_5_matches,
            over_2_5_matches=over_2_5_matches,
            over_3_5_matches=over_3_5_matches,
            over_0_5_percentage=self._percentage(
                over_0_5_matches,
                matches,
            ),
            over_1_5_percentage=self._percentage(
                over_1_5_matches,
                matches,
            ),
            over_2_5_percentage=self._percentage(
                over_2_5_matches,
                matches,
            ),
            over_3_5_percentage=self._percentage(
                over_3_5_matches,
                matches,
            ),
            under_1_5_matches=under_1_5_matches,
            under_2_5_matches=under_2_5_matches,
            under_3_5_matches=under_3_5_matches,
            under_1_5_percentage=self._percentage(
                under_1_5_matches,
                matches,
            ),
            under_2_5_percentage=self._percentage(
                under_2_5_matches,
                matches,
            ),
            under_3_5_percentage=self._percentage(
                under_3_5_matches,
                matches,
            ),
            both_teams_scored_matches=(
                both_teams_scored_matches
            ),
            both_teams_scored_percentage=(
                self._percentage(
                    both_teams_scored_matches,
                    matches,
                )
            ),
            both_teams_not_scored_matches=(
                both_teams_not_scored_matches
            ),
            both_teams_not_scored_percentage=(
                self._percentage(
                    both_teams_not_scored_matches,
                    matches,
                )
            ),
            win_or_draw_matches=(
                win_or_draw_matches
            ),
            win_or_draw_percentage=(
                self._percentage(
                    win_or_draw_matches,
                    matches,
                )
            ),
            win_or_loss_matches=(
                win_or_loss_matches
            ),
            win_or_loss_percentage=(
                self._percentage(
                    win_or_loss_matches,
                    matches,
                )
            ),
            draw_or_loss_matches=(
                draw_or_loss_matches
            ),
            draw_or_loss_percentage=(
                self._percentage(
                    draw_or_loss_matches,
                    matches,
                )
            ),
            team_over_0_5_matches=(
                team_over_0_5_matches
            ),
            team_over_1_5_matches=(
                team_over_1_5_matches
            ),
            team_over_2_5_matches=(
                team_over_2_5_matches
            ),
            team_over_0_5_percentage=(
                self._percentage(
                    team_over_0_5_matches,
                    matches,
                )
            ),
            team_over_1_5_percentage=(
                self._percentage(
                    team_over_1_5_matches,
                    matches,
                )
            ),
            team_over_2_5_percentage=(
                self._percentage(
                    team_over_2_5_matches,
                    matches,
                )
            ),
            opponent_over_0_5_matches=(
                opponent_over_0_5_matches
            ),
            opponent_over_1_5_matches=(
                opponent_over_1_5_matches
            ),
            opponent_over_2_5_matches=(
                opponent_over_2_5_matches
            ),
            opponent_over_0_5_percentage=(
                self._percentage(
                    opponent_over_0_5_matches,
                    matches,
                )
            ),
            opponent_over_1_5_percentage=(
                self._percentage(
                    opponent_over_1_5_matches,
                    matches,
                )
            ),
            opponent_over_2_5_percentage=(
                self._percentage(
                    opponent_over_2_5_matches,
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