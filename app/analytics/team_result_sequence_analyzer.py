from dataclasses import asdict, dataclass

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.fixture import Fixture
from app.models.team import Team


@dataclass
class TeamResultSequenceResult:
    team_id: int
    team_name: str
    venue: str
    requested_limit: int
    matches: int

    sequence: str
    current_result: str
    current_streak: int

    current_win_streak: int
    current_draw_streak: int
    current_loss_streak: int

    longest_win_streak: int
    longest_draw_streak: int
    longest_loss_streak: int

    current_unbeaten_streak: int
    current_winless_streak: int

    longest_unbeaten_streak: int
    longest_winless_streak: int

    wins: int
    draws: int
    losses: int

    win_percentage: float
    draw_percentage: float
    loss_percentage: float

    unbeaten_matches: int
    winless_matches: int

    unbeaten_percentage: float
    winless_percentage: float

    result_changes: int
    repeated_results: int

    result_change_percentage: float
    repeated_result_percentage: float

    wins_after_win: int
    draws_after_win: int
    losses_after_win: int

    wins_after_draw: int
    draws_after_draw: int
    losses_after_draw: int

    wins_after_loss: int
    draws_after_loss: int
    losses_after_loss: int

    recovery_after_loss_matches: int
    recovery_after_loss_percentage: float

    failed_recovery_after_loss_matches: int
    failed_recovery_after_loss_percentage: float

    def to_dict(self) -> dict:
        return asdict(self)


class TeamResultSequenceAnalyzer:
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
    ) -> TeamResultSequenceResult:
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

        chronological_fixtures = list(
            reversed(fixtures)
        )

        results: list[str] = []

        for fixture in chronological_fixtures:
            if fixture.home_team_id == team_id:
                goals_for = int(
                    fixture.home_goals
                )
                goals_against = int(
                    fixture.away_goals
                )

            else:
                goals_for = int(
                    fixture.away_goals
                )
                goals_against = int(
                    fixture.home_goals
                )

            results.append(
                self._get_result(
                    goals_for=goals_for,
                    goals_against=goals_against,
                )
            )

        matches = len(results)

        wins = results.count("W")
        draws = results.count("D")
        losses = results.count("L")

        longest_win_streak = self._longest_streak(
            results=results,
            target="W",
        )

        longest_draw_streak = self._longest_streak(
            results=results,
            target="D",
        )

        longest_loss_streak = self._longest_streak(
            results=results,
            target="L",
        )

        longest_unbeaten_streak = (
            self._longest_condition_streak(
                results=results,
                allowed={"W", "D"},
            )
        )

        longest_winless_streak = (
            self._longest_condition_streak(
                results=results,
                allowed={"D", "L"},
            )
        )

        current_result = (
            results[-1] if results else ""
        )

        current_streak = self._current_streak(
            results=results,
        )

        current_win_streak = (
            current_streak
            if current_result == "W"
            else 0
        )

        current_draw_streak = (
            current_streak
            if current_result == "D"
            else 0
        )

        current_loss_streak = (
            current_streak
            if current_result == "L"
            else 0
        )

        current_unbeaten_streak = (
            self._current_condition_streak(
                results=results,
                allowed={"W", "D"},
            )
        )

        current_winless_streak = (
            self._current_condition_streak(
                results=results,
                allowed={"D", "L"},
            )
        )

        result_changes = 0
        repeated_results = 0

        wins_after_win = 0
        draws_after_win = 0
        losses_after_win = 0

        wins_after_draw = 0
        draws_after_draw = 0
        losses_after_draw = 0

        wins_after_loss = 0
        draws_after_loss = 0
        losses_after_loss = 0

        for index in range(1, matches):
            previous_result = results[index - 1]
            current_match_result = results[index]

            if previous_result == current_match_result:
                repeated_results += 1
            else:
                result_changes += 1

            if previous_result == "W":
                if current_match_result == "W":
                    wins_after_win += 1
                elif current_match_result == "D":
                    draws_after_win += 1
                else:
                    losses_after_win += 1

            elif previous_result == "D":
                if current_match_result == "W":
                    wins_after_draw += 1
                elif current_match_result == "D":
                    draws_after_draw += 1
                else:
                    losses_after_draw += 1

            else:
                if current_match_result == "W":
                    wins_after_loss += 1
                elif current_match_result == "D":
                    draws_after_loss += 1
                else:
                    losses_after_loss += 1

        transitions = max(
            0,
            matches - 1,
        )

        losses_with_next_match = (
            wins_after_loss
            + draws_after_loss
            + losses_after_loss
        )

        recovery_after_loss_matches = (
            wins_after_loss + draws_after_loss
        )

        failed_recovery_after_loss_matches = (
            losses_after_loss
        )

        return TeamResultSequenceResult(
            team_id=team.id,
            team_name=team.name,
            venue=venue,
            requested_limit=limit,
            matches=matches,
            sequence="".join(results),
            current_result=current_result,
            current_streak=current_streak,
            current_win_streak=current_win_streak,
            current_draw_streak=current_draw_streak,
            current_loss_streak=current_loss_streak,
            longest_win_streak=longest_win_streak,
            longest_draw_streak=longest_draw_streak,
            longest_loss_streak=longest_loss_streak,
            current_unbeaten_streak=(
                current_unbeaten_streak
            ),
            current_winless_streak=(
                current_winless_streak
            ),
            longest_unbeaten_streak=(
                longest_unbeaten_streak
            ),
            longest_winless_streak=(
                longest_winless_streak
            ),
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
            unbeaten_matches=wins + draws,
            winless_matches=draws + losses,
            unbeaten_percentage=self._percentage(
                wins + draws,
                matches,
            ),
            winless_percentage=self._percentage(
                draws + losses,
                matches,
            ),
            result_changes=result_changes,
            repeated_results=repeated_results,
            result_change_percentage=(
                self._percentage(
                    result_changes,
                    transitions,
                )
            ),
            repeated_result_percentage=(
                self._percentage(
                    repeated_results,
                    transitions,
                )
            ),
            wins_after_win=wins_after_win,
            draws_after_win=draws_after_win,
            losses_after_win=losses_after_win,
            wins_after_draw=wins_after_draw,
            draws_after_draw=draws_after_draw,
            losses_after_draw=losses_after_draw,
            wins_after_loss=wins_after_loss,
            draws_after_loss=draws_after_loss,
            losses_after_loss=losses_after_loss,
            recovery_after_loss_matches=(
                recovery_after_loss_matches
            ),
            recovery_after_loss_percentage=(
                self._percentage(
                    recovery_after_loss_matches,
                    losses_with_next_match,
                )
            ),
            failed_recovery_after_loss_matches=(
                failed_recovery_after_loss_matches
            ),
            failed_recovery_after_loss_percentage=(
                self._percentage(
                    failed_recovery_after_loss_matches,
                    losses_with_next_match,
                )
            ),
        )

    @staticmethod
    def _get_result(
        goals_for: int,
        goals_against: int,
    ) -> str:
        if goals_for > goals_against:
            return "W"

        if goals_for == goals_against:
            return "D"

        return "L"

    @staticmethod
    def _longest_streak(
        results: list[str],
        target: str,
    ) -> int:
        longest = 0
        current = 0

        for result in results:
            if result == target:
                current += 1
                longest = max(
                    longest,
                    current,
                )
            else:
                current = 0

        return longest

    @staticmethod
    def _longest_condition_streak(
        results: list[str],
        allowed: set[str],
    ) -> int:
        longest = 0
        current = 0

        for result in results:
            if result in allowed:
                current += 1
                longest = max(
                    longest,
                    current,
                )
            else:
                current = 0

        return longest

    @staticmethod
    def _current_streak(
        results: list[str],
    ) -> int:
        if not results:
            return 0

        target = results[-1]
        streak = 0

        for result in reversed(results):
            if result != target:
                break

            streak += 1

        return streak

    @staticmethod
    def _current_condition_streak(
        results: list[str],
        allowed: set[str],
    ) -> int:
        streak = 0

        for result in reversed(results):
            if result not in allowed:
                break

            streak += 1

        return streak

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