from dataclasses import asdict, dataclass

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.fixture import Fixture
from app.models.team import Team


@dataclass
class TeamScoringPatternResult:
    team_id: int
    team_name: str
    venue: str
    requested_limit: int
    matches: int

    total_goals_for: int
    total_goals_against: int

    first_half_goals_for: int
    second_half_goals_for: int

    first_half_goals_against: int
    second_half_goals_against: int

    average_first_half_goals_for: float
    average_second_half_goals_for: float

    average_first_half_goals_against: float
    average_second_half_goals_against: float

    scored_first_half_matches: int
    scored_second_half_matches: int
    scored_both_halves_matches: int
    scored_neither_half_matches: int

    scored_first_half_percentage: float
    scored_second_half_percentage: float
    scored_both_halves_percentage: float
    scored_neither_half_percentage: float

    conceded_first_half_matches: int
    conceded_second_half_matches: int
    conceded_both_halves_matches: int
    conceded_neither_half_matches: int

    conceded_first_half_percentage: float
    conceded_second_half_percentage: float
    conceded_both_halves_percentage: float
    conceded_neither_half_percentage: float

    scored_only_first_half_matches: int
    scored_only_second_half_matches: int

    scored_only_first_half_percentage: float
    scored_only_second_half_percentage: float

    conceded_only_first_half_matches: int
    conceded_only_second_half_matches: int

    conceded_only_first_half_percentage: float
    conceded_only_second_half_percentage: float

    more_goals_first_half_matches: int
    equal_goals_each_half_matches: int
    more_goals_second_half_matches: int

    more_goals_first_half_percentage: float
    equal_goals_each_half_percentage: float
    more_goals_second_half_percentage: float

    conceded_more_first_half_matches: int
    conceded_equal_each_half_matches: int
    conceded_more_second_half_matches: int

    conceded_more_first_half_percentage: float
    conceded_equal_each_half_percentage: float
    conceded_more_second_half_percentage: float

    first_half_goal_share_percentage: float
    second_half_goal_share_percentage: float

    first_half_conceded_share_percentage: float
    second_half_conceded_share_percentage: float

    def to_dict(self) -> dict:
        return asdict(self)


class TeamScoringPatternAnalyzer:
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
    ) -> TeamScoringPatternResult:
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

        total_goals_for = 0
        total_goals_against = 0

        first_half_goals_for = 0
        second_half_goals_for = 0

        first_half_goals_against = 0
        second_half_goals_against = 0

        scored_first_half_matches = 0
        scored_second_half_matches = 0
        scored_both_halves_matches = 0
        scored_neither_half_matches = 0

        conceded_first_half_matches = 0
        conceded_second_half_matches = 0
        conceded_both_halves_matches = 0
        conceded_neither_half_matches = 0

        scored_only_first_half_matches = 0
        scored_only_second_half_matches = 0

        conceded_only_first_half_matches = 0
        conceded_only_second_half_matches = 0

        more_goals_first_half_matches = 0
        equal_goals_each_half_matches = 0
        more_goals_second_half_matches = 0

        conceded_more_first_half_matches = 0
        conceded_equal_each_half_matches = 0
        conceded_more_second_half_matches = 0

        for fixture in fixtures:
            if fixture.home_team_id == team_id:
                match_total_goals_for = int(
                    fixture.home_goals
                )
                match_total_goals_against = int(
                    fixture.away_goals
                )

                match_first_half_goals_for = int(
                    fixture.halftime_home
                )
                match_first_half_goals_against = int(
                    fixture.halftime_away
                )

            else:
                match_total_goals_for = int(
                    fixture.away_goals
                )
                match_total_goals_against = int(
                    fixture.home_goals
                )

                match_first_half_goals_for = int(
                    fixture.halftime_away
                )
                match_first_half_goals_against = int(
                    fixture.halftime_home
                )

            match_second_half_goals_for = max(
                0,
                match_total_goals_for
                - match_first_half_goals_for,
            )

            match_second_half_goals_against = max(
                0,
                match_total_goals_against
                - match_first_half_goals_against,
            )

            total_goals_for += match_total_goals_for
            total_goals_against += (
                match_total_goals_against
            )

            first_half_goals_for += (
                match_first_half_goals_for
            )
            second_half_goals_for += (
                match_second_half_goals_for
            )

            first_half_goals_against += (
                match_first_half_goals_against
            )
            second_half_goals_against += (
                match_second_half_goals_against
            )

            scored_first_half = (
                match_first_half_goals_for > 0
            )
            scored_second_half = (
                match_second_half_goals_for > 0
            )

            conceded_first_half = (
                match_first_half_goals_against > 0
            )
            conceded_second_half = (
                match_second_half_goals_against > 0
            )

            if scored_first_half:
                scored_first_half_matches += 1

            if scored_second_half:
                scored_second_half_matches += 1

            if scored_first_half and scored_second_half:
                scored_both_halves_matches += 1

            elif scored_first_half:
                scored_only_first_half_matches += 1

            elif scored_second_half:
                scored_only_second_half_matches += 1

            else:
                scored_neither_half_matches += 1

            if conceded_first_half:
                conceded_first_half_matches += 1

            if conceded_second_half:
                conceded_second_half_matches += 1

            if (
                conceded_first_half
                and conceded_second_half
            ):
                conceded_both_halves_matches += 1

            elif conceded_first_half:
                conceded_only_first_half_matches += 1

            elif conceded_second_half:
                conceded_only_second_half_matches += 1

            else:
                conceded_neither_half_matches += 1

            if (
                match_first_half_goals_for
                > match_second_half_goals_for
            ):
                more_goals_first_half_matches += 1

            elif (
                match_first_half_goals_for
                == match_second_half_goals_for
            ):
                equal_goals_each_half_matches += 1

            else:
                more_goals_second_half_matches += 1

            if (
                match_first_half_goals_against
                > match_second_half_goals_against
            ):
                conceded_more_first_half_matches += 1

            elif (
                match_first_half_goals_against
                == match_second_half_goals_against
            ):
                conceded_equal_each_half_matches += 1

            else:
                conceded_more_second_half_matches += 1

        matches = len(fixtures)

        return TeamScoringPatternResult(
            team_id=team.id,
            team_name=team.name,
            venue=venue,
            requested_limit=limit,
            matches=matches,
            total_goals_for=total_goals_for,
            total_goals_against=total_goals_against,
            first_half_goals_for=first_half_goals_for,
            second_half_goals_for=second_half_goals_for,
            first_half_goals_against=(
                first_half_goals_against
            ),
            second_half_goals_against=(
                second_half_goals_against
            ),
            average_first_half_goals_for=(
                self._average(
                    first_half_goals_for,
                    matches,
                )
            ),
            average_second_half_goals_for=(
                self._average(
                    second_half_goals_for,
                    matches,
                )
            ),
            average_first_half_goals_against=(
                self._average(
                    first_half_goals_against,
                    matches,
                )
            ),
            average_second_half_goals_against=(
                self._average(
                    second_half_goals_against,
                    matches,
                )
            ),
            scored_first_half_matches=(
                scored_first_half_matches
            ),
            scored_second_half_matches=(
                scored_second_half_matches
            ),
            scored_both_halves_matches=(
                scored_both_halves_matches
            ),
            scored_neither_half_matches=(
                scored_neither_half_matches
            ),
            scored_first_half_percentage=(
                self._percentage(
                    scored_first_half_matches,
                    matches,
                )
            ),
            scored_second_half_percentage=(
                self._percentage(
                    scored_second_half_matches,
                    matches,
                )
            ),
            scored_both_halves_percentage=(
                self._percentage(
                    scored_both_halves_matches,
                    matches,
                )
            ),
            scored_neither_half_percentage=(
                self._percentage(
                    scored_neither_half_matches,
                    matches,
                )
            ),
            conceded_first_half_matches=(
                conceded_first_half_matches
            ),
            conceded_second_half_matches=(
                conceded_second_half_matches
            ),
            conceded_both_halves_matches=(
                conceded_both_halves_matches
            ),
            conceded_neither_half_matches=(
                conceded_neither_half_matches
            ),
            conceded_first_half_percentage=(
                self._percentage(
                    conceded_first_half_matches,
                    matches,
                )
            ),
            conceded_second_half_percentage=(
                self._percentage(
                    conceded_second_half_matches,
                    matches,
                )
            ),
            conceded_both_halves_percentage=(
                self._percentage(
                    conceded_both_halves_matches,
                    matches,
                )
            ),
            conceded_neither_half_percentage=(
                self._percentage(
                    conceded_neither_half_matches,
                    matches,
                )
            ),
            scored_only_first_half_matches=(
                scored_only_first_half_matches
            ),
            scored_only_second_half_matches=(
                scored_only_second_half_matches
            ),
            scored_only_first_half_percentage=(
                self._percentage(
                    scored_only_first_half_matches,
                    matches,
                )
            ),
            scored_only_second_half_percentage=(
                self._percentage(
                    scored_only_second_half_matches,
                    matches,
                )
            ),
            conceded_only_first_half_matches=(
                conceded_only_first_half_matches
            ),
            conceded_only_second_half_matches=(
                conceded_only_second_half_matches
            ),
            conceded_only_first_half_percentage=(
                self._percentage(
                    conceded_only_first_half_matches,
                    matches,
                )
            ),
            conceded_only_second_half_percentage=(
                self._percentage(
                    conceded_only_second_half_matches,
                    matches,
                )
            ),
            more_goals_first_half_matches=(
                more_goals_first_half_matches
            ),
            equal_goals_each_half_matches=(
                equal_goals_each_half_matches
            ),
            more_goals_second_half_matches=(
                more_goals_second_half_matches
            ),
            more_goals_first_half_percentage=(
                self._percentage(
                    more_goals_first_half_matches,
                    matches,
                )
            ),
            equal_goals_each_half_percentage=(
                self._percentage(
                    equal_goals_each_half_matches,
                    matches,
                )
            ),
            more_goals_second_half_percentage=(
                self._percentage(
                    more_goals_second_half_matches,
                    matches,
                )
            ),
            conceded_more_first_half_matches=(
                conceded_more_first_half_matches
            ),
            conceded_equal_each_half_matches=(
                conceded_equal_each_half_matches
            ),
            conceded_more_second_half_matches=(
                conceded_more_second_half_matches
            ),
            conceded_more_first_half_percentage=(
                self._percentage(
                    conceded_more_first_half_matches,
                    matches,
                )
            ),
            conceded_equal_each_half_percentage=(
                self._percentage(
                    conceded_equal_each_half_matches,
                    matches,
                )
            ),
            conceded_more_second_half_percentage=(
                self._percentage(
                    conceded_more_second_half_matches,
                    matches,
                )
            ),
            first_half_goal_share_percentage=(
                self._percentage(
                    first_half_goals_for,
                    total_goals_for,
                )
            ),
            second_half_goal_share_percentage=(
                self._percentage(
                    second_half_goals_for,
                    total_goals_for,
                )
            ),
            first_half_conceded_share_percentage=(
                self._percentage(
                    first_half_goals_against,
                    total_goals_against,
                )
            ),
            second_half_conceded_share_percentage=(
                self._percentage(
                    second_half_goals_against,
                    total_goals_against,
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