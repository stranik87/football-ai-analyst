from dataclasses import asdict, dataclass

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.fixture import Fixture
from app.models.fixture_team_statistics import (
    FixtureTeamStatistics,
)
from app.models.team import Team


@dataclass
class TeamDisciplineResult:
    team_id: int
    team_name: str
    venue: str
    requested_limit: int
    matches: int

    fouls: int
    yellow_cards: int
    red_cards: int
    offsides: int

    average_fouls: float
    average_yellow_cards: float
    average_red_cards: float
    average_offsides: float

    matches_with_yellow_cards: int
    matches_with_red_cards: int
    matches_without_cards: int

    yellow_card_match_percentage: float
    red_card_match_percentage: float
    clean_discipline_percentage: float

    def to_dict(self) -> dict:
        return asdict(self)


class TeamDisciplineAnalyzer:
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
    ) -> TeamDisciplineResult:
        if limit <= 0:
            raise ValueError(
                "limit должен быть больше нуля"
            )

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

        fixture_query = (
            self.session.query(Fixture)
            .join(
                FixtureTeamStatistics,
                FixtureTeamStatistics.fixture_id
                == Fixture.id,
            )
            .filter(
                FixtureTeamStatistics.team_id == team_id,
                Fixture.status_short.in_(
                    self.FINISHED_STATUSES
                ),
                Fixture.home_goals.isnot(None),
                Fixture.away_goals.isnot(None),
            )
        )

        if venue == "home":
            fixture_query = fixture_query.filter(
                Fixture.home_team_id == team_id
            )

        elif venue == "away":
            fixture_query = fixture_query.filter(
                Fixture.away_team_id == team_id
            )

        else:
            fixture_query = fixture_query.filter(
                or_(
                    Fixture.home_team_id == team_id,
                    Fixture.away_team_id == team_id,
                )
            )

        fixtures = (
            fixture_query.order_by(
                Fixture.kickoff.desc(),
                Fixture.id.desc(),
            )
            .limit(limit)
            .all()
        )

        fouls = 0
        yellow_cards = 0
        red_cards = 0
        offsides = 0

        matches_with_yellow_cards = 0
        matches_with_red_cards = 0
        matches_without_cards = 0
        processed_matches = 0

        for fixture in fixtures:
            statistic = (
                self.session.query(
                    FixtureTeamStatistics
                )
                .filter(
                    FixtureTeamStatistics.fixture_id
                    == fixture.id,
                    FixtureTeamStatistics.team_id
                    == team_id,
                )
                .first()
            )

            if not statistic:
                continue

            match_fouls = statistic.fouls or 0
            match_yellow_cards = (
                statistic.yellow_cards or 0
            )
            match_red_cards = (
                statistic.red_cards or 0
            )
            match_offsides = statistic.offsides or 0

            fouls += match_fouls
            yellow_cards += match_yellow_cards
            red_cards += match_red_cards
            offsides += match_offsides

            if match_yellow_cards > 0:
                matches_with_yellow_cards += 1

            if match_red_cards > 0:
                matches_with_red_cards += 1

            if (
                match_yellow_cards == 0
                and match_red_cards == 0
            ):
                matches_without_cards += 1

            processed_matches += 1

        return TeamDisciplineResult(
            team_id=team.id,
            team_name=team.name,
            venue=venue,
            requested_limit=limit,
            matches=processed_matches,
            fouls=fouls,
            yellow_cards=yellow_cards,
            red_cards=red_cards,
            offsides=offsides,
            average_fouls=self._average(
                fouls,
                processed_matches,
            ),
            average_yellow_cards=self._average(
                yellow_cards,
                processed_matches,
            ),
            average_red_cards=self._average(
                red_cards,
                processed_matches,
            ),
            average_offsides=self._average(
                offsides,
                processed_matches,
            ),
            matches_with_yellow_cards=(
                matches_with_yellow_cards
            ),
            matches_with_red_cards=(
                matches_with_red_cards
            ),
            matches_without_cards=matches_without_cards,
            yellow_card_match_percentage=(
                self._percentage(
                    matches_with_yellow_cards,
                    processed_matches,
                )
            ),
            red_card_match_percentage=(
                self._percentage(
                    matches_with_red_cards,
                    processed_matches,
                )
            ),
            clean_discipline_percentage=(
                self._percentage(
                    matches_without_cards,
                    processed_matches,
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