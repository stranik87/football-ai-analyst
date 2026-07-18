import time
from datetime import datetime
from typing import Any

from app.api.fixtures import FixtureService
from app.core.logger import logger
from app.importers.base_importer import BaseImporter
from app.models.fixture import Fixture
from app.models.league import League
from app.models.league_season import LeagueSeason
from app.models.team import Team
from app.models.venue import Venue
from app.repositories.fixture_repository import FixtureRepository


class FixtureImporter(BaseImporter):
    """
    Импорт футбольных матчей для поддерживаемых лиг.

    Транзакцией управляет BaseImporter.
    """

    TARGET_LEAGUE_API_IDS = [
        39,
        140,
        135,
        78,
        61,
    ]

    SEASON = 2024
    REQUEST_DELAY = 2

    def import_data(self) -> None:
        api = FixtureService()
        db = self.session

        repository = FixtureRepository(db)

        added = 0
        skipped = 0
        missing_teams = 0
        invalid_records = 0
        empty_responses = 0

        league_seasons = (
            db.query(LeagueSeason)
            .join(
                League,
                League.id == LeagueSeason.league_id,
            )
            .filter(
                LeagueSeason.season == self.SEASON,
                League.api_id.in_(
                    self.TARGET_LEAGUE_API_IDS
                ),
            )
            .order_by(
                League.api_id.asc(),
                LeagueSeason.season.asc(),
            )
            .all()
        )

        logger.info(
            "Найдено сезонов для импорта матчей: "
            f"{len(league_seasons)}"
        )

        for season_number, league_season in enumerate(
            league_seasons,
            start=1,
        ):
            league = league_season.league

            if league is None:
                invalid_records += 1

                logger.warning(
                    "У сезона отсутствует связанная лига: "
                    f"league_season_id={league_season.id}"
                )
                continue

            logger.info(
                f"[{season_number}/{len(league_seasons)}] "
                f"Импорт матчей: {league.name}, "
                f"league={league.api_id}, "
                f"season={league_season.season}"
            )

            data = api.get_fixtures(
                league=league.api_id,
                season=league_season.season,
            )

            if (
                not data
                or not data.get("response")
            ):
                empty_responses += 1

                logger.warning(
                    "Не удалось получить матчи: "
                    f"league={league.name}, "
                    f"season={league_season.season}"
                )

                self._sleep_before_next_request(
                    current_number=season_number,
                    total=len(league_seasons),
                )
                continue

            for item in data["response"]:
                fixture_data = item.get("fixture") or {}
                teams_data = item.get("teams") or {}
                goals_data = item.get("goals") or {}
                score_data = item.get("score") or {}
                league_data = item.get("league") or {}

                fixture_api_id = fixture_data.get("id")

                if fixture_api_id is None:
                    invalid_records += 1

                    logger.warning(
                        "Пропущен матч без API ID: "
                        f"league={league.name}"
                    )
                    continue

                existing = repository.get_by_api_id(
                    fixture_api_id
                )

                if existing:
                    skipped += 1
                    continue

                home_data = teams_data.get("home") or {}
                away_data = teams_data.get("away") or {}

                home_team_api_id = home_data.get("id")
                away_team_api_id = away_data.get("id")

                if (
                    home_team_api_id is None
                    or away_team_api_id is None
                ):
                    invalid_records += 1

                    logger.warning(
                        "Пропущен матч без ID команд: "
                        f"fixture_api_id={fixture_api_id}"
                    )
                    continue

                home_team = (
                    db.query(Team)
                    .filter(
                        Team.api_id == home_team_api_id
                    )
                    .first()
                )

                away_team = (
                    db.query(Team)
                    .filter(
                        Team.api_id == away_team_api_id
                    )
                    .first()
                )

                if not home_team or not away_team:
                    missing_teams += 1

                    logger.warning(
                        "Команды матча не найдены в базе: "
                        f"fixture_api_id={fixture_api_id}, "
                        f"home_team_api_id="
                        f"{home_team_api_id}, "
                        f"away_team_api_id="
                        f"{away_team_api_id}"
                    )
                    continue

                venue = self._get_venue(
                    fixture_data=fixture_data,
                )

                kickoff = self._parse_datetime(
                    fixture_data.get("date")
                )

                if kickoff is None:
                    invalid_records += 1

                    logger.warning(
                        "Пропущен матч с некорректной датой: "
                        f"fixture_api_id={fixture_api_id}, "
                        f"date={fixture_data.get('date')}"
                    )
                    continue

                status_data = (
                    fixture_data.get("status") or {}
                )

                halftime_data = (
                    score_data.get("halftime") or {}
                )
                fulltime_data = (
                    score_data.get("fulltime") or {}
                )
                extratime_data = (
                    score_data.get("extratime") or {}
                )
                penalty_data = (
                    score_data.get("penalty") or {}
                )

                repository.add(
                    Fixture(
                        api_id=fixture_api_id,
                        league_season_id=(
                            league_season.id
                        ),
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        venue_id=(
                            venue.id
                            if venue is not None
                            else None
                        ),
                        kickoff=kickoff,
                        timestamp=self._to_int_or_none(
                            fixture_data.get("timestamp")
                        ),
                        timezone=(
                            fixture_data.get("timezone")
                            or "UTC"
                        ),
                        round=(
                            league_data.get("round")
                            or ""
                        ),
                        referee=fixture_data.get(
                            "referee"
                        ),
                        status_short=(
                            status_data.get("short")
                            or ""
                        ),
                        status_long=(
                            status_data.get("long")
                            or ""
                        ),
                        elapsed=self._to_int_or_none(
                            status_data.get("elapsed")
                        ),
                        extra_time=self._to_int_or_none(
                            status_data.get("extra")
                        ),
                        home_goals=self._to_int_or_none(
                            goals_data.get("home")
                        ),
                        away_goals=self._to_int_or_none(
                            goals_data.get("away")
                        ),
                        halftime_home=(
                            self._to_int_or_none(
                                halftime_data.get("home")
                            )
                        ),
                        halftime_away=(
                            self._to_int_or_none(
                                halftime_data.get("away")
                            )
                        ),
                        fulltime_home=(
                            self._to_int_or_none(
                                fulltime_data.get("home")
                            )
                        ),
                        fulltime_away=(
                            self._to_int_or_none(
                                fulltime_data.get("away")
                            )
                        ),
                        extratime_home=(
                            self._to_int_or_none(
                                extratime_data.get("home")
                            )
                        ),
                        extratime_away=(
                            self._to_int_or_none(
                                extratime_data.get("away")
                            )
                        ),
                        penalty_home=(
                            self._to_int_or_none(
                                penalty_data.get("home")
                            )
                        ),
                        penalty_away=(
                            self._to_int_or_none(
                                penalty_data.get("away")
                            )
                        ),
                    )
                )

                added += 1

            self._sleep_before_next_request(
                current_number=season_number,
                total=len(league_seasons),
            )

        logger.success(
            f"Добавлено матчей: {added}"
        )
        logger.info(
            f"Пропущено существующих матчей: {skipped}"
        )
        logger.warning(
            "Пропущено из-за отсутствующих команд: "
            f"{missing_teams}"
        )
        logger.warning(
            f"Некорректных записей: {invalid_records}"
        )
        logger.warning(
            f"Пустых ответов API: {empty_responses}"
        )

    def _get_venue(
        self,
        fixture_data: dict[str, Any],
    ) -> Venue | None:
        """
        Найти стадион матча в локальной базе.
        """

        venue_data = fixture_data.get("venue") or {}
        venue_api_id = venue_data.get("id")

        if venue_api_id is None:
            return None

        return (
            self.session.query(Venue)
            .filter(Venue.api_id == venue_api_id)
            .first()
        )

    def _sleep_before_next_request(
        self,
        current_number: int,
        total: int,
    ) -> None:
        """
        Подождать перед запросом следующего сезона.
        """

        if current_number >= total:
            return

        logger.info(
            f"Ожидание {self.REQUEST_DELAY} секунд "
            "перед следующей лигой..."
        )

        time.sleep(self.REQUEST_DELAY)

    @staticmethod
    def _parse_datetime(
        value: str | None,
    ) -> datetime | None:
        """
        Безопасно преобразовать ISO-дату API.
        """

        if not value:
            return None

        try:
            return datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_int_or_none(
        value: Any,
    ) -> int | None:
        """
        Преобразовать значение в int.

        Для отсутствующих значений возвращается None.
        """

        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None