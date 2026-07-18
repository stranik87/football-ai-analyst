import time
from typing import Any

from app.api.teams import TeamService
from app.core.logger import logger
from app.importers.base_importer import BaseImporter
from app.models.league import League
from app.models.team import Team
from app.models.venue import Venue
from app.repositories.team_repository import TeamRepository
from app.repositories.venue_repository import VenueRepository


class TeamImporter(BaseImporter):
    """
    Импорт команд и стадионов для поддерживаемых лиг.
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
        api = TeamService()
        db = self.session

        team_repository = TeamRepository(db)
        venue_repository = VenueRepository(db)

        added_teams = 0
        updated_teams = 0
        added_venues = 0
        updated_venues = 0
        invalid_records = 0
        empty_responses = 0

        leagues = (
            db.query(League)
            .filter(
                League.api_id.in_(
                    self.TARGET_LEAGUE_API_IDS
                )
            )
            .order_by(League.api_id.asc())
            .all()
        )

        logger.info(
            "Найдено лиг для импорта команд: "
            f"{len(leagues)}"
        )

        for league_number, league in enumerate(
            leagues,
            start=1,
        ):
            logger.info(
                f"[{league_number}/{len(leagues)}] "
                f"Импорт команд: {league.name}, "
                f"league={league.api_id}, "
                f"season={self.SEASON}"
            )

            data = api.get_teams(
                league=league.api_id,
                season=self.SEASON,
            )

            if (
                not data
                or not data.get("response")
            ):
                empty_responses += 1

                logger.warning(
                    "Нет данных по командам: "
                    f"league={league.name}, "
                    f"season={self.SEASON}"
                )

                self._sleep_before_next_request(
                    current_number=league_number,
                    total=len(leagues),
                )
                continue

            for item in data["response"]:
                team_data = item.get("team") or {}
                venue_data = item.get("venue") or {}

                team_api_id = team_data.get("id")

                if team_api_id is None:
                    invalid_records += 1

                    logger.warning(
                        "Пропущена команда без API ID: "
                        f"league={league.name}"
                    )
                    continue

                venue, venue_created = (
                    self._get_or_create_venue(
                        venue_data=venue_data,
                        repository=venue_repository,
                    )
                )

                if venue_created:
                    # Нужен ID нового стадиона,
                    # чтобы привязать его к команде.
                    db.flush()
                    added_venues += 1
                elif venue is not None:
                    updated_venues += 1

                team_values = {
                    "league_id": league.id,
                    "venue_id": (
                        venue.id
                        if venue is not None
                        else None
                    ),
                    "name": (
                        team_data.get("name")
                        or f"Team {team_api_id}"
                    ),
                    "code": (
                        team_data.get("code")
                        or ""
                    ),
                    "country": (
                        team_data.get("country")
                        or ""
                    ),
                    "founded": self._to_int(
                        team_data.get("founded")
                    ),
                    "logo": (
                        team_data.get("logo")
                        or ""
                    ),
                }

                existing_team = (
                    team_repository.get_by_api_id(
                        team_api_id
                    )
                )

                if existing_team:
                    self._update_model(
                        existing_team,
                        team_values,
                    )

                    updated_teams += 1
                    continue

                team_repository.add(
                    Team(
                        api_id=team_api_id,
                        **team_values,
                    )
                )

                added_teams += 1

            self._sleep_before_next_request(
                current_number=league_number,
                total=len(leagues),
            )

        logger.success(
            f"Добавлено команд: {added_teams}"
        )
        logger.info(
            f"Обновлено команд: {updated_teams}"
        )
        logger.success(
            f"Добавлено стадионов: {added_venues}"
        )
        logger.info(
            f"Обновлено существующих стадионов: "
            f"{updated_venues}"
        )
        logger.warning(
            f"Некорректных записей: "
            f"{invalid_records}"
        )
        logger.warning(
            f"Пустых ответов API: "
            f"{empty_responses}"
        )

    def _get_or_create_venue(
        self,
        venue_data: dict[str, Any],
        repository: VenueRepository,
    ) -> tuple[Venue | None, bool]:
        """
        Найти существующий стадион или создать новый.

        Возвращает:
            (стадион, создан_ли_новый_стадион)
        """

        venue_api_id = venue_data.get("id")

        if venue_api_id is None:
            return None, False

        values = {
            "name": (
                venue_data.get("name")
                or ""
            ),
            "address": (
                venue_data.get("address")
                or ""
            ),
            "city": (
                venue_data.get("city")
                or ""
            ),
            "capacity": self._to_int(
                venue_data.get("capacity")
            ),
            "surface": (
                venue_data.get("surface")
                or ""
            ),
            "image": (
                venue_data.get("image")
                or ""
            ),
        }

        venue = repository.get_by_api_id(
            venue_api_id
        )

        if venue:
            self._update_model(
                venue,
                values,
            )

            return venue, False

        venue = Venue(
            api_id=venue_api_id,
            **values,
        )

        repository.add(venue)

        return venue, True

    def _sleep_before_next_request(
        self,
        current_number: int,
        total: int,
    ) -> None:
        """
        Подождать перед запросом следующей лиги.
        """

        if current_number >= total:
            return

        logger.info(
            f"Ожидание {self.REQUEST_DELAY} секунд "
            "перед следующей лигой..."
        )

        time.sleep(self.REQUEST_DELAY)

    @staticmethod
    def _update_model(
        model: Any,
        values: dict[str, Any],
    ) -> None:
        """
        Обновить поля SQLAlchemy-модели.
        """

        for field, value in values.items():
            setattr(model, field, value)

    @staticmethod
    def _to_int(value: Any) -> int:
        """
        Безопасно преобразовать значение в целое число.
        """

        if value is None:
            return 0

        try:
            return int(value)
        except (TypeError, ValueError):
            return 0