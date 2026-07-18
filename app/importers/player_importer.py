import time
from datetime import date

from app.api.players import PlayerService
from app.core.logger import logger
from app.importers.base_importer import BaseImporter
from app.models.player import Player
from app.models.team import Team
from app.repositories.player_repository import PlayerRepository


class PlayerImporter(BaseImporter):
    """
    Импорт футболистов команд из API-Football.
    """

    SEASON = 2024
    TEAM_LIMIT = 1
    REQUEST_DELAY = 7
    MAX_PAGE = 3

    def import_data(self) -> None:
        api = PlayerService()
        repository = PlayerRepository(self.session)

        added_players = 0
        updated_players = 0
        skipped_players = 0

        teams = (
            self.session.query(Team)
            .filter(Team.league.has(api_id=39))
            .order_by(Team.id.asc())
            .limit(self.TEAM_LIMIT)
            .all()
        )

        if not teams:
            logger.warning(
                "Не найдены команды Premier League "
                "для импорта игроков."
            )
            return

        for team in teams:
            logger.info(
                f"Импорт игроков команды: "
                f"{team.name}, api_id={team.api_id}"
            )

            page = 1

            while True:
                data = api.get_players(
                    team=team.api_id,
                    season=self.SEASON,
                    page=page,
                )

                if not data:
                    logger.warning(
                        f"Нет ответа API для команды {team.name}"
                    )
                    break

                response = data.get("response", [])

                if not response:
                    logger.info(
                        f"На странице {page} нет игроков."
                    )
                    break

                for item in response:
                    player_data = item.get("player") or {}
                    statistics = item.get("statistics") or []

                    api_id = player_data.get("id")

                    if not api_id:
                        skipped_players += 1
                        continue

                    statistic = (
                        statistics[0]
                        if statistics
                        else {}
                    )

                    games = statistic.get("games") or {}
                    birth_data = player_data.get("birth") or {}

                    values = {
                        "team_id": team.id,
                        "name": player_data.get("name"),
                        "firstname": player_data.get("firstname"),
                        "lastname": player_data.get("lastname"),
                        "age": player_data.get("age"),
                        "birth_date": self._parse_date(
                            birth_data.get("date")
                        ),
                        "birth_place": birth_data.get("place"),
                        "birth_country": birth_data.get("country"),
                        "nationality": player_data.get(
                            "nationality"
                        ),
                        "height": player_data.get("height"),
                        "weight": player_data.get("weight"),
                        "position": games.get("position"),
                        "photo": player_data.get("photo"),
                        "is_injured": bool(
                            player_data.get("injured", False)
                        ),
                    }

                    existing_player = (
                        repository.get_by_api_id(api_id)
                    )

                    if existing_player:
                        self._update_player(
                            existing_player,
                            values,
                        )
                        updated_players += 1
                    else:
                        repository.add(
                            Player(
                                api_id=api_id,
                                **values,
                            )
                        )
                        added_players += 1

                paging = data.get("paging") or {}
                current_page = paging.get("current", page)
                total_pages = paging.get(
                    "total",
                    current_page,
                )

                logger.info(
                    f"Обработана страница "
                    f"{current_page}/{total_pages}"
                )

                if current_page >= total_pages:
                    break

                if current_page >= self.MAX_PAGE:
                    logger.warning(
                        f"Импорт остановлен на странице "
                        f"{self.MAX_PAGE}: бесплатный тариф API "
                        "не разрешает запрашивать страницу 4 и выше."
                    )
                    break

                page += 1
                time.sleep(self.REQUEST_DELAY)

        logger.success(
            f"Добавлено игроков: {added_players}"
        )
        logger.info(
            f"Обновлено игроков: {updated_players}"
        )
        logger.info(
            f"Пропущено записей: {skipped_players}"
        )

    @staticmethod
    def _parse_date(value: str | None) -> date | None:
        if not value:
            return None

        try:
            return date.fromisoformat(value)
        except ValueError:
            logger.warning(
                f"Некорректная дата рождения: {value}"
            )
            return None

    @staticmethod
    def _update_player(
        player: Player,
        values: dict,
    ) -> None:
        for field, value in values.items():
            setattr(player, field, value)