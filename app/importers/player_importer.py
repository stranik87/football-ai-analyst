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

    DEFAULT_SEASON = 2024
    DEFAULT_LEAGUE_API_ID = 39
    DEFAULT_TEAM_LIMIT = 1

    REQUEST_DELAY = 7
    MAX_PAGE = 3

    # Если у команды уже есть минимум 40 игроков,
    # команда считается импортированной.
    MIN_EXISTING_PLAYERS = 40

    def __init__(
        self,
        team_api_id: int | None = None,
        league_api_id: int = DEFAULT_LEAGUE_API_ID,
        season: int = DEFAULT_SEASON,
        team_limit: int = DEFAULT_TEAM_LIMIT,
        skip_existing: bool = False,
    ) -> None:
        super().__init__()

        self.team_api_id = team_api_id
        self.league_api_id = league_api_id
        self.season = season
        self.team_limit = team_limit
        self.skip_existing = skip_existing

    def import_data(self) -> None:
        api = PlayerService()
        repository = PlayerRepository(self.session)

        total_added = 0
        total_updated = 0
        total_skipped = 0
        skipped_teams = 0

        teams = self._get_teams()

        if not teams:
            if self.team_api_id is not None:
                logger.warning(
                    f"Команда с API ID {self.team_api_id} "
                    "не найдена в локальной базе."
                )
            else:
                logger.warning(
                    f"Не найдены команды лиги "
                    f"api_id={self.league_api_id}."
                )

            return

        logger.info(
            f"Найдено команд для обработки: {len(teams)}"
        )
        logger.info(f"Сезон: {self.season}")
        logger.info(
            f"Пропуск уже загруженных команд: "
            f"{'да' if self.skip_existing else 'нет'}"
        )

        for team_number, team in enumerate(
            teams,
            start=1,
        ):
            existing_count = repository.count_by_team_id(
                team.id
            )

            if (
                self.skip_existing
                and existing_count
                >= self.MIN_EXISTING_PLAYERS
            ):
                skipped_teams += 1

                logger.info(
                    f"[{team_number}/{len(teams)}] "
                    f"{team.name} пропущена: "
                    f"в базе уже есть "
                    f"{existing_count} игроков."
                )

                continue

            logger.info(
                f"[{team_number}/{len(teams)}] "
                f"Импорт игроков команды: "
                f"{team.name}, api_id={team.api_id}"
            )

            team_added = 0
            team_updated = 0
            team_skipped = 0

            page = 1

            while True:
                data = api.get_players(
                    team=team.api_id,
                    season=self.season,
                    page=page,
                )

                if not data:
                    logger.warning(
                        f"Нет ответа API для команды "
                        f"{team.name}."
                    )
                    break

                response = data.get("response") or []

                if not response:
                    logger.info(
                        f"Для команды {team.name} "
                        f"на странице {page} "
                        "игроки не найдены."
                    )
                    break

                for item in response:
                    player_data = (
                        item.get("player") or {}
                    )

                    statistics = (
                        item.get("statistics") or []
                    )

                    api_id = player_data.get("id")

                    if not api_id:
                        team_skipped += 1
                        total_skipped += 1
                        continue

                    statistic = (
                        statistics[0]
                        if statistics
                        else {}
                    )

                    games = (
                        statistic.get("games") or {}
                    )

                    birth_data = (
                        player_data.get("birth") or {}
                    )

                    values = {
                        "team_id": team.id,
                        "name": (
                            player_data.get("name")
                            or f"Player {api_id}"
                        ),
                        "firstname": player_data.get(
                            "firstname"
                        ),
                        "lastname": player_data.get(
                            "lastname"
                        ),
                        "age": player_data.get("age"),
                        "birth_date": self._parse_date(
                            birth_data.get("date")
                        ),
                        "birth_place": birth_data.get(
                            "place"
                        ),
                        "birth_country": birth_data.get(
                            "country"
                        ),
                        "nationality": player_data.get(
                            "nationality"
                        ),
                        "height": player_data.get(
                            "height"
                        ),
                        "weight": player_data.get(
                            "weight"
                        ),
                        "position": games.get(
                            "position"
                        ),
                        "photo": player_data.get(
                            "photo"
                        ),
                        "is_injured": bool(
                            player_data.get(
                                "injured",
                                False,
                            )
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

                        team_updated += 1
                        total_updated += 1
                    else:
                        repository.add(
                            Player(
                                api_id=api_id,
                                **values,
                            )
                        )

                        team_added += 1
                        total_added += 1

                paging = data.get("paging") or {}

                current_page = (
                    paging.get("current") or page
                )

                total_pages = (
                    paging.get("total")
                    or current_page
                )

                logger.info(
                    f"{team.name}: обработана страница "
                    f"{current_page}/{total_pages}"
                )

                if current_page >= total_pages:
                    break

                if current_page >= self.MAX_PAGE:
                    logger.warning(
                        f"{team.name}: импорт остановлен "
                        f"на странице {self.MAX_PAGE}. "
                        "Бесплатный тариф API "
                        "не разрешает запрашивать "
                        "страницу 4 и выше."
                    )
                    break

                page += 1
                time.sleep(self.REQUEST_DELAY)

            logger.success(
                f"{team.name}: "
                f"добавлено={team_added}, "
                f"обновлено={team_updated}, "
                f"пропущено={team_skipped}"
            )

            if team_number < len(teams):
                logger.info(
                    f"Ожидание "
                    f"{self.REQUEST_DELAY} секунд "
                    "перед следующей командой..."
                )
                time.sleep(self.REQUEST_DELAY)

        logger.success(
            f"Всего добавлено игроков: "
            f"{total_added}"
        )
        logger.info(
            f"Всего обновлено игроков: "
            f"{total_updated}"
        )
        logger.info(
            f"Всего пропущено записей: "
            f"{total_skipped}"
        )
        logger.info(
            f"Пропущено уже загруженных команд: "
            f"{skipped_teams}"
        )

    def _get_teams(self) -> list[Team]:
        query = self.session.query(Team)

        if self.team_api_id is not None:
            return (
                query
                .filter(
                    Team.api_id == self.team_api_id
                )
                .limit(1)
                .all()
            )

        return (
            query
            .filter(
                Team.league.has(
                    api_id=self.league_api_id
                )
            )
            .order_by(Team.id.asc())
            .limit(self.team_limit)
            .all()
        )

    @staticmethod
    def _parse_date(
        value: str | None,
    ) -> date | None:
        if not value:
            return None

        try:
            return date.fromisoformat(value)
        except ValueError:
            logger.warning(
                f"Некорректная дата рождения: "
                f"{value}"
            )
            return None

    @staticmethod
    def _update_player(
        player: Player,
        values: dict,
    ) -> None:
        for field, value in values.items():
            setattr(player, field, value)