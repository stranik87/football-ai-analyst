import time
from typing import Any

from app.api.standings import StandingsService
from app.core.logger import logger
from app.importers.base_importer import BaseImporter
from app.models.league import League
from app.models.league_season import LeagueSeason
from app.models.standing import Standing
from app.models.team import Team
from app.repositories.standing_repository import StandingRepository


class StandingImporter(BaseImporter):
    """
    Импорт турнирных таблиц по лигам и сезонам.

    Транзакцией управляет BaseImporter.
    """

    REQUEST_DELAY = 2

    SUPPORTED_LEAGUES = (
        39,
        61,
        78,
        135,
        140,
    )

    SUPPORTED_SEASONS = (
        2024,
    )

    def import_data(self) -> None:
        api = StandingsService()
        db = self.session

        repository = StandingRepository(db)

        league_seasons = (
            db.query(LeagueSeason, League)
            .join(
                League,
                League.id == LeagueSeason.league_id,
            )
            .filter(
                League.api_id.in_(
                    self.SUPPORTED_LEAGUES
                ),
                LeagueSeason.season.in_(
                    self.SUPPORTED_SEASONS
                ),
            )
            .order_by(
                League.api_id.asc(),
                LeagueSeason.season.asc(),
            )
            .all()
        )

        teams_by_api_id = {
            team.api_id: team
            for team in db.query(Team).all()
        }

        added = 0
        updated = 0
        skipped = 0
        missing_teams = 0
        invalid_records = 0
        empty_responses = 0

        logger.info(
            "Найдено сезонов для импорта таблиц: "
            f"{len(league_seasons)}"
        )

        logger.info(
            "Команд загружено в кэш: "
            f"{len(teams_by_api_id)}"
        )

        for index, (
            league_season,
            league,
        ) in enumerate(
            league_seasons,
            start=1,
        ):
            logger.info(
                f"[{index}/{len(league_seasons)}] "
                "Импорт турнирной таблицы: "
                f"league={league.name}, "
                f"league_api_id={league.api_id}, "
                f"season={league_season.season}"
            )

            data = api.get_by_league_and_season(
                league_api_id=league.api_id,
                season=league_season.season,
            )

            if (
                not data
                or not data.get("response")
            ):
                empty_responses += 1

                logger.warning(
                    "Турнирная таблица не получена: "
                    f"league={league.name}, "
                    f"season={league_season.season}"
                )

                self._sleep_before_next_request(
                    index=index,
                    total=len(league_seasons),
                )
                continue

            standings = self._extract_standings(
                data
            )

            if not standings:
                empty_responses += 1

                logger.warning(
                    "В ответе API отсутствуют строки "
                    "турнирной таблицы: "
                    f"league={league.name}, "
                    f"season={league_season.season}"
                )

                self._sleep_before_next_request(
                    index=index,
                    total=len(league_seasons),
                )
                continue

            league_added = 0
            league_updated = 0
            league_skipped = 0

            for item in standings:
                team_data = item.get("team") or {}
                team_api_id = team_data.get("id")

                if team_api_id is None:
                    invalid_records += 1

                    logger.warning(
                        "Пропущена строка таблицы "
                        "без ID команды: "
                        f"league={league.name}, "
                        f"season={league_season.season}"
                    )
                    continue

                team = teams_by_api_id.get(
                    team_api_id
                )

                if team is None:
                    missing_teams += 1

                    logger.warning(
                        "Команда из таблицы "
                        "не найдена в базе: "
                        f"team_api_id={team_api_id}, "
                        f"league={league.name}, "
                        f"season={league_season.season}"
                    )
                    continue

                values = self._build_values(item)

                existing = repository.get(
                    league_season_id=league_season.id,
                    team_id=team.id,
                )

                if existing:
                    changed = self._update_if_changed(
                        existing,
                        values,
                    )

                    if changed:
                        updated += 1
                        league_updated += 1
                    else:
                        skipped += 1
                        league_skipped += 1

                    continue

                repository.add(
                    Standing(
                        league_season_id=league_season.id,
                        team_id=team.id,
                        **values,
                    )
                )

                added += 1
                league_added += 1

            logger.success(
                "Турнирная таблица обработана: "
                f"league={league.name}, "
                f"season={league_season.season}, "
                f"добавлено={league_added}, "
                f"обновлено={league_updated}, "
                f"без изменений={league_skipped}"
            )

            self._sleep_before_next_request(
                index=index,
                total=len(league_seasons),
            )

        logger.success(
            "Импорт турнирных таблиц завершён."
        )
        logger.success(
            f"Добавлено записей таблицы: {added}"
        )
        logger.info(
            f"Обновлено записей таблицы: {updated}"
        )
        logger.info(
            f"Пропущено без изменений: {skipped}"
        )
        logger.warning(
            f"Команды не найдены: {missing_teams}"
        )
        logger.warning(
            f"Некорректных записей: {invalid_records}"
        )
        logger.warning(
            f"Пустых ответов API: {empty_responses}"
        )

    def _sleep_before_next_request(
        self,
        index: int,
        total: int,
    ) -> None:
        """
        Пауза между запросами к API.
        """

        if index >= total:
            return

        logger.info(
            f"Ожидание {self.REQUEST_DELAY} секунд "
            "перед следующей лигой..."
        )

        time.sleep(self.REQUEST_DELAY)

    @staticmethod
    def _extract_standings(
        data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Извлечь строки из всех групп турнирной таблицы.
        """

        result: list[dict[str, Any]] = []

        for response_item in data.get(
            "response",
            [],
        ):
            league_data = (
                response_item.get("league") or {}
            )

            groups = (
                league_data.get("standings") or []
            )

            for group in groups:
                if isinstance(group, list):
                    result.extend(group)

        return result

    @classmethod
    def _build_values(
        cls,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        all_stats = item.get("all") or {}
        home_stats = item.get("home") or {}
        away_stats = item.get("away") or {}

        return {
            "rank": cls._to_int(
                item.get("rank")
            ),
            "points": cls._to_int(
                item.get("points")
            ),
            "goals_diff": cls._to_int(
                item.get("goalsDiff")
            ),
            "group_name": item.get("group"),
            "form": item.get("form"),
            "status": item.get("status"),
            "description": item.get(
                "description"
            ),

            "played": cls._to_int(
                all_stats.get("played")
            ),
            "wins": cls._to_int(
                all_stats.get("win")
            ),
            "draws": cls._to_int(
                all_stats.get("draw")
            ),
            "losses": cls._to_int(
                all_stats.get("lose")
            ),
            "goals_for": cls._to_int(
                cls._nested_get(
                    all_stats,
                    "goals",
                    "for",
                )
            ),
            "goals_against": cls._to_int(
                cls._nested_get(
                    all_stats,
                    "goals",
                    "against",
                )
            ),

            "home_played": cls._to_int(
                home_stats.get("played")
            ),
            "home_wins": cls._to_int(
                home_stats.get("win")
            ),
            "home_draws": cls._to_int(
                home_stats.get("draw")
            ),
            "home_losses": cls._to_int(
                home_stats.get("lose")
            ),
            "home_goals_for": cls._to_int(
                cls._nested_get(
                    home_stats,
                    "goals",
                    "for",
                )
            ),
            "home_goals_against": cls._to_int(
                cls._nested_get(
                    home_stats,
                    "goals",
                    "against",
                )
            ),

            "away_played": cls._to_int(
                away_stats.get("played")
            ),
            "away_wins": cls._to_int(
                away_stats.get("win")
            ),
            "away_draws": cls._to_int(
                away_stats.get("draw")
            ),
            "away_losses": cls._to_int(
                away_stats.get("lose")
            ),
            "away_goals_for": cls._to_int(
                cls._nested_get(
                    away_stats,
                    "goals",
                    "for",
                )
            ),
            "away_goals_against": cls._to_int(
                cls._nested_get(
                    away_stats,
                    "goals",
                    "against",
                )
            ),
        }

    @staticmethod
    def _update_if_changed(
        model: Standing,
        values: dict[str, Any],
    ) -> bool:
        """
        Обновить только изменившиеся поля.
        """

        changed = False

        for field, value in values.items():
            if getattr(model, field) != value:
                setattr(model, field, value)
                changed = True

        return changed

    @staticmethod
    def _nested_get(
        data: dict[str, Any],
        *keys: str,
    ) -> Any:
        value: Any = data

        for key in keys:
            if not isinstance(value, dict):
                return None

            value = value.get(key)

        return value

    @staticmethod
    def _to_int(value: Any) -> int:
        if value is None:
            return 0

        try:
            return int(value)
        except (TypeError, ValueError):
            return 0