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
    """

    REQUEST_DELAY = 2
    SUPPORTED_LEAGUES = [39, 61, 78, 135, 140]
    SUPPORTED_SEASONS = [2024]

    def import_data(self) -> None:
        api = StandingsService()
        db = self.session
        repository = StandingRepository(db)

        league_seasons = (
            db.query(LeagueSeason)
            .join(
                League,
                League.id == LeagueSeason.league_id,
    )
            .filter(
                League.api_id.in_(self.SUPPORTED_LEAGUES),
                LeagueSeason.season.in_(self.SUPPORTED_SEASONS),
    )
            .order_by(
                League.api_id.asc(),
                LeagueSeason.season.asc(),
    )
            .all()
)

        added = 0
        updated = 0
        missing_teams = 0
        empty_responses = 0

        logger.info(
            f"Найдено сезонов для импорта таблиц: "
            f"{len(league_seasons)}"
        )

        for league_season in league_seasons:
            league = (
                db.query(League)
                .filter(League.id == league_season.league_id)
                .first()
            )

            if not league:
                logger.warning(
                    "Лига не найдена: "
                    f"league_season_id={league_season.id}"
                )
                continue

            logger.info(
                "Импорт турнирной таблицы: "
                f"league={league.name}, "
                f"season={league_season.season}"
            )

            data = api.get_by_league_and_season(
                league_api_id=league.api_id,
                season=league_season.season,
            )

            if not data or not data.get("response"):
                empty_responses += 1
                logger.warning(
                    "Турнирная таблица не получена: "
                    f"league={league.name}, "
                    f"season={league_season.season}"
                )

                time.sleep(self.REQUEST_DELAY)
                continue

            standings = self._extract_standings(data)

            for item in standings:
                team_data = item.get("team") or {}
                team_api_id = team_data.get("id")

                if team_api_id is None:
                    missing_teams += 1
                    continue

                team = (
                    db.query(Team)
                    .filter(Team.api_id == team_api_id)
                    .first()
                )

                if not team:
                    missing_teams += 1
                    logger.warning(
                        "Команда из таблицы не найдена: "
                        f"team_api_id={team_api_id}, "
                        f"league={league.name}"
                    )
                    continue

                values = self._build_values(item)

                existing = repository.get(
                    league_season_id=league_season.id,
                    team_id=team.id,
                )

                if existing:
                    for field, value in values.items():
                        setattr(existing, field, value)

                    updated += 1
                    continue

                db.add(
                    Standing(
                        league_season_id=league_season.id,
                        team_id=team.id,
                        **values,
                    )
                )

                added += 1

            db.commit()
            time.sleep(self.REQUEST_DELAY)

        logger.success(
            f"Добавлено записей таблицы: {added}"
        )
        logger.info(
            f"Обновлено записей таблицы: {updated}"
        )
        logger.warning(
            f"Команды не найдены: {missing_teams}"
        )
        logger.warning(
            f"Пустые ответы API: {empty_responses}"
        )

    @staticmethod
    def _extract_standings(
        data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Извлечь все группы турнирной таблицы из ответа API.
        """
        result: list[dict[str, Any]] = []

        for response_item in data.get("response", []):
            league_data = response_item.get("league") or {}

            for group in league_data.get("standings", []):
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
            "rank": cls._to_int(item.get("rank")),
            "points": cls._to_int(item.get("points")),
            "goals_diff": cls._to_int(
                item.get("goalsDiff")
            ),
            "group_name": item.get("group"),
            "form": item.get("form"),
            "status": item.get("status"),
            "description": item.get("description"),

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