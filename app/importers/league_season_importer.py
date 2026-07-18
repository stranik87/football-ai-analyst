from datetime import date
from typing import Any

from app.api.league_seasons import LeagueSeasonService
from app.core.logger import logger
from app.importers.base_importer import BaseImporter
from app.models.league import League
from app.models.league_season import LeagueSeason
from app.repositories.league_season_repository import (
    LeagueSeasonRepository,
)


class LeagueSeasonImporter(BaseImporter):
    """
    Импорт сезонов футбольных лиг.

    Транзакцией управляет BaseImporter.
    """

    def import_data(self) -> None:
        api = LeagueSeasonService()
        db = self.session

        repository = LeagueSeasonRepository(db)

        data = api.get_league_seasons()

        if (
            not data
            or not data.get("response")
        ):
            logger.warning(
                "API не вернул сезоны футбольных лиг."
            )
            return

        added = 0
        skipped = 0
        updated = 0
        missing_leagues = 0
        invalid_records = 0

        leagues_by_api_id = {
            league.api_id: league
            for league in db.query(League).all()
        }

        logger.info(
            "Лиг загружено в кэш: "
            f"{len(leagues_by_api_id)}"
        )

        for item in data["response"]:
            league_data = item.get("league") or {}
            seasons_data = item.get("seasons") or []

            league_api_id = league_data.get("id")

            if league_api_id is None:
                invalid_records += 1

                logger.warning(
                    "Пропущена запись сезона без API ID лиги."
                )
                continue

            league = leagues_by_api_id.get(
                league_api_id
            )

            if league is None:
                missing_leagues += 1
                continue

            for season_data in seasons_data:
                season_year = season_data.get("year")

                if season_year is None:
                    invalid_records += 1

                    logger.warning(
                        "Пропущен сезон без года: "
                        f"league_api_id={league_api_id}"
                    )
                    continue

                start_date = self._parse_date(
                    season_data.get("start")
                )
                end_date = self._parse_date(
                    season_data.get("end")
                )

                if (
                    start_date is None
                    or end_date is None
                ):
                    invalid_records += 1

                    logger.warning(
                        "Пропущен сезон с некорректными "
                        "датами: "
                        f"league_api_id={league_api_id}, "
                        f"season={season_year}, "
                        f"start={season_data.get('start')}, "
                        f"end={season_data.get('end')}"
                    )
                    continue

                values = {
                    "start": start_date,
                    "end": end_date,
                    "current": bool(
                        season_data.get(
                            "current",
                            False,
                        )
                    ),
                }

                existing = repository.get(
                    league_id=league.id,
                    season=season_year,
                )

                if existing:
                    changed = self._update_if_changed(
                        existing,
                        values,
                    )

                    if changed:
                        updated += 1
                    else:
                        skipped += 1

                    continue

                repository.add(
                    LeagueSeason(
                        league_id=league.id,
                        season=season_year,
                        **values,
                    )
                )

                added += 1

        logger.success(
            f"Добавлено сезонов: {added}"
        )
        logger.info(
            f"Обновлено сезонов: {updated}"
        )
        logger.info(
            f"Пропущено без изменений: {skipped}"
        )
        logger.warning(
            "Не найдены локальные лиги: "
            f"{missing_leagues}"
        )
        logger.warning(
            f"Некорректных записей: {invalid_records}"
        )

    @staticmethod
    def _parse_date(
        value: str | None,
    ) -> date | None:
        """
        Преобразовать дату формата YYYY-MM-DD.
        """

        if not value:
            return None

        try:
            return date.fromisoformat(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _update_if_changed(
        model: Any,
        values: dict[str, Any],
    ) -> bool:
        """
        Обновить только изменившиеся поля.

        Возвращает True, если хотя бы одно поле изменилось.
        """

        changed = False

        for field, value in values.items():
            if getattr(model, field) != value:
                setattr(model, field, value)
                changed = True

        return changed