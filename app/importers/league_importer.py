from app.api.leagues import LeagueService
from app.core.logger import logger
from app.importers.base_importer import BaseImporter
from app.models.league import League
from app.repositories.league_repository import LeagueRepository


class LeagueImporter(BaseImporter):
    """
    Импорт футбольных лиг из API-Football.
    """

    def import_data(self) -> None:
        api = LeagueService()
        repository = LeagueRepository(self.session)

        data = api.get_current_leagues()

        if not data or not data.get("response"):
            logger.warning(
                "API не вернул список футбольных лиг."
            )
            return

        added = 0
        skipped = 0
        invalid = 0

        # Загружаем существующие ID одним запросом,
        # чтобы не обращаться к базе для каждой лиги.
        existing_ids = {
            league.api_id
            for league in repository.get_all()
        }

        for item in data["response"]:
            league_data = item.get("league") or {}
            country_data = item.get("country") or {}

            api_id = league_data.get("id")

            if api_id is None:
                invalid += 1
                logger.warning(
                    "Пропущена лига без API ID."
                )
                continue

            if api_id in existing_ids:
                skipped += 1
                continue

            repository.add(
                League(
                    api_id=api_id,
                    name=(
                        league_data.get("name")
                        or f"League {api_id}"
                    ),
                    type=league_data.get("type") or "",
                    logo=league_data.get("logo") or "",
                    country=(
                        country_data.get("name")
                        or ""
                    ),
                    country_code=(
                        country_data.get("code")
                        or ""
                    ),
                    flag=country_data.get("flag") or "",
                )
            )

            # Защищает от повторного api_id
            # внутри одного ответа API.
            existing_ids.add(api_id)
            added += 1

        logger.success(
            f"Добавлено лиг: {added}"
        )
        logger.info(
            f"Пропущено существующих лиг: {skipped}"
        )
        logger.warning(
            f"Пропущено некорректных записей: {invalid}"
        )