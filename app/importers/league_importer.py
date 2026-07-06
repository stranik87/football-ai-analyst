from app.api.leagues import LeagueService
from app.core.logger import logger
from app.database.session import get_db
from app.importers.base_importer import BaseImporter
from app.models.league import League
from app.repositories.league_repository import LeagueRepository


class LeagueImporter(BaseImporter):

    def import_data(self):

        api = LeagueService()

        data = api.get_current_leagues()

        if not data:
            logger.error("Не удалось получить список лиг.")
            return

        db = next(get_db())

        repository = LeagueRepository(db)

        added = 0
        skipped = 0

        # Загружаем все существующие api_id одним запросом
        existing_ids = {
            league.api_id
            for league in repository.get_all()
        }

        try:

            for item in data["response"]:

                league = item["league"]
                country = item["country"]

                if league["id"] in existing_ids:
                    skipped += 1
                    continue

                repository.add(
                    League(
                        api_id=league["id"],
                        name=league["name"],
                        type=league["type"],
                        logo=league["logo"],
                        country=country["name"],
                        country_code=country["code"] or "",
                        flag=country["flag"] or "",
                    )
                )

                added += 1

            repository.commit()

            logger.success(f"Добавлено: {added}")
            logger.info(f"Пропущено: {skipped}")

        except Exception:
            repository.rollback()
            logger.exception("Ошибка импорта лиг")