from app.api.leagues import LeagueService as APLeagueService
from app.core.logger import logger
from app.database.session import get_db
from app.models.league import League
from app.repositories.league_repository import LeagueRepository


class LeagueImportService:

    def import_leagues(self):

        api = APLeagueService()

        data = api.get_current_leagues()

        if not data:
            logger.error("Не удалось получить список лиг.")
            return

        db = next(get_db())

        repository = LeagueRepository(db)

        added = 0
        skipped = 0

        try:

            for item in data["response"]:

                league = item["league"]
                country = item["country"]

                if repository.get_by_api_id(league["id"]):
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

        except Exception as e:

            repository.rollback()

            logger.exception(e)