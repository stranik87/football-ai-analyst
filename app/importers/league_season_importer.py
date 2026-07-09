from datetime import datetime

from app.api.league_seasons import LeagueSeasonService
from app.core.logger import logger
from app.database.session import get_db
from app.importers.base_importer import BaseImporter
from app.models.league import League
from app.models.league_season import LeagueSeason
from app.repositories.league_season_repository import LeagueSeasonRepository


class LeagueSeasonImporter(BaseImporter):

    def import_data(self):

        api = LeagueSeasonService()
        data = api.get_league_seasons()

        if not data:
            logger.error("Не удалось получить сезоны лиг.")
            return

        db = next(get_db())

        repository = LeagueSeasonRepository(db)

        added = 0
        skipped = 0

        try:

            for item in data["response"]:

                league_api_id = item["league"]["id"]

                league = (
                    db.query(League)
                    .filter_by(api_id=league_api_id)
                    .first()
                )

                if not league:
                    continue

                for season in item["seasons"]:

                    existing = repository.get(
                        league_id=league.id,
                        season=season["year"],
                    )

                    if existing:
                        skipped += 1
                        continue

                    repository.add(
                        LeagueSeason(
                            league_id=league.id,
                            season=season["year"],
                            start=datetime.strptime(
                                season["start"], "%Y-%m-%d"
                            ).date(),
                            end=datetime.strptime(
                                season["end"], "%Y-%m-%d"
                            ).date(),
                            current=season["current"],
                        )
                    )

                    added += 1

            repository.commit()

            logger.success(f"Добавлено сезонов: {added}")
            logger.info(f"Пропущено: {skipped}")

        except Exception:
            repository.rollback()
            logger.exception("Ошибка импорта сезонов")