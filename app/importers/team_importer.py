import time

from app.api.teams import TeamService
from app.core.logger import logger
from app.database.session import get_db
from app.importers.base_importer import BaseImporter
from app.models.league import League
from app.models.team import Team
from app.models.venue import Venue
from app.repositories.team_repository import TeamRepository
from app.repositories.venue_repository import VenueRepository


class TeamImporter(BaseImporter):
    """
    Безопасный импорт команд и стадионов только для топ-лиг.
    """

    TARGET_LEAGUE_API_IDS = [39, 140, 135, 78, 61]
    SEASON = 2024
    REQUEST_DELAY = 2

    def import_data(self):
        api = TeamService()
        db = next(get_db())

        team_repository = TeamRepository(db)
        venue_repository = VenueRepository(db)

        added_teams = 0
        skipped_teams = 0
        added_venues = 0

        leagues = (
            db.query(League)
            .filter(League.api_id.in_(self.TARGET_LEAGUE_API_IDS))
            .all()
        )

        try:
            for league in leagues:
                logger.info(
                    f"Импорт команд: {league.name} "
                    f"league={league.api_id}, season={self.SEASON}"
                )

                data = api.get_teams(
                    league=league.api_id,
                    season=self.SEASON,
                )

                time.sleep(self.REQUEST_DELAY)

                if not data:
                    logger.warning(f"Нет данных по лиге {league.name}")
                    continue

                for item in data["response"]:
                    team_data = item["team"]
                    venue_data = item["venue"]

                    venue = venue_repository.get_by_api_id(venue_data["id"])

                    if not venue:
                        venue = Venue(
                            api_id=venue_data["id"],
                            name=venue_data["name"] or "",
                            address=venue_data["address"] or "",
                            city=venue_data["city"] or "",
                            capacity=venue_data["capacity"] or 0,
                            surface=venue_data["surface"] or "",
                            image=venue_data["image"] or "",
                        )

                        venue_repository.add(venue)
                        db.flush()
                        added_venues += 1

                    existing_team = team_repository.get_by_api_id(
                        team_data["id"]
                    )

                    if existing_team:
                        skipped_teams += 1
                        continue

                    team = Team(
                        api_id=team_data["id"],
                        league_id=league.id,
                        venue_id=venue.id,
                        name=team_data["name"] or "",
                        code=team_data["code"] or "",
                        country=team_data["country"] or "",
                        founded=team_data["founded"] or 0,
                        logo=team_data["logo"] or "",
                    )

                    team_repository.add(team)
                    added_teams += 1

            team_repository.commit()

            logger.success(f"Добавлено команд: {added_teams}")
            logger.info(f"Пропущено команд: {skipped_teams}")
            logger.success(f"Добавлено стадионов: {added_venues}")

        except Exception:
            team_repository.rollback()
            logger.exception("Ошибка импорта команд")