import time
from datetime import datetime

from app.api.fixtures import FixtureService
from app.core.logger import logger
from app.database.session import get_db
from app.importers.base_importer import BaseImporter
from app.models.fixture import Fixture
from app.models.league import League
from app.models.league_season import LeagueSeason
from app.models.team import Team
from app.models.venue import Venue
from app.repositories.fixture_repository import FixtureRepository


class FixtureImporter(BaseImporter):
    """
    Импорт футбольных матчей.
    """

    TARGET_LEAGUE_API_IDS = [39, 140, 135, 78, 61]
    SEASON = 2024
    REQUEST_DELAY = 2

    def import_data(self):
        api = FixtureService()
        db = next(get_db())

        repository = FixtureRepository(db)

        added = 0
        skipped = 0
        missing_teams = 0

        league_seasons = (
            db.query(LeagueSeason)
            .join(League)
            .filter(
                LeagueSeason.season == self.SEASON,
                League.api_id.in_(self.TARGET_LEAGUE_API_IDS),
            )
            .all()
        )

        logger.info(f"Найдено сезонов для импорта матчей: {len(league_seasons)}")

        try:
            for league_season in league_seasons:
                league = league_season.league

                logger.info(
                    f"Импорт матчей: {league.name}, " f"season={league_season.season}"
                )

                data = api.get_fixtures(
                    league=league.api_id,
                    season=league_season.season,
                )

                time.sleep(self.REQUEST_DELAY)

                if not data:
                    logger.warning(f"Не удалось получить матчи: {league.name}")
                    continue

                for item in data["response"]:
                    fixture_data = item["fixture"]
                    teams_data = item["teams"]
                    goals_data = item["goals"]
                    score_data = item["score"]
                    league_data = item["league"]

                    existing = repository.get_by_api_id(fixture_data["id"])

                    if existing:
                        skipped += 1
                        continue

                    home_team = (
                        db.query(Team)
                        .filter_by(api_id=teams_data["home"]["id"])
                        .first()
                    )

                    away_team = (
                        db.query(Team)
                        .filter_by(api_id=teams_data["away"]["id"])
                        .first()
                    )

                    if not home_team or not away_team:
                        missing_teams += 1
                        continue

                    venue_api_id = fixture_data["venue"].get("id")

                    venue = None

                    if venue_api_id:
                        venue = db.query(Venue).filter_by(api_id=venue_api_id).first()

                    status = fixture_data["status"]

                    repository.add(
                        Fixture(
                            api_id=fixture_data["id"],
                            league_season_id=league_season.id,
                            home_team_id=home_team.id,
                            away_team_id=away_team.id,
                            venue_id=venue.id if venue else None,
                            kickoff=datetime.fromisoformat(fixture_data["date"]),
                            timestamp=fixture_data["timestamp"],
                            timezone=fixture_data["timezone"] or "UTC",
                            round=league_data["round"] or "",
                            referee=fixture_data["referee"],
                            status_short=status["short"] or "",
                            status_long=status["long"] or "",
                            elapsed=status["elapsed"],
                            extra_time=status["extra"],
                            home_goals=goals_data["home"],
                            away_goals=goals_data["away"],
                            halftime_home=score_data["halftime"]["home"],
                            halftime_away=score_data["halftime"]["away"],
                            fulltime_home=score_data["fulltime"]["home"],
                            fulltime_away=score_data["fulltime"]["away"],
                            extratime_home=score_data["extratime"]["home"],
                            extratime_away=score_data["extratime"]["away"],
                            penalty_home=score_data["penalty"]["home"],
                            penalty_away=score_data["penalty"]["away"],
                        )
                    )

                    added += 1

            repository.commit()

            logger.success(f"Добавлено матчей: {added}")
            logger.info(f"Пропущено матчей: {skipped}")
            logger.warning(f"Пропущено из-за отсутствующих команд: {missing_teams}")

        except Exception:
            repository.rollback()
            logger.exception("Ошибка импорта матчей")
            raise
