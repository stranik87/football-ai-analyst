import time


from app.api.fixture_statistics import FixtureStatisticsService
from app.core.logger import logger
from app.database.session import get_db
from app.importers.base_importer import BaseImporter
from app.models.fixture import Fixture
from app.models.fixture_team_statistics import FixtureTeamStatistics
from app.models.team import Team
from app.repositories.fixture_team_statistics_repository import (
    FixtureTeamStatisticsRepository,
)


class FixtureTeamStatisticsImporter(BaseImporter):
    """
    Импорт статистики команд по матчам.
    """

    REQUEST_DELAY = 2
    LIMIT = 10

    def import_data(self):
        api = FixtureStatisticsService()
        db = next(get_db())

        repository = FixtureTeamStatisticsRepository(db)

        fixtures = (
            db.query(Fixture)
            .filter(Fixture.status_short == "FT")
            .limit(self.LIMIT)
            .all()
        )

        added = 0
        skipped = 0
        missing_teams = 0

        logger.info(f"Найдено завершённых матчей: {len(fixtures)}")

        try:
            for fixture in fixtures:
                logger.info(f"Получение статистики матча: api_id={fixture.api_id}")

                data = api.get_by_fixture_id(fixture.api_id)

                time.sleep(self.REQUEST_DELAY)

                if not data or not data.get("response"):
                    logger.warning(f"Статистика не получена: api_id={fixture.api_id}")
                    continue

                for item in data["response"]:
                    team_api_id = item["team"]["id"]

                    team = db.query(Team).filter_by(api_id=team_api_id).first()

                    if not team:
                        missing_teams += 1
                        continue

                    existing = repository.get(
                        fixture_id=fixture.id,
                        team_id=team.id,
                    )

                    if existing:
                        skipped += 1
                        continue

                    values = {
                        statistic["type"]: statistic["value"]
                        for statistic in item["statistics"]
                    }

                    repository.add(
                        FixtureTeamStatistics(
                            fixture_id=fixture.id,
                            team_id=team.id,
                            shots_on_goal=self._to_int(values.get("Shots on Goal")),
                            shots_off_goal=self._to_int(values.get("Shots off Goal")),
                            total_shots=self._to_int(values.get("Total Shots")),
                            blocked_shots=self._to_int(values.get("Blocked Shots")),
                            shots_inside_box=self._to_int(
                                values.get("Shots insidebox")
                            ),
                            shots_outside_box=self._to_int(
                                values.get("Shots outsidebox")
                            ),
                            fouls=self._to_int(values.get("Fouls")),
                            corner_kicks=self._to_int(values.get("Corner Kicks")),
                            offsides=self._to_int(values.get("Offsides")),
                            ball_possession=self._to_float(
                                values.get("Ball Possession")
                            ),
                            yellow_cards=self._to_int(values.get("Yellow Cards")),
                            red_cards=self._to_int(values.get("Red Cards")),
                            goalkeeper_saves=self._to_int(
                                values.get("Goalkeeper Saves")
                            ),
                            total_passes=self._to_int(values.get("Total passes")),
                            passes_accurate=self._to_int(values.get("Passes accurate")),
                            passes_percentage=self._to_float(values.get("Passes %")),
                            expected_goals=self._to_float(values.get("expected_goals")),
                            goals_prevented=self._to_float(
                                values.get("goals_prevented")
                            ),
                        )
                    )

                    added += 1

            repository.commit()

            logger.success(f"Добавлено записей статистики: {added}")
            logger.info(f"Пропущено записей статистики: {skipped}")
            logger.warning(f"Команды не найдены: {missing_teams}")

        except Exception:
            repository.rollback()
            logger.exception("Ошибка импорта статистики матчей")
            raise

    @staticmethod
    def _to_int(value):
        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_float(value):
        if value is None:
            return None

        if isinstance(value, str):
            value = value.replace("%", "").strip()

        try:
            return float(value)
        except (TypeError, ValueError):
            return None
