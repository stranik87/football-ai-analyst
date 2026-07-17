import time

from sqlalchemy import func

from app.api.fixture_statistics import FixtureStatisticsService
from app.core.logger import logger
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

    Импорт выполняется небольшими пакетами.
    Каждый матч сохраняется отдельной транзакцией,
    чтобы при ошибке не потерять уже импортированные данные.
    """

    REQUEST_DELAY = 2
    LIMIT = 10

    FINISHED_STATUSES = (
        "FT",
        "AET",
        "PEN",
    )

    def import_data(self):
        api = FixtureStatisticsService()
        db = self.session

        repository = FixtureTeamStatisticsRepository(db)

        statistics_count = (
            db.query(
                FixtureTeamStatistics.fixture_id,
                func.count(
                    FixtureTeamStatistics.id
                ).label("statistics_count"),
            )
            .group_by(
                FixtureTeamStatistics.fixture_id
            )
            .subquery()
        )

        fixtures = (
            db.query(Fixture)
            .outerjoin(
                statistics_count,
                statistics_count.c.fixture_id
                == Fixture.id,
            )
            .filter(
                Fixture.status_short.in_(
                    self.FINISHED_STATUSES
                )
            )
            .filter(
                func.coalesce(
                    statistics_count.c.statistics_count,
                    0,
                )
                < 2
            )
            .order_by(
                Fixture.kickoff.asc()
            )
            .limit(self.LIMIT)
            .all()
        )

        total_added = 0
        total_skipped = 0
        total_missing_teams = 0
        completed_fixtures = 0
        empty_responses = 0

        logger.info(
            "Найдено завершённых матчей "
            f"без полной статистики: {len(fixtures)}"
        )

        for fixture in fixtures:
            try:
                logger.info(
                    "Получение статистики матча: "
                    f"id={fixture.id}, "
                    f"api_id={fixture.api_id}"
                )

                data = api.get_by_fixture_id(
                    fixture.api_id
                )

                if (
                    not data
                    or not data.get("response")
                ):
                    empty_responses += 1

                    logger.warning(
                        "Статистика не получена: "
                        f"api_id={fixture.api_id}"
                    )

                    time.sleep(self.REQUEST_DELAY)
                    continue

                fixture_added = 0
                fixture_skipped = 0
                fixture_missing_teams = 0

                for item in data["response"]:
                    team_data = item.get("team") or {}
                    team_api_id = team_data.get("id")

                    if team_api_id is None:
                        fixture_missing_teams += 1

                        logger.warning(
                            "В ответе отсутствует ID команды: "
                            f"fixture_api_id={fixture.api_id}"
                        )
                        continue

                    team = (
                        db.query(Team)
                        .filter(
                            Team.api_id == team_api_id
                        )
                        .first()
                    )

                    if not team:
                        fixture_missing_teams += 1

                        logger.warning(
                            "Команда не найдена в базе: "
                            f"team_api_id={team_api_id}, "
                            f"fixture_api_id={fixture.api_id}"
                        )
                        continue

                    existing = repository.get(
                        fixture_id=fixture.id,
                        team_id=team.id,
                    )

                    if existing:
                        fixture_skipped += 1
                        continue

                    statistics = (
                        item.get("statistics") or []
                    )

                    values = {
                        statistic.get("type"):
                        statistic.get("value")
                        for statistic in statistics
                        if statistic.get("type")
                    }

                    repository.add(
                        FixtureTeamStatistics(
                            fixture_id=fixture.id,
                            team_id=team.id,
                            shots_on_goal=self._to_int(
                                values.get(
                                    "Shots on Goal"
                                )
                            ),
                            shots_off_goal=self._to_int(
                                values.get(
                                    "Shots off Goal"
                                )
                            ),
                            total_shots=self._to_int(
                                values.get(
                                    "Total Shots"
                                )
                            ),
                            blocked_shots=self._to_int(
                                values.get(
                                    "Blocked Shots"
                                )
                            ),
                            shots_inside_box=self._to_int(
                                values.get(
                                    "Shots insidebox"
                                )
                            ),
                            shots_outside_box=self._to_int(
                                values.get(
                                    "Shots outsidebox"
                                )
                            ),
                            fouls=self._to_int(
                                values.get("Fouls")
                            ),
                            corner_kicks=self._to_int(
                                values.get(
                                    "Corner Kicks"
                                )
                            ),
                            offsides=self._to_int(
                                values.get("Offsides")
                            ),
                            ball_possession=self._to_float(
                                values.get(
                                    "Ball Possession"
                                )
                            ),
                            yellow_cards=self._to_int(
                                values.get(
                                    "Yellow Cards"
                                )
                            ),
                            red_cards=self._to_int(
                                values.get(
                                    "Red Cards"
                                )
                            ),
                            goalkeeper_saves=self._to_int(
                                values.get(
                                    "Goalkeeper Saves"
                                )
                            ),
                            total_passes=self._to_int(
                                values.get(
                                    "Total passes"
                                )
                            ),
                            passes_accurate=self._to_int(
                                values.get(
                                    "Passes accurate"
                                )
                            ),
                            passes_percentage=self._to_float(
                                values.get(
                                    "Passes %"
                                )
                            ),
                            expected_goals=self._to_float(
                                values.get(
                                    "expected_goals"
                                )
                            ),
                            goals_prevented=self._to_float(
                                values.get(
                                    "goals_prevented"
                                )
                            ),
                        )
                    )

                    fixture_added += 1

                repository.commit()

                total_added += fixture_added
                total_skipped += fixture_skipped
                total_missing_teams += (
                    fixture_missing_teams
                )
                completed_fixtures += 1

                logger.success(
                    "Матч обработан: "
                    f"api_id={fixture.api_id}, "
                    f"добавлено={fixture_added}, "
                    f"пропущено={fixture_skipped}, "
                    f"команд не найдено="
                    f"{fixture_missing_teams}"
                )

                time.sleep(self.REQUEST_DELAY)

            except Exception:
                repository.rollback()

                logger.exception(
                    "Ошибка импорта статистики матча: "
                    f"api_id={fixture.api_id}"
                )

                # Ошибка передаётся BaseImporter.
                # При исчерпании API-лимита импорт
                # корректно остановится, а уже сохранённые
                # матчи останутся в базе.
                raise

        logger.success(
            "Импорт статистики завершён. "
            f"Обработано матчей: {completed_fixtures}"
        )
        logger.info(
            f"Добавлено записей: {total_added}"
        )
        logger.info(
            f"Пропущено существующих: {total_skipped}"
        )
        logger.warning(
            f"Команды не найдены: {total_missing_teams}"
        )
        logger.warning(
            f"Пустых ответов API: {empty_responses}"
        )

    @staticmethod
    def _to_int(value):
        if value is None:
            return None

        if isinstance(value, str):
            value = value.replace("%", "").strip()

        try:
            return int(float(value))
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