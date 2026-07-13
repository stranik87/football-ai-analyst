from app.core.logger import logger
from app.importers.fixture_team_statistics_importer import (
    FixtureTeamStatisticsImporter,
)


def main():
    logger.info("Запуск импорта статистики одного матча...")
    FixtureTeamStatisticsImporter().run()
    logger.success("Проверка статистики завершена.")


if __name__ == "__main__":
    main()