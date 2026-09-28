import argparse

from app.core.logger import logger
from app.importers.fixture_team_statistics_importer import (
    FixtureTeamStatisticsImporter,
)


def main():
    parser = argparse.ArgumentParser(
        description="Импорт статистики матчей"
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Сезон для импорта, например 2026",
    )
    args = parser.parse_args()

    logger.info(
        f"Запуск импорта статистики матчей: season={args.season}"
    )

    importer = FixtureTeamStatisticsImporter(
        season=args.season
    )
    importer.run()

    logger.success("Проверка статистики завершена.")


if __name__ == "__main__":
    main()
