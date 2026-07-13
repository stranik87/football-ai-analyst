import argparse

from app.core.logger import logger
from app.database.database import create_database

from app.importers.fixture_importer import FixtureImporter
from app.importers.fixture_team_statistics_importer import (
    FixtureTeamStatisticsImporter,
)
from app.importers.league_importer import LeagueImporter
from app.importers.league_season_importer import LeagueSeasonImporter
from app.importers.team_importer import TeamImporter


IMPORTERS = {
    "leagues": ("Импорт лиг", LeagueImporter),
    "seasons": ("Импорт сезонов", LeagueSeasonImporter),
    "teams": ("Импорт команд", TeamImporter),
    "fixtures": ("Импорт матчей", FixtureImporter),
    "statistics": (
        "Импорт статистики матчей",
        FixtureTeamStatisticsImporter,
    ),
}


def run_importer(name: str) -> bool:
    """
    Запуск одного импортёра.
    """
    title, importer_class = IMPORTERS[name]

    logger.info(f"{title}...")

    return importer_class().run()


def run_all_importers() -> bool:
    """
    Последовательный запуск всех импортёров.
    """
    for name in IMPORTERS:
        success = run_importer(name)

        if not success:
            logger.warning(
                "Цепочка импортёров остановлена из-за лимита API."
            )
            return False

    return True


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Football AI Analyst"
    )

    parser.add_argument(
        "command",
        choices=[*IMPORTERS.keys(), "all"],
        nargs="?",
        default="all",
        help="Какой импортёр запустить",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    logger.info("Создание базы данных...")
    create_database()

    if args.command == "all":
        success = run_all_importers()
    else:
        success = run_importer(args.command)

    if success:
        logger.success("Работа завершена.")
    else:
        logger.warning("Работа остановлена раньше времени.")


if __name__ == "__main__":
    main()