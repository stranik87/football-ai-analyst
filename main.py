from app.core.logger import logger
from app.database.database import create_database
from app.importers.league_importer import LeagueImporter
from app.importers.league_season_importer import LeagueSeasonImporter
from app.importers.team_importer import TeamImporter


def main():
    logger.info("Создание базы данных...")
    create_database()

    logger.info("Импорт лиг...")
    LeagueImporter().run()

    logger.info("Импорт сезонов...")
    LeagueSeasonImporter().run()

    logger.info("Импорт команд...")
    TeamImporter().run()

    logger.success("Работа завершена.")


if __name__ == "__main__":
    main()