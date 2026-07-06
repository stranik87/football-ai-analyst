from app.core.logger import logger
from app.database.init_db import init_database
from app.importers.league_importer import LeagueImporter


def main():
    logger.info("Создание базы данных...")
    init_database()

    logger.info("Импорт лиг...")
    LeagueImporter().run()

    logger.success("Работа завершена.")


if __name__ == "__main__":
    main()