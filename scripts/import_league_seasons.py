from app.core.logger import logger
from app.importers.league_season_importer import LeagueSeasonImporter


def main() -> None:
    logger.info("Запуск импорта сезонов лиг...")

    success = LeagueSeasonImporter().run()

    if success:
        logger.success("Импорт сезонов лиг завершён.")
    else:
        logger.error("Импорт сезонов лиг завершился с ошибкой.")


if __name__ == "__main__":
    main()