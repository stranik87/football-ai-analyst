from app.core.logger import logger
from app.importers.player_importer import PlayerImporter


def main():
    logger.info("Запуск импорта игроков...")

    success = PlayerImporter().run()

    if success:
        logger.success("Импорт игроков завершён.")
    else:
        logger.error("Импорт игроков завершился с ошибкой.")


if __name__ == "__main__":
    main()