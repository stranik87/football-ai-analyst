from app.core.logger import logger
from app.importers.standing_importer import StandingImporter


def main() -> None:
    logger.info("Запуск импорта турнирных таблиц...")

    success = StandingImporter().run()

    if success:
        logger.success(
            "Импорт турнирных таблиц завершён."
        )
    else:
        logger.error(
            "Импорт турнирных таблиц завершился с ошибкой."
        )


if __name__ == "__main__":
    main()