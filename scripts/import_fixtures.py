from app.core.logger import logger
from app.importers.fixture_importer import FixtureImporter


def main():
    logger.info("Запуск отдельного импорта матчей...")
    FixtureImporter().run()
    logger.success("Импорт матчей завершён.")


if __name__ == "__main__":
    main()