from abc import ABC, abstractmethod

from app.api.client import APIDailyLimitError, FootballAPIClient
from app.core.logger import logger
from app.database.database import SessionLocal


class BaseImporter(ABC):
    """
    Базовый класс для всех импортёров.
    """

    def __init__(self):
        self.client = FootballAPIClient()
        self.session = SessionLocal()

    def run(self) -> bool:
        importer_name = self.__class__.__name__

        logger.info(f"Запуск импортёра: {importer_name}")

        try:
            self.import_data()
            self.session.commit()

            logger.success(f"Импорт завершён: {importer_name}")
            return True

        except APIDailyLimitError:
            self.session.rollback()

            logger.error(
                f"Импорт остановлен: {importer_name}. "
                "Дневной лимит API-Football исчерпан."
            )

            return False

        except Exception:
            self.session.rollback()

            logger.exception(
                f"Ошибка работы импортёра: {importer_name}"
            )

            raise

        finally:
            self.session.close()

    @abstractmethod
    def import_data(self) -> None:
        """
        Выполнить импорт данных.
        """
        raise NotImplementedError