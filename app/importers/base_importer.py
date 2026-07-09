from abc import ABC, abstractmethod

from app.api.client import FootballAPIClient
from app.core.logger import logger
from app.database.database import SessionLocal


class BaseImporter(ABC):
    """
    Базовый класс для всех импортёров.
    """

    def __init__(self):
        self.client = FootballAPIClient()
        self.session = SessionLocal()

    def run(self):
        logger.info(f"Запуск импортёра: {self.__class__.__name__}")

        try:
            self.import_data()
            self.session.commit()
            logger.success(f"Импорт завершён: {self.__class__.__name__}")

        except Exception as e:
            self.session.rollback()
            logger.exception(e)
            raise

        finally:
            self.session.close()

    @abstractmethod
    def import_data(self):
        pass