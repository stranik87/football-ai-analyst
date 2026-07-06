from abc import ABC, abstractmethod

from app.core.logger import logger


class BaseImporter(ABC):
    """
    Базовый класс для всех импортёров.
    """

    def run(self):
        logger.info(f"Запуск импортёра: {self.__class__.__name__}")
        self.import_data()
        logger.success(f"Импорт завершён: {self.__class__.__name__}")

    @abstractmethod
    def import_data(self):
        """
        Метод должен быть реализован в наследнике.
        """
        pass