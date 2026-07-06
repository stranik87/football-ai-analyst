from pathlib import Path

from loguru import logger

from config import Config


# Создаем папку logs, если ее нет
Path("logs").mkdir(exist_ok=True)

# Удаляем стандартный вывод Loguru
logger.remove()

# Вывод в консоль
logger.add(
    sink=lambda msg: print(msg, end=""),
    level=Config.LOG_LEVEL,
    colorize=True,
)

# Запись в файл
logger.add(
    "logs/football_ai.log",
    level=Config.LOG_LEVEL,
    rotation="10 MB",
    retention="30 days",
    encoding="utf-8",
)

__all__ = ["logger"]