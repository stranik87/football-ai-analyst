from dotenv import load_dotenv
import os

# Загружаем переменные окружения из .env
load_dotenv()


class Config:
    """
    Центральная конфигурация проекта.
    Все настройки приложения берутся только отсюда.
    """

    # -----------------------------
    # Информация о проекте
    # -----------------------------
    APP_NAME = os.getenv("APP_NAME", "Football AI Analyst")
    APP_VERSION = os.getenv("APP_VERSION", "1.0.0")

    # -----------------------------
    # Режим работы
    # -----------------------------
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    # -----------------------------
    # API Football
    # -----------------------------
    API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")

    API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"

    # -----------------------------
    # База данных
    # -----------------------------
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "sqlite:///football.db"
    )

    # -----------------------------
    # Логирование
    # -----------------------------
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # -----------------------------
    # Часовой пояс
    # -----------------------------
    TIMEZONE = os.getenv("TIMEZONE", "Asia/Tashkent")