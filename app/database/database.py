from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import Config
from app.database.base import Base

# Создаем подключение к базе данных
engine = create_engine(
    Config.DATABASE_URL,
    echo=Config.DEBUG
)

# Фабрика сессий
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def create_database():
    """
    Создает все таблицы базы данных.
    """
    Base.metadata.create_all(bind=engine)