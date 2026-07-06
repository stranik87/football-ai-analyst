from app.database.database import SessionLocal


def get_db():
    """
    Создает сессию базы данных.
    """
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()