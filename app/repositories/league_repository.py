from sqlalchemy.orm import Session

from app.models.league import League


class LeagueRepository:

    def __init__(self, db: Session):
        self.db = db

    def get_by_api_id(self, api_id: int):
        return (
            self.db.query(League)
            .filter(League.api_id == api_id)
            .first()
        )

    def add(self, league: League):
        """
        Добавить объект в сессию.
        Пока ничего не сохраняет.
        """
        self.db.add(league)

    def commit(self):
        """
        Один commit для всех изменений.
        """
        self.db.commit()

    def rollback(self):
        """
        Откат изменений при ошибке.
        """
        self.db.rollback()

    def get_all(self):
        return self.db.query(League).all()