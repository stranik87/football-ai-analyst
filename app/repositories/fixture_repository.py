from app.models.fixture import Fixture


class FixtureRepository:
    """
    Репозиторий матчей.
    """

    def __init__(self, db):
        self.db = db

    def get_by_api_id(self, api_id: int):
        return (
            self.db.query(Fixture)
            .filter_by(api_id=api_id)
            .first()
        )

    def add(self, fixture: Fixture):
        self.db.add(fixture)

    def commit(self):
        self.db.commit()

    def rollback(self):
        self.db.rollback()