from app.models.team import Team


class TeamRepository:
    """
    Репозиторий команд.
    """

    def __init__(self, db):
        self.db = db

    def get_by_api_id(self, api_id: int):
        return (
            self.db.query(Team)
            .filter_by(api_id=api_id)
            .first()
        )

    def get_all(self):
        return self.db.query(Team).all()

    def add(self, team: Team):
        self.db.add(team)

    def commit(self):
        self.db.commit()

    def rollback(self):
        self.db.rollback()