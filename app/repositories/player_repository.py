from app.models.player import Player


class PlayerRepository:
    """
    Репозиторий футболистов.
    """

    def __init__(self, db):
        self.db = db

    def get_by_api_id(self, api_id: int):
        return (
            self.db.query(Player)
            .filter_by(api_id=api_id)
            .first()
        )

    def get_by_team_id(self, team_id: int):
        return (
            self.db.query(Player)
            .filter_by(team_id=team_id)
            .all()
        )

    def get_all(self):
        return self.db.query(Player).all()

    def add(self, player: Player):
        self.db.add(player)

    def commit(self):
        self.db.commit()

    def rollback(self):
        self.db.rollback()