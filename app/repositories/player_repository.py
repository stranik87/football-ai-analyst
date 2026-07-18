from app.models.player import Player


class PlayerRepository:
    """
    Репозиторий футболистов.
    """

    def __init__(self, db):
        self.db = db

    def get_by_api_id(
        self,
        api_id: int,
    ) -> Player | None:
        return (
            self.db.query(Player)
            .filter(Player.api_id == api_id)
            .first()
        )

    def get_by_team_id(
        self,
        team_id: int,
    ) -> list[Player]:
        return (
            self.db.query(Player)
            .filter(Player.team_id == team_id)
            .all()
        )

    def count_by_team_id(
        self,
        team_id: int,
    ) -> int:
        return (
            self.db.query(Player)
            .filter(Player.team_id == team_id)
            .count()
        )

    def get_all(self) -> list[Player]:
        return self.db.query(Player).all()

    def get_all_by_api_id(self) -> dict[int, Player]:
        """
        Загружает игроков из базы в словарь:

        {
            api_id: Player
        }

        Это исключает повторные INSERT одного игрока
        во время одного запуска импортёра.
        """

        players = self.db.query(Player).all()

        return {
            player.api_id: player
            for player in players
        }

    def add(
        self,
        player: Player,
    ) -> None:
        self.db.add(player)

    def commit(self) -> None:
        self.db.commit()

    def rollback(self) -> None:
        self.db.rollback()