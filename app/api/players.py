from app.api.client import FootballAPIClient
from app.api.endpoints import APIEndpoint


class PlayerService:
    """
    Сервис для работы с футболистами.
    """

    def __init__(self):
        self.api = FootballAPIClient()

    def get_players(
        self,
        team: int,
        season: int,
        page: int = 1,
    ):
        return self.api.get(
            APIEndpoint.PLAYERS,
            {
                "team": team,
                "season": season,
                "page": page,
            },
        )