from app.api.client import FootballAPIClient
from app.api.endpoints import APIEndpoint


class TeamService:
    """
    Сервис для работы с командами.
    """

    def __init__(self):
        self.api = FootballAPIClient()

    def get_teams(self, league: int, season: int):
        """
        Получить команды лиги за указанный сезон.
        """
        return self.api.get(
            APIEndpoint.TEAMS,
            {
                "league": league,
                "season": season,
            }
        )

    def get_team_by_id(self, team_id: int):
        """
        Получить информацию о команде по ID.
        """
        return self.api.get(
            APIEndpoint.TEAMS,
            {
                "id": team_id,
            }
        )