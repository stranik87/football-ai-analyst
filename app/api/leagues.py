from app.api.client import FootballAPIClient
from app.api.endpoints import APIEndpoint


class LeagueService:
    """
    Сервис для работы с футбольными лигами.
    """

    def __init__(self):
        self.api = FootballAPIClient()

    def get_current_leagues(self):
        """
        Получить все текущие лиги.
        """
        return self.api.get(
            APIEndpoint.LEAGUES,
            {"current": "true"}
        )

    def get_league_by_id(self, league_id: int):
        """
        Получить информацию о лиге по ID.
        """
        return self.api.get(
            APIEndpoint.LEAGUES,
            {"id": league_id}
        )

    def get_country_leagues(self, country: str):
        """
        Получить все лиги страны.
        """
        return self.api.get(
            APIEndpoint.LEAGUES,
            {"country": country}
        )