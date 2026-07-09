from app.api.client import FootballAPIClient


class LeagueSeasonService:
    """
    Сервис получения сезонов лиг из API.
    """

    def __init__(self):
        self.client = FootballAPIClient()

    def get_league_seasons(self):
        """
        Возвращает список лиг вместе с сезонами.
        """
        return self.client.get("leagues")