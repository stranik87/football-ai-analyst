from app.api.client import FootballAPIClient


class StandingsService:
    """
    Работа с турнирными таблицами API-Football.
    """

    ENDPOINT = "standings"

    def __init__(self):
        self.client = FootballAPIClient()

    def get_by_league_and_season(
        self,
        league_api_id: int,
        season: int,
    ):
        """
        Получить турнирную таблицу лиги за сезон.
        """
        return self.client.get(
            self.ENDPOINT,
            params={
                "league": league_api_id,
                "season": season,
            },
        )