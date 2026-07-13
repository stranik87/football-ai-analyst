from app.api.client import FootballAPIClient
from app.api.endpoints import APIEndpoint


class FixtureService:
    """
    Сервис для работы с футбольными матчами.
    """

    def __init__(self):
        self.api = FootballAPIClient()

    def get_fixtures(self, league: int, season: int):
        """
        Получить матчи указанной лиги и сезона.
        """
        return self.api.get(
            APIEndpoint.FIXTURES,
            {
                "league": league,
                "season": season,
            },
        )

    def get_fixture_by_id(self, fixture_id: int):
        """
        Получить матч по ID API-Football.
        """
        return self.api.get(
            APIEndpoint.FIXTURES,
            {
                "id": fixture_id,
            },
        )