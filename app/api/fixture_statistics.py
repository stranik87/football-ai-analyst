from app.api.client import FootballAPIClient
from app.api.endpoints import APIEndpoint


class FixtureStatisticsService:
    """
    Сервис статистики матча.
    """

    def __init__(self):
        self.api = FootballAPIClient()

    def get_by_fixture_id(self, fixture_id: int):
        return self.api.get(
            APIEndpoint.FIXTURE_STATISTICS,
            {
                "fixture": fixture_id,
            },
        )