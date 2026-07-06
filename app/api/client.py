import requests
from requests.exceptions import RequestException

from config import Config
from app.core.logger import logger


class FootballAPIClient:

    BASE_URL = "https://v3.football.api-sports.io"

    def __init__(self):

        self.session = requests.Session()

        self.session.headers.update({
            "x-apisports-key": Config.API_FOOTBALL_KEY
        })

    def get(self, endpoint, params=None):

        url = f"{self.BASE_URL}/{endpoint}"

        try:

            logger.info(f"GET {url}")

            response = self.session.get(
                url,
                params=params,
                timeout=30
            )

            response.raise_for_status()

            return response.json()

        except RequestException as e:

            logger.error(f"API ERROR: {e}")

            return None