import time

import requests
from requests.exceptions import RequestException

from app.core.logger import logger
from config import Config


class FootballAPIClient:
    BASE_URL = "https://v3.football.api-sports.io"

    MAX_RETRIES = 3
    BASE_RETRY_DELAY = 5

    def __init__(self):
        self.session = requests.Session()

        self.session.headers.update(
            {
                "x-apisports-key": Config.API_FOOTBALL_KEY,
            }
        )

    def get(self, endpoint, params=None):
        url = f"{self.BASE_URL}/{endpoint}"

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                logger.info(f"GET {url} | попытка {attempt}/{self.MAX_RETRIES}")

                response = self.session.get(
                    url,
                    params=params,
                    timeout=30,
                )

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")

                    if retry_after:
                        delay = int(retry_after)
                    else:
                        delay = self.BASE_RETRY_DELAY * attempt

                    logger.warning(f"Лимит API. Повтор через {delay} сек.")

                    time.sleep(delay)
                    continue

                response.raise_for_status()

                data = response.json()

                api_errors = data.get("errors")

                if api_errors:
                    logger.error(f"API RESPONSE ERROR: {api_errors}")
                    return None

                return data

            except RequestException as e:
                logger.error(f"API ERROR: {e}")

                if attempt < self.MAX_RETRIES:
                    delay = self.BASE_RETRY_DELAY * attempt

                    logger.warning(f"Повтор запроса через {delay} сек.")

                    time.sleep(delay)
                    continue

                return None

        logger.error(
            f"Не удалось выполнить запрос после " f"{self.MAX_RETRIES} попыток: {url}"
        )

        return None
