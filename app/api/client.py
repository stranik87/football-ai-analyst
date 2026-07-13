import time
from typing import Any

import requests
from requests.exceptions import RequestException

from app.core.logger import logger
from config import Config


class APIDailyLimitError(Exception):
    """
    Дневной лимит запросов API-Football исчерпан.
    """


class FootballAPIClient:
    BASE_URL = "https://v3.football.api-sports.io"

    MAX_RETRIES = 3
    BASE_RETRY_DELAY = 5
    MAX_RETRY_DELAY = 60

    def __init__(self):
        self.session = requests.Session()

        self.session.headers.update(
            {
                "x-apisports-key": Config.API_FOOTBALL_KEY,
            }
        )

    def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                logger.info(
                    f"GET {url} | попытка "
                    f"{attempt}/{self.MAX_RETRIES}"
                )

                response = self.session.get(
                    url,
                    params=params,
                    timeout=30,
                )

                if response.status_code == 429:
                    self._handle_rate_limit(
                        response=response,
                        attempt=attempt,
                    )
                    continue

                response.raise_for_status()

                data = response.json()
                api_errors = data.get("errors")

                if api_errors:
                    if self._is_daily_limit_error(api_errors):
                        raise APIDailyLimitError(
                            "Дневной лимит API-Football исчерпан"
                        )

                    logger.error(
                        f"API RESPONSE ERROR: {api_errors}"
                    )
                    return None

                return data

            except APIDailyLimitError:
                logger.error(
                    "Дневной лимит API-Football исчерпан. "
                    "Импорт необходимо остановить."
                )
                raise

            except RequestException as error:
                logger.error(f"API ERROR: {error}")

                if attempt >= self.MAX_RETRIES:
                    break

                delay = min(
                    self.BASE_RETRY_DELAY * attempt,
                    self.MAX_RETRY_DELAY,
                )

                logger.warning(
                    f"Повтор запроса через {delay} сек."
                )

                time.sleep(delay)

            except ValueError as error:
                logger.error(
                    f"API вернул некорректный JSON: {error}"
                )
                return None

        logger.error(
            "Не удалось выполнить запрос после "
            f"{self.MAX_RETRIES} попыток: {url}"
        )

        return None

    def _handle_rate_limit(
        self,
        response: requests.Response,
        attempt: int,
    ) -> None:
        api_errors = self._get_response_errors(response)

        if self._is_daily_limit_error(api_errors):
            raise APIDailyLimitError(
                "Дневной лимит API-Football исчерпан"
            )

        retry_after = response.headers.get("Retry-After")

        try:
            delay = int(retry_after) if retry_after else 0
        except (TypeError, ValueError):
            delay = 0

        if delay <= 0:
            delay = self.BASE_RETRY_DELAY * attempt

        delay = min(delay, self.MAX_RETRY_DELAY)

        logger.warning(
            f"Временный лимит API. Повтор через {delay} сек."
        )

        time.sleep(delay)

    @staticmethod
    def _get_response_errors(
        response: requests.Response,
    ) -> Any:
        try:
            data = response.json()
        except ValueError:
            return response.text

        return data.get("errors", data)

    @staticmethod
    def _is_daily_limit_error(errors: Any) -> bool:
        if not errors:
            return False

        error_text = str(errors).lower()

        daily_limit_messages = (
            "request limit for the day",
            "requests limit for the day",
            "reached the request limit",
            "daily request limit",
        )

        return any(
            message in error_text
            for message in daily_limit_messages
        )