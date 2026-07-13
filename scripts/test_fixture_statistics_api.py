from pprint import pprint

from app.api.fixture_statistics import FixtureStatisticsService


def main():
    api = FixtureStatisticsService()

    data = api.get_by_fixture_id(1208021)

    if not data or not data.get("response"):
        print("Статистика матча не получена.")
        return

    pprint(data["response"])


if __name__ == "__main__":
    main()