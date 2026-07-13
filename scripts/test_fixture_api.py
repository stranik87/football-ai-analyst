from pprint import pprint

from app.api.fixtures import FixtureService


def main():
    api = FixtureService()

    data = api.get_fixtures(
        league=135,
        season=2024,
    )

    if not data or not data.get("response"):
        print("Матчи не получены.")
        return

    pprint(data["response"][0])


if __name__ == "__main__":
    main()