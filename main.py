from pprint import pprint

from app.api.client import FootballAPIClient


def main():
    client = FootballAPIClient()

    data = client.get("leagues")

    pprint(data["response"][0])


if __name__ == "__main__":
    main()